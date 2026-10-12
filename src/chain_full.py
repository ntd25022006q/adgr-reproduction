"""Post-extraction chain for the PAPER-CHUNKER rebuild (15,837 chunks).
Replaces the old chain which waited on the abandoned 11k extraction and built
an index from it. Stages run GPU-sequentially; Solar judging overlaps (separate
resource) and is launched inside each stage, never as a separate GPU job.
A stage that finds its output complete SKIPS (resume-safe). Any rc != 0 stops
the chain loudly — no silent gaps.
"""
import subprocess, sys, os, time, json
W = "/home/devlab1/adgr_workspace"
T = f"{W}/tmp"
P = "/home/devlab1/miniforge3/envs/unsloth/bin/python"
R = f"{W}/runs"
NEED_PAPER = 15837


def sh(cmd, log, env=None):
    print(f"=== RUN {cmd}", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    with open(log, "a") as f:
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=e)
    print(f"=== rc={r.returncode}", flush=True)
    return r.returncode


def wait_paper_extract():
    while True:
        try:
            n = sum(1 for r in json.load(open(f"{R}/extract_paper_full.json")).values()
                    if r.get("done"))
        except Exception:
            n = 0
        alive = any("extract_paper_full" in l
                    for l in os.popen("ps -eo cmd").read().splitlines() if "python" in l)
        print(f"paper extraction {n}/{NEED_PAPER} alive={alive}", flush=True)
        if n >= NEED_PAPER * 0.99 and not alive:
            return n
        if not alive and n < NEED_PAPER * 0.9:
            print("paper extraction died early; relaunching", flush=True)
            sh(f"cd {T} && setsid nohup {P} -u extract_paper_full.py "
               f"> {R}/extract_paper_full.log 2>&1 < /dev/null &", f"{R}/chain.log")
        time.sleep(300)


def free_ollama():
    for m in ["qwen3:8b", "qwen2.5:14b", "qwen3:14b", "qwen3:1.7b"]:
        os.system(f'curl -s --max-time 60 http://localhost:11434/api/generate '
                  f'-d "{{\\"model\\":\\"{m}\\",\\"keep_alive\\":0}}" > /dev/null 2>&1')
    print("ollama gen models unloaded", flush=True)


def stop_vllm():
    import signal
    for l in os.popen("ps -eo pid,cmd").read().splitlines():
        if "vllm.entrypoints.openai.api_server" in l:
            try:
                os.kill(int(l.split()[0]), signal.SIGTERM)
                print("stopped vLLM", l.split()[0], flush=True)
            except Exception as e:
                print("vllm kill err", e, flush=True)
    time.sleep(15)


n = wait_paper_extract()
print("paper extraction done:", n, flush=True)
free_ollama()
time.sleep(10)

stages = [
    # Tab.1: our KG index + quality metrics (DSER/SCR/LCR need our types)
    ("kg_build_paper", f"{P} -u {T}/kg_build_paper.py", f"{R}/kg_build_paper.log", None),
    # Tab.3 + Tab.4 ADGR + Tab.7 full arm, author index, full ctx, 14B gen
    # Tab.3 + Tab.4 ADGR + Tab.7 full arm, author index, full ctx, 14B gen
    ("e2e_v3", f"{P} -u {T}/e2e_v3.py", f"{R}/e2e_v3.log",
     {"ADGR_DATA": f"{W}/data", "CKPT": f"{R}/e2e_v3.json",
      "CK_ARMS": f"{R}/abl_v3_arms.json"}),
    # Tab.7: gen+judge the 4 non-full arms (e2e_v3 only gens full)
    ("ablation_gen", f"{P} -u {T}/ablation_gen.py", f"{R}/ablation_gen.log", None),
    # Tab.4/5/6 baselines need 32k ctx (41k-char native ctx); restart vLLM.
    # vllm_restart.py carries all fixes: --enforce-eager, wait_gpu_release,
    # wait_for_headroom (waits out external-job VRAM spikes instead of OOM).
    ("vllm_32k", f"{P} -u {T}/vllm_restart.py", f"{R}/vllm_32k.log",
     {"VLLM_UTIL": "0.72", "VLLM_LEN": "32768", "VLLM_LOG": f"{R}/vllm32k.log",
      "VLLM_NEED_FREE_MB": "17500"}),
    ("baselines_gen", f"{P} -u {T}/baselines_gen.py", f"{R}/baselines_gen.log",
     # CTX_CAP=60000: 7/600 baseline rows have ctx 60-68k chars (~27-34k tok);
     # uncapped they risk exceeding vLLM-32k max_model_len -> gen error ->
     # lost rows. 60k chars (~20-24k tok) is safe; only the 7 longest rows
     # lose ~10% tail context (immaterial for Tab.4/5/6).
     {"CTX_CAP": "60000"}),
    # Tab.4/5/6: re-judge the transient judge_fail rows (429 bursts under the
    # 100-req/min Solar cap) with the SAME answers - no regeneration.
    ("baselines_touchup", f"{P} -u {T}/baselines_touchup.py",
     f"{R}/baselines_touchup.log", None),
    # DOWNGRADE back to 12k before e2e_paper: 32k-vLLM (17GB) + ollama-kw (5GB)
    # + external (4.5GB) = 26.5GB > 24.5GB = guaranteed OOM inside e2e_paper.
    # util 0.55 (13.5G) + reranker-GPU 2.5G + CPU-keywords (OGEN_NUM_GPU=0,
    # see e2e_paper env) + embed 0.6G + ext 4.5G = 21.1G -> 3.4G slack.
    # util 0.50 + maxlen 8192: weights take 9.38G, leaving ~1.9G of KV -
    # enough for one 8192-token request (a 12288 request needs 2.25G and
    # failed with "not enough KV cache memory"). Nothing remaining needs
    # >8192: ADGR ctx is ~13k chars (~4.4k tok), baselines are already done,
    # ragas prompts are small. VRAM totals: fix_arms 12.3+5.5(kw GPU)+0.9+
    # 4.4(train)=23.1G; R11 (CPU kw) 12.3+2.5(rerank)+0.9+4.4=20.1G.
    ("vllm_12k", f"{P} -u {T}/vllm_restart.py", f"{R}/vllm_12k.log",
     {"VLLM_UTIL": "0.50", "VLLM_LEN": "8192", "VLLM_LOG": f"{R}/vllm12k.log",
      "VLLM_NEED_FREE_MB": "12800"}),
    # REDO the two Tab.7 arms R7 computed with broken ablation semantics
    # (reranker applied to the no_rerank arm; graph sources merged into the
    # dense_only pool). Paper Sec 3.6/5.8 definitions enforced. Runs without
    # the reranker, so VRAM stays at vLLM+ollama+ext = 23G.
    ("fix_arms", f"{P} -u {T}/fix_arms.py", f"{R}/fix_arms.log", None),
    # Tab.7: re-judge the few transient judge_fail rows in the two arms
    # fix_arms does not rewrite (ablation_gen redoes only those rows).
    ("tab7_touchup", f"{P} -u {T}/tab7_touchup.py && {P} -u {T}/ablation_gen.py",
     f"{R}/tab7_touchup.log", None),
    # Tab.3: recompute retrieval metrics from the CURRENT arms (the old
    # ir_single_gold/ir_metrics_v2 were computed from abl_n300v3.json, the
    # pre-fix retrieval with the 1500-char rerank cut).
    ("ir_recompute", f"{P} -u {T}/ir_recompute.py", f"{R}/ir_recompute.log", None),
    # E2E on OUR rebuilt paper-chunker index
    ("e2e_paper", f"{P} -u {T}/e2e_v3.py", f"{R}/e2e_paper.log",
     {"ADGR_DATA": f"{W}/data_paper", "CKPT": f"{R}/e2e_paper.json",
      "CK_ARMS": f"{R}/abl_paper_arms.json",
      # CPU keyword model (see adgr.ogen): keeps R11 at ~21G VRAM with the
      # reranker on GPU. Costs ~10-20s/question extra on CPU, still far
      # cheaper than a CPU reranker (which would add 10+ hours).
      "OGEN_NUM_GPU": "0"}),
    # Tab.4 (paper index): re-judge the transient judge_fail rows with the
    # row's own ctx_ids (no regeneration). Same 5-judge set as e2e_v3.
    ("paper_touchup", f"{P} -u {T}/paper_touchup.py", f"{R}/paper_touchup.log", None),
    # Tab.4 with the true RAGAS algorithms
    ("ragas_true", f"{P} -u {T}/ragas_true.py", f"{R}/ragas_true.log", None),
    # Tab.3 for the rebuilt paper-chunker index (after e2e_paper's arms exist).
    # GOLD_SINGLE must be recomputed on data_paper: chunk IDs differ from
    # data/, so reusing gold_single.json would score mismatched chunks.
    ("ir_recompute_paper", f"{P} -u {T}/ir_recompute.py",
     f"{R}/ir_recompute_paper.log",
     {"ARMS_SRC": "abl_paper_arms.json", "DATA": f"{W}/data_paper",
      "OUT_METRICS": "ir_metrics_paper.json",
      "OUT_SINGLE": "ir_single_gold_paper.json",
      "GOLD_SINGLE": "gold_single_paper.json"}),
    # Tab.4 (true RAGAS) for the rebuilt index. RAGAS_DATA switches the chunk
    # lookup to data_paper (R12 ragas_true uses data/ for e2e_v3.json).
    ("ragas_paper", f"{P} -u {T}/ragas_true.py", f"{R}/ragas_paper.log",
     {"RAGAS_SRC": f"{R}/e2e_paper.json", "RAGAS_OUT": f"{R}/ragas_paper.json",
      "RAGAS_DATA": f"{W}/data_paper"}),
]
def _kg_complete():
    try:
        s = json.load(open(f"{R}/kg_index_paper_stats.json"))
        return s.get("n_chunk") == 15837 and os.path.exists(f"{W}/data_paper/vdb_entities.json")
    except Exception:
        return False


def _e2e_v3_complete():
    # Only the checkpoint matters: the arms file is owned by fix_arms now
    # (it rebuilds the two broken ablation arms), so e2e_v3 must never be
    # re-run just because the arms file looks incomplete.
    try:
        d = json.load(open(f"{R}/e2e_v3.json"))
        ok = sum(1 for v in d.values()
                 if isinstance(v, dict) and v.get("done") and "error" not in v)
        return ok >= 300
    except Exception:
        return False


def _e2e_paper_complete():
    # Same ckpt-only rule: e2e_v3.py itself resumes the single error row when
    # run, so a restart must not pay the ~4-min index reload for 1 row.
    try:
        d = json.load(open(f"{R}/e2e_paper.json"))
        ok = sum(1 for v in d.values()
                 if isinstance(v, dict) and v.get("done"))
        return ok >= 300
    except Exception:
        return False


def _vllm_32k_ready():
    # Skip the 32k restart if a healthy 32k server is already up (e.g. booted
    # manually after an interrupted attempt). Deterministic: needs BOTH the
    # 32768 flag in the live cmdline AND a 200 /health.
    try:
        ps = os.popen("ps -eo cmd").read()
        if "vllm.entrypoints.openai.api_server" not in ps:
            return False
        if "max-model-len 32768" not in ps:
            return False
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8000/health",
                                    timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def _baselines_done():
    # 300 done rows in BOTH baseline checkpoints -> the 32k server's only
    # consumer (baselines_gen) has finished; no point waiting for 17.5GB of
    # headroom just to boot a 32k server nothing will use.
    try:
        for m in ["hybrid", "mix"]:
            d = json.load(open(f"{R}/e2e_base_{m}.json"))
            if sum(1 for v in d.values() if isinstance(v, dict) and v.get("done")) < 300:
                return False
        return True
    except Exception:
        return False


def _gate(name):
    """Post-stage contract check: a stage can exit rc=0 while its output is
    garbage (truncated arms file, empty metrics, half-written checkpoint) -
    and without gates that garbage silently poisons every downstream table
    (this is exactly how the 75-row arms and the stale Tab.3 survived for a
    day). A failed gate stops the chain LOUDLY (rc=99) instead. Thresholds
    allow a handful of genuinely-failed rows (transient 429s) but never a
    structural shortfall."""
    try:
        if name == "fix_arms":
            d = json.load(open(f"{R}/abl_gen.json"))
            for a in ("no_rerank", "dense_only", "dense_anchors", "no_edge_filter"):
                rows = d.get(a, {})
                n = sum(1 for r in rows.values() if r.get("done"))
                if n < 300:
                    return False, f"abl_gen[{a}] {n}/300 done"
            return True, "abl_gen 4x300 done"
        if name == "ir_recompute":
            for f in ("ir_metrics_v2.json", "ir_single_gold.json"):
                d = json.load(open(f"{R}/{f}"))
                if len(d) < 5:
                    return False, f"{f}: {len(d)} arms, want 5"
            return True, "Tab.3 metrics present (5 arms)"
        if name == "e2e_paper":
            d = json.load(open(f"{R}/e2e_paper.json"))
            ok = sum(1 for v in d.values() if isinstance(v, dict) and v.get("done"))
            if ok < 300:
                return False, f"e2e_paper {ok}/300"
            return True, "e2e_paper 300/300"
        if name == "ragas_true":
            d = json.load(open(f"{R}/ragas_true.json"))
            full = sum(1 for v in d.values() if isinstance(v, dict) and v)
            if full < 270:
                return False, f"ragas_true {full}/300 non-empty"
            return True, f"ragas_true {full}/300"
        if name == "ir_recompute_paper":
            for f in ("ir_metrics_paper.json", "ir_single_gold_paper.json"):
                d = json.load(open(f"{R}/{f}"))
                if len(d) < 5:
                    return False, f"{f}: {len(d)} arms"
            return True, "paper-index Tab.3 present"
        if name == "ragas_paper":
            d = json.load(open(f"{R}/ragas_paper.json"))
            full = sum(1 for v in d.values() if isinstance(v, dict) and v)
            if full < 270:
                return False, f"ragas_paper {full}/300"
            return True, f"ragas_paper {full}/300"
        if name == "standalone_local":
            d = json.load(open(f"{R}/standalone.json"))
            for m in ("qwen2.5-14b", "qwen3-14b"):
                n = sum(1 for v in d.values()
                        if isinstance(v, dict) and (v.get(m) or "").strip())
                if n < 290:
                    return False, f"standalone[{m}] {n}/300"
            return True, "standalone local models present"
        if name == "tab5_5run":
            d = json.load(open(f"{R}/tab5_5run_summary.json"))
            if len(d) < 10:
                return False, f"tab5 summary {len(d)}/10 methods"
            return True, "tab5 5-run summary complete"
    except Exception as e:
        return False, f"gate exception: {repr(e)[:120]}"
    return True, "no gate for this stage (rc governs)"


for name, cmd, log, env in stages:
    if name == "kg_build_paper" and _kg_complete():
        print("SKIP kg_build_paper: data_paper/ already complete (52,578 ent from full extraction)",
              flush=True)
        continue
    if name == "e2e_v3" and _e2e_v3_complete():
        print("SKIP e2e_v3: 300/300 done + 5 arms x 300 retrieval saved",
              flush=True)
        continue
    if name == "vllm_32k" and (_vllm_32k_ready() or _baselines_done()):
        print("SKIP vllm_32k: 32k server up OR baselines already complete "
              "(nothing left that needs 32k ctx)", flush=True)
        continue
    if name == "e2e_paper" and _e2e_paper_complete():
        print("SKIP e2e_paper: 300/300 rows done (error rows retried on demand)",
              flush=True)
        continue
    if name == "e2e_paper":
        # R11 runs keywords on CPU (OGEN_NUM_GPU=0), but ollama keeps the
        # qwen3:8b GPU instance (5.5G) resident until keep_alive expires. With
        # the reranker (2.5G) + vLLM-12k (12.3G) + external train.py (4.4G)
        # that would total 25.6G = OOM. Unload it first: 12.3+2.5+0.9+4.4=20.1G.
        free_ollama()
        time.sleep(5)
    rc = sh(cmd, log, env)
    if rc != 0:
        print(f"STAGE FAILED: {name} rc={rc}", flush=True)
        sys.exit(rc)
    g_ok, g_msg = _gate(name)
    print(f"GATE {name}: {'PASS' if g_ok else 'FAIL'} - {g_msg}", flush=True)
    if not g_ok:
        print(f"GATE FAILED: {name} - {g_msg} (stopping before garbage spreads)",
              flush=True)
        sys.exit(99)
    print(f"STAGE OK: {name}", flush=True)

print("=== GPU stages done; freeing vLLM for local-ollama Tab.5 models ===", flush=True)
stop_vllm()
free_ollama()
time.sleep(10)

tail = [
    # Tab.5 models #6-7 (need ~9GB each, only after vLLM stops)
    ("standalone_local", f"{P} -u {T}/standalone_local.py", f"{R}/standalone_local.log", None),
    # Tab.5 five-run protocol (Solar; paper Sec 5.3)
    ("tab5_5run", f"{P} -u {T}/tab5_5run.py", f"{R}/tab5_5run.log", None),
    ("FINAL", f"{P} -u {T}/final_merge.py", f"{R}/final_merge.log", None),
    # Figures from the merged report (CPU-only, matplotlib; never fails loud
    # data - it only plots what FINAL_v2.json contains).
    ("PLOTS", f"{P} -u {T}/plot_final.py", f"{R}/plot_final.log", None),
]
for name, cmd, log, env in tail:
    rc = sh(cmd, log, env)
    if rc != 0:
        print(f"STAGE FAILED: {name} rc={rc}", flush=True)
        sys.exit(rc)
    g_ok, g_msg = _gate(name)
    print(f"GATE {name}: {'PASS' if g_ok else 'FAIL'} - {g_msg}", flush=True)
    if not g_ok:
        print(f"GATE FAILED: {name} - {g_msg} (stopping before garbage spreads)",
              flush=True)
        sys.exit(99)
    print(f"STAGE OK: {name}", flush=True)
print("CHAIN COMPLETE", flush=True)
