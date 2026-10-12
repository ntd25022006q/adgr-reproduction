"""fix_arms.py - redo the two Tab.7 arms that were computed with broken
ablation semantics in R7, so they match paper Sec 3.6 / Sec 5.8:

  no_rerank  : rerank was applied to EVERY arm (e2e_v3 called the reranker
               unconditionally and adgr.retrieve never consumed the flag).
               Paper: "w/o Cross-Encoder Reranking: removes the reranking
               stage after multi-source evidence aggregation" -> top-10
               straight from the merged pool, in pool order.
  dense_only : the pool merged graph-derived sources (Eglob/Tg/Tv), i.e. it
               was still graph-guided. Paper: "Dense Retrieval Only: retrieves
               chunks using dense vector retrieval over the chunk index
               without graph-guided retrieval" -> pool = Ts (direct dense
               top-ns) only.

Same index (data/), same questions, same generator (vLLM 14B-AWQ, temp 0),
same five judge prompts and minimal-effort escalation as ablation_gen.py, so
the fixed rows are directly comparable with the untouched full/dense_anchors/
no_edge_filter arms. Fully checkpointed: phase 1 retrieval -> phase 2 gen+judge.
"""
import csv, json, os, re, sys, time, threading
sys.path.insert(0, "/home/devlab1/adgr_workspace/tmp")
os.environ.setdefault("ADGR_DATA", "/home/devlab1/adgr_workspace/data")
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from adgr import load_index, build_bm25, retrieve

R = "/home/devlab1/adgr_workspace/runs/"
D = "/home/devlab1/adgr_workspace/data/"
VLLM = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/home/devlab1/adgr_workspace/models/qwen25-awq"
CAP = int(os.environ.get("CTX_CAP", "30000"))
ARMS = {"no_rerank": {}, "dense_only": {"dense_only": True}}
_key = open(os.path.expanduser("~/.config/adgr/upstage_key")).read().strip()
_j = OpenAI(api_key=_key, base_url="https://api.upstage.ai/v1", timeout=300)
COUNT = {"supported", "total", "useful"}
P_F = ("Cho cau tra loi va nguc canh. Tach thanh menh de su that (toi da 10). "
       "Moi menh de 1 neu nguc canh ho tro truc tiep, 0 neu khong. "
       "Tra ve JSON {{\"supported\":s,\"total\":t}}.\nNguc canh:\n{ctx}\nCau tra loi:\n{ans}")
P_R = ("Cau hoi: {q}\nCau tra loi: {ans}\nDo lien quan 0-1. Tra ve JSON {{\"relevancy\":x}}.")
P_C = ("Cau hoi: {q}\nDap an mau: {ref}\nNguc canh:\n{ctx}\n"
       "Ty le y chinh co mat 0-1. Tra ve JSON {{\"recall\":x}}.")
P_P = ("Cau hoi: {q}\nNguc canh:\n{ctx}\nMoi doan 1 neu lien quan truc tiep, 0 neu khong. "
       "Tra ve JSON {{\"useful\":u,\"total\":10}}.")
P_4 = ("Ban la chuyen gia y khoa. Cham 0-1: Comprehensiveness, Diversity, Empowerment, "
       "Directness. Chi gia tri [0,1].\nCau hoi: {q}\nDap an mau: {ref}\nCan cham: {ans}\n"
       'JSON {{"comp":x,"div":x,"emp":x,"dir":x}}.')
GEN = ("Tra loi cau hoi y khoa tieng Viet CHI dua tren nguc canh duoi day. Ngan gon, chinh xac.\n\n"
       "Nguc canh:\n{ctx}\n\nCau hoi: {q}\nTra loi:")


def kv(j, k):
    m = re.search(r'["\']' + k + r'["\']\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', j or "")
    return float(m.group(1)) if m else None


def _fatal_api(e):
    t = re.sub(r"[^a-z]", "", repr(e).lower())
    return any(p in t for p in ("suspended", "insufficient", "invalidapikey",
                                "unauthorized", "accountdeactivated", "billing",
                                "paymentrequired", "outofcredit", "permissiondenied",
                                "zerobalance"))


def judge(prompt, keys):
    import os as _os
    p = prompt + "\nTra loi HOAN TOAN bang tieng Viet. Chi tra ve JSON."
    for eff in ("minimal", "minimal", "minimal", "low"):
        try:
            t = _j.chat.completions.create(
                model="solar-pro4", messages=[{"role": "user", "content": p}],
                reasoning_effort=eff, max_tokens=6000).choices[0].message.content or ""
            v = [(k, kv(t, k)) for k in keys]
            if all(x is not None and (k in COUNT or 0 <= x <= 1) for k, x in v):
                return t, True
        except Exception as e:
            if _fatal_api(e):
                print(f"FATAL solar billing/auth: {repr(e)[:150]} - aborting; "
                      f"checkpoint preserved.", flush=True)
                _os._exit(42)
            time.sleep(6)
    return "", False


def gen(q, ctx):
    import urllib.request
    body = {"model": MODEL, "messages": [{"role": "user",
                                         "content": GEN.format(ctx=ctx[:CAP], q=q)}],
            "temperature": 0, "max_tokens": 400}
    for _ in range(3):
        try:
            req = urllib.request.Request(VLLM, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                t = (json.load(r)["choices"][0]["message"]["content"] or "").strip()
            if len(t) >= 20:
                return t
        except Exception:
            time.sleep(8)
    return ""


def slug(u):
    m = re.search(r"/([^/]+)/?$", u or "")
    return m.group(1) if m else ""


print("loading index...", flush=True)
IDX = load_index()
eidx = list(range(len(IDX["edata"])))
bm25 = build_bm25([(i, r.get("entity_name", "") + " " + r.get("content", ""))
                   for i, r in enumerate(IDX["edata"])])
id2pos = {c["__id__"]: i for i, c in enumerate(IDX["cdata"])}
KEY = "url" if "url" in IDX["cdata"][0] else "file_path"
slug2chunks = {}
for c in IDX["cdata"]:
    slug2chunks.setdefault(str(c.get(KEY, "")).replace(".txt", ""), []).append(c["__id__"])
rows = list(csv.DictReader(open(D + "q300.csv", encoding="utf-8")))
print("index loaded;", len(rows), "rows", flush=True)

# ---------- load / repair checkpoints ----------
CK_ARMS = R + "abl_v3_arms.json"
try:
    armrows = json.load(open(CK_ARMS))["arms"]
except Exception:
    armrows = {}
try:
    out = json.load(open(R + "abl_gen.json"))
except Exception:
    out = {}
for a in ARMS:
    out.setdefault(a, {})
    armrows.setdefault(a, [])
# IDEMPOTENCY (bug #37 fix): the block below used to WIPE both arms on every
# run. Correct exactly once (clearing the buggy R7 rows); on every rerun it
# silently discarded up to 600 completed rows and redid them (~1h + ~$1 Solar
# wasted on chain-13's rerun). The buggy rows are long rebuilt, so reruns now
# RESUME via arm_needs()/todo2. The one-time migration backups are kept.
import shutil
import os as _os
if not _os.path.exists(R + "abl_gen.json.pre_fixarms_bak"):
    shutil.copy(R + "abl_gen.json", R + "abl_gen.json.pre_fixarms_bak")
if not _os.path.exists(R + "abl_v3_arms.json.pre_fixarms_bak"):
    shutil.copy(R + "abl_v3_arms.json", R + "abl_v3_arms.json.pre_fixarms_bak")


def arm_needs(qi):
    for a in ARMS:
        arr = armrows.get(a, [])
        if len(arr) <= qi or not arr[qi].get("top10"):
            return True
    return False


def save_arms():
    # abl_v3_arms.json wraps the arms in an "arms" key - write the wrapper,
    # otherwise ablation_gen/e2e_v3 (json.load(...)["arms"]) would KeyError.
    json.dump({"arms": armrows}, open(CK_ARMS, "w"), ensure_ascii=False)


def save_gen():
    json.dump(out, open(R + "abl_gen.json", "w"), ensure_ascii=False)


# ---------- PHASE 1: retrieval (no rerank for both arms) ----------
todo = [qi for qi in range(len(rows)) if arm_needs(qi)]
print(f"[phase1] retrieval todo {len(todo)}/{len(rows)}", flush=True)
for n, qi in enumerate(todo):
    row = rows[qi]
    for name, abl in ARMS.items():
        while len(armrows[name]) <= qi:
            armrows[name].append({})
        if armrows[name][qi].get("top10"):
            continue
        r = retrieve(IDX, bm25, eidx, row["question"], ablation=abl)
        dids = [c for c in r["pool"] if c in id2pos]
        top10 = dids[:10]           # no rerank: pool order (dense order for dense_only)
        g_ = set(slug2chunks.get(slug(row.get("article_url")), []))
        armrows[name][qi] = {"hit": len(set(top10) & g_) > 0 if g_ else None,
                             "pool": len(r["pool"]), "edges": r["n_edges"],
                             "ent": r["n_entities"], "ret_s": r["ret_s"],
                             "t_kw": r["t_kw"], "anchors": r["anchors"][:5],
                             "top10": top10}
    if (n + 1) % 25 == 0:
        save_arms()
        print(f"  [phase1] {n+1}/{len(todo)}", flush=True)
save_arms()
print("[phase1] retrieval complete", flush=True)

# ---------- PHASE 2: gen + judge (ablation_gen-identical prompts) ----------
IDXC = {c["__id__"]: c["content"] for c in
        json.load(open(D + "vdb_chunks.json"))["data"]}
lock = threading.Lock()
for name in ARMS:
    todo2 = [qi for qi in range(len(rows))
             if not out[name].get(str(qi), {}).get("done")]
    print(f"[{name}] phase2 todo {len(todo2)}/{len(rows)}", flush=True)

    def one(qi, _nm=name):
        if out[_nm].get(str(qi), {}).get("done"):
            return
        arr = armrows.get(_nm, [])
        if len(arr) <= qi or not arr[qi].get("top10"):
            return
        top = arr[qi]["top10"]
        docs = [IDXC[c] for c in top if c in IDXC]
        ctx = "\n\n".join(docs)
        q = rows[qi]["question"]
        a = gen(q, ctx)
        if len(a.strip()) < 20:
            with lock:
                out[_nm][str(qi)] = {"gen_failed": True, "done": True}
            return
        jobs = [("faith", P_F.format(ctx=ctx[:CAP], ans=a[:1500]), ["supported", "total"]),
                ("rel", P_R.format(q=q, ans=a[:1500]), ["relevancy"]),
                ("crec", P_C.format(q=q, ref=rows[qi]["answer"][:1200], ctx=ctx[:CAP]), ["recall"]),
                ("cprec", P_P.format(q=q, ctx="\n".join(
                    f"[{i}] {t}" for i, t in enumerate(docs))[:CAP]), ["useful"]),
                ("dim4", P_4.format(q=q, ref=rows[qi]["answer"][:1200], ans=a[:1500]),
                 ["comp", "div", "emp", "dir"])]
        res = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            for fu, nm in {ex.submit(judge, p, k): n for n, p, k in jobs}.items():
                try:
                    res[fu] = (nm, fu.result())
                except Exception:
                    res[fu] = (nm, ("", False))
        d = {v[0]: v[1] for v in res.values()}
        with lock:
            out[_nm][str(qi)] = {"ans": a[:2000], "faith": d["faith"][0][:300],
                                 "rel": d["rel"][0][:150], "crec": d["crec"][0][:150],
                                 "cprec": d["cprec"][0][:200], "dim4": d["dim4"][0][:400],
                                 "all_ok": all(d[x][1] for x in
                                               ["faith", "rel", "crec", "cprec", "dim4"]),
                                 "done": True}
            if len(out[_nm]) % 25 == 0:
                save_gen()
                print(f"  [{_nm}] {len(out[_nm])}/{len(rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, todo2))
    save_gen()
    print(f"[{name}] done {len(out[name])}", flush=True)
print("fix_arms complete: no_rerank + dense_only recomputed per paper Sec 3.6/5.8")
