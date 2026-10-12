"""Tab.5 five-run protocol (paper Sec 5.3: 'averaged across dimensions and five
independent runs', judge temp 0.0). Judges every (row, method) 5 times with
solar-pro4 and stores all runs + mean. Covers the 7 standalone models AND the 3
GraphRAG methods (Hybrid/Mix answers from baselines_gen, ADGR from e2e_v3).
Cost: ~10 methods x 300 rows x 5 runs = 15,000 calls. Checkpointed per method.
Run LAST per plan (1-run numbers already reported)."""
import json, time, csv, os, statistics as st
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
R = "/home/devlab1/adgr_workspace/runs/"
_key = open(os.path.expanduser("~/.config/adgr/upstage_key")).read().strip()
_j = OpenAI(api_key=_key, base_url="https://api.upstage.ai/v1", timeout=300)
P = ("Ban la chuyen gia y khoa. Cham 0-1: Comprehensiveness, Diversity, Empowerment, "
     "Directness. Chi [0,1]. Tra loi HOAN TOAN bang tieng Viet. Chi JSON.\n"
     "Cau hoi: {q}\nDap an mau: {ref}\nCan cham: {ans}\n"
     'JSON {{"comp":x,"div":x,"emp":x,"dir":x}}.')
DIMS = ["comp", "div", "emp", "dir"]
import sys
sys.path.insert(0, "/home/devlab1/adgr_workspace/tmp")
from eval_utils import kv
SA = ["qwen-local", "ultra", "super", "glm", "cf-llama", "qwen2.5-14b", "qwen3-14b"]
sa = json.load(open(R + "standalone.json"))
rows = list(csv.DictReader(open("/home/devlab1/adgr_workspace/data/q300.csv")))
RA = {}
for m, f in [("hybrid", "e2e_base_hybrid.json"), ("mix", "e2e_base_mix.json")]:
    try:
        RA[m] = {int(k): v.get("ans", "") for k, v in json.load(open(R + f)).items()}
    except Exception:
        RA[m] = {}
try:
    RA["adgr"] = {int(k): v.get("ans", "") for k, v in json.load(open(R + "e2e_v3.json")).items()}
except Exception:
    RA["adgr"] = {}
METHODS = [(m, "sa") for m in SA] + [(m, "rag") for m in ["hybrid", "mix", "adgr"]]
OUT = R + "tab5_5run.json"
try:
    out = json.load(open(OUT))
except Exception:
    out = {}
print(f"methods={len(METHODS)} resume={sum(len(v) for v in out.values())}", flush=True)


def _fatal_api(e):
    import re as _re
    t = _re.sub(r"[^a-z]", "", repr(e).lower())
    return any(p in t for p in ("suspended", "insufficient", "invalidapikey",
                                "unauthorized", "accountdeactivated", "billing",
                                "paymentrequired", "outofcredit", "permissiondenied",
                                "zerobalance"))


def judge(q, ref, ans):
    # Effort escalation: minimal (~116 out-tok) x2, then low fallback (28x saving).
    import os as _os
    last = ""
    for eff in ("minimal", "minimal", "low"):
        try:
            last = _j.chat.completions.create(
                model="solar-pro4",
                messages=[{"role": "user", "content": P.format(q=q, ref=ref[:1200], ans=ans[:1500])}],
                reasoning_effort=eff, max_tokens=6000).choices[0].message.content or ""
            v = [kv(last, k) for k in DIMS]
            if all(x is not None and 0 <= x <= 1 for x in v):
                return [kv(last, k) for k in DIMS]
        except Exception as e:
            if _fatal_api(e):
                print(f"FATAL solar billing/auth: {repr(e)[:150]} - aborting, "
                      f"checkpoint preserved, refill credit and resume.", flush=True)
                _os._exit(42)
            time.sleep(6)
    return [None] * 4


lock = __import__("threading").Lock()
for m, kind in METHODS:
    out.setdefault(m, {})
    todo = []
    for qi in range(len(rows)):
        ans = (sa.get(str(qi), {}).get(m, "") if kind == "sa" else RA.get(m, {}).get(qi, "")) or ""
        if not ans.strip():
            continue
        have = [r for r in out[m].get(str(qi), {}).get("runs", [])
                if any(v is not None for v in r)]
        if len(have) < 5:
            todo.append(qi)
    print(f"[{m}] todo {len(todo)}/{len(rows)}", flush=True)

    def one(qi, _m=m, _k=kind):
        ans = (sa.get(str(qi), {}).get(_m, "") if _k == "sa"
               else RA.get(_m, {}).get(qi, "")) or ""
        # only VALID runs count toward the 5: a transient 429/None run must be
        # retried, otherwise the row is permanently stuck below n=5 and the
        # summary silently drops it (older code appended [None]*4 and stopped).
        have = [r for r in out[_m].get(str(qi), {}).get("runs", [])
                if any(v is not None for v in r)]
        for _ in range(5 - len(have)):
            r = judge(rows[qi]["question"], rows[qi]["answer"], ans[:1500])
            if any(v is None for v in r):
                continue
            have.append(r)
        with lock:
            out[_m][str(qi)] = {"runs": have}
            if len(out[_m]) % 25 == 0:
                json.dump(out, open(OUT, "w"), ensure_ascii=False)

    # 8 workers (not 10): Solar caps at 100 req/min; 10 workers x ~3s/call
    # bursts past it and the 429s turn into [None] runs. 8 keeps ~160/min
    # peak with the 6s backoff absorbing spikes.
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"[{m}] done {len(out[m])}", flush=True)

res = {}
for m, _ in METHODS:
    vals = []
    for qi, r in out.get(m, {}).items():
        runs = [x for x in r.get("runs", []) if all(v is not None for v in x)]
        if runs:
            vals.append(sum(sum(x) / 4 for x in runs) / len(runs))
    res[m] = {"n": len(vals),
              "mean5": round(st.mean(vals), 4) if vals else None,
              "sd": round(st.stdev(vals), 4) if len(vals) > 1 else None}
json.dump(res, open(R + "tab5_5run_summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
