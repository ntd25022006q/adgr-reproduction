"""tab5_paperprompt.py - re-judge Tab.5 with the PAPER-EXACT judge prompt from
Online Resource 1 (adgr-supplementary/supplementary.tex), instead of our
short Vietnamese approximation. Differences that matter:
  - explicit anti-verbosity guards ("Do not favor longer answers by default",
    "Long but unfocused... lower Directness") - ours lacked these;
  - wording tolerance ("Do not penalize different wording if medical meaning
    preserved") - ours lacked it;
  - no-external-knowledge rule - ours lacked it;
  - reasons + scores (we append one JSON tail line for parseability; the
    paper's own instructions are otherwise verbatim).
Writes tab5_paperprompt.json (+ summary). Same answers, same 5-run protocol,
same temp 0, same minimal-escalation + quota guard. Comparing its means with
tab5_5run_summary isolates the PROMPT effect from everything else.
ROW_LIMIT env: pilot on 30 rows first (~1500 calls), then full 300.
"""
import json, time, csv, os, statistics as st
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
R = "/home/devlab1/adgr_workspace/runs/"
NROWS = int(os.environ.get("ROW_LIMIT", "30"))
OUT = os.environ.get("TAB5_PP_OUT", R + "tab5_paperprompt.json")
_key = open(os.path.expanduser("~/.config/adgr/upstage_key")).read().strip()
_j = OpenAI(api_key=_key, base_url="https://api.upstage.ai/v1", timeout=300)
P_PAPER = ("""You are an experienced medical expert and evaluator with strong clinical knowledge and the ability to assess medical responses objectively, fairly, and consistently.

Evaluate the generated answer below against both the Question and the Reference answer, independently on a continuous scale from 0 to 1:
- Comprehensiveness: essential medical information to fully address the question, especially key points in the reference.
- Diversity: broad relevant aspects (conditions, exceptions, context, considerations).
- Empowerment: supports understanding and informed judgments (implications, risks, warnings, next steps).
- Directness: clear, specific, focused; avoid unnecessary, redundant, unrelated information.

Principles: evaluate independently; compare with Question and Reference before scoring; do not assume one mode is better; Do not favor longer answers by default; long but unfocused answers get lower Directness; do not penalize different wording if medical meaning is preserved; do not introduce external medical knowledge unless clear errors/unsafe advice.

Question: {q}
Reference: {ref}
Answer to evaluate: {ans}

End your response with exactly one line of JSON: {{"comp":x,"div":x,"emp":x,"dir":x}}.""")
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
try:
    out = json.load(open(OUT))
except Exception:
    out = {}
print(f"methods={len(METHODS)} rows_limit={NROWS} resume={sum(len(v) for v in out.values())}", flush=True)


def _fatal_api(e):
    import re as _re
    t = _re.sub(r"[^a-z]", "", repr(e).lower())
    return any(p in t for p in ("suspended", "insufficient", "invalidapikey",
                                "unauthorized", "accountdeactivated", "billing",
                                "paymentrequired", "outofcredit", "permissiondenied",
                                "zerobalance"))


def judge(q, ref, ans):
    import os as _os
    last = ""
    for eff in ("minimal", "minimal", "low"):
        try:
            last = _j.chat.completions.create(
                model="solar-pro4", temperature=0,
                messages=[{"role": "user", "content": P_PAPER.format(
                    q=q, ref=ref[:1200], ans=ans[:1500])}],
                reasoning_effort=eff, max_tokens=6000).choices[0].message.content or ""
            v = [kv(last, k) for k in DIMS]
            if all(x is not None and 0 <= x <= 1 for x in v):
                return [kv(last, k) for k in DIMS]
        except Exception as e:
            if _fatal_api(e):
                print(f"FATAL solar billing/auth: {repr(e)[:150]} - aborting, "
                      f"checkpoint preserved.", flush=True)
                _os._exit(42)
            time.sleep(6)
    return [None] * 4


lock = __import__("threading").Lock()
for m, kind in METHODS:
    out.setdefault(m, {})
    todo = []
    for qi in range(min(NROWS, len(rows))):
        ans = (sa.get(str(qi), {}).get(m, "") if kind == "sa" else RA.get(m, {}).get(qi, "")) or ""
        if not ans.strip():
            continue
        have = [r for r in out[m].get(str(qi), {}).get("runs", [])
                if any(v is not None for v in r)]
        if len(have) < 5:
            todo.append(qi)
    print(f"[{m}] todo {len(todo)}/{min(NROWS, len(rows))}", flush=True)

    def one(qi, _m=m, _k=kind):
        ans = (sa.get(str(qi), {}).get(_m, "") if _k == "sa"
               else RA.get(_m, {}).get(qi, "")) or ""
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
json.dump(res, open(OUT.replace(".json", "_summary.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
