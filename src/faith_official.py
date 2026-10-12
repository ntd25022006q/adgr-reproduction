"""faith_official.py - official-style two-step faithfulness (RAGAS ref [10]) on
the e2e_v3 full arm, to isolate HOW MUCH of our faith gap (0.41 vs paper 0.89)
comes from our single-call shortcut (split+verify in one prompt) vs genuine
 strictness/data noise:
  step 1: extract discrete factual claims from the answer (own call);
  step 2: verify each claim against the context, batched CO/KHONG (own call);
  faith_official = supported / total.
Same judge model (solar-pro4, temp 0, minimal escalation, quota guard) and
same 300 answers/contexts as our single-call judge -> the ONLY difference is
the protocol. If two-step lands ~0.6-0.7, the shortcut explains much of the
gap; if it stays ~0.4, the gap is judge strictness + data noise.
Writes faith_official.json. Solar-only (no GPU). Checkpointed per 25 rows.
"""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
W = "/home/devlab1/adgr_workspace"
R = W + "/runs/"
_key = open(os.path.expanduser("~/.config/adgr/upstage_key")).read().strip()
_j = OpenAI(api_key=_key, base_url="https://api.upstage.ai/v1", timeout=300)
P_EXT = ("Split the medical answer below into distinct factual statements, "
         "one per line, at most 10 lines, no numbering, no extra text.\nAnswer:\n{a}")
P_VER = ("Context:\n{c}\nNumbered statements:\n{s}\nFor EACH statement, is it "
         "directly supported by the context? Reply with exactly that many lines, "
         "one word per line: CO or KHONG.")
OUT = R + "faith_official.json"


def _fatal_api(e):
    t = re.sub(r"[^a-z]", "", repr(e).lower())
    return any(p in t for p in ("suspended", "insufficient", "invalidapikey",
                                "unauthorized", "accountdeactivated", "billing",
                                "paymentrequired", "outofcredit", "permissiondenied",
                                "zerobalance"))


def call(prompt, mt=400):
    import os as _os
    last = ""
    for eff in ("minimal", "minimal", "low"):
        try:
            last = _j.chat.completions.create(
                model="solar-pro4", temperature=0,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=eff, max_tokens=mt).choices[0].message.content or ""
            if last.strip():
                return last, eff
        except Exception as e:
            if _fatal_api(e):
                print(f"FATAL solar billing/auth: {repr(e)[:150]} - aborting; "
                      f"checkpoint preserved.", flush=True)
                _os._exit(42)
            time.sleep(6)
    return "", "fail"


def yn(txt):
    t = (txt or "").strip().upper()
    if t.startswith("CO") and "KHONG" not in t[:8]:
        return 1
    if "KHONG" in t[:10]:
        return 0
    return None


src = {int(k): v for k, v in json.load(open(R + "e2e_v3.json")).items() if str(k).isdigit()}
CDATA = {c["__id__"]: c["content"] for c in
         json.load(open(W + "/data/vdb_chunks.json"))["data"]}
try:
    out = json.load(open(OUT))
except Exception:
    out = {}
todo = [qi for qi in sorted(src)
        if str(qi) not in out and src[qi].get("ans") and src[qi].get("ctx_ids")]
print(f"todo: {len(todo)}/300", flush=True)
lock = threading.Lock()
done_n = [0]


def one(qi):
    v = src[qi]
    ans = v["ans"][:1500]
    chs = [CDATA.get(cid, "")[:1200] for cid in v["ctx_ids"][:10]]
    ctx = "\n\n".join(chs)
    res = {"claims_n": 0, "supported": 0, "total": 0, "faith_official": None}
    cl, _ = call(P_EXT.format(a=ans), 400)
    claims = [l.strip("- ").strip() for l in cl.split("\n") if len(l.strip()) > 8][:10]
    if claims:
        at, _ = call(P_VER.format(
            c=ctx[:30000],
            s="\n".join(f"{i+1}. {s}" for i, s in enumerate(claims))), 300)
        vv = [yn(l) for l in at.split("\n")]
        vv = [x for x in vv if x is not None][:len(claims)]
        if vv:
            res = {"claims_n": len(claims), "supported": sum(vv),
                   "total": len(claims),
                   "faith_official": round(sum(vv) / len(claims), 4)}
    with lock:
        out[str(qi)] = res
        done_n[0] += 1
        if done_n[0] % 25 == 0:
            json.dump(out, open(OUT, "w"), ensure_ascii=False)
            print(f"[{done_n[0]}/{len(todo)}]", flush=True)


with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(one, todo))
json.dump(out, open(OUT, "w"), ensure_ascii=False)
vals = [v["faith_official"] for v in out.values()
        if isinstance(v, dict) and v.get("faith_official") is not None]
import statistics as st
print(f"done {len(out)} rows, official-faith mean={st.mean(vals):.4f} "
      f"(n={len(vals)}) vs single-call 0.4055 vs paper 0.8878", flush=True)
