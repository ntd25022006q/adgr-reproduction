"""baselines_touchup.py - re-judge the transient judge_fail rows of the two
baseline checkpoints. baselines_gen stores only the answer (not the context),
so the context is re-read from the same ctx_hybrid/ctx_mix.json the generation
stage used. Identical judge prompts and minimal-effort escalation, so the
re-judged rows are comparable with the rest of the table. Checkpointed."""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
R = "/home/devlab1/adgr_workspace/runs/"
CAP = int(os.environ.get("CTX_CAP", "60000"))
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


for mode in ["hybrid", "mix"]:
    OUT = R + f"e2e_base_{mode}.json"
    src = json.load(open(R + f"ctx_{mode}.json"))
    out = json.load(open(OUT))
    todo = [qi for qi, r in out.items()
            if r.get("done") and not r.get("all_ok") and not r.get("gen_failed")]
    print(f"[{mode}] re-judge {len(todo)} rows", flush=True)
    lock = threading.Lock()
    n_done = [0]

    def one(qi):
        r = out[qi]
        s = src[qi]
        a = r["ans"]
        q = s["question"]
        ctx = str(s.get("ctx", ""))[:CAP]
        jobs = [("faith", P_F.format(ctx=ctx, ans=a[:1500]), ["supported", "total"]),
                ("rel", P_R.format(q=q, ans=a[:1500]), ["relevancy"]),
                ("crec", P_C.format(q=q, ref=s["ref"][:1200], ctx=ctx), ["recall"]),
                ("cprec", P_P.format(q=q, ctx=ctx), ["useful"]),
                ("dim4", P_4.format(q=q, ref=s["ref"][:1200], ans=a[:1500]),
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
            out[qi].update({"faith": d["faith"][0][:300], "rel": d["rel"][0][:150],
                            "crec": d["crec"][0][:150],
                            "dim4": d["dim4"][0][:400],
                            "all_ok": all(d[x][1] for x in
                                          ["faith", "rel", "crec", "dim4"])})
            n_done[0] += 1
            json.dump(out, open(OUT, "w"), ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, todo))
    ok = sum(1 for r in out.values() if r.get("all_ok"))
    print(f"[{mode}] done: all_ok={ok}/{len(out)}", flush=True)
print("baselines_touchup complete")
