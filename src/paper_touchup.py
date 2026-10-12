"""paper_touchup.py - re-judge the transient judge_fail rows of e2e_paper.json.
Same 5 judge prompts and minimal-effort escalation as e2e_v3.py (all_ok covers
all FIVE keys there, unlike baselines_gen's four). Context is rebuilt from the
row's own ctx_ids via data_paper CDATA + ref from q300 - no regeneration.
Checkpointed per row."""
import json, os, re, time, threading, csv
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
W = "/home/devlab1/adgr_workspace"
R = W + "/runs/"
CAP = 30000
_key = open(os.path.expanduser("~/.config/adgr/upstage_key")).read().strip()
_j = OpenAI(api_key=_key, base_url="https://api.upstage.ai/v1", timeout=300)
COUNT = {"supported", "total", "useful"}
P_F = ("Cho cau tra loi va nguc canh. Tach thanh menh de su that (toi da 10). "
       "Moi menh de 1 neu nguc canh ho tro truc tiep, 0 neu khong. "
       "Tra ve JSON {{\"supported\":s,\"total\":t}}.\nNguc canh:\n{ctx}\nCau tra loi:\n{ans}")
P_R = ("Cau hoi: {q}\nCau tra loi: {ans}\nDo lien quan 0-1. Tra ve JSON {{\"relevancy\":x}}.")
P_C = ("Cau hoi: {q}\nDap an mau: {ref}\nNguc canh:\n{ctx}\nTy le y chinh co mat 0-1. "
       "Tra ve JSON {{\"recall\":x}}.")
P_P = ("Cau hoi: {q}\nNguc canh:\n{ctx}\nMoi doan 1 neu lien quan truc tiep, 0 neu khong. "
       "Tra ve JSON {{\"useful\":u,\"total\":10}}.")
P_4 = ("Ban la chuyen gia y khoa. Cham 0-1: Comprehensiveness, Diversity, Empowerment, "
       "Directness. Chi gia tri [0,1].\nCau hoi: {q}\nDap an mau: {ref}\nCan cham: {ans}\n"
       'JSON {{"comp":x,"div":x,"emp":x,"dir":x}}.')
KEYS5 = ["faith", "rel", "crec", "cprec", "dim4"]


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


CDATA = {c["__id__"]: c["content"] for c in
         json.load(open(W + "/data_paper/vdb_chunks.json"))["data"]}
# SAME csv order e2e_v3 enumerates (for qi, row in enumerate(rows)): qi IS the
# csv row position (question_idx holds disease_xxx IDs, NOT the position).
qrows = list(csv.DictReader(open(W + "/data/q300.csv", encoding="utf-8")))
OUT = R + "e2e_paper.json"
out = {int(k): v for k, v in json.load(open(OUT)).items()}
todo = [qi for qi, r in out.items()
        if r.get("done") and not r.get("all_ok") and not r.get("gen_failed")]
print(f"[paper_touchup] re-judge {len(todo)} rows", flush=True)
lock = threading.Lock()


def one(qi):
    r = out[qi]
    a = r.get("ans", "")
    docs = [CDATA[c] for c in r.get("ctx_ids", [])[:10] if c in CDATA]
    ctx = "\n\n".join(docs)
    q = r.get("q", "")
    ref = qrows[qi]["answer"] if qi < len(qrows) else ""
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(docs))
    jobs = [("faith", P_F.format(ctx=ctx[:CAP], ans=a[:1500]), ["supported", "total"]),
            ("rel", P_R.format(q=q, ans=a[:1500]), ["relevancy"]),
            ("crec", P_C.format(q=q, ref=ref[:1200], ctx=ctx[:CAP]), ["recall"]),
            ("cprec", P_P.format(q=q, ctx=numbered[:CAP]), ["useful"]),
            ("dim4", P_4.format(q=q, ref=ref[:1200], ans=a[:1500]),
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
                        "crec": d["crec"][0][:150], "cprec": d["cprec"][0][:200],
                        "dim4": d["dim4"][0][:400],
                        "all_ok": all(d[x][1] for x in KEYS5)})
        json.dump({str(k): v for k, v in out.items()}, open(OUT, "w"), ensure_ascii=False)


with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(one, todo))
ok = sum(1 for v in out.values() if v.get("all_ok"))
print(f"[paper_touchup] done: all_ok={ok}/{len(out)}", flush=True)
