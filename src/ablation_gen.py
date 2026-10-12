"""RAGAS per ablation arm (Tab.7). e2e_v3.py saves retrieval (top10/pool/edges) for
all 5 arms but generates+judges only the full arm. This stage generates answers
from each non-full arm's own top10 and judges them with the identical 5 RAGAS
prompts, so Tab.7 compares like-with-like. vLLM 14B generator + solar judges.
Checkpointed per arm per 25 rows."""
import json, time, csv, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
R = "/home/devlab1/adgr_workspace/runs/"
D = "/home/devlab1/adgr_workspace/data/"
VLLM = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/home/devlab1/adgr_workspace/models/qwen25-awq"
ARMS = ["dense_only", "dense_anchors", "no_edge_filter", "no_rerank"]
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
CAP = int(os.environ.get("CTX_CAP", "30000"))


def kv(j, k):
    import re
    m = re.search(r'["\']' + k + r'["\']\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', j or "")
    return float(m.group(1)) if m else None


def _fatal_api(e):
    # Billing/auth failures are PERMANENT: retrying burns nothing but marks rows
    # done with all_ok=False (silent data poisoning). Abort loudly instead so the
    # chain stops and a human refills credit. Plain 429 rate-limits are transient
    # and must keep retrying (handled by callers).
    import re as _re
    t = _re.sub(r"[^a-z]", "", repr(e).lower())
    return any(p in t for p in ("suspended", "insufficient", "invalidapikey",
                                "unauthorized", "accountdeactivated", "billing",
                                "paymentrequired", "outofcredit", "permissiondenied",
                                "zerobalance"))


def judge(prompt, keys):
    # Effort escalation: minimal (~116 out-tok, 5/6 valid) x3, then low fallback.
    # Measured: low=3273 tok/call vs minimal=116 (28x). temp=0 keeps values stable.
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
                print(f"FATAL solar billing/auth: {repr(e)[:150]} - aborting, "
                      f"checkpoint preserved, refill credit and resume.", flush=True)
                _os._exit(42)
            time.sleep(6)
    return "", False


def gen(q, ctx):
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


IDX = {c["__id__"]: c["content"] for c in
       json.load(open(D + "vdb_chunks.json"))["data"]}
rows = list(csv.DictReader(open(D + "q300.csv", encoding="utf-8")))
arms = json.load(open(R + "abl_v3_arms.json"))["arms"]
OUT = R + "abl_gen.json"
try:
    out = json.load(open(OUT))
except Exception:
    out = {}
for arm in ARMS:
    out.setdefault(arm, {})
    arr = arms[arm]
    todo = [qi for qi in range(len(rows))
            if str(qi) not in out[arm] or not out[arm][str(qi)].get("done")]
    print(f"[{arm}] todo {len(todo)}/{len(rows)}", flush=True)
    lock = __import__("threading").Lock()

    def one(qi, _arm=arm, _arr=arr):
        if str(qi) in out[_arm] and out[_arm][str(qi)].get("done"):
            return
        top = (_arr[qi].get("top10") or [])[:10]
        docs = [IDX[c] for c in top if c in IDX]
        ctx = "\n\n".join(docs)
        q = rows[qi]["question"]
        a = gen(q, ctx)
        if len(a.strip()) < 20:
            with lock:
                out[_arm][str(qi)] = {"gen_failed": True, "done": True}
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
            out[_arm][str(qi)] = {"ans": a[:2000], "faith": d["faith"][0][:300],
                                  "rel": d["rel"][0][:150], "crec": d["crec"][0][:150],
                                  "cprec": d["cprec"][0][:200], "dim4": d["dim4"][0][:400],
                                  "all_ok": all(d[x][1] for x in
                                                ["faith", "rel", "crec", "cprec", "dim4"]),
                                  "done": True}
            if len(out[_arm]) % 25 == 0:
                json.dump(out, open(OUT, "w"), ensure_ascii=False)
                print(f"  [{_arm}] {len(out[_arm])}/{len(rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, todo))
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"[{arm}] done {len(out[arm])}", flush=True)
print("ablation gen+judge complete")
