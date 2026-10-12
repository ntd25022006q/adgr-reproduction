"""RAGAS-true: reimplement the OFFICIAL RAGAS algorithms (ref [10]) instead of
direct judge ratings. Why: paper's AnsRel 0.7124 vs our 0.9814 is not a model
gap but a METHOD gap - official answer_relevancy = mean cosine(original,
LLM-generated questions from the answer), not a 0-1 judge rating. A direct
rating saturates near 1 for any on-topic answer.

  faithfulness : keep (already claim-extract + verify; matches official)
  ans_rel_true : 3 questions generated from the answer -> embed -> mean cos
  ctx_rec_true : reference -> claims -> each attributable to context? -> ratio
  ctx_prec_true: per-chunk relevant? verdicts (rank order) -> RAGAS weighted P

LLM backend env-switchable (VLLM shares the extraction server; runs queued).
Embeddings via local embeddinggemma. Checkpointed.
Reads e2e_v3.json (or CKPT env). Writes ragas_true.json.
"""
import json, os, re, time, statistics as st, urllib.request
R = "/home/devlab1/adgr_workspace/runs/"
CKPT = os.environ.get("RAGAS_SRC", R + "e2e_v3.json")
OUT = os.environ.get("RAGAS_OUT", R + "ragas_true.json")
# RAGAS_DATA: chunk index matching CKPT's ctx_ids. e2e_v3.json uses the author
# index (data/); e2e_paper.json uses the rebuilt index (data_paper/, whose
# chunk IDs differ - using data/ CDATA there would silently score empty
# strings). qrows stays data/q300.csv (same 300 questions everywhere).
LLM = os.environ.get("RAGAS_LLM", "vllm")
VLLM = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/home/devlab1/adgr_workspace/models/qwen25-awq"
VLLM_QWEN = os.environ.get("GEN_BACKEND", "vllm")


def llm_call(prompt, mt=600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": mt}
    req = urllib.request.Request(VLLM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except Exception:
            time.sleep(10)
    return ""


def emb(texts):
    import numpy as np
    out = []
    for i in range(0, len(texts), 32):
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embed",
            data=json.dumps({"model": "embeddinggemma:300m",
                             "input": texts[i:i + 32]}).encode(),
            headers={"Content-Type": "application/json"})
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out.extend(json.load(r)["embeddings"])
                break
            except Exception:
                time.sleep(5)
    m = np.asarray(out, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


P_QGEN = ("Doc cau tra loi y khoa duoi day. Viet DUNG 3 cau hoi ngan, moi cau "
          "hoi ve mot y khac nhau trong cau tra loi. Chi tra ve 3 dong, moi dong "
          "mot cau hoi, khong danh so.\nCau tra loi:\n{a}")
P_CLAIMS = ("Tach doan van duoi day thanh cac menh de su that doc lap, moi menh de "
            "mot dong, toi da 8 dong, khong danh so.\nDoan van:\n{t}")
P_ATTR = ("Ngu canh:\n{c}\nMenh de: {s}\nMenh de tren co duoc ngu canh ho tro "
          "truc tiep khong? Chi tra ve DUNG mot tu: CO hoac KHONG.")
P_CHUNK = ("Cau hoi: {q}\nDap an mau: {ref}\nDoan trich:\n{ch}\nDoan trich tren "
           "co lien quan truc tiep de tra loi cau hoi khong? Chi tra ve DUNG mot "
           "tu: CO hoac KHONG.")


def yn(txt):
    t = (txt or "").strip().upper()
    if t.startswith("CO") and "KHONG" not in t[:8]:
        return 1
    if "KHONG" in t[:10]:
        return 0
    return None


# PROLOGUE: free ollama gen models before loading embeddinggemma.
# e2e_paper leaves qwen3:8b (~5GB) resident; 15 (vLLM-12k) + 5 + 1 (embed) +
# 4.5 (external) = 25.5GB > 24.5GB = OOM. R12 needs ONLY embeddinggemma.
for _m in ("qwen3:8b", "qwen2.5:14b", "qwen3:14b", "qwen3:1.7b"):
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps({"model": _m, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"}), timeout=30).read()
    except Exception:
        pass
print("ragas prologue: ollama gen models unloaded (only embeddinggemma needed)",
      flush=True)
import csv
_RDATA = os.environ.get("RAGAS_DATA", "/home/devlab1/adgr_workspace/data")
qrows = list(csv.DictReader(open("/home/devlab1/adgr_workspace/data/q300.csv", encoding="utf-8")))
src = {int(k): v for k, v in json.load(open(CKPT)).items() if str(k).isdigit()}
CDATA = {c["__id__"]: c["content"] for c in
         json.load(open(_RDATA + "/vdb_chunks.json"))["data"]}
try:
    out = json.load(open(OUT))
except Exception:
    out = {}
print(f"rows: {len(src)} resume: {len(out)}", flush=True)
todo = [qi for qi in sorted(src)
        # empty-dict rows (all LLM calls failed that pass) must be retried,
        # not skipped: `str(qi) in out` alone would drop them silently.
        if (str(qi) not in out or not out[str(qi)]) and src[qi].get("ans") and src[qi].get("ctx_ids")]
print(f"todo: {len(todo)}", flush=True)
import numpy as np
P_ATTRB = ("Ngu canh:\n{c}\nCac menh de (moi dong mot menh de):\n{s}\n"
           "Voi MOI menh de, no co duoc ngu canh ho tro truc tiep khong? Tra ve "
           "dung so dong tuong ung, moi dong chi mot tu CO hoac KHONG.")
P_PREC = ("Cau hoi: {q}\nDap an mau: {ref}\nCac doan trich (danh so):\n{s}\n"
          "Voi MOI doan, no co lien quan truc tiep de tra loi cau hoi khong? Tra ve "
          "dung so dong tuong ung, moi dong chi mot tu CO hoac KHONG.")
# 4 workers (was: sequential loop, ~2h per 300 rows). Rows are independent;
# vLLM-8k serves 4 concurrent short prompts easily, ollama embed batches too.
# Same math per row, only the driver loop is threaded. ~35 min per 300 rows.
import threading as _th
from concurrent.futures import ThreadPoolExecutor as _TPE
_lock = _th.Lock()
_done_n = [0]


def one(qi):
    v = src[qi]
    ans = v["ans"][:1500]
    q = qrows[qi]["question"]
    ref = qrows[qi]["answer"][:1200]
    chs = [CDATA.get(cid, "")[:1200] for cid in v["ctx_ids"][:10]]
    res = {}
    try:
        qs_txt = llm_call(P_QGEN.format(a=ans), 300)
        gen_qs = [l.strip("- ").strip() for l in qs_txt.split("\n") if len(l.strip()) > 8][:3]
        if gen_qs:
            M = emb([q] + gen_qs)
            res["ans_rel_true"] = round(float(np.mean(M[1:] @ M[0])), 4)
            res["ans_rel_nq"] = len(gen_qs)
        cl = llm_call(P_CLAIMS.format(t=ref), 400)
        claims = [l.strip("- ").strip() for l in cl.split("\n") if len(l.strip()) > 8][:8]
        if claims:
            at = llm_call(P_ATTRB.format(
                c="\n\n".join(chs)[:8000],
                s="\n".join(f"{i+1}. {s}" for i, s in enumerate(claims))), 200)
            vv = [yn(l) for l in at.split("\n")]
            vv = [x for x in vv if x is not None][:len(claims)]
            if vv:
                res["ctx_rec_true"] = round(sum(vv) / len(claims), 4)
                res["claims_n"] = len(claims)
        pv = llm_call(P_PREC.format(
            q=q, ref=ref,
            s="\n".join(f"[{i}] {c}" for i, c in enumerate(chs))[:8000]), 200)
        pv = [yn(l) for l in pv.split("\n")]
        pv = [x for x in pv if x is not None][:len(chs)]
        if pv and len(pv) == len(chs):
            prec_at = [sum(pv[:k + 1]) / (k + 1) for k in range(len(pv))]
            res["ctx_prec_true"] = round(
                sum(p * w for p, w in zip(prec_at, pv)) / max(1, sum(pv)), 4)
        with _lock:
            out[str(qi)] = res
    except Exception as e:
        with _lock:
            out[str(qi)] = {"error": repr(e)[:120]}
    with _lock:
        _done_n[0] += 1
        if _done_n[0] % 25 == 0:
            json.dump(out, open(OUT, "w"), ensure_ascii=False)
            print(f"[{_done_n[0]}/{len(todo)}]", flush=True)


with _TPE(max_workers=4) as _ex:
    list(_ex.map(one, todo))
json.dump(out, open(OUT, "w"), ensure_ascii=False)
print("done", len(out))
