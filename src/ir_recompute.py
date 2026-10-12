"""ir_recompute.py - recompute retrieval metrics (slug-gold Hit/MRR/nDCG with
the FIXED IDCG, plus the single-gold proxy) from a GIVEN arm checkpoint.
Replaces the stale ir_single_gold.json / ir_metrics_v2.json which were computed
from abl_n300v3.json (the pre-fix retrieval: 1500-char rerank cut, NFC issues).
Env: ARMS_SRC (default abl_v3_arms.json), DATA (default data/),
     OUT_METRICS (default ir_metrics_v2.json), OUT_SINGLE (default ir_single_gold.json).
CPU/numpy only + one small embeddinggemma pass for the single-gold proxy.
"""
import json, csv, re, math, os, urllib.request
import numpy as np
D = os.environ.get("DATA", "/home/devlab1/adgr_workspace/data")
R = "/home/devlab1/adgr_workspace/runs"
# Questions are index-independent (same ViMedAQA-300 for both indexes) and live
# ONLY in data/q300.csv - data_paper/ holds just the rebuilt chunk/entity
# matrices. Building this path from DATA broke the paper run with
# FileNotFoundError (bug #38: never tested the non-default env path).
ARMS_SRC = os.environ.get("ARMS_SRC", "abl_v3_arms.json")
OUT_M = os.environ.get("OUT_METRICS", "ir_metrics_v2.json")
OUT_S = os.environ.get("OUT_SINGLE", "ir_single_gold.json")
rows = list(csv.DictReader(open("/home/devlab1/adgr_workspace/data/q300.csv", encoding="utf-8")))
cdata = json.load(open(f"{D}/vdb_chunks.json"))["data"]
# vdb_chunks of the rebuilt index uses "url"; the author index uses "file_path"
CHUNK_KEY = "url" if "url" in cdata[0] else "file_path"
id2pos = {c["__id__"]: i for i, c in enumerate(cdata)}
mat = np.frombuffer(__import__("base64").b64decode(
    json.load(open(f"{D}/vdb_chunks.json"))["matrix"]),
    dtype=np.float32).reshape(len(cdata), 768)
C = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def slug(u):
    # robust to BOTH key styles: bare "slug(.txt)" (author index) and full
    # "https://.../slug/" urls (rebuilt index). The previous version returned
    # "" for slash-less strings, emptying the whole gold mapping (bug #39:
    # ir_metrics_v2 came out n=0 on the author index).
    u = (u or "").rstrip("/")
    m = re.search(r"/([^/]+)$", u)
    return (m.group(1) if m else u).replace(".html", "")


slug2 = {}
for c in cdata:
    # same slug normalization as e2e_v3: the rebuilt index stores full URLs,
    # the author index stores bare "slug.txt" - both must map to the slug or
    # every gold lookup misses (seen: hit=None 100% on data_paper).
    slug2.setdefault(slug(str(c.get(CHUNK_KEY, "")).replace(".txt", "")), []).append(c["__id__"])
gold = [set(slug2.get(slug(r.get("article_url")), [])) for r in rows]


def metrics(arms, goldfn, tag):
    out = {}
    for arm, arr in arms.items():
        H = {k: [] for k in (1, 3, 5, 10)}
        mrr, nd = [], []
        n = 0
        for qi, a in enumerate(arr):
            g = goldfn(qi)
            top = (a.get("top10") or [])[:10]
            if not g or "error" in a or not top:
                continue
            n += 1
            rel = [1.0 if c in g else 0.0 for c in top]
            for k in H:
                H[k].append(max(rel[:k]))
            rr = next((1.0 / (i + 1) for i, v in enumerate(rel) if v), 0.0)
            mrr.append(rr)
            row = []
            for k in (5, 10):
                dcg = sum(v / math.log2(i + 2) for i, v in enumerate(rel[:k]))
                ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), k)))
                row.append(dcg / ideal if ideal else 0.0)
            nd.append(row)
        m = lambda v: round(sum(v) / len(v), 4) if v else None
        out[arm] = {"n": n, "hit1": m(H[1]), "hit3": m(H[3]), "hit5": m(H[5]),
                    "hit10": m(H[10]), "mrr10": m(mrr),
                    "ndcg5": m([x[0] for x in nd]), "ndcg10": m([x[1] for x in nd])}
        print(f"  [{tag}] {arm}: n={n} hit10={out[arm]['hit10']} "
              f"ndcg10={out[arm]['ndcg10']}", flush=True)
    return out


raw = json.load(open(f"{R}/{ARMS_SRC}"))
arms = raw["arms"] if "arms" in raw else raw["rows"]
print(f"source: {ARMS_SRC} arms={list(arms)}", flush=True)
m = metrics(arms, lambda qi: gold[qi], "slug")
json.dump(m, open(f"{R}/{OUT_M}", "w"), indent=1)

# ---- single-gold proxy: article chunk most similar to the reference answer ----
GS = os.environ.get("GOLD_SINGLE", "gold_single.json")
try:
    sg = {int(k): v for k, v in json.load(open(f"{R}/{GS}")).items()}
    print(f"reusing {GS}", flush=True)
except Exception:
    def emb(texts):
        out = []
        for i in range(0, len(texts), 32):
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/embed",
                data=json.dumps({"model": "embeddinggemma:300m",
                                 "input": texts[i:i + 32]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                out.extend(json.load(r)["embeddings"])
        mm = np.asarray(out, dtype=np.float32)
        return mm / (np.linalg.norm(mm, axis=1, keepdims=True) + 1e-9)
    Z = emb([r["answer"][:1500] or " " for r in rows])
    sg = {}
    for i, r in enumerate(rows):
        ids = slug2.get(slug(r.get("article_url")), [])
        if not ids:
            continue
        M = C[[id2pos[j] for j in ids]]
        sg[i] = [ids[int(np.argmax(M @ Z[i]))]]
    json.dump({str(k): v for k, v in sg.items()}, open(f"{R}/{GS}", "w"))
    print(f"computed {GS}: {len(sg)}/300", flush=True)
s = metrics(arms, lambda qi: sg.get(qi) or [], "single")
json.dump(s, open(f"{R}/{OUT_S}", "w"), indent=1)
print("ir_recompute done ->", OUT_M, "+", OUT_S)
