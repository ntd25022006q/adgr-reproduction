"""ORIGINAL_REPRODUCTION of ADGR (Nguyen-Tien et al., submitted to Applied Intelligence).

Faithful to paper Sec 3 / Alg.1 with documented substitutions (see SUBSTITUTIONS).
Index: ViMedGraph artifacts (vietmed-rag-dataset) on A30. No corpus re-embedding:
uses precomputed NanoVectorDB matrices + embeddinggemma:300m (paper's model) via
ollama for query-side embeddings only.
"""
import json, base64, re, time, unicodedata, os
from functools import lru_cache
import numpy as np

DATA = os.environ.get("ADGR_DATA", "/home/devlab1/adgr_workspace/data")
CFG = {"tau_anchor": 0.6, "delta": 0.1, "d": 10, "k": 15,
       "alpha": 0.3, "beta": 0.7, "tau_edge": 0.3, "m": 10, "Nmax": 50,
       "ng": 15, "ns": 10, "r": 5, "nc": 10}

SUBSTITUTIONS = {
    "gen_model": "qwen3:8b via ollama (paper: Qwen2.5-14B-Instruct-AWQ via vLLM)",
    "keyword_llm": "qwen3:8b via ollama (paper: LLM, model unspecified for keywords)",
    "judge": "solar-pro4 via Upstage (paper: GPT-5.5; RAGAS LLM likewise)",
    "bm25_tokenizer": "whitespace lowercase (LightRAG impl detail not pinned)",
}

_OLLAMA = "http://127.0.0.1:11434/api/embed"
_OLLAMA_GEN = "http://127.0.0.1:11434/api/generate"


def oembed(texts):
    import urllib.request
    out = []
    for t in texts:
        req = urllib.request.Request(
            _OLLAMA, data=json.dumps({"model": "embeddinggemma:300m",
                                      "input": t[:1500] or " "}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.append(json.load(r)["embeddings"][0])
    m = np.asarray(out, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def ogen(prompt, max_tokens=400, counts=False):
    import urllib.request
    # num_ctx=4096: keyword prompts are ~100 tok in / ~50 out. Ollama's default
    # 32k ctx made qwen3:8b hold 10GB VRAM; 4k keeps it ~5GB. Outputs identical
    # (temp=0, same weights) as long as content fits, which it always does here.
    opts = {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}
    # OGEN_NUM_GPU=0 runs the keyword model on CPU: R11 (e2e_paper) then holds
    # vLLM-12k (13.5G) + reranker-GPU (2.5G) + embed (0.6G) + external (4.5G)
    # = 21G, leaving 3.4G slack. GPU keywords would push it to 26G = OOM.
    if os.environ.get("OGEN_NUM_GPU", "").strip():
        opts["num_gpu"] = int(os.environ["OGEN_NUM_GPU"])
    req = urllib.request.Request(
        _OLLAMA_GEN, data=json.dumps({"model": "qwen3:8b", "prompt": prompt,
                                      "stream": False, "think": False,
                                      "options": opts}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
        if counts:
            return d["response"].strip(), int(d.get("prompt_eval_count") or 0), int(d.get("eval_count") or 0)
        return d["response"].strip()


def load_index():
    def mat(fname, n=None, dim=768):
        d = json.load(open(f"{DATA}/{fname}"))
        raw = base64.b64decode(d["matrix"])
        if n is None:
            n = len(raw) // (dim * 4)
        m = np.frombuffer(raw, dtype=np.float32).reshape(n, dim)
        return (m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)).astype(np.float32), d["data"]
    C, cdata = mat("vdb_chunks.json")
    E, edata = mat("vdb_entities.json")
    R, rdata = mat("vdb_relationships.json")
    names = json.load(open(f"{DATA}/name_list.json"))
    N = np.load(f"{DATA}/name_matrix_normed.npy", mmap_mode="r")
    ent_chunks = json.load(open(f"{DATA}/kv_store_entity_chunks.json"))
    rel_chunks = json.load(open(f"{DATA}/kv_store_relation_chunks.json"))
    chunks = json.load(open(f"{DATA}/kv_store_text_chunks.json"))
    rel_by_id = {r["__id__"]: (i, r) for i, r in enumerate(rdata)}
    adj = {}
    for i, r in enumerate(rdata):
        adj.setdefault(r["src_id"], []).append(i)
        if r["tgt_id"] != r["src_id"]:
            adj.setdefault(r["tgt_id"], []).append(i)
    return {"C": C, "cdata": cdata, "E": E, "edata": edata, "R": R, "rdata": rdata,
            "names": names, "N": np.asarray(N, dtype=np.float32),
            "ent_chunks": ent_chunks, "rel_chunks": rel_chunks, "chunks": chunks, "adj": adj}


def tok(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.findall(r"\w+", s, flags=re.UNICODE)


KW_PROMPT = ("Trich xuat tu khoa truy van tieng Viet.\nCau hoi: {q}\n"
             "Tra ve JSON gom 2 truong: \"low\" (danh sach 3-6 cum tu NGAN, moi cum toi da 4 tu, "
             "chi ten benh/thuoc/trieu chung/thu thuat cu the, KHONG viet cau dai, KHONG danh so), "
             "va \"high\" (1 cau tom tat y dinh truy van). Chi tra ve JSON.")


def keywords(q):
    # Memoized: e2e_v3 calls retrieve() once per arm (5x) per question, and
    # every call re-extracts keywords for the SAME question. With the keyword
    # model on CPU (OGEN_NUM_GPU=0, needed to fit R11's VRAM budget) that is
    # 5 identical LLM calls per question -> 5x latency. Same temp-0 model,
    # same input string => identical output, so caching changes nothing but
    # wall-clock: ~4-6h -> ~1h for R11's retrieval phase.
    return _keywords(q)


@lru_cache(maxsize=512)
def _keywords(q):
    try:
        raw = ogen(KW_PROMPT.format(q=q), max_tokens=200)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        d = json.loads(m.group(0)) if m else {}
        low_raw = d.get("low", [])
        if isinstance(low_raw, str):
            low_raw = [low_raw]
        low = []
        for x in low_raw:
            if isinstance(x, str):
                parts = [p for p in re.split(r"[,;]", x) if p.strip()]
                low.extend(parts if len(parts) > 1 else [x])
        low = [s.strip() for s in low if isinstance(s, str) and len(s.strip()) > 1][:6]
        high = str(d.get("high", "")).strip()
    except Exception:
        low, high = [], ""
    if not low:
        low = [w for w in tok(q) if len(w) > 3][:4] or [q[:40]]
    qtok = [w for w in tok(q) if len(w) > 3][:6]
    for w in qtok:
        if w not in low and len(low) < 10:
            low.append(w)

    if not high:
        high = q
    else:
        ht = tok(high)
        if ht and sum(1 for w in ht if w in set(tok(q))) / max(1, len(ht)) < 0.3:
            high = q
    return low, high


def build_bm25(idxs_contents):
    from rank_bm25 import BM25Okapi
    corpus = [tok(c) for _, c in idxs_contents]
    return BM25Okapi(corpus)


def anchors(IDX, bm25, eidx, Kl):
    N = IDX["N"]
    Z = {kappa: oembed([kappa])[0] for kappa in Kl}
    B = {}
    for kappa in Kl:
        scores = bm25.get_scores(tok(kappa))
        ranked = np.argsort(-scores)
        B[kappa] = set(int(eidx[i]) for i in ranked[:10])
    cand = {}
    for kappa in Kl:
        sims = N @ Z[kappa]
        order = np.argsort(-sims)
        for v in B[kappa]:
            cand.setdefault(int(v), {"kws": set()})
            cand[int(v)]["kws"].add(kappa)
        added = 0
        for v in order[: 4 * CFG["d"] + len(B[kappa])].tolist():
            if v in B[kappa]:
                continue
            if sims[v] >= CFG["tau_anchor"] and added < CFG["d"]:
                cand.setdefault(int(v), {"kws": set()})
                cand[int(v)]["kws"].add(kappa)
                added += 1
    scored = []
    for v, info in cand.items():
        best = -1.0
        for kappa in info["kws"]:
            c = float(N[v] @ Z[kappa])
            best = max(best, c)
        f = 0
        for kappa in info["kws"]:
            if v in B.get(kappa, set()) and float(N[v] @ Z[kappa]) >= CFG["tau_anchor"]:
                f = 1
                break
        scored.append((min(1.0, best + CFG["delta"] * f), v))
    scored.sort(reverse=True)
    return scored[:CFG["k"]]


def retrieve(IDX, bm25, eidx, q, ablation=None):
    abl = ablation or {}
    t0 = time.time()
    Kl, Kh = keywords(q)
    t_kw = time.time()
    zq = oembed([q])[0]
    if abl.get("dense_only"):
        A = []
    elif abl.get("dense_anchors"):
        Z = {kappa: oembed([kappa])[0] for kappa in Kl}
        agg = np.zeros(len(IDX["names"]), dtype=np.float32)
        for kappa in Kl:
            agg += IDX["N"] @ Z[kappa]
        topk = np.argsort(-agg)[:CFG["k"]]
        A = [(float(agg[v]), int(v)) for v in topk]
    else:
        A = anchors(IDX, bm25, eidx, Kl)
    scored_edges = []
    R, rdata = IDX["R"], IDX["rdata"]
    if abl.get("dense_only"):
        pass
    for s, v in A:
        cands = IDX["adj"].get(IDX["names"][v], [])
        if not cands:
            continue
        if abl.get("no_edge_filter"):
            top = [(0.0, c) for c in cands[:CFG["Nmax"]]]
        else:
            sims = R[cands] @ zq
            keep = [(float(sims[i]), cands[i]) for i in range(len(cands)) if float(sims[i]) >= CFG["tau_edge"]]
            keep.sort(reverse=True)
            top = keep[:CFG["m"]]
        for ss, ri in top:
            scored_edges.append((CFG["alpha"] * s + CFG["beta"] * ss, ri, v))
    scored_edges.sort(reverse=True)
    seen_e, Esel = set(), []
    for sc, ri, v in scored_edges:
        if ri not in seen_e:
            seen_e.add(ri)
            Esel.append((sc, ri, v))
        if len(Esel) >= CFG["Nmax"]:
            break
    zkh = oembed([Kh])[0]
    gsim = R @ zkh
    Eglob = np.argsort(-gsim)[:CFG["ng"]].tolist()
    csim = IDX["C"] @ zq
    Ts_idx = np.argsort(-csim)[:CFG["ns"]].tolist()
    Aprime = {v for _, _, v in Esel}
    id2pos = {c["__id__"]: i for i, c in enumerate(IDX["cdata"])}
    C = IDX["C"]

    def pi_rank(cids, n_elements):
        cap = max(1, (CFG["r"] * n_elements) // 2)
        scored = []
        for cid in cids:
            pos = id2pos.get(cid)
            if pos is None:
                continue
            scored.append((float(C[pos] @ zq), cid))
        scored.sort(reverse=True)
        return [c for _, c in scored[:cap]]

    if abl.get("dense_only"):
        ent_keys = {rdata[int(ri)]["src_id"] for ri in Eglob} | \
            {rdata[int(ri)]["tgt_id"] for ri in Eglob}
    else:
        ent_keys = {IDX["names"][v] for v in Aprime} | \
            {rdata[int(ri)]["src_id"] for ri in Eglob} | {rdata[int(ri)]["tgt_id"] for ri in Eglob}
    Tv_ids = set()
    for nm in ent_keys:
        Tv_ids.update(IDX["ent_chunks"].get(nm, {}).get("chunk_ids", []))
    for ri in Eglob:
        r = rdata[int(ri)]
        Tv_ids.update(IDX["rel_chunks"].get(f"{r['src_id']}<SEP>{r['tgt_id']}", {}).get("chunk_ids", []))
    Tv = pi_rank(Tv_ids, len(ent_keys))
    Te_ids = set()
    for _, ri, _ in Esel:
        r = rdata[int(ri)]
        Te_ids.add(r["source_id"])
        Te_ids.update(IDX["rel_chunks"].get(f"{r['src_id']}<SEP>{r['tgt_id']}", {}).get("chunk_ids", []))
    Te = pi_rank(Te_ids, len(Esel))
    Tg_ids = {rdata[int(ri)]["source_id"] for ri in Eglob}
    Tg = pi_rank(Tg_ids, len(Eglob))
    Ts = [IDX["cdata"][i]["__id__"] for i in Ts_idx]
    if abl.get("dense_only"):
        # Paper Sec 5.8: "Dense Retrieval Only: retrieves chunks using dense
        # vector retrieval over the chunk index without graph-guided retrieval."
        # Eglob/Tg/Tv are graph-derived sources, so the ablation must NOT merge
        # them in - the pool IS the direct dense set Ts (already similarity
        # ordered, cap ns).
        pool, seen = [], set()
        for cid in Ts:
            if cid not in seen:
                seen.add(cid)
                pool.append(cid)
        dt = time.time() - t0
        return {"q": q, "Kl": Kl, "Kh": Kh, "anchors": [], "n_edges": 0,
                "n_entities": 0, "pool": pool, "ret_s": round(dt, 2),
                "t_kw": round(t_kw - t0, 2), "Eglob": [], "Ts": [int(x) for x in Ts_idx]}
    pool, seen = [], set()
    for i in range(max(len(Tv), len(Te), len(Tg), len(Ts))):
        for lst in (Tv, Te, Tg, Ts):
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                pool.append(lst[i])
    dt = time.time() - t0
    return {"q": q, "Kl": Kl, "Kh": Kh, "anchors": [(IDX["names"][v], s) for s, v in A],
            "n_edges": len(Esel), "n_entities": len(Aprime), "pool": pool,
            "ret_s": round(dt, 2), "t_kw": round(t_kw - t0, 2),
            "Eglob": [int(x) for x in Eglob], "Ts": [int(x) for x in Ts_idx]}
