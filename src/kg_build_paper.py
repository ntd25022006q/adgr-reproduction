"""Build the PAPER-chunker KG index (data_paper/) for the FINAL rebuilt-index numbers.
Generated from kg_build.py by the exact relaunch_kg.py recipe: 7 path
replacements (chunks_full->chunks_paper, extract_full->extract_paper_full,
data_ours->data_paper, plus 4 stats/cache renames). Same code path that built
the 52,578-entity / 106,279-relation index behind Tables 1/3/4/6/7/8.
Duplicate titleish() helper removed (dead override, identical behavior).
"""

import json, os, re, hashlib, base64, unicodedata, time, urllib.request
import numpy as np

R = "/home/devlab1/adgr_workspace/runs"
OUT = "/home/devlab1/adgr_workspace/data_paper"
os.makedirs(OUT, exist_ok=True)
SEP = "<|#|>"
MAXDESC = 1200


BOILER = {"youmed", "đại học y dược tphcm", "bệnh viện", "trung tâm y tế",
          "viện", "khoa", "bài viết", "bạn đọc", "liên hệ", "bình luận",
          "tác giả", "danh mục", "trang chủ", "google", "facebook", "youtube"}


def nfc(s):
    """Encoding audit found 484 chunks and 1317 crawl paragraphs in mixed
    NFC/NFD form; identical words then produce different KG keys.
    Canonicalize everything to NFC before any keying or embedding."""
    return unicodedata.normalize("NFC", s or "")
try:
    CANON = json.load(open(f"{R}/kg_canon.json"))
except Exception:
    CANON = {}


def _fold(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).strip().lower())


def nkey(s):
    k = nfc(s).strip().lower()
    if CANON:
        return CANON.get(_fold(k), k)
    return _fold(k)


def titleish(s):
    s = unicodedata.normalize("NFC", s).strip()
    return s[:1].upper() + s[1:] if s else s


def h(pre, s):
    return pre + hashlib.md5(s.encode("utf-8")).hexdigest()


# ---------- stage 1: merge ----------
print("stage1 merge", flush=True)
chunks = json.load(open(f"{R}/chunks_paper.json"))
ex = json.load(open(f"{R}/extract_paper_full.json"))
chunks = [{**c, "content": nfc(c.get("content", ""))} for c in chunks]
cid = [h("chunk-", c["url"] + "|" + str(i)) for i, c in enumerate(chunks)]
ent_name, ent_desc, ent_type, ent_ch = {}, {}, {}, {}
rel_desc, rel_kw, rel_ch = {}, {}, {}
n_ent_lines = n_rel_lines = n_bad = 0
for k, row in ex.items():
    ci = int(k)
    if ci >= len(chunks):
        continue
    c = cid[ci]
    for line in row.get("ents", []):
        p = line.split(SEP)
        if len(p) != 4 or not p[1].strip():
            n_bad += 1
            continue
        nm, ty, ds = nfc(p[1]).strip(), p[2].strip(), nfc(p[3]).strip()
        if nkey(nm) in BOILER:
            continue
        key = nkey(nm)
        if not key:
            n_bad += 1
            continue
        n_ent_lines += 1
        if key not in ent_name:
            ent_name[key] = titleish(nm)
            ent_desc[key] = []
            ent_type[key] = ty
            ent_ch[key] = set()
        if ds and len(ent_desc[key]) < 6:
            ent_desc[key].append(ds)
        ent_ch[key].add(c)
    for line in row.get("rels", []):
        p = line.split(SEP)
        if len(p) != 5 or not p[1].strip() or not p[2].strip():
            n_bad += 1
            continue
        s, t, kw, ds = nfc(p[1]).strip(), nfc(p[2]).strip(), nfc(p[3]).strip(), nfc(p[4]).strip()
        ks, kt = nkey(s), nkey(t)
        if not ks or not kt or ks == kt:
            n_bad += 1
            continue
        n_rel_lines += 1
        # both endpoints must survive into the final entity set, else drop the
        # relation here (avoids KeyError later AND orphan edges in the index)
        if ks not in ent_desc or kt not in ent_desc:
            n_bad += 1
            continue
        rk = ks + "<SEP>" + kt
        if rk not in rel_desc:
            rel_desc[rk] = []
            rel_kw[rk] = kw or "related"
            rel_ch[rk] = set()
        if ds and len(rel_desc[rk]) < 6:
            rel_desc[rk].append(ds)
            rel_ch[rk].add(c)
ent_keys = sorted(k for k in ent_name if k in ent_desc and k in ent_ch)
idx_of = {k: i for i, k in enumerate(ent_keys)}
print(f"entities={len(ent_keys)} relations={len(rel_desc)} "
      f"ent_lines={n_ent_lines} rel_lines={n_rel_lines} bad={n_bad}", flush=True)
json.dump({"n_ent": len(ent_keys), "n_rel": len(rel_desc), "n_chunk": len(chunks),
           "ent_lines": n_ent_lines, "rel_lines": n_rel_lines, "bad": n_bad},
          open(f"{R}/kg_build_paper_stats.json", "w"), indent=2)

# ---------- embedding helper ----------
_cache_p = f"{R}/kg_emb_paper.npy"
_txt_p = f"{R}/kg_emb_paper_texts.json"
if os.path.exists(_cache_p) and os.path.exists(_txt_p):
    cache_txt = json.load(open(_txt_p))
    cache_m = np.load(_cache_p)
    print("emb cache loaded", cache_m.shape, flush=True)
else:
    cache_txt, cache_m = [], np.zeros((0, 768), dtype=np.float32)


def embed_all(texts, tag):
    global cache_txt, cache_m
    todo = [t for t in texts if t not in set(cache_txt)]
    if todo:
        base = len(cache_txt)
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            for att in range(3):
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:11434/api/embed",
                        data=json.dumps({"model": "embeddinggemma:300m", "input": batch}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        v = np.asarray(json.load(r)["embeddings"], dtype=np.float32)
                    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                    cache_txt.extend(batch)
                    cache_m = np.vstack([cache_m, v]) if len(cache_m) else v
                    np.save(_cache_p, cache_m)
                    json.dump(cache_txt, open(_txt_p, "w"), ensure_ascii=False)
                    break
                except Exception:
                    time.sleep(5)
            if (i // 64) % 50 == 0:
                print(f"  {tag} emb {i}/{len(todo)} cache={len(cache_txt)}", flush=True)
    pos = {t: i for i, t in enumerate(cache_txt)}
    return np.asarray([cache_m[pos[t]] for t in texts], dtype=np.float32)


names = [ent_name[k] for k in ent_keys]
ent_content = [f"{ent_name[k]}\n{ent_type[k]}\n" + " ".join(ent_desc[k])[:MAXDESC] for k in ent_keys]
rel_keys = sorted(rel_desc)
rel_rows = []
for rk in rel_keys:
    ks, kt = rk.split("<SEP>")
    if ks not in ent_name or kt not in ent_name:
        continue
    body = " ".join(rel_desc[rk])[:MAXDESC]
    rel_rows.append({"src_id": ent_name[ks], "tgt_id": ent_name[kt],
                     "content": f"{rel_kw[rk]}\t{ent_name[ks]}\n{ent_name[kt]}\n{body}",
                     "source_id": sorted(rel_ch[rk])[0],
                     "chunk_ids": sorted(rel_ch[rk])})
rel_content = [r["content"] for r in rel_rows]
chunk_content = [c["content"][:3000] for c in chunks]

print("stage2 embed names/entities/relations/chunks", flush=True)
N = embed_all(names, "names")
E = embed_all(ent_content, "ents")
R_ = embed_all(rel_content, "rels")
C = embed_all(chunk_content, "chunks")
print("matrices", N.shape, E.shape, R_.shape, C.shape, flush=True)


def pack(mat, data):
    return {"embedding_dim": 768, "data": data,
            "matrix": base64.b64encode(np.ascontiguousarray(mat, dtype=np.float32).tobytes()).decode()}


ent_data = [{"__id__": h("ent-", k), "entity_name": ent_name[k],
             "content": ent_content[i], "chunk_ids": sorted(ent_ch[k])}
            for i, k in enumerate(ent_keys)]
rel_data = [{"__id__": h("rel-", rk), "src_id": rel_rows[i]["src_id"],
             "tgt_id": rel_rows[i]["tgt_id"], "content": rel_content[i],
             "source_id": rel_rows[i]["source_id"]}
            for i, rk in enumerate(rel_keys)]
chunk_data = [{"__id__": cid[i], "content": chunk_content[i], "url": chunks[i]["url"]}
              for i in range(len(chunks))]
json.dump(pack(E, ent_data), open(f"{OUT}/vdb_entities.json", "w"), ensure_ascii=False)
json.dump(pack(R_, rel_data), open(f"{OUT}/vdb_relationships.json", "w"), ensure_ascii=False)
json.dump(pack(C, chunk_data), open(f"{OUT}/vdb_chunks.json", "w"), ensure_ascii=False)
json.dump(names, open(f"{OUT}/name_list.json", "w"), ensure_ascii=False)
np.save(f"{OUT}/name_matrix_normed.npy", N)
json.dump({ent_name[k]: {"chunk_ids": sorted(ent_ch[k])} for k in ent_keys},
          open(f"{OUT}/kv_store_entity_chunks.json", "w"), ensure_ascii=False)
json.dump({rk: {"chunk_ids": sorted(rel_ch[rk])} for rk in rel_keys},
          open(f"{OUT}/kv_store_relation_chunks.json", "w"), ensure_ascii=False)
json.dump({cid[i]: {"tokens": len(chunks[i]["content"].split()),
                    "content": chunk_content[i], "url": chunks[i]["url"]}
           for i in range(len(chunks))}, open(f"{OUT}/kv_store_text_chunks.json", "w"), ensure_ascii=False)
json.dump({"n_ent": len(ent_data), "n_rel": len(rel_data), "n_chunk": len(chunk_data)},
          open(f"{R}/kg_index_paper_stats.json", "w"), indent=2)
deg = {}
for r in rel_data:
    deg[r["src_id"]] = deg.get(r["src_id"], 0) + 1
    deg[r["tgt_id"]] = deg.get(r["tgt_id"], 0) + 1
dv = list(deg.values())
print("INDEX OK  avg_degree=", round(sum(dv) / max(1, len(dv)), 2),
      " max=", max(dv) if dv else 0, flush=True)