"""E2E v3 — fixes self-inflicted context truncation found in audit, and merges
bench_ours.py so retrieval is done ONCE per question for all 4 arms
(full / dense_anchors / no_edge_filter / no_rerank).
Audit evidence: full top-10 context = 13,050 chars mean (~4,350 tok) vs paper's
30k budget -> never binding; v2 fed gen only 9,000 chars (89% rows truncated,
32% of context lost) and judges only 7,000 chars. v3 feeds full context.
Reranker no longer cuts docs at 1,500 chars (14% of chunk text was dropped), so
results are NOT comparable to the old abl_n300v3 table.
Also: generation runs on the vLLM server already resident for extraction ->
zero extra GPU memory (paper-faithful Qwen2.5-14B-Instruct-AWQ)."""
import sys, json, csv, time, os, re
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("ADGR_DATA", "/home/devlab1/adgr_workspace/data")
sys.path.insert(0, "/home/devlab1/adgr_workspace/tmp")
from adgr import load_index, build_bm25, retrieve, ogen as _ogen
from rerank import rerank as cereank
from eval_utils import valid_ans, kv
from openai import OpenAI

CKPT = os.environ.get("CKPT", "/home/devlab1/adgr_workspace/runs/e2e_v3.json")
CK_ARMS = os.environ.get("CK_ARMS", "/home/devlab1/adgr_workspace/runs/abl_ours_arms.json")
JUDGE_WORKERS = 5
CTX_CAP = int(os.environ.get("CTX_CAP", "30000"))
_key = open(os.path.expanduser("~/.config/adgr/upstage_key")).read().strip()
_judge = OpenAI(api_key=_key, base_url="https://api.upstage.ai/v1", timeout=300)
COUNT_KEYS = {"supported", "total", "useful"}


def _fatal_api(e):
    import re as _re
    t = _re.sub(r"[^a-z]", "", repr(e).lower())
    return any(p in t for p in ("suspended", "insufficient", "invalidapikey",
                                "unauthorized", "accountdeactivated", "billing",
                                "paymentrequired", "outofcredit", "permissiondenied",
                                "zerobalance"))


def judge_ok(prompt, keys, tries=3):
    # Effort escalation: minimal (~116 out-tok) then low fallback (28x saving).
    import os as _os
    prompt = prompt + "\nTra loi HOAN TOAN bang tieng Viet. Chi tra ve JSON."
    last = ""
    for eff in (["minimal"] * max(1, tries - 1) + ["low"]):
        try:
            last = _judge.chat.completions.create(
                model="solar-pro4", messages=[{"role": "user", "content": prompt}],
                reasoning_effort=eff, max_tokens=6000).choices[0].message.content or ""
            vals = [(k, kv(last, k)) for k in keys]
            if all(v is not None and (k in COUNT_KEYS or 0 <= v <= 1) for k, v in vals):
                return last, True
        except Exception as e:
            if _fatal_api(e):
                print(f"FATAL solar billing/auth: {repr(e)[:150]} - aborting, "
                      f"checkpoint preserved, refill credit and resume.", flush=True)
                _os._exit(42)
            time.sleep(5)
    return last, False


P_FAITH = ("Cho cau tra loi va nguc canh. Tach thanh menh de su that (toi da 10). "
           "Moi menh de 1 neu nguc canh ho tro truc tiep, 0 neu khong. "
           "Tra ve JSON {{\"supported\":s,\"total\":t}}.\nNguc canh:\n{ctx}\nCau tra loi:\n{ans}")
P_REL = ("Cau hoi: {q}\nCau tra loi: {ans}\nDo lien quan 0-1. Tra ve JSON {{\"relevancy\":x}}.")
P_REC = ("Cau hoi: {q}\nDap an mau: {ref}\nNguc canh:\n{ctx}\nTy le y chinh co mat 0-1. "
         "Tra ve JSON {{\"recall\":x}}.")
P_PREC = ("Cau hoi: {q}\nNguc canh:\n{ctx}\nMoi doan 1 neu lien quan truc tiep, 0 neu khong. "
          "Tra ve JSON {{\"useful\":u,\"total\":10}}.")
P_4DIM = ("Ban la chuyen gia y khoa. Cham 0-1: Comprehensiveness (bao phu y chinh dap an mau), "
          "Diversity, Empowerment, Directness (gon, trung cau). Chi gia tri [0,1].\n"
          "Cau hoi: {q}\nDap an mau: {ref}\nCan cham: {ans}\nJSON {{\"comp\":x,\"div\":x,\"emp\":x,\"dir\":x}}.")
GEN = ("Tra loi cau hoi y khoa tieng Viet CHI dua tren nguc canh duoi day. Ngan gon, chinh xac.\n\n"
       "Nguc canh:\n{ctx}\n\nCau hoi: {q}\nTra loi:")

print("loading...", flush=True)
IDX = load_index()
eidx = list(range(len(IDX["edata"])))
bm25 = build_bm25([(i, r.get("entity_name", "") + " " + r.get("content", ""))
                   for i, r in enumerate(IDX["edata"])])
id2pos = {c["__id__"]: i for i, c in enumerate(IDX["cdata"])}
KEY = "url" if "url" in IDX["cdata"][0] else "file_path"


def slug(u):
    # same robust form as ir_recompute (bug #39): bare slugs must survive.
    u = (u or "").rstrip("/")
    m = re.search(r"/([^/]+)$", u)
    return (m.group(1) if m else u).replace(".html", "")


slug2chunks = {}
for c in IDX["cdata"]:
    # normalize BOTH sides to the URL slug (last path component): the author
    # index stores bare "slug.txt" file_paths (which matched by accident),
    # but the rebuilt index stores full "https://.../slug/" urls - without
    # normalization every lookup misses and hit=None for all rows (seen R11).
    slug2chunks.setdefault(slug(str(c.get(KEY, "")).replace(".txt", "")), []).append(c["__id__"])
rows = list(csv.DictReader(open("/home/devlab1/adgr_workspace/data/q300.csv")))
print("index:", {k: (v.shape if hasattr(v, "shape") else len(v))
                 for k, v in IDX.items() if k in ("C", "E", "R", "names")},
      "rows:", len(rows), flush=True)

ARMS = {"full": {}, "dense_anchors": {"dense_anchors": True},
        "dense_only": {"dense_only": True},
        "no_edge_filter": {"no_edge_filter": True}, "no_rerank": {"no_rerank": True}}
try:
    _raw = json.load(open(CK_ARMS))
    # CK_ARMS is stored WRAPPED as {"arms": {...}}. Loading the whole file and
    # treating it as the arms dict made every rerun of this script (chain
    # restarts) mangle the file: need_arms saw all arms empty, and the final
    # write re-wrapped the wrapper ({"arms": {"arms": ...}}), which then broke
    # ablation_gen's json.load(...)["arms"][arm]. Unwrap defensively: accept
    # both the wrapped and the (hand-repaired) unwrapped form.
    armrows = _raw["arms"] if ("arms" in _raw and "full" not in _raw) else _raw
except Exception:
    armrows = {k: [] for k in ARMS}


VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_MODEL = "/home/devlab1/adgr_workspace/models/qwen25-awq"


def gen_vllm(prompt, max_tokens=400):
    import urllib.request
    body = {"model": VLLM_MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    u = d.get("usage") or {}
    return (d["choices"][0]["message"]["content"] or "").strip(), \
        int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


GEN_BACKEND = os.environ.get("GEN_BACKEND", "vllm")
gen_fn = gen_vllm if GEN_BACKEND == "vllm" else (lambda p, m=400: _ogen(p, m, True))
print("gen backend:", GEN_BACKEND, flush=True)

try:
    out = {int(k): v for k, v in json.load(open(CKPT)).items()}
    print("resumed:", len(out), flush=True)
except Exception:
    out = {}


def save():
    json.dump({str(k): v for k, v in out.items()}, open(CKPT, "w"), ensure_ascii=False)


for qi, row in enumerate(rows):
    if qi in out and out[qi].get("done") and "error" not in out[qi]:
        continue
    q = row["question"]
    t00 = time.time()
    try:
        need_arms = not all(len(armrows.get(name, [])) > qi for name in ARMS)
        if need_arms:
            for name in ARMS:
                armrows[name] = armrows.get(name, [])[:qi]
            for name, a in ARMS.items():
                r = retrieve(IDX, bm25, eidx, q, ablation=a)
                dids = [c for c in r["pool"] if c in id2pos]
                docs = [IDX["cdata"][id2pos[c]]["content"] for c in dids]
                t_rr = time.time()
                # Paper Sec 3.6/5.8: the cross-encoder rerank stage must be
                # REMOVED for the "w/o Cross-Encoder Reranking" arm (top-10
                # straight from the merged pool, in pool order). dense_only's
                # pool is already the dense similarity order, so rerank would
                # only reorder the same set - skip it there too (also keeps the
                # reranker off the GPU for those two arms).
                if a.get("no_rerank") or a.get("dense_only"):
                    s = [0.0] * len(docs)
                else:
                    s = cereank(q, docs) if docs else []
                t_rr = round(time.time() - t_rr, 2)
                order = sorted(range(len(docs)), key=lambda i: -s[i])[:10]
                top_ids = [dids[i] for i in order]
                g_ = set(slug2chunks.get(slug(row.get("article_url")), []))
                RET = {}
                RET[name] = {"ids": top_ids, "rr_s": t_rr,
                             "hit": len(set(top_ids) & g_) > 0 if g_ else None}
                armrows[name].append({"hit": RET[name]["hit"], "pool": len(r["pool"]),
                                      "edges": r["n_edges"], "ent": r["n_entities"],
                                      "ret_s": r["ret_s"], "t_kw": r["t_kw"],
                                      "anchors": r["anchors"][:5],
                                      "top10": top_ids})
            json.dump(armrows, open(CK_ARMS, "w"), ensure_ascii=False)
        dids = armrows["full"][qi]["top10"]
        docs = [IDX["cdata"][id2pos[c]]["content"] for c in dids if c in id2pos]
        t_rr = armrows["full"][qi].get("rr_s", 0)
        ctx = "\n\n".join(docs)
        t_g = time.time()
        ans, tok_in, tok_out, gr = "", 0, 0, 0
        for _ in range(3):
            try:
                a, ti, to = gen_fn(GEN.format(ctx=ctx[:CTX_CAP], q=q), 400)
            except Exception:
                time.sleep(2)
                continue
            if valid_ans(a):
                ans, tok_in, tok_out = a, ti, to
                break
            ans = a
            gr += 1
            time.sleep(2)
        t_g = round(time.time() - t_g, 2)
        if not valid_ans(ans):
            if out.get(qi, {}).get("gentries", 0) >= 2:
                out[qi] = {"gen_failed": True, "done": True}
                save()
                print(f"[{qi+1}/300] GEN_FAIL (final)", flush=True)
                continue
            prior = out.get(qi, {})
            prior["gentries"] = prior.get("gentries", 0) + 1
            prior.pop("gen_failed", None)
            out[qi] = prior
            save()
            print(f"[{qi+1}/300] GEN_FAIL retry attempt", flush=True)
            continue
        g = set(slug2chunks.get(slug(row.get("article_url")), []))
        numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(docs))
        jobs = [("faith", P_FAITH.format(ctx=ctx[:CTX_CAP], ans=ans[:1500]), ["supported", "total"]),
                ("rel", P_REL.format(q=q, ans=ans[:1500]), ["relevancy"]),
                ("crec", P_REC.format(q=q, ref=row["answer"][:1200], ctx=ctx[:CTX_CAP]), ["recall"]),
                ("cprec", P_PREC.format(q=q, ctx=numbered[:CTX_CAP]), ["useful"]),
                ("dim4", P_4DIM.format(q=q, ref=row["answer"][:1200], ans=ans[:1500]),
                 ["comp", "div", "emp", "dir"])]
        res = {}
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            futs = {ex.submit(judge_ok, p, k): n for n, p, k in jobs}
            for fu in futs:
                try:
                    res[futs[fu]] = fu.result()
                except Exception:
                    res[futs[fu]] = ("", False)
        (jf, a), (jr, b), (jc, c), (jp, d_), (j5, e_) = (
            res["faith"], res["rel"], res["crec"], res["cprec"], res["dim4"])
        out[qi] = {"hit": len(set(dids) & g) > 0 if g else None,
                   "edges": armrows["full"][qi]["edges"],
                   "anchors": armrows["full"][qi]["anchors"],
                   "ans": ans[:2000], "ctx_ids": dids, "ctx_chars": len(ctx),
                   "faith": jf[:300], "rel": jr[:150], "crec": jc[:150],
                   "cprec": jp[:200], "dim4": j5[:400],
                   "all_ok": all([a, b, c, d_, e_]), "done": True,
                   "tok": {"in": tok_in, "out": tok_out},
                   "t": {"ret": armrows["full"][qi]["ret_s"],
                         "kw": armrows["full"][qi]["t_kw"], "rr": t_rr, "gen": t_g,
                         "total": round(time.time() - t00, 1)}}
        save()
        print(f"[{qi+1}/300] ok={out[qi]['all_ok']} ctx={len(ctx)} "
              f"tok_in={tok_in}", flush=True)
    except Exception as e:
        old = out.get(qi, {}).get("tries", 0)
        out[qi] = {"error": repr(e)[:200], "tries": old + 1}
        if old >= 2:
            out[qi]["done"] = True
        save()
        print(f"[{qi+1}/300] ERROR try{old+1} {repr(e)[:80]}", flush=True)
json.dump({"arms": {k: v for k, v in armrows.items()}}, open(CK_ARMS, "w"), ensure_ascii=False)
print("wrote", CKPT, "and", CK_ARMS)
