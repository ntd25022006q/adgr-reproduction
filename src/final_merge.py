"""FINAL merge: one paper-mirrored report from EVERY validated artifact.

Reads only version-locked outputs; anything missing is listed under `pending`,
never silently skipped (the old version silently omitted e2e_paper/ragas_paper/
ir_*_paper because it predated those stages). Writes runs/FINAL_v2.json and
prints a compact human-readable diff vs the paper numbers.

Per-table metrics are computed here from the raw judge strings (faith =
supported/total, dim4 = mean of the 4 dimensions) so the merged report does
not depend on any intermediate script having been run.
"""
import json, os, re, statistics as st
R = "/home/devlab1/adgr_workspace/runs/"


def load(f):
    p = R + f
    return json.load(open(p)) if os.path.exists(p) else None


def _num(t, k):
    m = re.search(r'["\']' + k + r'["\']\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', t or "")
    return float(m.group(1)) if m else None


def faith_ratio(rows):
    """official-style faithfulness = sum(supported)/sum(total) over rows.
    Per-row ratios are clamped to [0,1]: the judge occasionally returns
    incoherent JSON (e.g. supported=6/total=1, seen in 2/1200 Tab.7 rows) and
    an unbounded ratio would let 2 rows dominate the sum. Clamping is the
    standard winsorization for a quantity that is definitionally a fraction."""
    s = t = 0.0
    for r in rows:
        a, b = _num(r.get("faith", ""), "supported"), _num(r.get("faith", ""), "total")
        if a is not None and b:
            s += min(1.0, max(0.0, a / b)) * b
            t += b
    return round(s / t, 4) if t else None


def dim4_mean(rows):
    vals = []
    for r in rows:
        v = [_num(r.get("dim4", ""), k) for k in ("comp", "div", "emp", "dir")]
        if all(x is not None for x in v):
            vals.append(sum(v) / 4)
    return round(st.mean(vals), 4) if vals else None


def judge_means(rows):
    rel = [_num(r.get("rel", ""), "relevancy") for r in rows]
    rel = [x for x in rel if x is not None]
    return {"faith": faith_ratio(rows),
            "ans_rel": round(st.mean(rel), 4) if rel else None,
            "recall": None, "dim4": dim4_mean(rows),
            "n": len(rows)}


def recall_mean(rows):
    v = [_num(r.get("crec", ""), "recall") for r in rows]
    v = [x for x in v if x is not None]
    return round(st.mean(v), 4) if v else None


rep = {"tab1": {}, "tab3": {}, "tab4": {}, "tab5": {}, "tab6": {}, "tab7": {},
       "tab8": {}, "pending": []}
PAPER = {
    "tab1": {"chunks": 17236, "ents": 49274, "rels": 106666},
    "tab3_hit10_full": 0.71,
    "tab4_ragas": {"adgr": 0.8792, "hybrid": None, "mix": None},
    "tab7_ragas": {"dense_only": 0.5978, "dense_anchors": 0.7185,
                   "no_edge_filter": 0.8517, "no_rerank": 0.7714, "full": 0.8792},
    "tab6_in_tok_adgr": 4490.5,
}

# ---- Tab.1: corpus + our rebuilt KG ----
pc, ki = load("paper_chunk_stats.json"), load("kg_index_paper_stats.json")
if pc and ki:
    rep["tab1"] = {"paper": PAPER["tab1"],
                   "ours": {"chunks": ki.get("n_chunk", pc.get("n_chunks")),
                            "ents": ki.get("n_ent"), "rels": ki.get("n_rel")}}
    # KG quality (DSER/SCR/LCR/clustering) from kg_quality_paper.json, computed
    # from raw extraction type fields. Kept INSIDE final_merge (not a manual
    # patch) so reruns never drop it again (lost once on 10-12, bug #42).
    kq = load("kg_quality_paper.json")
    if kq:
        rep["tab1"]["ours_quality"] = {
            "dser": kq.get("dser"), "scr": kq.get("scr"), "lcr": kq.get("lcr"),
            "clustering_sample3000": kq.get("clustering_sample"),
            "paper": {"dser": 0.9363, "scr": 0.9277, "lcr": 0.9059, "clustering": 0.029},
            "note": kq.get("note", "computed from raw extraction type fields")}
else:
    rep["pending"].append("tab1 (paper_chunk_stats/kg_index_paper_stats)")

# ---- Tab.3: retrieval metrics (author index + rebuilt index) ----
for tag, fm, fs in [("author_index", "ir_metrics_v2.json", "ir_single_gold.json"),
                    ("paper_index", "ir_metrics_paper.json", "ir_single_gold_paper.json")]:
    m, s = load(fm), load(fs)
    if m and s:
        rep["tab3"][tag] = {"slug_gold": m.get("full"), "single_gold": s.get("full"),
                            "arms_single": s}
        if tag == "author_index" and s["full"]["hit10"] is not None:
            rep["tab3"][tag]["paper_full_hit10"] = PAPER["tab3_hit10_full"]
    else:
        rep["pending"].append(f"tab3 {tag} ({fm}/{fs})")

# ---- Tab.4: RAGAS comparison ----
def _ragas_means(rdict):
    rows = [v for v in (rdict or {}).values() if isinstance(v, dict) and v]
    comp = {}
    for k in ("ans_rel_true", "ctx_rec_true", "ctx_prec_true"):
        vs = [v[k] for v in rows if isinstance(v.get(k), (int, float))]
        comp[k] = round(st.mean(vs), 4) if vs else None
    return comp, len(rows)


ra = load("ragas_true.json")
if ra:
    ae = load("e2e_v3.json") or {}
    rows = [v for v in ae.values() if isinstance(v, dict) and v.get("ans")]
    comp, _ = _ragas_means(ra)
    jm = judge_means(rows)
    fo = load("faith_official.json") or {}
    fov = [v["faith_official"] for v in fo.values()
           if isinstance(v, dict) and v.get("faith_official") is not None]
    entry = {"ragas_true": ra, "judge_means": jm, "ragas_components": comp}
    if fov:
        # two-step official faithfulness on identical data (closes ~60% of the
        # single-call gap: 0.41 -> 0.77 vs paper 0.89). Lives HERE, not as a
        # manual patch (those get wiped by reruns - bug #43).
        entry["faith_protocol_ablation"] = {
            "single_call_solar": jm.get("faith"),
            "two_step_solar": round(st.mean(fov), 4), "two_step_n": len(fov),
            "paper_official_gpt55": 0.8878}
        vals = [jm.get("faith"), comp.get("ans_rel_true"), comp.get("ctx_rec_true"),
                comp.get("ctx_prec_true")]
        if all(v is not None for v in vals):
            entry["overall_ragas_like"] = round(sum(vals) / 4, 4)
    rep["tab4"]["adgr"] = entry
else:
    rep["pending"].append("tab4 adgr (ragas_true.json)")
for m in ("hybrid", "mix"):
    d = load(f"e2e_base_{m}.json")
    if d:
        rows = [v for v in d.values() if isinstance(v, dict) and v.get("ans")]
        jm = judge_means(rows)
        jm["recall"] = recall_mean(rows)
        rep["tab4"][m] = {"judge_means": jm, "all_ok": sum(
            1 for v in rows if v.get("all_ok")), "n": len(rows)}
    else:
        rep["pending"].append(f"tab4 {m} (e2e_base_{m}.json)")
rp = load("ragas_paper.json")
ep = load("e2e_paper.json")
if rp and ep:
    rows = [v for v in ep.values() if isinstance(v, dict) and v.get("ans")]
    comp, _ = _ragas_means(rp)
    jm = judge_means(rows)
    entry = {"ragas_true": rp, "judge_means": jm, "ragas_components": comp}
    vals = [jm.get("faith"), comp.get("ans_rel_true"), comp.get("ctx_rec_true"),
            comp.get("ctx_prec_true")]
    if all(v is not None for v in vals):
        entry["overall_ragas_like"] = round(sum(vals) / 4, 4)
    rep["tab4"]["adgr_paper_index"] = entry
else:
    rep["pending"].append("tab4 adgr_paper_index (ragas_paper/e2e_paper)")

# ---- Tab.5: standalone + 5-run (+ paper-prompt variant, persistent) ----
t5 = load("tab5_5run_summary.json") or load("tab5_summary.json")
if t5:
    rep["tab5"] = {"summary": t5, "source": "5run" if load("tab5_5run_summary.json") else "1run"}
    # paper-prompt re-judge lives here permanently (a manual merge_patch once
    # got wiped by the next FINAL rerun because it lived outside this script -
    # bug #43: merge inputs must live IN the merge script, never as manual
    # post-edits, or reruns silently drop them).
    pp = load("tab5_paperprompt_summary.json")
    if pp:
        rep["tab5"]["paper_prompt"] = {
            "summary": pp,
            "note": "same answers/judge-model/runs; paper-exact prompt (Online Resource 1)"}
else:
    rep["pending"].append("tab5 (tab5_5run_summary/tab5_summary)")

# ---- Tab.6: token cost + latency (paper reports both) ----
for tag, f in (("adgr", "e2e_v3.json"), ("adgr_paper_index", "e2e_paper.json")):
    d = load(f)
    if d:
        toks = [v.get("tok", {}).get("in") for v in d.values()
                if isinstance(v, dict) and v.get("tok", {}).get("in")]
        if toks:
            rep["tab6"][tag] = {"n": len(toks), "mean_in_tok": round(st.mean(toks), 1)}
        lat = {}
        for k in ("ret", "gen", "total"):
            vs = [v["t"][k] for v in d.values()
                  if isinstance(v, dict) and isinstance(v.get("t", {}).get(k), (int, float))]
            if vs:
                lat[f"mean_{k}_s"] = round(st.mean(vs), 2)
        if lat:
            rep["tab6"].setdefault(tag, {"n": len(d)}).update(lat)
if not rep["tab6"]:
    rep["pending"].append("tab6 (token counts)")

# ---- Tab.7: ablation means (raw judge strings) ----
# Paper Tab.7 has FIVE rows (4 ablations + Full ADGR); abl_gen.json holds only
# the 4 ablation arms (full-arm answers live in e2e_v3.json). Add the full arm
# from e2e_v3 so the table is complete.
ag = load("abl_gen.json")
if ag:
    t7 = {}
    for arm, rows in ag.items():
        rs = [v for v in rows.values() if isinstance(v, dict) and v.get("ans")]
        jm = judge_means(rs)
        jm["recall"] = recall_mean(rs)
        jm["paper_ragas"] = PAPER["tab7_ragas"].get(arm)
        t7[arm] = jm
    e3f = load("e2e_v3.json")
    if e3f:
        rs = [v for v in e3f.values() if isinstance(v, dict) and v.get("ans")]
        jm = judge_means(rs)
        jm["recall"] = recall_mean(rs)
        jm["paper_ragas"] = PAPER["tab7_ragas"].get("full")
        t7["full"] = jm
    rep["tab7"] = t7
else:
    rep["pending"].append("tab7 (abl_gen.json)")

# ---- Tab.8: Aescin qualitative case (paper Sec 5.9) + judge agreement ----
# Paper Tab.8: Hybrid 7 ent/139 rel, Mix 7/80, ADGR 5/57 for the Aescin
# contraindication query. Ours: qi=288 ("Aescin + thuoc chong dong") from the
# saved arm rows (ent=n_entities, rel=n_edges). Baseline ent/rel counts are
# LightRAG internals we never logged -> cite paper's, reproduce ADGR's only.
t8 = {"paper": {"hybrid": {"ent": 7, "rel": 139}, "mix": {"ent": 7, "rel": 80},
                "adgr": {"ent": 5, "rel": 57}},
      "ours_qi288": {}}
try:
    for src, tag in (("abl_v3_arms.json", "author_index"),
                     ("abl_paper_arms.json", "paper_index")):
        a = load(src)
        arms = a.get("arms", a) if isinstance(a, dict) else {}
        if arms.get("full") and len(arms["full"]) > 288:
            r = arms["full"][288]
            t8["ours_qi288"][tag] = {"ent": r.get("ent"), "edges": r.get("edges"),
                                     "pool": r.get("pool"), "hit": r.get("hit")}
except Exception as e:
    t8["ours_qi288_error"] = repr(e)[:100]
ka = load("kappa_all_fixed.json")
if ka:
    t8["judge_agreement"] = ka
else:
    rep["pending"].append("tab8 judge agreement (kappa_all_fixed.json)")
rep["tab8"] = t8

json.dump(rep, open(R + "FINAL_v2.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(rep, indent=1, ensure_ascii=False)[:4000])
print("\npending:", rep["pending"] if rep["pending"] else "NONE - all tables present")
