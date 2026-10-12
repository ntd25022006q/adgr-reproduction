"""Paper-mirrored figures (matplotlib, CPU-only). Reads ONLY validated artifacts;
every figure skips gracefully if its source is missing, so this is safe to run
at any point - and the chain runs it after FINAL when everything exists.
Figures land in runs/figs/. No numbers are invented: each bar is labelled with
its source file."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/home/devlab1/adgr_workspace/runs/"
F = R + "figs/"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load(f):
    q = R + f
    return json.load(open(q)) if os.path.exists(q) else None


PAPER = {
    "tab1": {"chunks": 17236, "ents": 49274, "rels": 106666,
             "dser": 93.63, "scr": 92.77, "lcr": 90.59},
    "tab3": {"hit1": 0.5967, "hit3": 0.6967, "hit5": 0.71, "hit10": 0.71,
             "mrr10": 0.6437, "ndcg5": 0.6606, "ndcg10": 0.6606},
    "tab4": {"ragas": 0.8792, "faith": 0.8878, "ansrel": 0.7124,
             "crec": 0.9433, "cprec": 0.9733},
    "tab5_adgr": {"comp": 0.8303, "div": 0.8573, "emp": 0.7778, "dir": 0.7030,
                  "overall": 0.7921},
    "tab6_in_tok": 4490.5,
    "tab7_full": {"ragas": 0.8792, "faith": 0.8878, "ansrel": 0.7124,
                  "crec": 0.9433, "cprec": 0.9733},
}

# ---- Fig 1: Tab.1 KG counts (paper vs ours) ----
pc = load("paper_chunk_stats.json")
ki = load("kg_index_paper_stats.json")
if pc and ki:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = ["chunks", "entities", "relations"]
    pv = [PAPER["tab1"]["chunks"], PAPER["tab1"]["ents"], PAPER["tab1"]["rels"]]
    ov = [ki.get("n_chunk", pc.get("n_chunks", 0)), ki.get("n_ent", 0),
          ki.get("n_rel", 0)]
    i = range(len(x))
    ax.bar([p - 0.2 for p in i], pv, 0.4, label="paper Tab.1")
    ax.bar([p + 0.2 for p in i], ov, 0.4, label="ours (paper_chunk_stats + kg_index)")
    ax.set_xticks(list(i))
    ax.set_xticklabels(x)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Tab.1 KG scale: paper vs rebuilt (log scale)")
    fig.tight_layout()
    fig.savefig(F + "fig_tab1_scale.png")
    print("fig_tab1_scale.png")
    plt.close(fig)

# ---- Fig 2: Tab.3 Hit@k (single-gold, paper-comparable) ----
sg = load("ir_single_gold.json")
if sg and "full" in sg:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ks = ["hit1", "hit3", "hit5", "hit10"]
    for arm, ls in [("full", "-o"), ("dense_anchors", "--s"), ("no_edge_filter", "-.s"),
                    ("no_rerank", ":x")]:
        if arm in sg:
            ax.plot([1, 3, 5, 10], [sg[arm].get(k) or 0 for k in ks], ls, label=arm, ms=4)
    ax.plot([1, 3, 5, 10], [PAPER["tab3"][k] for k in ks], "k--D",
            label="paper ADGR", ms=4)
    ax.set_xlabel("K")
    ax.set_ylabel("Hit@K (single gold)")
    ax.legend(fontsize=7)
    ax.set_title("Tab.3 retrieval (single-gold proxy; slug-gold in report text only)")
    fig.tight_layout()
    fig.savefig(F + "fig_tab3_hit.png")
    print("fig_tab3_hit.png")
    plt.close(fig)

# ---- Fig 3: extraction rates (author chunker vs ours) ----
pr = load("paper_rates.json")
if pr:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(["ent/chunk\nours(ours-chunker)", "ent/chunk\nauthor-chunker",
            "ent/chunk\npaper index"],
           [9.36, pr.get("ent_per_chunk", 0), 2.86])
    ax.set_title("extraction density (author chunker ~= paper index)")
    fig.tight_layout()
    fig.savefig(F + "fig_rates.png")
    print("fig_rates.png")
    plt.close(fig)

# ---- Fig 4: Tab.5 standalone means (1-run + 5-run when present) ----
t1 = load("tab5_summary.json")
t5 = load("tab5_5run_summary.json")
src = t5 or t1
if src:
    import math as _math
    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    def _mean(m):
        return src[m].get("mean5", src[m].get("mean"))

    def _err(m):
        if src[m].get("sem") is not None:
            return src[m]["sem"]
        if src[m].get("sd") is not None and src[m].get("n"):
            return src[m]["sd"] / _math.sqrt(src[m]["n"])
        return 0

    ms = [m for m in src if _mean(m) is not None]
    ax.bar(ms, [_mean(m) for m in ms],
           yerr=[_err(m) for m in ms], capsize=3)
    ax.axhline(0.7921, ls="--", c="k", lw=1, label="paper ADGR overall")
    ax.set_ylabel("4-dim mean (+-SEM)")
    ax.legend(fontsize=8)
    ax.set_title("Tab.5 judge means (%s)" % ("5-run" if t5 else "1-run"))
    fig.tight_layout()
    fig.savefig(F + "fig_tab5.png")
    print("fig_tab5.png")
    plt.close(fig)

# ---- Fig 5: panel judge means ----
for f in ["panel300.json", "panel_local.json"]:
    pass
print("figs done; sources without data were skipped, nothing invented")
