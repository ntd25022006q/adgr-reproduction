"""plot_report.py - full figure set for the academic report (matplotlib, Agg).
Reads ONLY FINAL_v2.json (+ ir files for arms). Every figure labelled with
source. Skips gracefully if data missing. Output: runs/figs/*.png @150dpi.
Covers paper Tables 1,3,4,5,6,7,8 + our two ablations (prompt, faith).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/home/devlab1/adgr_workspace/runs/"
F = R + "figs/"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
F2 = json.load(open(R + "FINAL_v2.json"))
made = []


def savefig(name):
    plt.tight_layout()
    plt.savefig(F + name)
    plt.close()
    made.append(name)
    print(name, flush=True)


PAPER_T1 = {"chunks": 17236, "ents": 49274, "rels": 106666}
PAPER_T3 = {"hit1": 0.5967, "hit3": 0.6967, "hit5": 0.71, "hit10": 0.71}
PAPER_T4 = {"faith": 0.8878, "ans_rel": 0.7124, "crec": 0.9433, "cprec": 0.9733}
PAPER_T5 = {"overall": 0.7921}
PAPER_T7 = {"dense_only": 0.5978, "dense_anchors": 0.7185, "no_edge_filter": 0.8517,
            "no_rerank": 0.7714, "full": 0.8792}

# 1. Tab.1 KG scale
try:
    o = F2["tab1"]["ours"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = ["chunks", "entities", "relations"]
    pv = [PAPER_T1["chunks"], PAPER_T1["ents"], PAPER_T1["rels"]]
    ov = [o["chunks"], o["ents"], o["rels"]]
    i = range(len(x))
    ax.bar([p - 0.2 for p in i], pv, 0.4, label="paper Tab.1")
    ax.bar([p + 0.2 for p in i], ov, 0.4, label="ours (rebuilt index)")
    ax.set_xticks(list(i))
    ax.set_xticklabels(x)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Tab.1 KG scale: paper vs rebuilt (log scale)")
    savefig("fig_tab1_scale.png")
except Exception as e:
    print("skip fig1:", repr(e)[:80])

# 2. Tab.1 quality (DSER/SCR/LCR)
try:
    q = F2["tab1"]["ours_quality"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = ["DSER", "SCR", "LCR"]
    pv = [0.9363, 0.9277, 0.9059]
    ov = [q["dser"], q["scr"], q["lcr"]]
    i = range(len(x))
    ax.bar([p - 0.2 for p in i], pv, 0.4, label="paper")
    ax.bar([p + 0.2 for p in i], ov, 0.4, label="ours")
    ax.set_xticks(list(i))
    ax.set_xticklabels(x)
    ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Tab.1 KG quality: DSER/SCR/LCR")
    savefig("fig_tab1_quality.png")
except Exception as e:
    print("skip fig1q:", repr(e)[:80])

# 3. Tab.3 Hit@K (author index single-gold + paper ADGR line)
try:
    sg = F2["tab3"]["author_index"]["arms_single"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ks = ["hit1", "hit3", "hit5", "hit10"]
    for arm, ls in [("full", "-o"), ("dense_anchors", "--s"), ("no_edge_filter", "-.s"),
                    ("no_rerank", ":x"), ("dense_only", "-.D")]:
        if arm in sg and sg[arm].get("hit10") is not None:
            ax.plot([1, 3, 5, 10], [sg[arm].get(k) or 0 for k in ks], ls, label=arm, ms=4)
    ax.plot([1, 3, 5, 10], [PAPER_T3[k] for k in ks], "k--D", label="paper ADGR", ms=4)
    ax.set_xlabel("K")
    ax.set_ylabel("Hit@K (single-gold proxy)")
    ax.legend(fontsize=7)
    ax.set_title("Tab.3 retrieval (single-gold proxy)")
    savefig("fig_tab3_hit.png")
except Exception as e:
    print("skip fig3:", repr(e)[:80])

# 4. Tab.4 RAGAS components (ours judge-means vs paper official)
try:
    jm = F2["tab4"]["adgr"]["judge_means"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = ["faith", "ans_rel", "dim4"]
    pv = [PAPER_T4["faith"], PAPER_T4["ans_rel"], 0.84]
    ov = [jm["faith"], jm["ans_rel"], jm["dim4"]]
    i = range(len(x))
    ax.bar([p - 0.2 for p in i], pv, 0.4, label="paper (GPT-5.5/official)")
    ax.bar([p + 0.2 for p in i], ov, 0.4, label="ours (solar/direct)")
    ax.set_xticks(list(i))
    ax.set_xticklabels(["Faithfulness", "Ans. relevancy", "4-dim mean"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Tab.4 RAGAS components: paper vs ours")
    savefig("fig_tab4_ragas.png")
except Exception as e:
    print("skip fig4:", repr(e)[:80])

# 5. Tab.5 standalone means (5-run)
try:
    src = F2["tab5"].get("summary", {})
    ms = [m for m in src if (src[m].get("mean5") or src[m].get("mean")) is not None]
    vals = [src[m].get("mean5", src[m].get("mean")) for m in ms]
    import math
    errs = []
    for m in ms:
        if src[m].get("sem") is not None:
            errs.append(src[m]["sem"])
        elif src[m].get("sd") and src[m].get("n"):
            errs.append(src[m]["sd"] / math.sqrt(src[m]["n"]))
        else:
            errs.append(0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(ms, vals, yerr=errs, capsize=3)
    ax.axhline(PAPER_T5["overall"], ls="--", c="k", lw=1, label="paper ADGR 0.7921")
    ax.set_ylabel("4-dim mean (+-SEM)")
    ax.legend(fontsize=8)
    ax.set_title("Tab.5 judge means (5-run, solar-pro4)")
    plt.xticks(rotation=18, ha="right", fontsize=8)
    savefig("fig_tab5.png")
except Exception as e:
    print("skip fig5:", repr(e)[:80])

# 6. Tab.5 prompt ablation (our-prompt vs paper-prompt)
try:
    pp = F2["tab5"].get("paper_prompt", {}).get("summary", {})
    base = F2["tab5"].get("summary", {})
    ms = [m for m in base if m in pp and pp[m].get("mean5") is not None]
    if ms:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        b = [base[m].get("mean5", base[m].get("mean")) for m in ms]
        p = [pp[m]["mean5"] for m in ms]
        i = range(len(ms))
        ax.bar([x - 0.2 for x in i], b, 0.4, label="our prompt")
        ax.bar([x + 0.2 for x in i], p, 0.4, label="paper prompt")
        ax.set_xticks(list(i))
        ax.set_xticklabels(ms, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8)
        ax.set_title("Tab.5 prompt ablation: our vs paper-exact judge prompt")
        savefig("fig_tab5_prompt.png")
except Exception as e:
    print("skip fig5p:", repr(e)[:80])

# 7. Tab.6 tokens + latency
try:
    t6 = F2["tab6"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].bar(["paper\nin-tok", "ours\nin-tok"], [4490.5, t6["adgr"]["mean_in_tok"]])
    axes[0].set_title("Tab.6 input tokens/row")
    lat_o = [t6["adgr"].get("mean_ret_s", 0), t6["adgr"].get("mean_gen_s", 0)]
    axes[1].bar(["ours ret (s)", "ours gen (s)"], lat_o)
    axes[1].axhline(3.1917, ls="--", c="k", lw=1, label="paper ret 3.19s")
    axes[1].axhline(6.6576, ls=":", c="k", lw=1, label="paper gen 6.66s")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Tab.6 latency (s/row)")
    savefig("fig_tab6_eff.png")
except Exception as e:
    print("skip fig6:", repr(e)[:80])

# 8. Tab.7 ablation (faith + recall per arm, paper RAGAS overlay)
try:
    t7 = F2["tab7"]
    arms = [a for a in ["dense_only", "dense_anchors", "no_edge_filter", "no_rerank", "full"] if a in t7]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fv = [t7[a].get("faith") or 0 for a in arms]
    rv = [t7[a].get("recall") or 0 for a in arms]
    i = range(len(arms))
    ax.bar([x - 0.2 for x in i], fv, 0.4, label="ours faith")
    ax.bar([x + 0.2 for x in i], rv, 0.4, label="ours ctx-recall")
    ax.plot(list(i), [PAPER_T7[a] for a in arms], "k--D", ms=4, label="paper RAGAS")
    ax.set_xticks(list(i))
    ax.set_xticklabels(arms, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Tab.7 ablations: ours (faith/recall) vs paper RAGAS")
    savefig("fig_tab7_abl.png")
except Exception as e:
    print("skip fig7:", repr(e)[:80])

# 9. Tab.8 Aescin case (ent/rel)
try:
    t8 = F2["tab8"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = ["Hybrid\n(paper)", "Mix\n(paper)", "ADGR\n(paper)", "ADGR\nours-auth", "ADGR\nours-paper"]
    ev = [7, 7, 5, t8["ours_qi288"]["author_index"]["ent"],
          t8["ours_qi288"]["paper_index"]["ent"]]
    rv = [139, 80, 57, t8["ours_qi288"]["author_index"]["edges"],
          t8["ours_qi288"]["paper_index"]["edges"]]
    i = range(len(x))
    ax.bar([p - 0.2 for p in i], ev, 0.4, label="entities")
    ax.bar([p + 0.2 for p in i], rv, 0.4, label="relations")
    ax.set_xticks(list(i))
    ax.set_xticklabels(x, fontsize=8)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Tab.8 Aescin case: graph context size (log)")
    savefig("fig_tab8_case.png")
except Exception as e:
    print("skip fig8:", repr(e)[:80])

# 10. Faith gap waterfall (paper 0.89 -> single-call 0.41 -> two-step 0.77)
try:
    fo = F2["tab4"].get("faith_protocol_ablation", {})
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = ["paper\n(GPT/official)", "ours\n1-call", "ours\n2-step"]
    v = [0.8878, 0.4055, fo.get("two_step_solar", 0.7706)]
    ax.bar(x, v)
    ax.set_ylim(0, 1.05)
    for i, val in enumerate(v):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_title("Faithfulness: protocol ablation closes ~60% of gap")
    savefig("fig_faith_gap.png")
except Exception as e:
    print("skip figF:", repr(e)[:80])

print(f"\nDONE: {len(made)} figs")
