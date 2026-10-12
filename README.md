# ADGR Reproduction — Vietnamese Medical GraphRAG

**🌐 Language: [English](README.md) | [Tiếng Việt](README.vi.md)**
**📄 Full report (VI, PDF): [report_vi.pdf](report/report_vi.pdf) — 12 pages, 11 figures, every figure with a reading guide**
**📝 English version (LaTeX source, sentence-aligned translation of the VI report): [report.tex](report/report.tex)**

Independent, from-scratch reproduction of *Adaptive Directed Graph Retrieval*
(paper submitted to *Applied Intelligence*): answering Vietnamese medical
questions by retrieving evidence from a knowledge graph (diseases, drugs,
symptoms, relations). The Vietnamese report is the single master; the English
version translates it sentence by sentence.

Standing rule: **no identical numbers demanded under different conditions —
but zero differences without a measured cause.** 9 verdicts MATCH/CLOSE/
SUPERIOR, 5 EXPLAINED, 0 UNEXPLAINED.

## Pipeline overview

Question → memoized LLM keywords (1 call/question) → hybrid anchors (BM25
top-10 ∪ dense top-10, overlap bonus, top-15) → one-hop edges, edge filter,
quota m, cap Nmax → global + direct safety nets → merge (⌊rn/2⌋ per source,
round-robin) → top-10 rerank (Qwen3-0.6B) → Qwen2.5-14B generation (temp 0) →
Hit/MRR/nDCG scoring, formula-faithful RAGAS, 4-dimension × 5-run judging.

## Figure 1 — Graph scale: same league as the paper (Table 1)

<img src="report/figs/fig_tab1_scale.png" width="700">

52,578 entities / 106,279 relations rebuilt vs 49,274 / 106,666 in the paper
(107% / −0.4%), from 15,837 chunks (91.9% — 5 URLs died at crawl time, 4
rescued from alternates, all disclosed). Log axis is mandatory: the ~100k
relation bars crush the 16k chunk bars on a linear axis.

## Figure 2 — Graph quality (Table 1)

<img src="report/figs/fig_tab1_quality.png" width="700">

DSER 0.913 / SCR 0.884 / LCR 0.816 (paper: 0.936 / 0.928 / 0.906), recomputed
from raw extraction. Deepest gap is LCR −9pp: coarse chunks leave 18% of
entities on disconnected islands (paper: 9%).

## Figure 3 — Extraction density: the chunker-size evidence (new)

<img src="report/figs/fig_rates.png" width="650">

Our chunker yields ~9.4 entity hits/chunk — 3× the author chunker (~3.3) and
the paper (~2.9). The extractor is not greedier: run on author-style chunks,
its density matches the paper immediately. This single measurement explains
both the low LCR and why 3.3 distinct entities coexist with 9.4 hits (same
entities repeat across chunks).

## Figure 4 — Retrieval coverage tracks the paper (Table 3)

<img src="report/figs/fig_tab3_hit.png" width="700">

Full-pipeline Hit@10 **0.79 / 0.68** on two indexes (paper: 0.71). Optimistic
slug-gold: 0.68 / 0.85 / 0.89 / 0.94 across K=1/3/5/10. The odd order
(dense_only > no_rerank > full) is a mathematical consequence of the
single-gold definition — two independent proxies agree, so it is not a system
fault. Real rank-1 weakness (Hit@1 0.43 vs 0.60) converges from Hit@5: the
evidence IS retrieved, just ranked lower at the very top.

## Figure 5 — RAGAS: the gap sits in exactly one cell (Table 4)

<img src="report/figs/fig_tab4_ragas.png" width="700">

Three pairs (faithfulness, answer relevancy, 4-dim mean): two track, only
faithfulness diverges (0.41 vs 0.89). Standard-algorithm RAGAS on identical
data: ans.rel. 0.66 (300 samples), ctx.rec. 0.64 (275), ctx.prec. 0.86 (46).

## Figure 6 — Decomposing the faithfulness gap by experiment (Table 4)

<img src="report/figs/fig_faith_gap.png" width="650">

0.41 → 0.77 → 0.89. Short answers match the paper (pipeline intact); proper
two-step scoring reaches 0.77 (~75% of the gap is the one-step shortcut); the
rest is a stricter judge plus noise (no GPT-5.5 — a stated limit, never fixed
by softening the judging).

## Figure 7 — Highest and stable under both prompts (Table 5)

<img src="report/figs/fig_tab5.png" width="750">

ADGR 0.858 ≈ Mix 0.858 > Hybrid 0.829 > standalone, mean of 5 runs (SEM 0.046,
second-shortest after Mix 0.042). Highest bar with near-shortest whiskers —
high and stable.

## Figure 8 — The original prompt punishes verbosity by design (Table 5)

<img src="report/figs/fig_tab5_prompt.png" width="700">

Original prompt scores ~0.08 lower (ADGR 0.778) with **zero rank flips across
10 methods**. Verbose models (14B pair near 500 words/answer, cf-llama 647)
drop hardest; concise ones barely move.

## Figure 9 — Cheaper than paper on every axis (Table 6)

<img src="report/figs/fig_tab6_eff.png" width="700">

Input 3,441 vs 4,490 tokens (−23%); retrieval 2.36s vs 3.19s; generation 5.82s
vs 6.66s; total 9.85s → 8.18s. Spend retrieval compute to save generation
cost — ADGR's advertised trade-off, reproduced in both cost dimensions.

## Figure 10 — Every disabled module hurts in paper order (Table 7)

<img src="report/figs/fig_tab7_abl.png" width="750">

Paper RAGAS per arm: dense_only 0.60; dense_anchors 0.72; no_rerank 0.77;
no_edge_filter 0.85; full 0.88 — the exact descending importance order. Our
bars sit lower (different judge) but the spread shape matches. On our faith
scale, dense_anchors is lowest (0.356) — both scales agree: the anchor stage
is load-bearing.

## Figure 11 — The Aescin case on a log axis (Table 8)

<img src="report/figs/fig_tab8_case.png" width="700">

Hybrid drags 139 relations (including other drugs' contraindications — right
concept, wrong entity, dangerously misleading); ADGR keeps just the 3 needed
ones. Ours: 12–14 entities / 39–41 relations — the same compact regime. Judge
agreement Fleiss κ 0.19/0.14 (270 samples, factor-2 bug fixed) — a disclosed
limit of automated judging.

## Results vs the paper (summary)

| Table | Metric | Paper | Ours | Verdict |
|---|---|---|---|---|
| 1 | chunks / entities / relations | 17236 / 49274 / 106666 | 15837 / 52578 / 106279 | MATCH (92% / −0.4%) |
| 1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | CLOSE–EXPLAINED |
| 3 | Hit@1 / Hit@10 full | 0.5967 / 0.7100 | 0.4333 / 0.7933 ; 0.6833* | MATCH (in proxy range) |
| 3 | MRR / nDCG@10 full | 0.6437 / 0.6606 | 0.5538* / 0.6118* | CLOSE |
| 4 | RAGAS / faithfulness | 0.8792 / 0.8878 | 0.64 / 0.41 (0.77 two-step) | EXPLAINED, 3 layers |
| 5 | ADGR overall | 0.7921 | 0.858 / 0.778 (orig. prompt) | MATCH, rankings kept |
| 6 | input tokens/row; ret/gen | 4490.5; 3.19s / 6.66s | 3441.4; 2.36s / 5.82s | SUPERIOR (−23%, 9.85s→8.18s) |
| 7 | 5 arms (paper RAGAS) | 0.60–0.88 | spread confirmed | MATCH direction |
| 8 | Aescin H/M/ADGR | 7/139, 7/80, 5/57 | 12–14 / 39–41 | MATCH regime |

\*automatic proxies (the paper uses hand-made gold labels). The full verdict
table with one-line causes is in the report.

## Documented substitutions (all measured, none hidden)

| Paper | Here | Measured effect |
|---|---|---|
| Generator / embedding / reranker | SAME models per paper Sec 5.1 (Qwen2.5-14B, embeddinggemma:300m, Qwen3-Reranker-0.6B, vLLM, temp 0) | identical; only vector store differs (numpy+BM25 vs NanoVectorDB) |
| GPT-5.5 judge | solar-pro4, same protocol | ~0.08 scale shift, rankings kept |
| 7 paid standalone LLMs | 7 free models (zero overlap) | references only |
| Manual gold labels | slug-gold + single-gold proxies (agree) | paper 0.71 inside range |
| Released chunker | authors' H2 code (thesis_src, external), 91.9% by count | 8% gap from dead URLs + filtering; ~9.4 hits/chunk |
| Official RAGAS library | formulas reimplemented | ans.rel 0.66 / rec 0.64 / prec 0.86 |
| Two-step faithfulness | one-step (cheap) + two-step check | 0.41 vs 0.77 on same data |

## Repository layout

```
├── README.md / README.vi.md  # this file, English + Vietnamese
├── AGENTS.md                  # operating rules distilled from 43 verified fixes
├── LICENSE                    # MIT
├── report/                    # reports (VI PDF + VI/EN LaTeX) + 11 figures
├── src/                       # pipeline: retrieval, evaluation, 16-stage chain
├── results/                   # FINAL_v2.json (8 tables) + all raw answers/judgments
└── corpus/                    # articles, chunks, extraction, contexts (<100MB each)
```

Large artifacts live outside GitHub by design (rebuild instructions in
`AGENTS.md`): author index matrices (2.3 GB), rebuilt index (960 MB), model
weights (9.4 GB).

## Reproduce

```bash
python -u src/kg_build_paper.py   # build the knowledge-graph index
python -u src/chain_full.py       # run everything (per-question checkpoints,
                                  # resume-safe, validated stage gates)
```

Needs: 1 GPU with 24GB, vLLM + ollama, Python 3.11, an Upstage (Solar) key
(~$5 for the full judging protocol). Read `AGENTS.md` first — every rule
there exists because violating it cost GPU hours or corrupted a table.

> KG-build note: `src/paper_chunker.py` calls the authors' chunking code
> (`thesis_src`, external, not redistributed) and `src/extract_paper_full.py`
> needs the 3 prompt files `tmp/ext_*.txt` (authors' extraction prompt). Every
> output of this path (`corpus/chunks_paper.json`, extraction, index) already
> ships in the repo, so the chain skips the KG-build stage when outputs exist.

## Cite the original paper

```bibtex
@article{nguyentienha2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {{Nguyen Tien Ha} and {Vu Duy Linh} and {Nguyen Thi Minh Huyen}
             and {Le Ngoc Toan} and {Kim Dinh Thai}},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```
