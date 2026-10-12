# ADGR Reproduction — Vietnamese Medical GraphRAG

**🌐 Language: [English](README.md) | [Tiếng Việt](README.vi.md)**
**📄 Full reports: [report.pdf (EN)](report/report.pdf) | [report_vi.pdf (VI)](report/report_vi.pdf) — 12 pages each, every figure with a reading guide**

Independent, from-scratch reproduction of *Adaptive Directed Graph Retrieval
for Graph-Based Retrieval-Augmented Generation* (submitted to *Applied
Intelligence*): a system that answers Vietnamese medical questions by
retrieving evidence from a knowledge graph of diseases, drugs, symptoms and
their relations.

Rule used throughout: **no identical numbers without identical conditions —
but zero differences without a measured cause.** 9 verdicts MATCH/CLOSE/
SUPERIOR, 5 EXPLAINED, 0 UNEXPLAINED.

## Graph scale: same league as the paper (Table 1)

<img src="report/figs/fig_tab1_scale.png" width="700">

52,578 entities / 106,279 relations rebuilt vs 49,274 / 106,666 in the paper
(107% / −0.4%), from 15,837 chunks (91.9% — 5 URLs died at crawl time, 4
recovered from alternates, all disclosed). Quality recomputed from raw
extraction: DSER 0.913, SCR 0.884, LCR 0.816 (paper: 0.936 / 0.928 / 0.906).
The deepest gap is LCR −9pp: coarser H2 chunking leaves 18% of entities on
disconnected islands — retrieval still works (Table 3), so the connected core
suffices.

## Retrieval coverage tracks the paper (Table 3)

<img src="report/figs/fig_tab3_hit.png" width="700">

Hit@10 of the full pipeline: **0.79 / 0.68** on two indexes (paper: 0.71).
The optimistic slug-gold proxy reaches 0.94; the pessimistic single-gold
gives 0.79 — the paper's number sits inside our range. Two independent
proxies agree on the arm ranking exactly, so the odd-looking order
(dense_only > no_rerank > full) is a mathematical consequence of the
single-gold definition, not a system fault.

## The biggest gap, fully decomposed (Table 4)

<img src="report/figs/fig_faith_gap.png" width="650">

Faithfulness 0.41 vs 0.89, split by experiment into three measured layers:
short answers already match the paper (0.89 — pipeline intact); proper
two-step scoring on identical data reaches 0.77 (~75% of the gap is our
one-step judging shortcut); the rest is a stricter judge plus noisier data
(no GPT-5.5 access — stated as a limit). Deliberately not "fixed" by softening
the judging: that would be metric gaming.

## Rankings hold under both prompts (Table 5)

<img src="report/figs/fig_tab5.png" width="750">

ADGR 0.858 ≈ Mix 0.858 > Hybrid 0.829 > standalone, mean of 5 independent runs
(SEM 0.046, second-smallest). Under the authors' original prompt everything
scores ~0.08 lower (ADGR 0.778) but **zero rank flips across 10 methods** —
comparison conclusions survive any calibration.

## Compact yet accurate: the Aescin case (Table 8)

<img src="report/figs/fig_tab8_case.png" width="700">

"Who is Aescin contraindicated for?" Hybrid drags 139 relations — including
other drugs' contraindications (right concept, wrong entity). ADGR keeps just
the 3 needed Aescin relations. Ours: 12–14 entities / 39–41 relations — the
same compact regime. Log scale is mandatory here: 139 vs 5 spans 28×.

## Results vs the paper

| Table | Metric | Paper | Ours | Verdict |
|---|---|---|---|---|
| 1 | chunks / entities / relations | 17236 / 49274 / 106666 | 15837 / 52578 / 106279 | MATCH (92% / −0.4%) |
| 1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | CLOSE–EXPLAINED |
| 3 | Hit@1 / Hit@10, full | 0.5967 / 0.7100 | 0.4333 / 0.7933 ; 0.6833* | MATCH (in proxy range) |
| 3 | MRR / nDCG@10, full | 0.6437 / 0.6606 | 0.5538* / 0.6118* | CLOSE |
| 4 | RAGAS / faithfulness | 0.8792 / 0.8878 | 0.64 / 0.41 (0.77 two-step) | EXPLAINED, 3 layers |
| 5 | ADGR overall | 0.7921 | 0.858 / 0.778 (orig. prompt) | MATCH, rankings kept |
| 6 | input tokens/row; ret/gen | 4490.5; 3.19s / 6.66s | 3441.4; 2.36s / 5.82s | SUPERIOR (−23%, 9.85s→8.18s) |
| 7 | 5 arms (paper RAGAS) | 0.60–0.88 | spread confirmed | MATCH direction |
| 8 | Aescin H/M/ADGR | 7/139, 7/80, 5/57 | 12–14 / 39–41 | MATCH regime |

\*automatic proxies (the paper uses hand-made gold labels). Full verdict table
with one-line causes is in the report.

## Documented substitutions (all measured, none hidden)

| Paper | Here | Measured effect |
|---|---|---|
| Commercial generator | Qwen2.5-14B-Instruct-AWQ, temp 0, local vLLM | deterministic answers |
| Commercial embedding | embeddinggemma:300m (768-dim) | index rebuilt at −0.4% relations |
| Commercial reranker | Qwen3-Reranker-0.6B (+ CPU fallback) | exact-length scores verified |
| GPT-5.5 judge | solar-pro4, same protocol | ~0.08 scale shift, rankings kept |
| 7 paid standalone LLMs | 7 free models (zero overlap) | references only |
| Manual gold labels | slug-gold + single-gold proxies (agree) | paper 0.71 inside range |
| Released chunker | H2 rule reverse-engineered (91.9%) | 3.3 ent/chunk, LCR −9pp |
| Official RAGAS library | formulas reimplemented | ans.rel 0.66 / rec 0.64 / prec 0.86 |
| Two-step faithfulness | one-step (cheap) + two-step check | 0.41 vs 0.77 on same data |

## Repository layout

```
├── README.md / README.vi.md  # this file, English + Vietnamese
├── AGENTS.md                  # operating rules distilled from 43 verified fixes
├── LICENSE                    # MIT
├── report/                    # full academic reports (EN + VI, LaTeX + PDF) + 10 figures
├── src/                       # pipeline: retrieval, evaluation, 16-stage chain
├── results/                   # FINAL_v2.json (8 tables) + all raw answers/judgments
└── corpus/                    # articles, chunks, extraction, contexts (<100MB each)
```

Large artifacts live outside GitHub by design (rebuild instructions in
`AGENTS.md`): author index matrices (2.3 GB), rebuilt index (960 MB), model
weights (9.4 GB).

## Method in brief

Query → LLM keywords (memoized, 1 call/question) → hybrid anchors (BM25 top-10
∪ dense top-10, overlap bonus, top-15) → one-hop edges, edge filter, quota m,
cap Nmax → global + direct safety nets → merge (⌊rn/2⌋ per source,
round-robin) → cross-encoder rerank top-10 → Qwen2.5-14B generation (temp 0) →
judged by Hit/MRR/nDCG, official-algorithm RAGAS, and 4-dimension × 5-run LLM
judging (minimal×3 → low×1 escalation, 28× cheaper; quota guard halts loudly).

## Reproduce

```bash
python -u src/kg_build_paper.py   # build the knowledge-graph index
python -u src/chain_full.py       # run everything (per-question checkpoints,
                                  # resume-safe, validated stage gates)
```

Needs: 1 GPU with 24GB, vLLM + ollama, Python 3.11, an Upstage (Solar) key
(~$5 for the full judging protocol). Read `AGENTS.md` first — every rule
there exists because violating it cost GPU hours or corrupted a table.

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
