# ADGR Reproduction — Vietnamese Medical GraphRAG

**🌐 Language: [English](README.md) | [Tiếng Việt](README.vi.md)**
**📄 Full reports: [report.pdf (EN)](report/report.pdf) | [report_vi.pdf (VI)](report/report_vi.pdf)**

Independent, from-scratch reproduction of *Adaptive Directed Graph Retrieval
for Graph-Based Retrieval-Augmented Generation* (submitted to *Applied
Intelligence*): a system that answers Vietnamese medical questions by
retrieving evidence from a knowledge graph of diseases, drugs, symptoms and
their relations.

## Results vs the paper

| Table | Metric | Paper | Ours | Reading |
|---|---|---|---|---|
| 1 | chunks / entities / relations | 17236 / 49274 / 106666 | 15837 / 52578 / 106279 | 92–107%, same scale |
| 1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | close |
| 3 | Hit@10, full ADGR | 0.71 | 0.79 / 0.68 (two indexes) | same range |
| 4 | RAGAS overall ADGR | 0.8792 | 0.64 equiv. (decomposed: two-step 0.77) | gap measured, see report |
| 5 | ADGR overall (judge) | 0.7921 | 0.858 (ours) / 0.778 (orig. prompt) | ranking kept |
| 6 | input tokens/row | 4490.5 | 3441.4 | cheaper |
| 7 | 4 ablations | spread 0.60–0.88 | spread confirmed | same directions |
| 8 | Aescin: 5 ent / 57 rel | — | 12–14 ent / ~40 rel | same compact regime |

Every difference is quantified by cause in the report. Nothing mocked,
nothing cherry-picked — including the numbers below the paper's.

## Repository layout

```
├── README.md / README.vi.md  # this file, English + Vietnamese
├── AGENTS.md                  # operating rules from 43 verified fixes
├── LICENSE                    # MIT
├── report/                    # full academic reports (EN + VI, LaTeX + PDF) + 10 figures
├── src/                       # pipeline: retrieval, evaluation, 16-stage chain
├── results/                   # FINAL_v2.json (8 tables) + all raw answers/judgments
└── corpus/                    # articles, chunks, extraction, contexts (<100MB each)
```

Large artifacts live outside GitHub by design (with rebuild instructions in
`AGENTS.md`): author index matrices (2.3 GB), rebuilt index (960 MB), model
weights (9.4 GB).

## Method in brief

Query → LLM keywords → hybrid anchors (BM25 + dense, overlap bonus) →
one-hop edges scored and filtered by query relevance → 4-source chunk merge →
cross-encoder rerank top-10 → Qwen2.5-14B generation (temp 0) → judged by
Hit/MRR/nDCG, official-algorithm RAGAS, and 4-dimension × 5-run LLM judging.

## Documented substitutions

| Paper | Here | Measured effect |
|---|---|---|
| GPT-5.5 judge | solar-pro4, same protocol | −0.09 scale shift, rankings kept |
| 7 paid standalone LLMs | 7 free models (different set) | references only |
| Manual gold labels | dual automatic proxies (agree) | disclosed |
| Official RAGAS library | official algorithms reimplemented | components match |

## Reproduce

```bash
python -u src/kg_build_paper.py   # build the knowledge-graph index
python -u src/chain_full.py       # run everything (per-question checkpoints,
                                  # resume-safe, validated stage gates)
```

Needs: 1 GPU with 24GB, vLLM + ollama, Python 3.11, an Upstage (Solar) key
(~$5 for the full judging protocol). Read `AGENTS.md` first.

## Cite the original paper

```bibtex
@article{nguyentienha2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {{Nguyen Tien Ha}, Vu Duy Linh, Nguyen Thi Minh Huyen,
              Le Ngoc Toan and Kim Dinh Thai},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```
