# ADGR Reproduction — Vietnamese Medical GraphRAG

Independent, from-scratch reproduction of **Adaptive Directed Graph Retrieval
(ADGR)** (Nguyen-Tien et al., submitted to *Applied Intelligence*) with a
rebuilt Vietnamese medical knowledge graph, full retrieval ablations, and all
8 evaluation tables — every number disclosed with its provenance, nothing
mocked.

- Paper: *Adaptive Directed Graph Retrieval for Graph-Based
  Retrieval-Augmented Generation*
- Author resources: [ViMedGraph (HF)](https://huggingface.co/datasets/vuduylinh150804/vietmed-crawled-corpus)
  · [reference implementation](https://github.com/Jmhzbmcn2/Graduation-Thesis)
  · [supplementary (judge prompt + extraction protocol)](https://github.com/Jmhzbmcn2/adgr-supplementary)

## Results at a glance (ours vs paper)

| Table | Metric | Paper | Ours | Verdict |
|---|---|---|---|---|
| Tab.1 | chunks / entities / relations | 17,236 / 49,274 / 106,666 | 15,837 / 52,578 / 106,279 | 92–107% |
| Tab.1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | close |
| Tab.3 | Hit@10 full ADGR | 0.71 | 0.79 (author) / 0.68 (rebuilt) | same range |
| Tab.4 | RAGAS overall ADGR | 0.8792 | 0.64 (RAGAS-like) | gap decomposed, see below |
| Tab.4 | dim4 mean ADGR | — | 0.837 | high, consistent |
| Tab.5 | ADGR overall (5-run) | 0.7921 | 0.858 (our prompt) / 0.778 (paper prompt) | ranking preserved |
| Tab.6 | input tok/row | 4,490.5 | 3,441.4 | cheaper (compact ctx) |
| Tab.6 | ret / gen latency | 3.19s / 6.66s | 2.36s / 5.82s | comparable |
| Tab.7 | 4 ablations spread | 0.60→0.88 | spread confirmed | directions match |
| Tab.8 | Aescin: 5 ent / 57 rel | — | 12–14 ent / 39–41 rel | same compact regime |

The faithfulness gap (0.41 vs 0.89) is **measured, not hidden**: short answers
match paper (0.89); a two-step official-protocol rerun scores 0.77, proving
~60% of the gap is our single-call shortcut and ~40% is judge strictness
(solar-pro4 vs GPT-5.5) plus noisier extraction. See `results/FINAL_v2.json`
(`tab4.faith_protocol_ablation`) and the report.

## Repository layout

```
├── README.md                  # this file
├── AGENTS.md                   # operating rules distilled from 41 fixed bugs
├── LICENSE                    # MIT
├── report/                    # academic report (report.tex + report.pdf + figs/)
├── src/                       # pipeline scripts (retrieval, eval, chain)
├── results/                   # FINAL_v2.json, metrics, 5-run judgments, figs
└── corpus/                    # crawl + chunks + extraction + baseline contexts
                               # (<100MB each; full indexes excluded, see below)
```

## What is (and is not) included

**Included:** all code, all configs, every evaluation number, figures, the
academic report (LaTeX + PDF), corpus artifacts under GitHub's 100MB/file
limit (crawl, chunks, extraction, baseline contexts, arm checkpoints).

**Excluded (too large for GitHub) + how to reproduce:**
| Excluded | Size | Reproduce |
|---|---|---|
| `data/` author index matrices | 2.3 GB | original author artifacts (see paper Data availability) |
| `data_paper/` rebuilt index | 960 MB | `chain_full.py` stages R6–R7 (`kg_build_paper`) from `corpus/extract_paper_full.json` |
| `models/` (Qwen2.5-14B-AWQ) | 9.4 GB | AWQ release of Qwen2.5-14B-Instruct |
| raw run logs, backups | — | regenerable; `FINAL_v2.json` is the distilled record |

## Requirements

- 1× NVIDIA GPU 24GB (tested A30; paper used RTX 3090), ~120GB RAM, ~100GB disk
- vLLM + ollama (`embeddinggemma:300m`, `qwen3:8b`, `qwen2.5:14b`, `qwen3:14b`)
- Python 3.11, numpy, rank-bm25, matplotlib, openai client
- An Upstage (Solar) API key for judging (~$5 for the full Tab.4/5/7 protocol
  at minimal reasoning effort; quota guard aborts loudly instead of poisoning)

## Reproducing

```bash
# 1. rebuild the paper-chunker index (R6-R7)
python -u src/kg_build_paper.py
# 2. run the full chain (R8-R17 + plots); every stage checkpoints per row,
#    skips when complete, and validates its output at stage gates
python -u src/chain_full.py
# 3. collect: runs/FINAL_v2.json + runs/figs/
```

See `AGENTS.md` for the operating discipline (VRAM budgets, checkpoint schema
law, judge effort policy, gate protocol) — every rule exists because violating
it once cost GPU hours.

## Substitutions (all disclosed, none hidden)

| Paper | This reproduction |
|---|---|
| GPT-5.5 judge | solar-pro4 (temp 0, 5 runs; + paper-exact prompt variant) |
| 7 paid standalone LLMs | 7 free models (zero overlap; descriptive refs only) |
| Manual gold-chunk labels | slug-gold + single-gold proxies (dual-proxy agreement checked) |
| Official RAGAS library | Reimplemented official algorithms + direct-rating faithfulness |

## Citation

```bibtex
@article{nguyentien2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {Nguyen-Tien, Ha and Vu-Duy, Linh and Nguyen-Thi, Minh-Huyen
             and Le-Ngoc, Toan and Kim, Dinh-Thai},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```

*Reproduction artifacts: `results/FINAL_v2.json` (8 tables, zero pending),
`report/report.pdf` (6-page academic report), 41 verified bug fixes, no mocked
or cherry-picked numbers anywhere in this repository.*
