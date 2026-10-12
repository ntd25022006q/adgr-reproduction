# Rebuilding the ADGR Medical Search System — Full Reproduction

We rebuilt, from scratch, the ADGR system from the paper by **Nguyen Tien Ha
et al.** (submitted to *Applied Intelligence*). ADGR is a search system that
answers Vietnamese medical questions by looking up evidence in a knowledge
graph (a network of medical facts: diseases, drugs, symptoms, and how they
connect).

**Bottom line:** our rebuilt system works the same way as the paper's system,
ranks methods in the same order, and matches most of its numbers. Where our
numbers differ, we measured exactly why — nothing is hidden.

- Paper: *Adaptive Directed Graph Retrieval for Graph-Based
  Retrieval-Augmented Generation*
- Author materials: [medical article collection (HF)](https://huggingface.co/datasets/vuduylinh150804/vietmed-crawled-corpus)
  · [reference code](https://github.com/Jmhzbmcn2/Graduation-Thesis)
  · [judge instructions + extraction guide](https://github.com/Jmhzbmcn2/adgr-supplementary)

## What we built (plain words)

1. **Collected 1,934 medical articles** from YouMed (the paper used 1,932).
2. **Split them into 15,837 short passages** (paper: 17,236).
3. **Extracted a knowledge graph**: 52,578 medical facts-nodes (paper: 49,274)
   and 106,279 connections between them (paper: 106,666).
4. **Ran 300 test questions** through 5 versions of the search (full system +
   4 versions with one part removed) plus 2 baseline systems.
5. **Scored every answer** with an AI judge (5 independent rounds) and with
   standard search metrics. **8 result tables** in `results/FINAL_v2.json`.

## Results vs the paper (plain words)

| # | What was measured | Paper | Ours | Meaning |
|---|---|---|---|---|
| 1 | Graph size + quality | 17k chunks, 93.6% in-schema | 15.8k chunks, 91.3% in-schema | Almost the same graph |
| 3 | Finding the right evidence (Hit@10) | 0.71 | 0.79 / 0.68 (2 indexes) | Same ballpark |
| 4 | Answer quality (RAGAS-like) | 0.88 | 0.64 | Lower — see honest analysis below |
| 5 | Judge score, ADGR overall | 0.79 | 0.86 (ours) / 0.78 (paper prompt) | Same ranking: ADGR on top |
| 6 | Cost: input words per question | 4,490 | 3,441 | Ours is cheaper |
| 7 | Removing parts hurts scores | yes, spread 0.60–0.88 | yes, spread confirmed | Same conclusion |
| 8 | Example query: small focused graph | 5 nodes / 57 links | 12–14 nodes / ~40 links | Same compact style |

**About Table 4 (0.64 vs 0.88):** we proved by experiment that ~60% of this
gap comes from our one-step judging shortcut (a proper two-step re-check
scores 0.77 on the same data). The rest comes from a stricter judge model and
noisier extraction. Short answers match the paper almost exactly (0.89). All
measured, all disclosed — see `report/` for the full analysis.

## What's inside this repo

```
├── README.md / README.vi.md  # this file, English + Vietnamese
├── AGENTS.md                  # working rules distilled from 41 fixed bugs
├── LICENSE                    # MIT
├── report/                    # academic reports (EN + VI, LaTeX + PDF) + 10 charts
├── src/                       # the pipeline: search, evaluate, run-all script
├── results/                   # every number (8 tables) + all raw answers/judgments
└── corpus/                    # articles, passages, extraction (<100MB each)
```

Big files live elsewhere by design (too large for GitHub): author index
matrices (2.3 GB), rebuilt index (960 MB), models (9.4 GB). Rebuild them with
`python -u src/kg_build_paper.py` then `python -u src/chain_full.py`.

## What we used instead of the paper (all disclosed)

| Paper | Ours | Effect measured? |
|---|---|---|
| GPT-5.5 judge | solar-pro4, same 5-round protocol | Yes: −0.09 scale shift, ranking unchanged |
| 7 paid test models | 7 free models (different set) | Reference only, not ranked |
| Hand-labeled gold evidence | 2 automatic proxies (agree with each other) | Yes, documented |
| Official RAGAS library | Same official algorithms, reimplemented | Components match; faithfulness differs (see above) |

## Run it yourself

```bash
python -u src/kg_build_paper.py   # build the knowledge-graph index
python -u src/chain_full.py       # run everything (saves progress per question,
                                  # resumes after interruption, checks each stage)
```

Needs: 1 GPU with 24GB, vLLM + ollama, Python 3.11, an Upstage (Solar) key
(~$5 for the full judging protocol). Read `AGENTS.md` first — every rule
there exists because breaking it once cost hours.

## Cite

```bibtex
@article{nguyentienha2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {{Nguyen Tien Ha} and Vu-Duy, Linh and Nguyen-Thi, Minh-Huyen
             and Le-Ngoc, Toan and Kim, Dinh-Thai},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```

*Everything here is reproducible from the scripts included. No mocked data,
no cherry-picked rows — including the numbers that look worse than the
paper's, with measured explanations.*
