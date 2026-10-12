# AGENTS.md — Operating rules for the ADGR reproduction workspace

Distilled from ~37 fixed bugs across chain-1..13. Every rule below exists
because violating it cost GPU hours or corrupted a table. Read before touching
anything in `tmp/` or `runs/`.

## 1. GPU discipline (A30 24GB, shared with external jobs)

- ONE GPU job at a time. Never launch a second vLLM/ollama-heavy process.
- Before ANY vLLM (re)start: `nvidia-smi` free MB must exceed NEED (32k: 17500,
  12k: 15000, 8k: 12800). The restart script enforces this — never bypass it.
- vLLM flags that work: `--enforce-eager` ALWAYS (without it: EngineCore fail).
  Known-good: 32k/0.72, 12k/0.62, 8k/0.50. util 0.50 + maxlen 12288 FAILS
  (KV 1.91G < 2.25G needed for one request) — see vllm12k.log 10-10.
- Max sentence length served must cover the longest stage prompt
  (ADGR ctx ~4.4k tok -> maxlen 8192 minimum).
- Miniforge lives at `$HOME/miniforge3`, NOT under the workspace. The server
  launch MUST use `/home/devlab1/miniforge3/envs/unsloth/bin/python`.
- Always `python -u` for servers (buffered logs hid every boot failure once).
- Kill by explicit PID only. Never `pkill -f python`. Never `pkill ollama`.
- One vllm_restart at a time (flock in the script). Stale restarts from killed
  chains MUST be hunted (`ps | grep vllm_restart`) before relaunching anything.

## 2. Checkpoints: schema is law

- Canonical arm file shape: `{"arms": {arm: [rows]}}`. NEVER write it bare.
- Load defensively: accept wrapped AND unwrapped (`raw["arms"] if ... else raw`).
- Backup (`*.pre_<op>_bak`) BEFORE any script that rewrites a checkpoint.
- Every row: `done` flag + `gen_failed`/`all_ok` distinction. `done` without
  `all_ok` and without `gen_failed` = judge failed = retriable, NOT complete.
- Empty-dict rows (`{}`) must be treated as MISSING on resume, never skipped.
- Stages must be idempotent: rerun = no-op when output complete. NEVER wipe
  completed rows inside a stage script (one-time migrations go in throwaway
  scripts, not in the stage).
- Dump checkpoints on a counter (every 25 rows), never only at the end.

## 3. Judges (Solar/upstage, rate limit 100 req/min)

- Effort escalation ONLY: `minimal x3 -> low x1` (~116 tok vs 3273, validated
  5/6 -> ~6/6). Never bare `low` (28x cost), never bare `minimal` (validity).
- Quota guard in EVERY judge function: `suspended/insufficient/invalidapikey/
  unauthorized/billing/...` -> `os._exit(42)` LOUDLY. A dead key must STOP the
  chain, never mark rows done+failed (that poisoning cost 123 rows once).
- 429/rate-limit = transient: sleep + retry, and do NOT count failed runs
  toward n (tab5_5run pattern).
- Concurrency: row-workers(4) x judge-pool(5) = 20 concurrent is the proven
  safe point. tab5: 8 workers max.
- Cost math before launch: rows x calls x ~150 tok. No balance API exists.

## 4. Retrieval fidelity (paper Sec 3.3-3.6, 5.8)

- `CFG` in adgr.py is locked to paper Table 2. Change = new experiment, new files.
- Ablation semantics (enforced, verified):
  - no_rerank = pool order, NO cross-encoder (e2e_v3 skips cereank).
  - dense_only = Ts (direct dense top-ns) ONLY, no Eglob/Tg/Tv merge.
  - dense_anchors / no_edge_filter consumed inside retrieve().
- Slug normalization: chunk keys (file_path `slug.txt` OR full `https://.../slug/`)
  and `article_url` must BOTH reduce to the last path component before matching.
  Un-normalized = hit=None 100% on the rebuilt index (silent, caught by audit).
- `question_idx` is an ID (`disease_746`), NOT the row position. Join by
  enumerate order, never by text match.
- Reranker returns EXACTLY len(docs) scores; GPU needs >=4GB free else CPU from
  the start (batch 4 GPU / 16 CPU). Mid-run OOM -> CPU fallback per row.

## 5. Chain operation

- Edit files freely (stages load at execution). Restarting the chain costs
  ~1 in-flight row + index reload (~4 min) — cheap, do it rather than run stale code.
- Killing a stage: kill chain parent AND stage child; verify NONE remain.
- Never touch a checkpoint file while its stage runs (reader is fine, writer races).
- Every stage boundary SHOULD have a `_gate()` entry: assert output shape/counts,
  exit 99 LOUDLY on mismatch. Add gates for new stages, never remove.
- Log lines are progress, JSON files are results. Truncate stale logs freely;
  never delete a .json without backup.

## 6. Numbers before narrative

- After any metric lands, diff it against paper values AND cross-arm sanity
  (ablations must spread; clustering = methodology bug until proven otherwise).
- Proxy metrics (single-gold, slug-gold) are disclosed as proxies, never
  presented as paper numbers.
- Substitutions are fixed and disclosed: gen Qwen2.5-14B-AWQ/vLLM, embed
  embeddinggemma:300m, rerank Qwen3-0.6B, judge solar-pro4 (paper: GPT-5.5),
  7 free standalone models (paper: 7 paid models, zero overlap).

## 7. ETA methodology (no guessing)

- ETA = measured_pace x remaining + fixed_overheads, quoted as a RANGE from
  the min/max of the last 3 pace windows. Never carry old pace forward, never
  give point estimates. State the pace, the math, and what could shift it.
- Known ceilings: Solar 100 req/min (judging floor), 1 GPU (no parallel stages),
  ~1.9GB KV on the 8k server (concurrency ~2 for gen).
