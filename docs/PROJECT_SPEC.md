# EMR-ACH Project Specification

**Status:** draft for fresh session handoff (2026-04-23).
**Repository:** `E:/Projects/ACH/`, remote `https://github.com/ApartsinProjects/EMR-ACH` (master).
**Latest tag:** `v2.1-data-ready` at `8ffba6f`.
**Latest master HEAD:** `c50161e` (v2.2 [H7] filing + audit).

---

## 1. Project mission

Produce a **leakage-free, multi-domain LLM-forecasting benchmark** with an accompanying paper. Deliverables, in priority order:

1. A public benchmark dataset of **Forecast Dossiers (FDs)**: resolved forecasting questions paired with pre-event news evidence, under strict temporal leakage control.
2. A **baseline battery** (B1 through B9 + a majority-class reference) on a shared pick-only response contract, with reproducible Batch API run commands.
3. A **proposed analytical framework (EMR-ACH)** layered on the benchmark: contrastive indicators, diagnostic weighting, multi-agent debate, hybrid retrieval. As of 2026-04-23 this is **proposed + partially prototyped**, not a shipped method; paper treats it as future work (see §10).
4. A **camera-ready HTML paper** (`paper/index.html`) documenting 1-3, including gold subset, ETD evidence structure, and ablation slots.

---

## 2. Research questions

1. Do top LLMs beat chance on multi-domain forecasting under a realistic 14-day horizon with 30-day news lookback?
2. Does contrastive ACH-style evidence structuring measurably improve over plain RAG + CoT baselines?
3. Does multi-agent debate reduce variance relative to self-consistency at fixed cost?
4. Where does the "change subset" (FDs whose prior expectation breaks) gap open between baselines and richer methods?

**v2.1** establishes the benchmark + baselines slot. **v2.2** fixes benchmark correctness bugs (horizon, leakage). **v2.3** (future) ships the EMR-ACH method and answers RQ2-RQ4.

---

## 3. Current state (as of 2026-04-23 end-of-day)

### 3.1 Shipped on remote `master`
- **v2.1 benchmark** published at `benchmark/data/2026-01-01/`: 6,294 FDs across 3 tracks (134 forecastbench, 5,975 gdelt-cameo, 185 earnings) with 28,945 articles. Tagged `v2.1-data-ready`.
- **v2.1 gold subset** at `benchmark/data/2026-01-01-gold/`: 81 FDs (60+17 forecastbench + 2+1 gdelt + 1 earnings), 690 articles, 2,107 ETD facts. Self-contained (schema + examples + README + CITATION).
- **ETD fact pipeline**: Stage-1 extraction (`articles_to_facts.py`), Stage-2 dedup with G8 date-bucketed FAISS kNN (`etd_dedup.py`), Stage-3 FD linkage (`etd_link.py`), filter (`etd_filter.py`), audit (`etd_audit.py`), compare (`etd_compare_facts_vs_articles.py`), orchestrator (`etd_post_publish.py`). Produces `facts.v1_canonical.jsonl`, `facts.v1_linked.jsonl`, `facts.v1_production_{cutoff}.jsonl`.
- **Baselines B1 through B9** with pick-only + plurality-vote + per-sample seed for Batch API dedup.
- **Paper skeleton** in `paper/index.html` with Appendices A through G (data construction, reuse contract, retrieval routing, gold subset, ETD, baseline reference, worked examples). Tables 1, 2, 3, 4, 5, 6, 6a, 7 numbered in reading order. `[PENDING]` + `[TBD]` placeholders for result cells.
- **v2.2 H6 plumbing**: `--horizon-days` (default 14) + `--lookback-days` (default 90, will be 30 at build time) CLI flags, `temporal:` YAML config block, `forecast_point = resolution_date - horizon_days` in FD constructors, per-article leakage assertions in 3 fetchers + earnings linker, 7 passing regression tests.
- **OpenAI embeddings helper** (`src/common/openai_embeddings.py`) with auto-chunk above 50k cap, parallel polling, salvage-on-cancel, and LOCAL-COMPLETED short-circuit for offline-synced chunks.
- **Offline sync helper** (`scripts/offline_sync_chunk.py`) to bypass a stuck OpenAI Batch by sync-encoding all inputs locally.

### 3.1.1 v2.2 build report (2026-04-23)

Backlog H6 executed via reuse-first strategy (§6.1A). Built by
`scripts/build_h14_from_v21.py` on top of the v2.1 publish.

**h14 pool** (`benchmark/data/2026-01-01-h14/`, commit `16150fe`):
- FD in:  134 forecastbench + 185 earnings (gdelt-cameo excluded per §10.1).
- FD out: 40 forecastbench + 49 earnings = **89 FDs**.
- Articles out: **291** (reused from the 28,945-article v2.1 pool).
- Drop reasons: `empty_after_leakage:forecastbench=94`, `empty_after_leakage:earnings=136`.
- Dangling article ids: **1004** (H2 residual; all earnings). Written to
  `meta/dangling_article_ids.txt` for paper appendix.
- Leakage assertion: **0 violations**. Every article passes
  `publish_date <= forecast_point = resolution_date - 14d`.
- Horizon assertion: **0 violations**. Every FD has
  `(resolution_date - forecast_point).days == 14`.

**ETD post-publish** (commit `5faa187`):
- Stage-3 link: 445 facts linked (histogram: 0=78185, 1=435, 2=10) from
  78,630 canonical Stage-2 facts across 291 h14 articles, 89 FDs.
- Production filter: **403 facts** (drops: `no_linked_fd=67221`,
  `polarity=9951`, `min_confidence=1017`, `source_blocklist=38`).
- Audit: 4.07 facts/article, 99 articles with facts, 0 schema fails,
  0 future-dated facts, 0 bad polarity.
- Non-fatal: `etd_compare_facts_vs_articles.py` hit OpenAI 429 quota on
  synchronous audit calls (compare reports preserved but empty).

**Gold subset** (`benchmark/data/2026-01-01-h14-gold/`, commit `c6c7f07`):
- Filters: `--min-articles 3 --min-distinct-days 2 --horizon-days 14
  --min-avg-chars 300 --min-source-diversity 1 --keep-unknown`.
- After filter cascade: **22 FDs** (21 fb-stability, 1 fb-change).
  All earnings FDs drop out at `min_articles>=3` and `min_avg_chars>=300`
  due to the H2 dangling-id shrinkage + v2.1's resolution-day article bias.
- Articles: 98. ETD facts: 269 (from `facts.v1_production_2026-01-01-h14.jsonl`).
- Two source patches landed under commit `c6c7f07`:
  - `build_gold_subset.py` horizon filter: was `fp - horizon_days`; now
    `rd - horizon_days`. Same value under v2.1 (`fp == rd`), fixes the
    double-subtract bug under v2.2.
  - `_load_etd_facts_for_articles` takes an optional cutoff and prefers
    the cutoff-specific production facts file before legacy fallback.

**Scope reduction vs task target**: task asked for 150-300 gold FDs; yield
is 22 because (a) v2.1 article curation biased toward resolution-day
articles (forecastbench p50 gap = 0 days), so the 14-day leakage shift
drops 55-74% at pool stage; (b) H2 leaves 1049 earnings article_ids
dangling; (c) `min_articles>=3` removes all earnings FDs from gold. Real
fix is strategy 6.1B (CC-News rebuild with lookback=30d), blocked on H8.

**Baselines eval prep** (commit `1a60320`): all 10 baselines (B1-B9 +
b3b_rag_claims) dry-run clean against h14 gold; 472 total requests;
cost estimate ~$0.11 at gpt-4o-mini Batch API rates. Plan in
`docs/V2_2_EVAL_PLAN.md` with ready-to-fire commands.

**Next session — ready to fire**:
1. Pre-flight sync smoke: `python -m evaluation.baselines.runner --method b1
   --fds data/2026-01-01-h14-gold/forecasts.jsonl --articles
   data/2026-01-01-h14-gold/articles.jsonl --smoke 3 --sync`.
2. Full B1-B9 Batch API submission per `docs/V2_2_EVAL_PLAN.md`.
3. Regenerate paper Table 1, 4 cells for the v2.2 row.
4. Tag `v2.2-data-ready`.

**Open blockers**: OpenAI .env key hit 429 in B2's synchronous ETD
compare; verify Batch API quota before firing (Batch quota is separate
from sync quota).

### 3.2 Known bugs filed in `docs/V2_2_REFACTOR_BACKLOG.md`
- **H1** GDELT-CAMEO `publish_date = event_date`. **FIXED in code** (URL-slug parser fallback to event date); v2.1 publish retrofitted (1,130 of 27,805 articles corrected; most gdelt URLs don't expose a date slug).
- **H2** 1,049 of 1,299 earnings article_ids dangle in the published pool (unify-vs-link race). **DIAGNOSTIC SHIPPED** (step_publish now dumps a `dangling_article_ids.txt`); **HARD FIX DEFERRED** to v2.2 via reuse-check cache invalidation.
- **H3** GDELT-CAMEO 74% Comply majority-class skew. **OPEN** (v2.2 actor-pair rebalancing).
- **H6** v2.1 horizon is 0-3 days for every FD (`forecast_point == resolution_date`). **PLUMBING SHIPPED** (CLI + FD builder changes); **ACTUAL REBUILD NOT RUN** yet.
- **H7** EMR-ACH method on the gold subset is a dry-run stub; contrastive indicators are MIRAI-locked 4-CAMEO; diagnostic weighting and hybrid retrieval (MMR/RRF/temporal decay) are MISSING; multi-agent debate is single-round. **DEFERRED TO v2.3** per 2026-04-23 scope decision.

### 3.3 Abandoned in-flight
- CC-News serial fetch (`scripts/fetch_cc_news_archive.py` at `--workers 1`) is ~3 min per shard, ~30 h per month. Stopped 2026-04-23 after 6 shards. Needs H8 (parallelizer, see §6.1) before any real CC-News run.
- Paper reframe and CC-News parallelizer agents were launched, then stopped when this spec replaced them; no commits landed from those.
- 2024-04-01 probe benchmark agent was launched, then stopped; no commits landed.

---

## 4. Architecture and data contracts

### 4.1 Forecast Dossier (FD) schema
Canonical fields (see `docs/FORECAST_DOSSIER.md` + `benchmark/schema/forecast_dossier.md`):
- `id` (str), `benchmark` (one of forecastbench, gdelt-cameo, earnings), `source` (str), `hypothesis_set` (list[str], typically Comply/Surprise pair), `hypothesis_definitions` (dict), `question` (str), `background` (str), `forecast_point` (UTC timestamp), `resolution_date` (UTC timestamp), `ground_truth` (one of hypothesis_set), `ground_truth_idx` (int), `crowd_probability` (float or null), `lookback_days` (int), `default_horizon_days` (int), `article_ids` (list[str]), `prior_state_30d` (str), `prior_state_stability` (str), `prior_state_n_events` (int), `fd_type` (stability, change, unknown), `metadata` (dict; ticker, actors, CAMEO code, x_multiclass_gt, etc.).

Invariants:
- `forecast_point = resolution_date - horizon_days` (H6; default 14 in v2.2).
- Every article in `article_ids` satisfies `publish_date <= forecast_point` (leakage).
- `ground_truth in hypothesis_set`.
- `fd_type in {stability, change, unknown}`.

### 4.2 Article schema
`id` (sha1-prefixed art_*), `url`, `title`, `text`, `title_text`, `publish_date` (YYYY-MM-DD; MUST be the real article publish date, not the event date), `source_domain`, `gdelt_themes`, `gdelt_tone`, `actors`, `cameo_code`, `char_count`, `provenance` (list[str]).

### 4.3 ETD fact schema
See `docs/ETD_SPEC.md` + `benchmark/data/*-gold/schema/etd_fact.schema.json`. Key fields: `id` (f_*), `fact` (str), `time` (YYYY-MM-DD), `primary_article_id` (points at article), `article_ids` (all articles reporting same fact), `source`, `entities` (list), `metrics`, `polarity` (asserted, negated, uncertain), `extraction_confidence` (low, medium, high), `canonical_id` (Stage-2 dedup), `variant_ids` (Stage-2 cluster), `linked_fd_ids` (Stage-3).

### 4.4 Pick-only response contract (baselines)
Every baseline returns:
```
{"prediction": "<one hypothesis from hypothesis_set>", "reasoning": "..."}
```
Multi-sample methods (B4 self-consistency, B5 multi-agent, B6 ToT, B7 Reflexion, B9 ensemble) aggregate via **plurality vote** (ties broken by hypothesis_set order). No probability distributions. Full methodology: `benchmark/evaluation/BASELINES.md`.

---

## 5. Pipeline (orchestrator = `scripts/build_benchmark.py`)

```
1  per-track raw build   (GDELT-CAMEO KG, earnings via yfinance, forecastbench repo clone)
2  per-track text fetch  (trafilatura; GDELT DOC; NYT/Guardian/Google; earnings sources)
3  unify_articles.py     -> data/unified/articles.jsonl
4  unify_forecasts.py    -> data/unified/forecasts.jsonl   (applies forecast_point = resolution_date - horizon_days)
5  compute_relevance.py  per benchmark; OpenAI Batch API (text-embedding-3-small)
6  annotate_prior_state  (earnings ticker-date join; gdelt pre-event window; fb via hypothesis_definitions)
7  relink_gdelt_context  (pre-event context substitution)
8  quality_filter.py     --model-cutoff X --cutoff-buffer-days Y
9  diagnostic_report.py + build_eda_report.py
10 step_publish          -> benchmark/data/{cutoff}/ (forecasts.jsonl + articles.jsonl subset by FD article_ids + meta/ + integrity dangling-ref check)
```

Post-publish (in `scripts/etd_post_publish.py`):
```
A  delta computation    (articles published but not yet in facts.v1.jsonl)
B  ETD Stage-1 extract  (articles_to_facts.py, gpt-4o-mini Batch API)
C  ETD Stage-2 dedup    (etd_dedup.py --knn-mode bucket, FAISS-CPU)
D  ETD Stage-3 link     (etd_link.py --cutoff)
E  production filter    (source-blocklist + medium+ confidence + asserted + no-future + require-linked-fd)
F  audit                (etd_audit.py on production set)
G  per-benchmark compare (etd_compare_facts_vs_articles.py)
```

Gold-subset build (`scripts/build_gold_subset.py`):
```
1  filter cascade    (min-articles, min-distinct-days, horizon-days, min-avg-chars, min-source-diversity, benchmark-specific topic filter)
2  stratified sample to per-(benchmark, fd_type) quotas (DEFAULT_TARGETS)
3  write self-contained folder: forecasts + articles + facts + schema + examples + README + LICENSE + CITATION + checksums + build_manifest + selection_criteria + meta/distribution
```

---

## 6. v2.2 rebuild plan

**Goal:** Produce a horizon=14 / lookback=30 benchmark at cutoff 2026-01-01 suitable for the paper, scoped to forecastbench + earnings first (319 FDs) with gdelt-cameo deferred until pipeline is validated.

### 6.1 Two article-pool strategies
- **(A) Reuse-first (fast path)**: Use existing `benchmark/data/2026-01-01/articles.jsonl` as starting pool; apply leakage filter `publish_date <= forecast_point = resolution_date - 14d`; re-link FD article_ids after filtering; rebuild gold. Est. ~1 hour, zero new downloads. Expected reuse rate: forecastbench ~85-90%, earnings ~60-70%, gdelt-cameo ~17% (H1 residual).
- **(B) CC-News rebuild (quality path)**: H8 parallelize fetcher first; download 1-month slice (2025-12); rebuild. Est. 3-5 h for download, plus ~2 h for rebuild. Blocked on H8.

**Decision (2026-04-23):** pursue (A) immediately; H8 parallelizer + CC-News done in parallel for future v2.2 refinement.

### 6.2 v2.2 acceptance criteria
- Every FD satisfies `resolution_date - forecast_point == horizon_days` within 1 day.
- Every article in every `article_ids` list satisfies `publish_date <= forecast_point`.
- Gold subset yields at least 150 FDs across forecastbench + earnings with balanced fd_type stratification.
- Leakage regression tests (`tests/test_v2_2_leakage.py`) stay green.
- `dangling_article_ids.txt` is empty or contains an explicit note explaining the residuals.
- Published at `benchmark/data/2026-01-01-h14/` to disambiguate from v2.1 `benchmark/data/2026-01-01/`.

### 6.3 v2.2 build command
```
python scripts/build_benchmark.py \
    --cutoff 2026-01-01 \
    --benchmarks forecastbench,earnings \
    --horizon-days 14 --lookback-days 30 \
    --skip-raw \
    --embedder openai --openai-mode batch \
    --skip-news-fetch   # under strategy (A); omit under (B)
```

### 6.4 v2.2 post-publish + gold
```
python scripts/etd_post_publish.py --cutoff 2026-01-01
python scripts/build_gold_subset.py --cutoff 2026-01-01 \
    --min-articles 3 --min-distinct-days 2 --horizon-days 14 \
    --min-avg-chars 300 --min-source-diversity 1
```

---

## 7. Paper scope (scope decision 2026-04-23)

Paper is **benchmark + baselines**, not method. EMR-ACH is proposed + future work.

- **Title**: `EMR-ACH Forecasting Benchmark: Leakage-Free Multi-Domain Forecast Dossiers for LLM Evaluation`.
- **Abstract**: lead with the benchmark + baselines; mention EMR-ACH framework as proposed + deferred.
- **§1 Intro**: close the "horizon = 0 contamination" gap; deliver a reproducible benchmark.
- **§4 Analytical Framework and Baseline Protocol**: pick-only contract, baseline battery, proposed EMR-ACH as future work; point at H7.
- **§5 Dataset construction**: v2.1 vs v2.2 distinction (retrospective vs prospective).
- **§6-§7**: gold subset, ETD evidence structure, cost analysis.
- **§8 Future Work**: v2.3 EMR-ACH implementation (H7 subtasks).
- **Appendices A-G**: unchanged.
- **Tables**: baseline rows fillable from Batch API runs; EMR-ACH rows tagged `[Baseline only; EMR-ACH deferred to v2.3]`.

Current `[PENDING]` counts (14:00 UTC 2026-04-23): 13 `PENDING` + 22 `TBD` cells across Tables 1, 4, 5, 7.

---

## 8. Evaluation protocol

### 8.1 Metrics
- Accuracy (overall and on `fd_type=change` subset).
- Macro-F1.
- 95% bootstrap CI on accuracy (1000 resamples).
- Per-benchmark breakdown (forecastbench, gdelt-cameo, earnings separately).
- For calibration (B8 deprecated; majority-class reference replaces).

### 8.2 Budget
- Baselines B1 through B9 on 81-FD gold via Batch API: projected ~$31 at current `baselines.yaml` defaults; trim to ~$10-15 via smaller context window + fewer self-consistency samples.
- ETD Stage-1 re-extract on v2.2 deltas: ~$2-5 (319 FDs scope).
- OpenAI embeddings (Batch): ~$0.15 per 78k facts.
- Total v2.2 + first paper-results pass: ~$15-25.

### 8.3 Fairness defaults (`benchmark/configs/baselines.yaml`)
- Same model (gpt-4o-mini) across all baselines; B9 ensemble is the only method allowed to deviate.
- Same temperature defaults; per-sample seed for Batch API dedup.
- Context window cap identical.
- Retrieval preserved for RAG-using baselines; B1/B2 see no articles by design.

---

## 9. Environment

- **Host OS**: Windows, using Git Bash for commands.
- **Python**: `/c/Python314/python` = Python 3.14 (path `C:\Python314\python.exe`). Alt: `C:\Users\apart\AppData\Local\Programs\Python\Python311\python.exe`.
- **GPU**: NVIDIA RTX 2060, 6GB VRAM. **NOT NEEDED for v2.2 or v2.3 per current plan** (every pipeline component is OpenAI API or CPU). Existing MIRAI-era code that imports torch/sentence-transformers is dead-code unless someone invokes `--embedder sbert` explicitly.
- **OpenAI key**: `OPENAI_API_KEY` in `.env` (starts `sk-proj-FflqlR...`). **Do NOT use** `.env.all`'s stale key `sk-OCcEe...`; it's on a different account and cannot see the batches this project submitted.
- **Disk**: 930 GB total on `E:`, 542 GB free.
- **Package deps**: `requirements.txt` (faiss-cpu added 2026-04-23 for G8 kNN).

---

## 10. Tracking and backlog

- Canonical backlog: `docs/V2_2_REFACTOR_BACKLOG.md` (22 P0, 29 P1, 29 P2 = 80 items as of 2026-04-23).
- Audit docs: `docs/EMRACH_IMPLEMENTATION_AUDIT.md`, `docs/GOLD_EVALUATION_AUDIT.md`.
- Architecture: `docs/V2_2_ARCHITECTURE.md` (pipeline contracts), `docs/PIPELINE.md`, `docs/FORECAST_DOSSIER.md`, `docs/ETD_SPEC.md`.
- Decision log: append to backlog as SHIPPED / OPEN / DEFERRED status lines.

### 10.1 Immediate next-session priorities
1. **Push unpushed commits** (`c50161e` ahead of origin/master) and tag status.
2. **Fire v2.2 reuse-first build** (strategy 6.1A): produces `benchmark/data/2026-01-01-h14/`. ~1 h wall-clock.
3. **Build v2.2 gold subset** on the new pool. ~1 min.
4. **Baselines Batch API run** on v2.2 gold (not v2.1 gold; v2.2 has the correct horizon). Fill Table 1 rows. ~1 h wall-clock, ~$15.
5. **Paper reframe** to benchmark + baselines contribution (§§1, 4, 8, abstract, Table 1/4 placeholder swap). 4-6 commits.
6. **H8 CC-News parallelizer** (background; unblocks strategy 6.1B for a later refinement). 3 commits + tests.
7. Tag `v2.2-data-ready` once (2)-(5) are green.

### 10.2 Deferred to v2.3
- H7 EMR-ACH port (contrastive indicators off MIRAI, MMR+RRF+temporal retrieval, diagnosticity matrix A, N-round multi-agent, wire `emrach_on_gold.py` live). 4-8 h dev + $5-10 eval.
- H3 gdelt-cameo actor-pair rebalancing. Paper doesn't need it while gdelt is deferred.
- Probe cutoff 2024-04-01 (v2.1.1) once benchmark rebuild is stable.

---

## 11. Conventions

- **Writing style**: no em dashes (`—`), no double dashes (`--`) in prose. Use commas, semicolons, parentheses, or separate sentences.
- **Commit style**:
  - `v2.2 [<tag>]: short description` (e.g. `v2.2 [H6-1]: ...`).
  - Body: HEREDOC multiline; final trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
  - Never skip hooks, never force-push, never amend a published commit.
- **Gitignore**:
  - `data/etd/facts.v*.jsonl` (sizable; gold subset carries the sampled copy).
  - `data/cc_news/` (per-shard zst output; regenerable).
  - `data/etd_openai_batches/` (request + output JSONLs; regenerable).
  - `data/etd/audit/*.log`, `data/etd/audit/delta_*.jsonl`, `*_diffs.jsonl`, `verifier_*` (ephemeral audit output).
- **Test discipline**: `pytest tests/ -q` must stay green. Current suites: `test_etd_dedup_knn.py` (6), `test_v2_2_leakage.py` (7), `test_emrach_facts_rows.py` (8), `test_gdelt_doc_vs_baseline.py` (skipped-by-design), `test_etd_hallucination_gate.py` (8), `test_emrach_on_gold_adapter.py` (9), plus pre-existing `test_etd_date_validators.py` and `test_etd_schema_invariants.py`.

---

## 12. Open decisions for the fresh session

1. **CC-News strategy** before v2.2 tagging: reuse-first only (ship v2.2 at H1-residual gdelt quality), or H8 + real CC-News rebuild (slower, better gdelt).
2. **v2.2 scope**: forecastbench + earnings only (319 FDs), or add gdelt after H1/H3 stronger fixes (~6k FDs).
3. **Paper reframe depth**: lightweight (abstract + §4 + Table placeholder swap, 4 commits) or full (all EMR-ACH prose rewritten as proposed-framework, 6-8 commits).
4. **Budget cap** for the next Batch API pass: default $20, or stricter?
5. **Tag cadence**: one `v2.2-data-ready` after everything, or intermediate `v2.2-h14-pool` after the build then `v2.2-baselines-done` after the eval?

---

## 13. Glossary

- **FD** — Forecast Dossier. One resolved question + pre-event article bundle.
- **ETD** — Event Timeline Dossier. Atomic facts extracted from articles; Stage-1 LLM, Stage-2 SBERT/OpenAI dedup, Stage-3 link to FDs.
- **CC-News** — Common Crawl news archive. Monthly WARC shards at `commoncrawl.org/crawl-data/CC-NEWS/`.
- **Pick-only** — Response contract where every baseline returns one hypothesis label per FD (no probability distribution).
- **Horizon** — Days between `forecast_point` and `resolution_date`. v2.1=0, v2.2=14.
- **Lookback** — Days of article-pool window ending at `forecast_point`. v2.1 sourced through trafilatura/GDELT; v2.2 default 30 days.
- **Gold subset** — Curated subset with self-contained folder (schema, examples, facts, README, LICENSE, CITATION).
