# EMR-ACH v2.2 Refactor Backlog

**Status**: prioritized actionable work items, drafted 2026-04-22 alongside [`docs/V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md).
**How to read**: each item lists title, files affected, effort (S = under a day, M = 1 to 3 days, L = 3+ days), priority (P0 = blocks v2.2 launch, P1 = should land in v2.2, P2 = nice-to-have or follow-on), and dependencies on other backlog items. Within a category, items are ordered by priority then dependency.

Items reference v2.1 line numbers. v2.1 entrypoint is [`scripts/build_benchmark.py`](../scripts/build_benchmark.py); the four in-flight files (`build_benchmark.py`, `compute_relevance.py`, `annotate_prior_state.py`, `configs/default_config.yaml`) are not to be touched while the current build is running.

---

## Category A, Performance (the v2.2 wall-clock thesis)

### A1. Ship `fetch_gdelt_doc_archive.py`
Files: new `scripts/fetch_gdelt_doc_archive.py`, `data/gdelt_doc/raw/`. Effort: M. Priority: P0. Deps: none.
One-time bulk download of GDELT DOC monthly archives to local zstd-JSONL shards. Per-shard sidecar `_progress.json` for resume. See [`docs/V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) §3.1.

### A2. Ship `build_gdelt_doc_index.py`
Files: new `scripts/build_gdelt_doc_index.py`, `data/gdelt_doc/index/`. Effort: M. Priority: P0. Deps: A1, A6.
SBERT encode plus FAISS flat-IP per-shard, plus aligned parquet metadata. Manifest pins model name and revision so the index is gated by config. See §3.2.

### A3. Ship `query_gdelt_doc_index.py`
Files: new `scripts/query_gdelt_doc_index.py`. Effort: M. Priority: P0. Deps: A2, B1.
Per-FD lookup with date-window plus language plus domain-blocklist filters; optional `--fetch-bodies` post-pass. Replaces the per-FD HTTP cascade for survivors. See §3.3.

### A4. Encode-once, score-many in `compute_relevance.py`
Files: `scripts/compute_relevance.py` (in-flight; do not touch until current build completes), `scripts/build_benchmark.py` (rewire `step_relevance_parallel`). Effort: M. Priority: P0. Deps: A6.
Replace per-benchmark invocations with a single `--benchmarks fb,gdelt-cameo,earnings` call that encodes once and applies per-benchmark masks. See §5.

### A5. Parallelize trafilatura body fetch
Files: `scripts/fetch_article_text.py` (117 lines, single-threaded), `scripts/retry_article_text.py` (191 lines). Effort: S. Priority: P1. Deps: none.
Wrap the URL loop in `concurrent.futures.ThreadPoolExecutor`; tune workers per the v2.1 24-worker observation. Body extraction is the second hottest path after Google News.

### A6. `src/common/embeddings_backend.py`
Files: new module. Effort: M. Priority: P0. Deps: none.
Single `encode(texts, model, backend)` API for SBERT and OpenAI Batch; reusable from compute_relevance, build_gdelt_doc_index, and any future encoder caller. SBERT path mirrors current `compute_relevance.py:embed` behavior (batch=256, FP16). See §6.

### A7. OpenAI Batch backend in `embeddings_backend.py`
Files: same module as A6. Effort: M. Priority: P1. Deps: A6.
50k-input chunked Batch jobs, async-with-resume; truncate-and-renormalize from 1536 to 768 dim for SBERT cache compatibility. See §6.2.

### A8. FAISS shard-prune by date intersection
Files: `query_gdelt_doc_index.py`. Effort: S. Priority: P1. Deps: A3.
Skip entire FAISS shards whose [start_month, end_month] does not intersect the FD's `[forecast_point - lookback_days, forecast_point)` window. Cheap, large speedup for narrow lookback values.

### A9. Per-FD query-embedding cache
Files: `query_gdelt_doc_index.py`. Effort: S. Priority: P2. Deps: A3.
Memoize SBERT-encoded queries by `MD5(question + background)`. Most builds re-query the same FDs.

### A10. SBERT `batch_size=256` plus FP16 plumbed via config
Files: `compute_relevance.py` (in-flight), `default_config.yaml` (in-flight). Effort: S. Priority: P1. Deps: A6.
Today the speedup is hardcoded; expose as `relevance.encoder.batch_size` and `relevance.encoder.fp16`. Defer until current build completes.

### A11. GDELT KG download is fine; do not touch
Files: `scripts/build_gdelt_cameo.py` (518 lines). Effort: 0. Priority: P2. Deps: none.
The 3-minute one-time cost is below the optimization waterline. Documented for completeness; no work item.

### A12. Bulk-fetch bodies only for survivors
Files: `query_gdelt_doc_index.py`. Effort: S. Priority: P0. Deps: A3.
Inverts v2.1's "fetch bodies first, filter later" to "filter first, fetch survivors". This is half the wall-clock win; called out separately because it is easy to forget when implementing A3.

---

## Category B, Architecture (drift containment)

### B1. Hybrid retrieval contract module
Files: new `src/retrieval/contract.py`. Effort: S. Priority: P0. Deps: none.
Single source of truth for the per-benchmark source cascade (Section 4 table). All fetchers and `query_gdelt_doc_index.py` import from here. Avoids the "which source for which benchmark" knowledge being scattered across three fetcher files.

### B2. `NewsFetcher` base class
Files: new `src/common/news_fetcher.py`; `scripts/fetch_forecastbench_news.py`, `scripts/fetch_gdelt_cameo_news.py`, `scripts/fetch_earnings_news.py` become subclasses. Effort: M. Priority: P1. Deps: B1.
Shared `art_id`, `domain_of`, append-loop, dedup, spam filter, HEADERS, TIMEOUT. Subclasses override `build_query`, `eligible_sources`, `art_id_prefix`. Net delta roughly -400 lines. See [`docs/V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) §8.1.

### B3. `src/unify/` package
Files: new package; `scripts/unify_articles.py` (329 lines), `scripts/unify_forecasts.py` (379 lines) become CLI shims. Effort: M. Priority: P1. Deps: none.
Mirrors the v2.1 deprecation pattern used for `benchmark/build.py` (commit `d60a4ba`). See §8.2.

### B4. `src/common/paths.py` single owner
Files: new module; gradually migrate every `Path("data/...")` literal. Effort: M (incremental). Priority: P1. Deps: none.
Eliminates the `data/unified/` versus `data/{benchmark}/` split-knowledge across scripts.

### B5. Per-stage config slices and config-hash
Files: new `src/common/config_slices.py`. Effort: S. Priority: P1. Deps: none.
Each stage declares which config keys it depends on; resume logic hashes the slice. Enables Section 7.1 snapshot semantics.

### B6. Atomic stage-meta writes
Files: every `step_*` in `scripts/build_benchmark.py` (in-flight). Effort: S. Priority: P1. Deps: B5.
Write to `.tmp` then rename. Currently some stages write meta.json non-atomically; on `Ctrl-C` mid-write the next build resumes from corrupted state.

### B7. `src/etd/date_validators.py`
Files: new module; `scripts/articles_to_facts.py:230` plus the `--strict-dates` flag plumbing. Effort: S. Priority: P1. Deps: none.
Pure-function validators (`is_post_publish`, `is_within_window`, `is_iso_format`, `is_calendar_valid`), individually unit-tested. The `>` versus `>=` bug recovered in commit `7237553` would have been caught by a per-validator test.

### B8. `src/common/gdelt_aggregator_domains.py`
Files: new module. Effort: S. Priority: P0. Deps: none.
Block list for GDELT-affiliated and aggregator domains; consumed by the editorial-only filter for GDELT-CAMEO in the bulk path. See §4.

### B9. Single `Layout` dataclass for cutoff outputs
Files: new `src/common/layout.py`; `step_publish` plus all audit scripts. Effort: M. Priority: P2. Deps: B4.
Replaces the per-script knowledge of where files live in `benchmark/data/{cutoff}/`.

### B10. Retire deprecated `benchmark/build.py` shim entirely
Files: `benchmark/build.py`. Effort: S. Priority: P2. Deps: B4 (so docs are in sync).
v2.1 left the shim as a deprecation forwarder. After v2.2 release announcement, delete.

### B11. Retire `configs/relevance.yaml` legacy fallback
Files: `scripts/compute_relevance.py:load_cfg` falls back to a standalone `configs/relevance.yaml`. Effort: S. Priority: P2. Deps: A4.
The file does not exist in the current tree; the fallback is dead code. Remove for clarity.

### B12. Single `Source` enum
Files: new `src/common/sources.py`. Effort: S. Priority: P2. Deps: B1, B2.
String literals like `"gdelt-doc"`, `"google-news"`, `"nyt"`, `"guardian"`, `"finnhub"`, `"yfinance"`, `"sec-edgar"` are scattered as bare strings. Promote to enum; consume from contract module B1 and from base fetcher B2.

### B13. Pull GDELT BigQuery fetcher into the cascade or delete
Files: `scripts/fetch_gdelt_cameo_bigquery.py` (346 lines). Effort: S. Priority: P2. Deps: B1.
Currently orphaned (not invoked from build_benchmark). Either wire it as a fallback in B1 or move to `scripts/archive/`.

### B14. Move `download_gdelt_news.py`, `download_metaculus.py`, `prepare_data.py`, `scrape_market_pages.py` to `scripts/archive/`
Files: as listed; check call sites in the orchestrator. Effort: S. Priority: P2. Deps: confirmed unreferenced.
These are pre-v2.0 artifacts. Surveyed but not invoked by `build_benchmark.py` orchestrator.

---

## Category C, Quality and rigor (carry forward from v2.1)

### C1. Two-layer leakage enforcement at score-time and publish-time
Files: `compute_relevance.py` (mask), `quality_filter.py` (audit). Effort: S. Priority: P0. Deps: A4.
Mask at score-time is for performance; audit at publish-time is for correctness. v2.2 must keep both. See §5.3.

### C2. Side-by-side acceptance test for GDELT DOC index
Files: new `tests/integration/test_v22_v21_parity.py`. Effort: M. Priority: P0. Deps: A1, A2, A3.
Build at the same cutoff with `--use-gdelt-doc-index` on and off; assert ≥ 80% Jaccard on per-FD `article_ids`; assert total survivor count within ±5%.

### C3. Embedding-backend identity in `build_manifest.json`
Files: `scripts/build_benchmark.py` (in-flight) `step_publish`. Effort: S. Priority: P0. Deps: A6.
Write `embedding_backend`, `embedding_model`, `embedding_model_revision`. Treat backend as part of the cutoff identity; refuse to mix backends within a single `{cutoff}/` directory.

### C4. Resume protocol invariants test
Files: new `tests/test_resume_invariants.py`. Effort: M. Priority: P1. Deps: B6.
For each long-running stage: kill mid-loop, restart, verify final output equals an uninterrupted run.

### C5. Config-hash regression test
Files: new `tests/test_config_slices.py`. Effort: S. Priority: P1. Deps: B5.
Mutating an unrelated config key does not invalidate a stage; mutating a related key does.

### C6. Domain blocklist round-trip test for GDELT-CAMEO editorial filter
Files: new `tests/test_gdelt_editorial_filter.py`. Effort: S. Priority: P1. Deps: B8.
Aggregator domains are filtered; legitimate editorial outlets are kept; ambiguous cases fail closed.

### C7. ETD Stage-1 hallucination floor monitoring
Files: `scripts/etd_audit.py` (already exists, 283 lines). Effort: S. Priority: P1. Deps: none.
Current Phase C v3 production: 11.8% unsupported at high confidence. Add CI threshold: fail if a new ETD batch exceeds 13%.

### C8. Per-row fingerprint collision test
Files: new `tests/test_fingerprint_collisions.py`. Effort: S. Priority: P2. Deps: A6.
Exercise `_per_row_fingerprints` over the full 220k-article pool; assert zero collisions. MD5 is cryptographically broken but fine for this use; the test is a tripwire.

### C9. Schema-version bump policy
Files: `docs/FORECAST_DOSSIER.md` §7 (already documented). Effort: 0. Priority: P2. Deps: none.
v2.2 keeps the v2.1 FD schema; no bump. Documented for clarity.

### C10. Production filter recipe in `build_manifest.json`
Files: `scripts/build_benchmark.py` (in-flight) `step_publish`. Effort: S. Priority: P1. Deps: none.
Persist the production filter recipe (`--source-blocklist news.fjsen.com,world.people.com.cn --min-confidence high --polarity asserted --no-future --require-linked-fd`) into the manifest. Reproducibility.

---

## Category D, Documentation

### D1. v2.2 migration guide in `benchmark/RECREATE.md`
Files: `benchmark/RECREATE.md`. Effort: S. Priority: P0. Deps: most of A and B landed.
Section: "Switching from v2.1 to v2.2 mid-cycle"; covers the GDELT DOC archive prerequisite, the cache-compatibility matrix, and the rollback procedure.

### D2. Update `docs/PIPELINE.md` to reference both v2.1 and v2.2 paths
Files: `docs/PIPELINE.md`. Effort: S. Priority: P1. Deps: A2.
Add a Section 3.5 documenting the GDELT DOC index path; do not duplicate [`docs/V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md), reference it.

### D3. New `docs/CACHE_INVARIANTS.md`
Files: new doc. Effort: S. Priority: P1. Deps: A6, A2, A3, B5.
Tabular cache-invalidation matrix from [`docs/V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) §7.2 elevated to its own doc, with worked examples.

### D4. Update `benchmark/DATASET.md` if `provenance` tag set changes
Files: `benchmark/DATASET.md`. Effort: S. Priority: P1. Deps: B1.
v2.2 adds `gdelt-doc`, `gdelt-doc-editorial`, possibly `sec-edgar` as new provenance tags. Document.

### D5. Update `benchmark/README.md` quickstart for v2.2 flags
Files: `benchmark/README.md`. Effort: S. Priority: P1. Deps: A1, A2, A3 wired into orchestrator.
Add `--use-gdelt-doc-index`, `--no-gdelt-doc-index`, `--embedding-backend {sbert,openai_batch}` to the documented flag set.

### D6. Archive `docs/SUMMARY.md` and `docs/CONTRIBUTION_ANALYSIS.md`
Files: as listed; move to `docs/archive/`. Effort: S. Priority: P2. Deps: none.
Pre-existing housekeeping item from the v2.1 backlog; carry forward.

### D7. v2.2 changelog entry in [`docs/PIPELINE.md`](PIPELINE.md) §12 and [`docs/FORECAST_DOSSIER.md`](FORECAST_DOSSIER.md) §8
Files: as listed. Effort: S. Priority: P0 (at v2.2 release). Deps: v2.2 ships.
One row each: "2.2 / 2026-MM-DD / GDELT DOC bulk pipeline, hybrid retrieval contract, encode-once-score-many, optional OpenAI Batch backend. No FD schema changes."

---

## Category E, Drift cleanup discovered during the audit

### E1. Three near-duplicate `_sys.path.insert(...)` blocks in `fetch_*_news.py`
Files: `scripts/fetch_forecastbench_news.py`, `scripts/fetch_gdelt_cameo_news.py`, `scripts/fetch_earnings_news.py`. Effort: S. Priority: P2. Deps: B2 (subsumed).
Each fetcher has two `_sys.path.insert(0, str(_Path(__file__).parent.parent))` blocks. Cleaned up automatically by B2.

### E2. `unify_articles.py` `_with_raised_csv_field_limit` decorator pattern
Files: `scripts/unify_articles.py:31`. Effort: S. Priority: P2. Deps: B3.
Move into the `src/unify/` package as a private helper.

### E3. `compute_relevance.py` legacy `configs/relevance.yaml` fallback
Files: `scripts/compute_relevance.py:43`. Effort: S. Priority: P2. Deps: A4.
Same as B11; called out here too because it surfaced during the audit of the relevance script.

### E4. `articles_to_facts.py` validator surface scattered across phases
Files: `scripts/articles_to_facts.py`. Effort: S. Priority: P1. Deps: B7.
Same as B7; called out here for the audit trail.

### E5. `art_id` prefix knowledge replicated in unify
Files: `scripts/unify_articles.py:art_id`. Effort: S. Priority: P2. Deps: B2, B3.
The unifier hardcodes `"art_"` while fetchers use `"fbn_"`, `"gdc_"`, `"earn_"`. Centralize.

### E6. Per-fetcher HEADERS plus TIMEOUT constants
Files: all three fetchers. Effort: S. Priority: P2. Deps: B2 (subsumed).
Identical across the fetchers. Subsumed by base class.

### E7. `_fast_jsonl.py` is a script-local module
Files: `scripts/_fast_jsonl.py` (74 lines). Effort: S. Priority: P2. Deps: B4.
Move to `src/common/fast_jsonl.py`; update imports. Currently each consumer does its own `_sys.path.insert` to reach it.

### E8. `optional_imports.py` proxy pattern is sound; document but do not change
Files: `src/common/optional_imports.py`. Effort: 0. Priority: P2. Deps: none.
Surveyed; well-designed lazy proxy; no work.

### E9. Build orchestrator step ordering is implicit
Files: `scripts/build_benchmark.py` (in-flight). Effort: M. Priority: P2. Deps: B5, B6.
The 17-step pipeline order in `main()` is procedural; no DAG. Consider promoting to a declarative DAG (e.g. dict of `step_name -> {func, deps}`) so resume can target an arbitrary step. Defer until v2.3 unless it falls out naturally from B5 plus B6.

### E10. Spam blocklist `src/common/spam_domains.py` only filters fetch-time
Files: as named. Effort: S. Priority: P2. Deps: none.
Document that spam filtering happens at fetch time only; articles already in the unified pool from earlier builds are not retroactively filtered. Either retro-filter on every load or document the limitation.

### E11. Earnings annotator metadata key inconsistency
Files: `scripts/annotate_prior_state.py` (in-flight); already fixed in commit `8076a54`. Effort: 0. Priority: P0 (already done). Deps: none.
Listed for the audit trail: `_earnings_meta.ticker` versus `metadata.ticker` mix has been resolved. Add a regression test (covered by C5 above).

### E12. Articles-audit and FD-audit field-name mismatches
Files: `scripts/articles_audit.py`, `scripts/fd_audit.py`; fixed in commit `e22395c`. Effort: 0. Priority: P0 (already done). Deps: none.
For audit trail.

### E13. ETD post-publish orchestrator entry point
Files: `scripts/etd_post_publish.py` (254 lines, shipped today in `e22395c`). Effort: 0. Priority: P0 (already done). Deps: none.
Listed for completeness. Not a v2.2 work item; v2.1 closed it.

### E14. Two parallel `scripts/build_gdelt_cameo.py` versus `benchmark/scripts/gdelt_cameo/`
Files: as named. Effort: S. Priority: P2. Deps: none.
The v2.1 deprecation kept the canonical at `scripts/build_gdelt_cameo.py`; verify the `benchmark/scripts/gdelt_cameo/` tree is fully retired. If not, archive.

### E15. `fetch_text_multi.py` versus `fetch_article_text.py` versus `retry_article_text.py`
Files: three scripts, 490 plus 117 plus 191 lines. Effort: M. Priority: P2. Deps: A5.
Three text-fetch utilities with overlapping responsibilities. After A5 parallelizes the primary fetcher, consider collapsing the trio into one module.

### E16. `gdelt_retry_orphans.py` purpose unclear
Files: `scripts/gdelt_retry_orphans.py` (210 lines). Effort: S. Priority: P2. Deps: none.
Surveyed but role versus `relink_gdelt_context.py` plus `fetch_gdelt_text.py` is unclear from the file alone. Either document inline header or archive.

### E17. `debug_flows.py` is a development utility
Files: `scripts/debug_flows.py` (289 lines). Effort: S. Priority: P2. Deps: none.
Should live under `scripts/dev/` or `tools/`, not alongside production scripts.

### E18. Per-script `ROOT = Path(__file__).parent.parent`
Files: many scripts. Effort: S. Priority: P2. Deps: B4.
Subsumed by B4's `src/common/paths.py`.

---

## Summary by priority

- **P0 (blocks v2.2 launch)**: A1, A2, A3, A4, A6, A12, B1, B8, C1, C2, C3, D1, D7, E11, E12, E13.
- **P1 (should land in v2.2)**: A5, A7, A8, A10, B2, B3, B4, B5, B6, B7, C4, C5, C6, C7, C10, D2, D3, D4, D5, E4.
- **P2 (nice-to-have or follow-on)**: A9, A11, B9, B10, B11, B12, B13, B14, C8, C9, D6, E1, E2, E3, E5, E6, E7, E8, E9, E10, E14, E15, E16, E17, E18.

Total: 16 P0, 20 P1, 25 P2 = 61 items.
