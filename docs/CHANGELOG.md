# EMR-ACH Changelog

## 2.2 (in progress, 2026-04-23+)

Focus: wall-clock and reproducibility. No FD schema changes.

### Added
- `src/common/paths.py` with `bootstrap_sys_path()` (B4 + B4a); single
  owner of the `sys.path.insert` idiom that caused three production
  crashes on 2026-04-23.
- `src/common/config_slices.py` per-stage slice declaration + cache
  hashing (B5 + C5).
- `src/retrieval/contract.py` per-benchmark source cascade contract
  (B1) with `Benchmark`, `Source`, `RetrievalMode` enums.
- `src/common/embeddings_backend.py` unified `encode(..., backend=...)`
  API wrapping the shipped `src.common.openai_embeddings` (A6).
- `src/common/gdelt_aggregator_domains.py` editorial-only filter
  blocklist (B8 + C6).
- `src/etd/date_validators.py` pure-function ETD date predicates (B7 +
  E4); pins the `> vs >=` leakage rule from commit 7237553.
- `src/common/article_checksums.py` + `scripts/preflight_publish.py`
  (G2 + G3 + G5); per-benchmark article-file integrity gate that
  closes the 2026-04-22 silent-deletion failure.
- `scripts/check_quality_filter.py` audit-level gate that refuses to
  proceed when `forecasts_filtered.jsonl` / `quality_meta.json` are
  absent or quality_filter was a no-op (G4).
- `scripts/reuse_check.py` read-only dry-audit of stage-cache reuse
  (G6).
- `benchmark/evaluation/baselines/methods/b10_hybrid_facts_articles.py`
  (F1) and `b10b_facts_only.py` (F2) ablation-triangle baselines.
- `src/common/layout.py` typed `CutoffLayout` dataclass for per-cutoff
  output paths (B9).
- `src/common/retrieval_router.py` per-benchmark retrieval dispatch
  with SBERT fallback (A13).
- `src/common/stage_cache.py` reuse-contract primitives with atomic
  meta-file writes (G1 + B6 + C4).
- `src/common/news_fetcher.py` base class plus `art_id_for`,
  `domain_of`, `FetchedArticle` (B2 + E1 + E5 + E6).
- `src/unify/csv_helpers.py` `with_raised_csv_field_limit` decorator
  lifted from `scripts/unify_articles.py` (B3 + E2).
- `src/common/fast_jsonl.py` promoted from `scripts/_fast_jsonl.py`
  with `write_jsonl_atomic` helper (E7).
- `scripts/fetch_gdelt_doc_archive.py` (A1),
  `scripts/build_gdelt_doc_index.py` (A2),
  `scripts/query_gdelt_doc_index.py` (A3 + A8 + A12) skeleton CLIs with
  shard-prune logic and `--dry-run` defaults.
- `src/common/parallel_body_fetch.py` bounded
  `ThreadPoolExecutor` body-fetch helper (A5).
- `src/common/query_embedding_cache.py` per-FD query embedding cache
  (A9) keyed by backend + model.
- `src/common/sources.py` Source enum re-export (B12).
- `docs/CACHE_INVARIANTS.md` invalidation matrix (D3).
- `scripts/archive/` quarantine directory for pre-v2.0 scripts (B14).

### Fixed (not in this pass but relevant to v2.2)
- ETD Phase A: `>= -> >` leakage rule (commit 7237553).
- earnings annotator metadata key mismatch (commit 8076a54).
- audit field-name mismatches (commit e22395c).

### Deferred to v2.3
- E9: declarative DAG orchestration for `build_benchmark.py`.
- B18: ETD Stage 2 incremental dedup.
- B4b: completion of the long-tail `sys.path.insert` migration.
- B9b: full migration of audit scripts to `CutoffLayout`.
- F3: EMR-ACH analysis matrix accepting facts as rows (rescoped to L).

### Acceptance
- The full pytest suite grows from 46 pre-existing tests to a v2.2
  baseline of 170+; every test green.
- No in-flight files edited in this pass (see the tasking hard rules).

## 2.1 (2026-04-22)

See [`PIPELINE.md`](PIPELINE.md) and [`FORECAST_DOSSIER.md`](FORECAST_DOSSIER.md)
§8 for the 2.1 framing. Key change: unified Comply/Surprise primary
target across the three benchmarks.
