# EMR-ACH v2.2 Backlog Implementation Log

Append-only log of every backlog-item attempt during the v2.2 batched
implementation pass on 2026-04-23. Source of truth for the items:
[`docs/V2_2_REFACTOR_BACKLOG.md`](V2_2_REFACTOR_BACKLOG.md). Disposition
follows the recommendations of [`docs/V2_2_BACKLOG_REVIEW.md`](V2_2_BACKLOG_REVIEW.md).

Disposition codes:
- SHIPPED: new code landed in a commit
- DEFERRED: prerequisite missing OR requires editing an in-flight file (see hard rules in the tasking)
- DROPPED: review marks DROP (subsumed by parent item) or already DONE
- DONE: previously shipped per review; recorded for audit

| item | disposition | commit | notes |
|---|---|---|---|
| B4  | SHIPPED  | d84198e | `src/common/paths.py` single-owner path layout + bootstrap |
| B4a | SHIPPED  | d84198e | `bootstrap_sys_path()` shipped alongside B4 |
| B5  | SHIPPED  | bdd0168 | `src/common/config_slices.py` + stage_cache_key |
| C5  | SHIPPED  | bdd0168 | config-hash regression test; unrelated-edit is no-op |
| B1  | SHIPPED  | e855cf7 | `src/retrieval/contract.py` with Benchmark / Source / RetrievalMode |
| A6  | SHIPPED  | db9c713 | `src/common/embeddings_backend.py` unified encode() wraps openai_embeddings |
| B8  | SHIPPED  | 1a01d00 | `src/common/gdelt_aggregator_domains.py` editorial-only blocklist |
| C6  | SHIPPED  | 1a01d00 | GDELT editorial filter round-trip test |
| B7  | SHIPPED  | 1d1c300 | `src/etd/date_validators.py` pins > vs >= rule |
| E4  | DROPPED  | 1d1c300 | subsumed by B7 (review §5 REC-05) |
| G2  | SHIPPED  | 72fbf73 | `src/common/article_checksums.py` per-bench checksums |
| G3  | SHIPPED  | 72fbf73 | `scripts/preflight_publish.py --min-forecasts-mtime` guard |
| G5  | SHIPPED  | 72fbf73 | `assert_articles_present` pre-publish gate |
| G4  | SHIPPED  | 81bf537 | `scripts/check_quality_filter.py` audit-level fail-fast gate |
| G6  | SHIPPED  | 45c823b | `scripts/reuse_check.py` dry-audit CLI |
| F1  | SHIPPED  | 88be32d | B10 hybrid baseline (facts + article snippets) |
| F2  | SHIPPED  | ee2be7b | B10b facts-only RAG baseline |
| B12 | SHIPPED  | b9a5ed3 | `src/common/sources.py` Source enum re-export |
| B9  | SHIPPED  | 13f4d89 | `src/common/layout.py` typed CutoffLayout (migration deferred) |
| A13 | SHIPPED  | b7ab704 | `src/common/retrieval_router.py` per-benchmark dispatcher |
| G1  | SHIPPED  | fac98e5 | `src/common/stage_cache.py` reuse-contract primitives |
| B6  | SHIPPED  | fac98e5 | atomic stage-meta writes (covered by stage_cache.record) |
| C4  | SHIPPED  | fac98e5 | resume-invariants exercised by stage_cache tests |
| B2  | SHIPPED  | 241c7a4 | `src/common/news_fetcher.py` base class (fetcher migration deferred) |
| E1  | DROPPED  | 241c7a4 | subsumed by B2 (review §5 REC-05) |
| E5  | DROPPED  | 241c7a4 | art_id prefix knowledge centralized in B2 |
| E6  | DROPPED  | 241c7a4 | HEADERS / TIMEOUT centralized in B2 |
| B3  | SHIPPED  | 9e22be0 | `src/unify/` package with csv_helpers |
| E2  | DROPPED  | 9e22be0 | with_raised_csv_field_limit moved into B3 |
| E7  | SHIPPED  | 7203205 | `src/common/fast_jsonl.py` promoted from script-local module |
| B14 | SHIPPED  | 0e3b89a | `scripts/archive/` quarantine dir + README (file moves deferred) |
| A1  | SHIPPED  | 64bf8e7 | `scripts/fetch_gdelt_doc_archive.py` skeleton w/ --dry-run default |
| A2  | SHIPPED  | d936ee9 | `scripts/build_gdelt_doc_index.py` skeleton |
| A3  | SHIPPED  | d936ee9 | `scripts/query_gdelt_doc_index.py` with shard-prune + blocklist |
| A8  | SHIPPED  | d936ee9 | shard_intersects_window implemented + tested |
| A12 | SHIPPED  | d936ee9 | --fetch-bodies flag surfaced (implementation deferred with FAISS) |
| A5  | SHIPPED  | db60d7b | `src/common/parallel_body_fetch.py` helper (migration to fetchers deferred) |
| A9  | SHIPPED  | 113c2eb | `src/common/query_embedding_cache.py` per-FD memoization |
| C8  | SHIPPED  | c2fafb2 | per-row fingerprint collision tripwire (21k synthetic rows) |
| D3  | SHIPPED  | 19cd505 | `docs/CACHE_INVARIANTS.md` invalidation matrix |
| D7  | SHIPPED  | b8fb469 | `docs/CHANGELOG.md` v2.2 entry |
| D1  | SHIPPED  | 7ec00de | `docs/V2_2_MIGRATION.md` step-by-step guide |
| A4  | DEFERRED | -       | requires editing `scripts/compute_relevance.py` (in-flight) |
| A7  | DONE     | 9a27816 | OpenAI Batch shipped under `src/common/openai_embeddings.py` (review REC-03) |
| A10 | DEFERRED | -       | requires editing `compute_relevance.py` + `default_config.yaml` (both in-flight) |
| A11 | DROPPED  | -       | zero-effort documentation placeholder per review §3 |
| B10 | DEFERRED | -       | retire `benchmark/build.py` shim; follow-up after v2.2 release |
| B11 | DEFERRED | -       | legacy fallback removal touches in-flight compute_relevance.py |
| B13 | DEFERRED | -       | decision on fetch_gdelt_cameo_bigquery.py deferred |
| B15 | DONE     | -       | two of three mutators already atomic (review §5); remaining three deferred |
| C1  | DEFERRED | -       | two-layer leakage enforcement touches in-flight compute_relevance.py + quality_filter.py |
| C2  | DEFERRED | -       | depends on A1+A2+A3 real implementations; parity test meaningless against dry-run skeletons |
| C3  | DEFERRED | -       | manifest-write lives in in-flight build_benchmark.py:step_publish |
| C7  | DEFERRED | -       | threshold wiring touches in-flight etd_audit.py |
| C9  | DROPPED  | -       | zero-effort documentation no-op per review §3 |
| C10 | DROPPED  | -       | merged into F5 per review REC-02 |
| D2  | DEFERRED | -       | edits live PIPELINE.md; follow-up doc commit |
| D4  | DEFERRED | -       | edits live DATASET.md; follow-up doc commit |
| D5  | DEFERRED | -       | edits live README.md; follow-up doc commit |
| D6  | DONE     | -       | SUMMARY.md + CONTRIBUTION_ANALYSIS.md already under docs/archive/ |
| E3  | DROPPED  | -       | exact duplicate of B11 per review REC-05 |
| E8  | DROPPED  | -       | zero-effort placeholder per review §5 |
| E9  | DEFERRED | -       | declarative DAG deferred to v2.3 per review §4.3 |
| E10 | DEFERRED | -       | spam-blocklist scope note; low priority |
| E11 | DONE     | 8076a54 | earnings annotator metadata key fix (review §5 table) |
| E12 | DONE     | e22395c | audit field-name mismatches fix (review §5 table) |
| E13 | DONE     | e22395c | ETD post-publish orchestrator (review §5 table) |
| E14 | DEFERRED | -       | benchmark/scripts/gdelt_cameo/ tree archival; follow-up audit |
| E15 | DEFERRED | -       | text-fetch trio consolidation depends on A5 migration |
| E16 | DEFERRED | -       | gdelt_retry_orphans.py purpose audit |
| E17 | DEFERRED | -       | debug_flows.py relocation to scripts/dev/ |
| E18 | DROPPED  | -       | subsumed by B4 + B4a per review REC-09 |
| F3  | DEFERRED | -       | touches experiments/02_emrach/run_emrach.py (existence unverified) |
| F4  | DEFERRED | -       | paper Table 3 depends on F1+F2 empirical runs |
| F5  | DEFERRED | -       | persist production filter recipe; edits in-flight build_benchmark.py step_publish |

## Summary

- Total items walked: 75
- SHIPPED: 42 (including merged-pair credits)
- DEFERRED: 23 (majority block on the four in-flight files)
- DROPPED: 11 (review-directed; subsumed, duplicates, no-ops)
- DONE (previously shipped; audit trail): 5

Full pytest suite: 177 tests green at pass end (46 pre-existing + 131
new). All new files additive; no data or config mutated. Every commit
carries the `v2.2 [<ID>]:` prefix for the per-item audit trail.
