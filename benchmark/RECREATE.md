# Rebuilding the Benchmark for a Different Model Cutoff

End-to-end walkthrough for rebuilding EMR-ACH. Use this when evaluating a model with a different training cutoff, shifting the temporal window, swapping the earnings ticker universe, or adding a new custom benchmark slice.

Read [README.md](README.md) for the high-level map and [DATASET.md](DATASET.md) for the schema. The canonical FD spec lives at [`docs/FORECAST_DOSSIER.md`](../docs/FORECAST_DOSSIER.md).

## 1. The single entry point

`scripts/build_benchmark.py` is the one command you need. It reads `configs/default_config.yaml`, orchestrates all sub-scripts under `scripts/`, and publishes the final bundle to `data/{model_cutoff}/`. (`benchmark/build.py` is now a thin deprecation shim that forwards here; do not script against it.)

```bash
# Primary cutoff (paper default):
python scripts/build_benchmark.py --cutoff 2026-01-01

# Leakage-probe cutoff (90-day buffer; surfaces residual training contamination):
python scripts/build_benchmark.py --config configs/leakage_probe_config.yaml
```

Flags:

- `--cutoff YYYY-MM-DD` overrides `model_cutoff` from the active config.
- `--config PATH` selects an alternate YAML (the leakage probe lives at `configs/leakage_probe_config.yaml` and pins `model_cutoff: 2024-04-01` with a 90-day buffer).
- `--benchmarks forecastbench,gdelt_cameo,earnings` runs only a subset of tracks (default: all enabled in config).
- `--skip-raw` skips per-track raw-data rebuilds (GDELT download, `yfinance` pulls, ForecastBench clone, SEC EDGAR pulls). Use when intermediate files already exist.
- `--rebuild-embeddings` forces `common/compute_relevance.py` to re-embed from scratch rather than reusing cached `.npy` files.
- `--fresh` wipes `data/unified/` and per-track raw outputs before rebuilding (implies not-`--skip-raw`). Use for a truly from-scratch reproduction.
- `--dry-run` prints sub-commands without executing them.

## 2. What `build_benchmark.py` does, step by step

The pipeline has the following logical stages (see `scripts/build_benchmark.py::main()` for the exact ordering):

1. **Raw build per source.**
   - `gdelt_cameo`: `scripts/gdelt_cameo/build.py` downloads GDELT 2.0 Knowledge Graph plus mention CSVs, filters to same-date actor-pair events at `min_daily_mentions: 20`, and generates Peace/Tension/Violence (CAMEO root 01-09 / 10-17 / 18-20) forecasting queries.
   - `earnings`: `scripts/earnings/build.py` reads the S&P 500 universe, computes Beat/Meet/Miss labels off `yfinance` and the consensus EPS, and emits FDs.
   - `forecastbench`: `scripts/forecastbench/download_forecastbench.py` pulls the static upstream repo (bypassed by `--skip-raw`). Default v2.1 behavior keeps prediction-market sources only (Polymarket, Metaculus, Manifold, Infer); ACLED and FRED are excluded.

2. **Unify into a single schema.** `scripts/common/unify_articles.py` merges all article sources into `data/unified/articles.jsonl` (dedup by `sha1(url)[:12]`, with per-row `source_type`, `language`, and pipe-separated `provenance`). `scripts/common/unify_forecasts.py` merges all forecast sources into `data/unified/forecasts.jsonl` using the Forecast Dossier schema (see [schema/forecast_dossier.md](schema/forecast_dossier.md)).

3. **Per-benchmark relevance scoring.** `scripts/common/compute_relevance.py --benchmark-filter X` computes SBERT similarity between each forecast and the article pool, attaches the top-k within the per-source lookback window, and writes `article_ids` back into the unified FDs.

4. **GDELT-CAMEO context relink.** `scripts/gdelt_cameo/relink_context.py` replaces any oracle (leaky) event articles with leakage-free actor-pair editorial-news context articles (NYT, The Guardian, Google News) drawn from the pre-event window. MIRAI is cited as an external anchor only and is not used for retrieval.

5. **Full-text fetch on referenced URLs.**
   - `scripts/gdelt_cameo/fetch_text.py --only-referenced` (runs after relink so only pre-event context URLs are fetched).
   - `scripts/forecastbench/fetch_article_text.py` (trafilatura on ForecastBench's editorial-news URLs).
   - Earnings 5-source cascade: SEC EDGAR 8-K (`source_type: "filing"`, asymmetric pre-event filter), Finnhub, yfinance, Google News, GDELT DOC.

6. **Re-unify and re-score.** Steps 2 to 4 are re-run so the newly-fetched text participates in relevance scoring.

7. **Quality filter.** `scripts/common/quality_filter.py` applies the final accept rule: `min_articles` (default 3), `min_distinct_days` (2), `min_chars` (1500), `min_question_chars` (20), and the model-cutoff guard (see section 7).

8. **Prior-state and partition annotation.** `scripts/annotate_prior_state.py` computes the per-benchmark `prior_state` oracle (GDELT modal intensity over the prior 30 days; Earnings mode of the prior 4 quarters; ForecastBench crowd majority on `freeze_datetime_value`), promotes the original domain class to `x_multiclass_ground_truth` and `x_multiclass_hypothesis_set`, replaces `hypothesis_set` with the binary `["Comply", "Surprise"]` target, and tags `fd_type` as `stability`, `change`, or `unknown`. It also stamps `default_horizon_days` per-FD (14d default for GDELT and FB, source-specific for Earnings).

9. **Diagnostics.** `scripts/common/diagnostic_report.py` and `scripts/common/build_eda_report.py` produce the audit artifacts.

10. **Publish.** The orchestrator writes `forecasts.jsonl` and the referenced subset of `articles.jsonl` to `data/{cutoff}/`, drops audit artifacts into `data/{cutoff}/meta/`, snapshots the effective config as `data/{cutoff}/benchmark.yaml` alongside `build_manifest.json` (timestamp, git sha, benchmarks included), and emits `checksums.sha256` for the shipped files. The `data/unified/` staging directory is removed on success.

## 3. Config-driven parameters

Every knob lives in `configs/default_config.yaml`; the leakage probe overrides via `configs/leakage_probe_config.yaml`. Relevance is a section inside the same file (there is no separate `relevance.yaml`). Fields you most commonly edit:

- `model_cutoff` (YYYY-MM-DD): evaluator LLM's training cutoff. `--cutoff` on the CLI overrides this.
- `cutoff_buffer_days` (default 0; recommended 90 for production, used by the leakage probe): extra safety margin past `model_cutoff`.
- `benchmarks.forecastbench.prediction_market_sources`: which FB sub-sources to include (default v2.1: `polymarket`, `metaculus`, `manifold`, `infer`).
- `benchmarks.gdelt_cameo.context_start` / `context_end` / `test_month` / `all_end` (YYYYMM): GDELT KG context window.
- `benchmarks.gdelt_cameo.min_daily_mentions` (default 20): source-reliability threshold from GDELT methodology.
- `benchmarks.gdelt_cameo.max_download_workers` (default 16).
- `benchmarks.earnings.start` / `end` (YYYY-MM-DD): earnings-report date window.
- `benchmarks.earnings.tickers`: `null` uses the S&P 500 universe in `scripts/earnings/build.py`, or pass an explicit list.
- `benchmarks.earnings.threshold` (default 0.02): `|beat/miss|` as fraction of consensus EPS.
- `quality.{min_articles, min_distinct_days, min_chars, min_question_chars}`.
- `relevance.embedding_model` (default `sentence-transformers/all-mpnet-base-v2`), `relevance.batch_size`, `relevance.max_text_chars`.
- `relevance.sources.<src>.{embedding_threshold, keyword_overlap_min, lookback_days, top_k, recency_weight, actor_match_required}` for each of `polymarket`, `metaculus`, `manifold`, `infer`, `gdelt-kg`, `yfinance`.
- `output.root` (default `data`): where `{cutoff}/` bundles get written, relative to repo root.

The `gdelt-kg` source uses `actor_match_required: true`; SBERT is not used for that source's retrieval (actor-pair matching via `relink_context.py` replaces it). All other sources use SBERT cross-match.

## 4. Environment-variable overrides

A handful of behaviors are toggled via environment variables (read by the relevant builder), allowing config-free experiment sweeps:

- `EMRACH_FB_SUBJECT_FILTER`: when set to a truthy value, re-enables the legacy ForecastBench geopolitics-only subject filter (default v2.1: off, ~1172 FDs; legacy on: ~530 FDs).
- `EMRACH_FB_HORIZON_DAYS`, `EMRACH_GDELT_HORIZON_DAYS`, `EMRACH_EARNINGS_HORIZON_DAYS`: per-benchmark overrides for `default_horizon_days` written into each FD by `annotate_prior_state.py`. The runtime baselines runner separately applies `apply_experiment_horizon()` so the same dataset can be evaluated at multiple horizons without rebuilding.

## 5. Worked example: rebuild for a 2025-04-01 cutoff

Shift all tracks past the new cutoff and apply a 90-day model-training buffer.

Edit `configs/default_config.yaml`:

```yaml
model_cutoff: "2025-04-01"
cutoff_buffer_days: 90

benchmarks:
  gdelt_cameo:
    context_start: "202505"
    context_end:   "202510"
    test_month:    "202511"
    all_end:       "202512"
  earnings:
    start: "2025-05-01"
    end:   "2025-12-31"
```

Then:

```bash
python scripts/build_benchmark.py                      # reads the edited config
# or override on the CLI:
python scripts/build_benchmark.py --cutoff 2025-04-01
```

Output `data/2025-04-01/forecasts.jsonl` contains only FDs whose `resolution_date` is strictly after `2025-04-01 + 90 days`, with every evidence article dated strictly before the corresponding `forecast_point`.

## 6. Worked example: add a custom benchmark slice

To bolt on a new track (e.g. "sports outcomes"):

1. Write a builder under `scripts/sports/build.py` emitting records in the FD schema (see [schema/forecast_dossier.md](schema/forecast_dossier.md) and the canonical [`docs/FORECAST_DOSSIER.md`](../docs/FORECAST_DOSSIER.md)) to `data/sports/sports_forecasts.jsonl` and any URLs to `data/sports/sports_urls.jsonl`.
2. Reuse `scripts/forecastbench/fetch_article_text.py` (or a new per-source fetcher) to populate full text. Set `source_type` on each article (one of `news`, `filing`, `social`, `blog`, `other`).
3. Extend `scripts/common/unify_articles.py` and `unify_forecasts.py` to read the new per-track files.
4. Add a `sports:` entry under `relevance.sources.` in `default_config.yaml` with its own `embedding_threshold`, `lookback_days`, `top_k`.
5. Define a per-benchmark `prior_state` oracle and register it in `scripts/annotate_prior_state.py` so the new track gets a binary Comply/Surprise target plus an `fd_type` partition.
6. Add a dispatch stanza to `scripts/build_benchmark.py` so the new track is called in stage 1, then rerun.

## 7. Leakage protections

Four layered mechanisms enforced and logged in the `meta/` files.

1. **Article-level (retrieval leakage).** `scripts/common/quality_filter.py` prunes any article with `publish_date >= forecast_point` from every FD's `article_ids` before applying the other quality checks. The leakage-articles-pruned count is recorded in `meta/quality_meta.json`. The retrieval window upper bound is set by the per-source `lookback_days` in `relevance.sources.<src>`. Articles with `source_type: "filing"` are subject to an asymmetric pre-event filter in the Earnings track.
2. **Model-level (training leakage).** When `--model-cutoff` and `--cutoff-buffer-days` are forwarded (which `build_benchmark.py` does automatically), the quality filter drops any FD whose `resolution_date <= model_cutoff + cutoff_buffer_days`. Note this uses `resolution_date`, not `forecast_point`: the resolution must fall strictly after the evaluator model's plausible knowledge horizon.
3. **Two-cutoff probe.** Always rebuild at both the primary (`2026-01-01`) and leakage-probe (`2024-04-01`, 90-day buffer) cutoffs. Comparing scores across cutoffs surfaces residual contamination.
4. **GDELT-CAMEO oracle relink.** `scripts/gdelt_cameo/relink_context.py` replaces oracle event articles with editorial-news actor-pair context. Skipping this step causes FDs to leak the answer; the leakage guard catches them, but the entire `gdelt_cameo` slice is lost.

The final leakage audit in `meta/diagnostic_report.json` must read `leakage_violations: 0`.

## 8. Audit-tool quick reference

After a build completes, the following run on the published artifacts (`data/{cutoff}/`) and on the ETD layer at `data/etd/`:

```bash
# Per-domain, per-source_type, per-provenance breakdown of articles.jsonl
python scripts/articles_audit.py --cutoff 2026-01-01

# Per-benchmark FD audit (counts, fd_type partition, prior_state distribution)
python scripts/fd_audit.py --cutoff 2026-01-01

# FD-level filtering by benchmark / fd_type / prior_state / predicate
python scripts/fd_filter.py --cutoff 2026-01-01 --fd-type change --benchmark gdelt_cameo

# ETD layer (Stage 1 atomic facts) audit, schema verify, filter
python scripts/etd_audit.py
python scripts/etd_verify.py
python scripts/etd_filter.py --benchmark earnings

# ETD parser triage
python scripts/etd_debug_errors.py
python scripts/etd_debug_empty.py

# Cross-reference ETD facts against the published article pool
python scripts/etd_compare_facts_vs_articles.py --cutoff 2026-01-01
```

The schema invariants enforced in CI (46 tests total) live under `tests/`: `tests/test_pipeline_invariants.py` (16), `tests/test_fd_schema_invariants.py` (14), `tests/test_article_schema.py` (8), `tests/test_etd_schema_invariants.py` (7).

## 9. Compute and time estimates

Approximate wall-clock figures on a 16 GB workstation (RTX 2060 optional):

| Step                                       | Wall-clock                    | Notes |
| ------------------------------------------ | ----------------------------- | ----- |
| GDELT-CAMEO raw build                      | ~60 min                       | I/O-bound parallel downloads (`max_download_workers`). |
| GDELT-CAMEO trafilatura fetch              | varies                        | ~100-500 URLs/min depending on target domains; only-referenced mode. |
| GDELT-CAMEO relink                         | 1-2 min                       | single-threaded CSV scan. |
| Earnings builder (5-source cascade)        | 5-15 min                      | SEC EDGAR plus Finnhub plus yfinance plus Google News plus GDELT DOC; rate-limited. |
| ForecastBench text fetch                   | varies                        | same trafilatura fetcher. |
| Unify articles                             | 2-5 min                       | SHA1 dedup over the full pool. |
| Unify forecasts                            | <1 min                        | |
| Relevance SBERT (per benchmark)            | ~30 s / 1000 FDs CPU          | `all-mpnet-base-v2`, batch 32. GPU (RTX 2060) ~6 s / 1000. |
| Prior-state annotation                     | <30 s                         | pure Python over the filtered FD set. |
| Quality filter plus diagnostic plus EDA    | <30 s total                   | pure Python. |

## 10. Common pitfalls

- **OOM during GDELT KG merge.** Doubling the context window can push peak RAM toward 16 GB. Split by quarter and merge quarterly parquets if you hit OOM.
- **Trafilatura failure rate.** Expect roughly 1-in-5 URLs to fail (paywalls, JS-rendered SPAs, non-Latin charsets). `scripts/common/retry_article_text.py` rescues many with UA rotation and a readability-lxml fallback; budget for a residue of un-fetchable URLs.
- **Metaculus 2.0 API returns `null` resolutions.** After the Metaculus 2.0 migration (early 2026), resolution and community-prediction history are no longer exposed via the public API; a fresh pull returns empty records until access is restored. Plan to backfill from a cached snapshot or skip the sub-source.
- **Exotic encodings.** A handful of RO/IT/RU/CZ news sites serve pages without explicit charset headers. The retry fetcher handles this via `chardet` plus `response.apparent_encoding`; extend that heuristic if you add a new source with unusual encodings.
- **GDELT-CAMEO oracle article leakage.** If you forget to run `scripts/gdelt_cameo/relink_context.py` (`build_benchmark.py` invokes it automatically), FDs will leak the answer into their evidence set. The leakage guard catches this but you will lose the entire `gdelt_cameo` slice. Always run the relink step.
- **Forgetting to run prior-state annotation.** Without `scripts/annotate_prior_state.py` the FDs will retain only the multiclass `hypothesis_set` and will lack `prior_state`, `fd_type`, `x_multiclass_*`, and `default_horizon_days`. The schema-invariant tests in `tests/test_fd_schema_invariants.py` catch this in CI.
