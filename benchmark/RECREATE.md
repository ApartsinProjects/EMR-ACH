# Rebuilding the Benchmark for a Different Model Cutoff

End-to-end walkthrough for rebuilding EMR-ACH. Use this when evaluating a model with a different training cutoff, shifting the temporal window, swapping the earnings ticker universe, or adding a new custom benchmark slice.

Read [README.md](README.md) for the high-level map and [DATASET.md](DATASET.md) for the schema.

## 1. The single entry point

`benchmark/build.py` is the one command you need. It reads `configs/default_config.yaml`, orchestrates all sub-scripts under `scripts/`, and publishes the final bundle to `data/{model_cutoff}/`.

```bash
cd benchmark
python build.py --cutoff 2026-01-01
```

Flags:

- `--cutoff YYYY-MM-DD` overrides `model_cutoff` from the config.
- `--benchmarks forecastbench,gdelt_cameo,earnings` runs only a subset of tracks (default: all enabled in config).
- `--skip-raw` skips per-track raw-data rebuilds (GDELT download, `yfinance` pulls, ForecastBench clone). Use when intermediate files already exist.
- `--rebuild-embeddings` forces `common/compute_relevance.py` to re-embed from scratch rather than reusing cached `.npy` files.
- `--fresh` wipes `data/unified/` and per-track raw outputs before rebuilding (implies not-`--skip-raw`). Use for a truly from-scratch reproduction.
- `--dry-run` prints sub-commands without executing them.

## 2. What `build.py` does, step by step

The pipeline has the following logical stages (see `build.py::main()` for the exact ordering):

1. **Raw build per source.**
   - `gdelt_cameo`: `scripts/gdelt_cameo/build.py` downloads GDELT 2.0 Knowledge Graph + mention CSVs, filters to same-date actor-pair events, and generates forecasting queries.
   - `earnings`: `scripts/earnings/build.py` reads `yfinance` quarterly earnings, computes Beat / Meet / Miss labels, and emits FDs.
   - `forecastbench`: `scripts/forecastbench/download_forecastbench.py` pulls the static upstream repo (bypassed by `--skip-raw`).

2. **Unify into a single schema.** `scripts/common/unify_articles.py` merges all article sources into `data/unified/articles.jsonl` (dedup by `sha1(url)[:12]`). `scripts/common/unify_forecasts.py` merges all forecast sources into `data/unified/forecasts.jsonl` using the Forecast Dossier schema (see [schema/forecast_dossier.md](schema/forecast_dossier.md)).

3. **Per-benchmark relevance scoring.** `scripts/common/compute_relevance.py --benchmark-filter X` computes SBERT similarity between each forecast and the article pool, attaches the top-k within the per-source lookback window, and writes `article_ids` back into the unified FDs.

4. **GDELT-CAMEO context relink.** `scripts/gdelt_cameo/relink_context.py` replaces any oracle (leaky) event articles with leakage-free actor-pair context articles drawn from the pre-event window. This step uses actor-pair matching (`actor_match_required: true`), not SBERT.

5. **Full-text fetch on referenced URLs.**
   - `scripts/gdelt_cameo/fetch_text.py --only-referenced` (runs after relink so only pre-event context URLs are fetched).
   - `scripts/forecastbench/fetch_article_text.py` (trafilatura on ForecastBench's GDELT DOC URLs).

6. **Re-unify + re-score.** Steps 2-4 are re-run so the newly-fetched text participates in relevance scoring.

7. **Quality filter.** `scripts/common/quality_filter.py` applies the final accept rule: `min_articles` (default 3), `min_distinct_days` (2), `min_chars` (1500), `min_question_chars` (20), and the model-cutoff guard (see section 6).

8. **Diagnostics.** `scripts/common/diagnostic_report.py` and `scripts/common/build_eda_report.py` produce the audit artifacts.

9. **Publish.** The orchestrator writes `forecasts.jsonl` and the referenced subset of `articles.jsonl` to `data/{cutoff}/`, drops audit artifacts into `data/{cutoff}/meta/`, and snapshots the effective config as `data/{cutoff}/benchmark.yaml` alongside `build_manifest.json` (timestamp, git sha, benchmarks included). The `data/unified/` staging directory is removed on success.

## 3. Config-driven parameters

Every knob lives in `configs/default_config.yaml`. Relevance is a section inside that same file (there is no separate `relevance.yaml`). The fields you most commonly edit:

- `model_cutoff` (YYYY-MM-DD): evaluator LLM's training cutoff. `--cutoff` on the CLI overrides this.
- `cutoff_buffer_days` (default 0, recommended 90 for production): extra safety margin past `model_cutoff`.
- `benchmarks.forecastbench.prediction_market_sources`: which FB sub-sources to include.
- `benchmarks.gdelt_cameo.context_start` / `context_end` / `test_month` / `all_end` (YYYYMM): GDELT KG context window.
- `benchmarks.gdelt_cameo.min_daily_mentions` (default 50): source-reliability threshold from GDELT methodology.
- `benchmarks.gdelt_cameo.max_download_workers` (default 16).
- `benchmarks.earnings.start` / `end` (YYYY-MM-DD): earnings-report date window.
- `benchmarks.earnings.tickers`: `null` uses the default list in `scripts/earnings/build.py`, or pass a list.
- `benchmarks.earnings.threshold` (default 0.02): `|beat/miss|` as fraction of consensus EPS.
- `quality.{min_articles, min_distinct_days, min_chars, min_question_chars}`.
- `relevance.embedding_model` (default `sentence-transformers/all-mpnet-base-v2`), `relevance.batch_size`, `relevance.max_text_chars`.
- `relevance.sources.<src>.{embedding_threshold, keyword_overlap_min, lookback_days, top_k, recency_weight, actor_match_required}` for each of `polymarket`, `metaculus`, `manifold`, `infer`, `gdelt-kg`, `yfinance`.
- `output.root` (default `data`): where `{cutoff}/` bundles get written, relative to `benchmark/`.

The `gdelt-kg` source uses `actor_match_required: true`; SBERT is not used for that source's retrieval (actor-pair matching via `relink_context.py` replaces it). All other sources use SBERT cross-match.

## 4. Worked example: rebuild for a 2025-04-01 cutoff

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
python build.py                    # reads the edited config
# or override on the CLI:
python build.py --cutoff 2025-04-01
```

Output `data/2025-04-01/forecasts.jsonl` contains only FDs whose `resolution_date` is strictly after `2025-04-01 + 90 days`, with every evidence article dated strictly before the corresponding `forecast_point`.

## 5. Worked example: add a custom benchmark slice

To bolt on a new track (e.g. "sports outcomes"):

1. Write a builder under `scripts/sports/build.py` emitting records in the FD schema (see [schema/forecast_dossier.md](schema/forecast_dossier.md)) to `data/sports/sports_forecasts.jsonl` and any URLs to `data/sports/sports_urls.jsonl`.
2. Reuse `scripts/forecastbench/fetch_article_text.py` (or a new per-source fetcher) to populate full text.
3. Extend `scripts/common/unify_articles.py` and `unify_forecasts.py` to read the new per-track files.
4. Add a `sports:` entry under `relevance.sources.` in `default_config.yaml` with its own `embedding_threshold`, `lookback_days`, `top_k`.
5. Add a dispatch stanza to `build.py` so the new track is called in stage 1, then rerun `python build.py --cutoff ...`.

## 6. Leakage protections

Two tiers of guard are enforced and logged in the `meta/` files.

1. **Article-level (retrieval leakage).** `scripts/common/quality_filter.py` prunes any article with `publish_date >= forecast_point` from every FD's `article_ids` before applying the other quality checks. The leakage-articles-pruned count is recorded in `meta/quality_meta.json`. The retrieval window upper bound is set by the per-source `lookback_days` in `relevance.sources.<src>`.

2. **Model-level (training leakage).** When `--model-cutoff` and `--cutoff-buffer-days` are forwarded (which `build.py` does automatically), the quality filter drops any FD whose `resolution_date <= model_cutoff + cutoff_buffer_days`. Note this uses `resolution_date`, not `forecast_point` — the resolution must fall strictly after the evaluator model's plausible knowledge horizon.

The final leakage audit in `meta/diagnostic_report.json` must read `leakage_violations: 0`.

## 7. Compute and time estimates

Approximate wall-clock figures on a 16 GB workstation (RTX 2060 optional):

| Step                                       | Wall-clock                    | Notes |
| ------------------------------------------ | ----------------------------- | ----- |
| GDELT-CAMEO raw build                      | ~60 min                       | I/O-bound parallel downloads (`max_download_workers`). |
| GDELT-CAMEO trafilatura fetch              | varies                        | ~100-500 URLs/min depending on target domains; only-referenced mode. |
| GDELT-CAMEO relink                         | 1-2 min                       | single-threaded CSV scan. |
| Earnings builder                           | 2-5 min                       | `yfinance`, rate-limited. |
| ForecastBench text fetch                   | varies                        | same trafilatura fetcher. |
| Unify articles                             | 2-5 min                       | SHA1 dedup over the full pool. |
| Unify forecasts                            | <1 min                        | |
| Relevance SBERT (per benchmark)            | ~30 s / 1000 FDs CPU          | `all-mpnet-base-v2`, batch 32. GPU (RTX 2060) ~6 s / 1000. |
| Quality filter + diagnostic + EDA          | <30 s total                   | pure Python. |

## 8. Common pitfalls

- **OOM during GDELT KG merge.** Doubling the context window can push peak RAM toward 16 GB. Split by quarter and merge quarterly parquets if you hit OOM.
- **Trafilatura failure rate.** Expect roughly 1-in-5 URLs to fail (paywalls, JS-rendered SPAs, non-Latin charsets). `scripts/common/retry_article_text.py` rescues many with UA rotation and a readability-lxml fallback; budget for a residue of un-fetchable URLs.
- **Metaculus 2.0 API returns `null` resolutions.** After the Metaculus 2.0 migration (early 2026), resolution and community prediction history are no longer exposed via the public API; a fresh pull returns empty records until access is restored. Plan to backfill from a cached snapshot or skip the sub-source.
- **Exotic encodings.** A handful of RO/IT/RU/CZ news sites serve pages without explicit charset headers. The retry fetcher handles this via `chardet` + `response.apparent_encoding`; extend that heuristic if you add a new source with unusual encodings.
- **GDELT-CAMEO oracle article leakage.** If you forget to run `scripts/gdelt_cameo/relink_context.py` (build.py invokes it automatically), FDs will leak the answer into their evidence set. The leakage guard catches this but you will lose the entire `gdelt_cameo` slice. Always run the relink step.
