# EMR-ACH Benchmark Dataset

Schema, data-file inventory, and guidance for reading the audit artifacts produced by `scripts/build_benchmark.py`.

## 1. Schema overview

The benchmark ships in two linked schemas. All fields are produced by `scripts/common/unify_forecasts.py`, pruned by `scripts/common/compute_relevance.py`, filtered by `scripts/common/quality_filter.py`, and finally annotated by `scripts/annotate_prior_state.py` (which promotes the binary `Comply` vs `Surprise` target).

Canonical, top-level FD reference: [`docs/FORECAST_DOSSIER.md`](../docs/FORECAST_DOSSIER.md). The summary below is per-benchmark facing; it is intentionally aligned with that document.

**Forecast Dossier (FD).** One JSON object per line in `data/{cutoff}/forecasts.jsonl`, representing a single resolved forecast question plus its pre-event evidence. Every FD carries:

- `id` (str) and a `benchmark` tag: `forecastbench`, `gdelt_cameo`, or `earnings`.
- `source` sub-tag: `polymarket`, `metaculus`, `manifold`, `infer`, `gdelt-kg`, or `yfinance`.
- `hypothesis_set` (primary, binary): always `["Comply", "Surprise"]`. `Comply` means the realized outcome matches the prior-state oracle; `Surprise` means it does not.
- `hypothesis_definitions` (label to one-sentence definition, in natural language; no codes).
- `question` (natural-language text, no codes) and optional `background`.
- `forecast_point` (YYYY-MM-DD, when a model is asked to predict) and `resolution_date` (YYYY-MM-DD).
- `ground_truth` (one of `hypothesis_set`) and `ground_truth_idx` (int).
- `prior_state` (str): the per-benchmark oracle baseline. GDELT-CAMEO is the modal Peace/Tension/Violence intensity over the prior 30 days; Earnings is the mode of Beat/Meet/Miss across the prior 4 quarters; ForecastBench is the crowd majority class on `freeze_datetime_value`.
- `fd_type` (str): partition tag, one of `stability`, `change`, `unknown`. Headline metrics are reported on the `change` subset (cases where the realized class differs from `prior_state`); `stability` and `unknown` are tracked separately.
- `default_horizon_days` (int): per-FD default forecast horizon (14 for ForecastBench and GDELT-CAMEO, source-specific for Earnings). Runtime experiments override this via `apply_experiment_horizon()` in the baselines runner.
- `x_multiclass_ground_truth` (str): the original domain class label preserved for secondary analysis (`Yes`/`No` for ForecastBench, `Peace`/`Tension`/`Violence` for GDELT-CAMEO, `Beat`/`Meet`/`Miss` for Earnings).
- `x_multiclass_hypothesis_set` (list[str]): the original domain hypothesis set, e.g. `["Peace", "Tension", "Violence"]`.
- `crowd_probability` (float or null), `lookback_days` (int).
- `article_ids` (list of `art_*` IDs resolving to rows in the same folder's `articles.jsonl`).

Pipeline-internal fields (`metadata.*`, `_relevance_*`, `_earnings_meta`, `_drop_reasons`) are stripped by the quality filter and **do not** appear in the shipped `forecasts.jsonl`. Full per-field reference: [schema/forecast_dossier.md](schema/forecast_dossier.md) (per-benchmark surface) and [`docs/FORECAST_DOSSIER.md`](../docs/FORECAST_DOSSIER.md) (canonical spec).

**Article.** One JSON object per line in `data/{cutoff}/articles.jsonl`, which contains only the subset of the unified article pool referenced by at least one accepted FD. Each article has:

- `id` (`art_` + `sha1(url)[:12]`) and `url`.
- `title`, `text` (trafilatura-extracted, may be empty if fetch failed), `title_text` (`title + "\n" + text`, embed-input convenience).
- `publish_date` (YYYY-MM-DD), `source_domain` (lowercased netloc without `www.`).
- `char_count` (int).
- `source_type` (enum): one of `news`, `filing`, `social`, `blog`, `other`. Used by retrieval logic (e.g. SEC EDGAR 8-K rows carry `filing` and are subject to an asymmetric pre-event filter in the Earnings track).
- `language` (str): ISO 639-1 code, default `en`.
- `provenance` (str): pipe-separated multi-source tag string, e.g. `forecastbench` or `gdelt_cameo|earnings`, recording which pipeline(s) surfaced the row.

Full per-field reference: [schema/article.md](schema/article.md) and the machine-readable [`docs/article.schema.json`](../docs/article.schema.json). Because the unifier deduplicates by `sha1(url)[:12]`, an article surfaced by multiple tracks appears as one row with a multi-source `provenance` string.

## 2. Files produced by a build

Everything at `data/{cutoff}/` is produced by a single `scripts/build_benchmark.py` run. Primary deliverables live at the root; audit artifacts live in `meta/`. (Note: `benchmark/build.py` is now a thin deprecation shim that forwards to `scripts/build_benchmark.py`; new code and CI should target the script directly.)

| File | Role | Produced by | Purpose |
|---|---|---|---|
| `forecasts.jsonl` | **Primary #1** | `quality_filter.py` + `annotate_prior_state.py` + orchestrator | The accepted FD set. One JSON object per line. Shipped fields only (no pipeline internals); binary target promoted, multiclass preserved as `x_multiclass_*`. |
| `articles.jsonl` | **Primary #2** | orchestrator `step_publish` | Subset of the unified article pool referenced by at least one accepted FD. |
| `benchmark.yaml` | Config snapshot | orchestrator `write_run_manifest` | Effective config used for this exact build. Drop-in to reproduce: `python scripts/build_benchmark.py --config data/{cutoff}/benchmark.yaml`. |
| `build_manifest.json` | Run metadata | orchestrator `write_run_manifest` | Timestamp, git sha, resolved `model_cutoff`, `cutoff_buffer_days`, benchmarks included, Python executable. |
| `checksums.sha256` | Integrity | orchestrator | SHA-256 of every shipped artifact for downstream verification. |
| `meta/eda_report.html` | Audit | `build_eda_report.py` | Self-contained inline-SVG EDA (coverage waterfall, per-source counts, histograms, class balance, worked FD examples). Not primary benchmark data. |
| `meta/diagnostic_report.md` | Audit | `diagnostic_report.py` | Human-readable QA: per-source table, class balance, spot-check, leakage audit. |
| `meta/diagnostic_report.json` | Audit | `diagnostic_report.py` | Machine-readable twin of the above; feeds the EDA report. |
| `meta/quality_meta.json` | Audit | `quality_filter.py` | Accepted and dropped totals, per-source kept and dropped, top drop-reason patterns, thresholds, leakage-articles-pruned count. |
| `meta/relevance_meta.json` | Audit | `compute_relevance.py` | Before-and-after coverage and per-source `avg_k` after SBERT cross-match. |

Not shipped in `data/{cutoff}/`:

- The unified pre-filter pool under `data/unified/` (articles + forecasts + embeddings). This staging directory is removed on successful completion.
- Per-track raw downloads (`data/gdelt_cameo/...`, `data/earnings/...`, ForecastBench clone). These are regenerable from the sub-scripts.
- The ETD layer at `data/etd/facts.v1.jsonl` (Stage 1 atomic facts), with optional canonical and linked outputs at `facts.v1_canonical.jsonl` and `facts.v1_linked.jsonl`. See [`docs/ETD_SPEC.md`](../docs/ETD_SPEC.md) and [`docs/etd.schema.json`](../docs/etd.schema.json).

## 3. Headline statistics

For the latest build's audit, open `data/{cutoff}/meta/diagnostic_report.md`: the per-source table, class balance (both binary `Comply`/`Surprise` and the preserved multiclass), `fd_type` partition counts, articles-per-FD histogram, day-spread, evidence-character totals, and crowd-probability distribution are all computed there and mirrored as JSON in `diagnostic_report.json`.

Headline metrics in the paper are reported on the `change` subset of each benchmark; the `stability` and `unknown` partitions are reported separately.

Cross-reference during analysis:

- **Per-source accepted and dropped, plus top drop patterns**: `meta/quality_meta.json`.
- **SBERT cross-match coverage and per-source `avg_k`**: `meta/relevance_meta.json`.
- **Leakage audit** (`leakage_violations` must be `0`; `leakage_articles_pruned` reports how many were caught by the guard): both `quality_meta.json` and `diagnostic_report.json`.
- **Visualizations, worked examples, and a self-describing reading guide**: `meta/eda_report.html`.

## 4. Per-benchmark notes

- **GDELT-CAMEO.** 3-class Peace/Tension/Violence ordinal (CAMEO root codes 01-09 / 10-17 / 18-20), with `min_daily_mentions: 20`. Evidence retrieval is editorial-news-only (NYT, The Guardian, Google News); MIRAI is cited as an external anchor only and is not used for retrieval. Prior-state oracle: modal intensity over the prior 30 days.
- **ForecastBench.** Prediction-market-only sources: Polymarket, Metaculus, Manifold, Infer. ACLED and FRED are excluded (algorithmic pseudo-probabilities are not crowd forecasts). The v2.1 default disables the legacy geopolitics-only subject filter, yielding roughly 1172 FDs versus the older roughly 530. Prior-state oracle: crowd majority class on `freeze_datetime_value`.
- **Earnings.** S&P 500 universe with a 5-source evidence cascade: SEC EDGAR 8-K (`source_type: "filing"`, asymmetric pre-event filter), Finnhub, yfinance, Google News, GDELT DOC. Prior-state oracle: mode of Beat/Meet/Miss across the prior 4 quarters.

## 5. Leakage guarantees

Four layered mechanisms enforce leakage-free evaluation; all are audited in `meta/`:

1. **Retrieval leakage (article-level).** `scripts/common/quality_filter.py` ensures every article in `article_ids` satisfies `publish_date < forecast_point`. Violators are pruned before other quality checks apply. `source_type: "filing"` evidence (e.g. SEC EDGAR 8-K) carries an asymmetric pre-event window enforced by the Earnings retrieval stage.
2. **Training leakage (model-level).** Every accepted FD satisfies `resolution_date > model_cutoff + cutoff_buffer_days`, dropping FDs whose ground truth lands inside the evaluator model's plausible training window.
3. **Two-cutoff leakage probe.** Builds run at the primary `2026-01-01` cutoff and a leakage-probe `2024-04-01` cutoff (config: `configs/leakage_probe_config.yaml`, 90-day buffer). Comparing scores across cutoffs surfaces residual contamination.
4. **GDELT-CAMEO oracle relink.** `scripts/gdelt_cameo/relink_context.py` replaces any oracle (event-source) articles with leakage-free actor-pair editorial-news context articles drawn from the pre-event window.

The final leakage audit in `diagnostic_report.json` reports `leakage_violations: 0`.

## 6. Article-pool and coverage stats

Per-build article-pool counts (input counts before dedup, post-dedup totals, full-text extraction rate, domain mix, per-`source_type` breakdown) are in `meta/diagnostic_report.json` under the article-pool section and visualized in `meta/eda_report.html` under the "Article Pool" heading.

The coverage-recovery pipeline (per-source baseline retrieval, then SBERT cross-match lift, then LLM keyword-rewrite retry, then quality filter) is documented with stage-by-stage FDs-with-articles counts in the EDA report. Consult `meta/relevance_meta.json` for the post-SBERT view and `meta/quality_meta.json` for the post-filter view.

## 7. Reading the HTML EDA report

Open `data/{cutoff}/meta/eda_report.html` in a browser. The report is self-contained (inline SVG plus CSS, no network calls) and contains:

- Pipeline coverage waterfall and per-source FD counts.
- Article-per-FD histogram, day-spread histogram, evidence-character histogram.
- Crowd-probability distribution (overall and per-source).
- Publish-date density, top source domains, ground-truth class balance (binary and multiclass), `fd_type` partition counts, accepted-vs-dropped per source, drop-reason breakdown.
- A handful of worked FD examples rendered with question, background excerpt, ground truth, prior state, resolution date, and the first few linked articles.
- A preamble keyed to the underlying `*_meta.json` files so every number in the HTML is traceable.

## 8. Audit tooling

The `scripts/` tree includes dedicated auditors that operate on the shipped artifacts and on the ETD layer:

- `scripts/articles_audit.py`: per-domain, per-`source_type`, and per-`provenance` breakdowns of `articles.jsonl`.
- `scripts/fd_audit.py` and `scripts/fd_filter.py`: FD-level audit and filtering by `benchmark`, `fd_type`, `prior_state`, or arbitrary predicates.
- `scripts/etd_audit.py`, `scripts/etd_verify.py`, `scripts/etd_filter.py`: audit, schema-verify, and filter the ETD facts file.
- `scripts/etd_debug_errors.py`, `scripts/etd_debug_empty.py`: triage parser failures and empty-fact records.
- `scripts/etd_compare_facts_vs_articles.py`: cross-reference ETD coverage against the published article pool.
