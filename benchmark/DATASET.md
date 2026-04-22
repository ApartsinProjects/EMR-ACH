# EMR-ACH Benchmark Dataset

Schema, data-file inventory, and guidance for reading the audit artifacts produced by `build.py`.

## 1. Schema overview

The benchmark ships in two linked schemas. All fields are produced by `scripts/common/unify_forecasts.py`, pruned by `scripts/common/compute_relevance.py`, and filtered by `scripts/common/quality_filter.py`.

**Forecast Dossier (FD).** One JSON object per line in `data/{cutoff}/forecasts.jsonl`, representing a single resolved forecast question plus its pre-event evidence. Every FD carries:

- `id` (str) and a `benchmark` tag: `forecastbench`, `gdelt_cameo`, or `earnings`.
- `source` sub-tag: `polymarket`, `metaculus`, `manifold`, `infer`, `gdelt-kg`, or `yfinance`.
- `hypothesis_set` (natural-language labels, no codes) and `hypothesis_definitions` (label to one-sentence definition).
- `question` (natural-language text, no codes) and optional `background`.
- `forecast_point` (YYYY-MM-DD, when a model is asked to predict) and `resolution_date` (YYYY-MM-DD).
- `ground_truth` (one of `hypothesis_set`) and `ground_truth_idx` (int).
- `crowd_probability` (float or null), `lookback_days` (int).
- `article_ids` (list of `art_*` IDs resolving to rows in the same folder's `articles.jsonl`).

Pipeline-internal fields (`metadata.*`, `_relevance_*`, `_earnings_meta`, `_drop_reasons`) are stripped by the quality filter and **do not** appear in the shipped `forecasts.jsonl`. Full per-field reference: [schema/forecast_dossier.md](schema/forecast_dossier.md).

**Article.** One JSON object per line in `data/{cutoff}/articles.jsonl`, which contains only the subset of the unified article pool referenced by at least one accepted FD. Each article has:

- `id` (`art_` + `sha1(url)[:12]`) and `url`.
- `title`, `text` (trafilatura-extracted, may be empty if fetch failed), `title_text` (`title + "\n" + text`, embed-input convenience).
- `publish_date` (YYYY-MM-DD), `source_domain` (lowercased netloc without `www.`).
- `char_count` (int) and `provenance` (list of tags, e.g. `["forecastbench"]` or `["gdelt_cameo"]`).

Full per-field reference: [schema/article.md](schema/article.md). Because the unifier deduplicates by `sha1(url)[:12]`, an article surfaced by multiple tracks appears as one row with a multi-tag `provenance`.

## 2. Files produced by a build

Everything at `data/{cutoff}/` is produced by a single `build.py` run. Primary deliverables live at the root; audit artifacts live in `meta/`.

| File | Role | Produced by | Purpose |
|---|---|---|---|
| `forecasts.jsonl` | **Primary #1** | `quality_filter.py` + orchestrator | The accepted FD set. One JSON object per line. Shipped fields only (no pipeline internals). |
| `articles.jsonl` | **Primary #2** | orchestrator `step_publish` | Subset of the unified article pool referenced by at least one accepted FD. |
| `benchmark.yaml` | Config snapshot | orchestrator `write_run_manifest` | Effective config used for this exact build. Drop-in to reproduce: `python build.py --config data/{cutoff}/benchmark.yaml`. |
| `build_manifest.json` | Run metadata | orchestrator `write_run_manifest` | Timestamp, git sha, resolved `model_cutoff`, `cutoff_buffer_days`, benchmarks included, Python executable. |
| `meta/eda_report.html` | Audit | `build_eda_report.py` | Self-contained inline-SVG EDA (coverage waterfall, per-source counts, histograms, class balance, worked FD examples). Not primary benchmark data. |
| `meta/diagnostic_report.md` | Audit | `diagnostic_report.py` | Human-readable QA: per-source table, class balance, spot-check, leakage audit. |
| `meta/diagnostic_report.json` | Audit | `diagnostic_report.py` | Machine-readable twin of the above; feeds the EDA report. |
| `meta/quality_meta.json` | Audit | `quality_filter.py` | Accepted / dropped totals, per-source kept and dropped, top drop-reason patterns, thresholds, leakage-articles-pruned count. |
| `meta/relevance_meta.json` | Audit | `compute_relevance.py` | Before-and-after coverage and per-source `avg_k` after SBERT cross-match. |

Not shipped in `data/{cutoff}/`:

- The unified pre-filter pool under `data/unified/` (articles + forecasts + embeddings). This staging directory is removed on successful completion.
- Per-track raw downloads (`data/gdelt_cameo/...`, `data/earnings/...`, ForecastBench clone). These are regenerable from the sub-scripts.

## 3. Headline statistics

A from-scratch rebuild is currently in progress; the last set of committed numbers is stale. For the current build's audit, open `data/{cutoff}/meta/diagnostic_report.md` — the per-source table, class balance, articles-per-FD histogram, day-spread, evidence-character totals, and crowd-probability distribution are all computed there and mirrored as JSON in `diagnostic_report.json`.

Cross-reference during analysis:

- **Per-source accepted / dropped and top drop patterns**: `meta/quality_meta.json`.
- **SBERT cross-match coverage and per-source `avg_k`**: `meta/relevance_meta.json`.
- **Leakage audit** (`leakage_violations` must be `0`; `leakage_articles_pruned` reports how many were caught by the guard): both `quality_meta.json` and `diagnostic_report.json`.
- **Visualizations, worked examples, and a self-describing reading guide**: `meta/eda_report.html`.

## 4. Leakage guarantees

Two tiers, both enforced by `scripts/common/quality_filter.py` and audited downstream:

1. **Retrieval leakage (article-level)**: every article referenced in `article_ids` satisfies `publish_date < forecast_point`. Articles failing this are pruned before the other quality checks apply.
2. **Training leakage (model-level)**: every accepted FD satisfies `resolution_date > model_cutoff + cutoff_buffer_days`. This drops FDs whose ground truth falls inside the evaluator model's plausible training window.

The final leakage audit in `diagnostic_report.json` reports `leakage_violations: 0`.

## 5. Article-pool and coverage stats

Per-build article-pool counts (input counts before dedup, post-dedup totals, full-text extraction rate, domain mix) are in `meta/diagnostic_report.json` under the article-pool section and visualized in `meta/eda_report.html` under the "Article Pool" heading.

The coverage-recovery pipeline (GDELT DOC API baseline -> SBERT cross-match lift -> LLM keyword-rewrite retry -> quality filter) is documented with stage-by-stage FDs-with-articles counts in the EDA report. Consult `meta/relevance_meta.json` for the post-SBERT view and `meta/quality_meta.json` for the post-filter view.

## 6. Reading the HTML EDA report

Open `data/{cutoff}/meta/eda_report.html` in a browser. The report is self-contained (inline SVG + CSS, no network calls) and contains:

- Pipeline coverage waterfall and per-source FD counts.
- Article-per-FD histogram, day-spread histogram, evidence-character histogram.
- Crowd-probability distribution (overall and per-source).
- Publish-date density, top source domains, ground-truth class balance, accepted-vs-dropped per source, drop-reason breakdown.
- A handful of worked FD examples rendered with question, background excerpt, ground truth, resolution date, and the first few linked articles.
- A preamble keyed to the underlying `*_meta.json` files so every number in the HTML is traceable.
