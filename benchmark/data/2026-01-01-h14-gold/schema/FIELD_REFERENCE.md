# Per-field reference, EMR-ACH Gold Subset

## Forecast Dossier (`forecasts.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable per-FD identifier. |
| `benchmark` | enum | One of `forecastbench`, `gdelt-cameo`, `earnings`. |
| `source` | string | Origin source within the benchmark. |
| `hypothesis_set` | string[] | Primary hypotheses; v2.1-gold uses `["Comply", "Surprise"]`. |
| `hypothesis_definitions` | object | Plain-English definition per hypothesis. Inline so the FD is self-contained. |
| `question` | string | The forecasting question. |
| `background` | string | Context / resolution criteria; may be empty. |
| `forecast_point` | date | Cutoff for evidence; predictions are made at this date. |
| `resolution_date` | date | When the outcome is known. |
| `ground_truth` | string | The realized hypothesis (one of `hypothesis_set`). |
| `ground_truth_idx` | int | Index into `hypothesis_set`. |
| `article_ids` | string[] | Article IDs that constitute the evidence pool; resolve via `articles.jsonl`. |
| `fd_type` | enum | One of `stability` (status-quo holds; ground_truth=Comply), `change` (status-quo breaks; ground_truth=Surprise), `unknown` (insufficient prior history). |
| `prior_state_30d` | string | Domain-specific prior expectation: modal CAMEO intensity (GDELT) / mode of prior 4 quarters Beat-Meet-Miss (earnings) / crowd majority Yes-No (FB). |
| `prior_state_stability` | float | Confidence in the prior expectation, 0-1. |
| `prior_state_n_events` | int | History sample size used to compute the prior. |
| `x_multiclass_hypothesis_set` | string[] | Original domain-multiclass labels (3-class CAMEO, Beat/Meet/Miss, Yes/No), preserved for ablation. |
| `x_multiclass_ground_truth` | string | Original multiclass label that resolved. |
| `default_horizon_days` | int | Recommended experiment horizon h; the published bundle was built so every article is dated <= forecast_point - h. |
| `lookback_days` | int | The retrieval lookback window used at build time. |
| `crowd_probability` | number | ForecastBench markets only: prediction-market freeze-time probability. |
| `metadata` | object | Per-benchmark metadata (e.g. CAMEO actor codes, event_base_code). |
| `_earnings_meta` | object | Earnings only: ticker, sector, industry, eps_estimate, eps_actual, surprise_pct, report_date. |

## Article (`articles.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable URL-hash identifier; format `art_<12-hex>`. |
| `url` | string | Canonical URL. |
| `title` | string | Article title. |
| `text` | string | Trafilatura-extracted body. May be empty for title-only records. |
| `publish_date` | date | ISO publication date (YYYY-MM-DD). |
| `source_domain` | string | Hostname (lowercase, `www.` stripped). |
| `language` | string | BCP-47 language tag (default: `en`). |
| `provenance` | string|string[] | Pipe-separated tags from each fetcher that surfaced this URL. |

## Comply vs Surprise contract

Every FD's primary target is binary:
- `Comply` if the resolved outcome matches the prior-state expectation.
- `Surprise` if it breaks the expectation.

Headline metrics in this dataset should be reported on the **change** subset (where the prior is wrong and the model must read evidence to win).

## ETD Atomic Fact (`facts.jsonl`)

ETD = Event Timeline Dossier. Each fact is one atomic dated event extracted by an LLM (gpt-4o-mini, v3 prompt with anchor table + verbatim-quote requirements) from one or more articles. Facts in this file are filtered to those whose `primary_article_id` is present in `articles.jsonl`, so the gold subset stays self-contained.

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable per-fact identifier; format `f_<12-hex>`. |
| `time` | string | The event's date: ISO date (YYYY-MM-DD), YYYY-MM, YYYY, or `unknown`. Bounded above by the source article's `publish_date`. |
| `time_precision` | enum | `day` / `week` / `month` / `quarter` / `year` / `unknown`. |
| `time_type` | enum | `point` / `interval` / `ongoing` / `periodic`. |
| `fact` | string | One-sentence atomic claim. |
| `evidence_quote` | string | Verbatim substring of the article body that grounds the fact (v3 prompt requirement). |
| `entities` | object[] | Named actors involved: `[{name, type}, ...]`. |
| `metrics` | object[] | Numeric quantities: `[{name, value, unit}, ...]`. |
| `kind` | string | Short tag, e.g. `military-deployment`, `earnings-release`, `policy-statement`. |
| `polarity` | enum | `asserted` (default) / `negated` / `hypothetical` / `reported` (cited; requires `attribution`). |
| `attribution` | string | Named source if `polarity == reported`. |
| `extraction_confidence` | enum | `high` / `medium` / `low`. Production filter typically requires `high`. |
| `primary_article_id` | string | Article ID; resolves to a record in `articles.jsonl`. |
| `article_date` | string | The source article's publish_date (denormalized for convenience). |
| `linked_fd_ids` | string[] | If Stage-3 link ran, the FD IDs whose `article_ids` include this fact's `primary_article_id`. May be empty if Stage 1 or Stage 2 only. |
| `canonical_id` | string | If Stage-2 dedup ran, points at the canonical fact for a near-duplicate cluster (null if this fact IS the canonical or no dedup ran). |
| `extractor` | string | Model identifier, e.g. `gpt-4o-mini-2024-07-18`. |
| `extracted_at` | string | ISO timestamp of the extraction. |

### How to use ETD facts as evidence

The default baseline runner uses `articles.jsonl` text as evidence. ETD facts are an **additive** channel, useful for:
- Building a "facts-only" baseline (B10b in the v2.2 design): pass the per-FD fact subset as the evidence block.
- Building a "hybrid" baseline (B10): include both article snippets AND structured facts in the prompt.
- Time-travel queries: filter `facts.jsonl` by `time` to reconstruct what was known at any past date.

See `examples/load.py:fd_facts()` for a one-call helper that returns the facts linked to a given FD via its `article_ids`.
