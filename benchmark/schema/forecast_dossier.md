# Forecast Dossier (FD) Schema

A **Forecast Dossier (FD)** is one row in `data/{cutoff}/forecasts.jsonl`. Each FD pairs a single resolved forecast question with its pre-event evidence article IDs.

Fields are emitted by `scripts/common/unify_forecasts.py`, populated / pruned by `scripts/common/compute_relevance.py` (and by `scripts/gdelt_cameo/relink_context.py` for the GDELT-CAMEO track), and filtered by `scripts/common/quality_filter.py`. The quality filter strips pipeline-internal fields before writing; the schema below lists exactly what ships.

## Shipped fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable ID. Prefix encodes source: `fb_*` (ForecastBench), `gdc_*` or similar (GDELT-CAMEO), `earn_*` (earnings). |
| `benchmark` | string | One of `forecastbench`, `gdelt_cameo`, `earnings`. |
| `source` | string | Sub-source: `polymarket`, `metaculus`, `manifold`, `infer`, `gdelt-kg`, or `yfinance`. |
| `hypothesis_set` | list[string] | Ordered natural-language class labels (no raw codes). Examples: `["Yes","No"]` for binary forecastbench markets; `["Verbal cooperation","Material cooperation","Verbal conflict","Material conflict"]` for GDELT-CAMEO; `["Beat","Meet","Miss"]` for earnings. |
| `hypothesis_definitions` | dict[string, string] | Mapping from each label in `hypothesis_set` to a one-sentence natural-language definition. Enables self-contained evaluation without external codebooks. |
| `question` | string | Natural-language question text. Never contains raw codes. |
| `background` | string | Long-form resolution criteria, market description, or actor context. May be empty. |
| `forecast_point` | string (YYYY-MM-DD) | Date at which a model is asked to forecast. Articles with `publish_date >= forecast_point` are retrieval leakage. |
| `resolution_date` | string (YYYY-MM-DD) | Date of ground-truth resolution. May equal `forecast_point` for snapshot markets. |
| `ground_truth` | string | Realized label. Always a member of `hypothesis_set`. |
| `ground_truth_idx` | int | Index of `ground_truth` within `hypothesis_set`. |
| `crowd_probability` | float (0-1) or null | Crowd- or market-implied probability (of class 0 for binary; `null` for GDELT-CAMEO and earnings tracks). |
| `lookback_days` | int | Retrieval window before `forecast_point`. Set per-source in `configs/default_config.yaml` (30d for ForecastBench sources, 90d for `gdelt-kg`, 14d for `yfinance`). |
| `article_ids` | list[string] | `art_*` IDs resolvable in the same folder's `articles.jsonl`. |

## Fields that do NOT ship

The quality filter whitelists only the fields above when writing `forecasts.jsonl`. The following pipeline-internal fields exist in `data/unified/forecasts.jsonl` during the build but are stripped before publication and must not be expected in the shipped dataset:

- `metadata.*` (source-specific raw payloads)
- `_relevance_original_count`, `_relevance_added_count` (SBERT diagnostics)
- `_earnings_meta` (yfinance surprise metadata)
- `_drop_reasons` (only present on dropped FDs in `forecasts_dropped.jsonl` under `data/unified/`)

## Example

```json
{
  "id": "fb_KtGRsTouYWiD",
  "benchmark": "forecastbench",
  "source": "manifold",
  "hypothesis_set": ["Yes", "No"],
  "hypothesis_definitions": {
    "Yes": "The event described in the question occurred by the resolution date.",
    "No": "The event did not occur by the resolution date."
  },
  "question": "Will an artificial intelligence system play a major artistic role in the 2024 Olympic opening ceremonies?",
  "background": "Olympic ceremonies have often featured technologies ... Otherwise, the market resolves to NO.",
  "forecast_point": "2024-07-27",
  "resolution_date": "2024-07-27",
  "ground_truth": "No",
  "ground_truth_idx": 1,
  "crowd_probability": 0.133,
  "lookback_days": 30,
  "article_ids": ["art_1e7cb2b2f8c8", "art_84abbb67c0b4", "art_b84361ef487e"]
}
```

## Leakage guarantees

1. **Retrieval leakage (article-level).** Every article referenced by `article_ids` satisfies `article.publish_date < forecast_point`. The quality filter prunes any violating article before applying the other accept-rule checks, and the leakage-pruned count is logged in `meta/quality_meta.json`.
2. **Training leakage (model-level).** When `--model-cutoff` and `--cutoff-buffer-days` are forwarded by `build.py` (always, by default), the quality filter drops any FD whose `resolution_date <= model_cutoff + cutoff_buffer_days`. The guard uses `resolution_date` (not `forecast_point`) so the ground truth is guaranteed to land strictly after the evaluator model's plausible knowledge horizon.
