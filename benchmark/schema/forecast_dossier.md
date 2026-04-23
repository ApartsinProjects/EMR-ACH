# Forecast Dossier (FD) Schema

A **Forecast Dossier (FD)** is one row in `data/{cutoff}/forecasts.jsonl`. Each FD pairs a single resolved forecast question with its pre-event evidence article IDs and a per-benchmark prior-state oracle.

This page is the per-benchmark facing schema reference. The canonical, top-level spec is [`docs/FORECAST_DOSSIER.md`](../../docs/FORECAST_DOSSIER.md); when in doubt, defer to that document.

Fields are emitted by `scripts/common/unify_forecasts.py`, populated and pruned by `scripts/common/compute_relevance.py` (and by `scripts/gdelt_cameo/relink_context.py` for the GDELT-CAMEO track), filtered by `scripts/common/quality_filter.py`, and finally annotated by `scripts/annotate_prior_state.py`, which is the step that promotes the binary `Comply` vs `Surprise` target and tags `fd_type`. The quality filter and annotator together strip pipeline-internal fields before writing; the schema below lists exactly what ships.

## Shipped fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable ID. Prefix encodes source: `fb_*` (ForecastBench), `gdc_*` or similar (GDELT-CAMEO), `earn_*` (earnings). |
| `benchmark` | string | One of `forecastbench`, `gdelt_cameo`, `earnings`. |
| `source` | string | Sub-source: `polymarket`, `metaculus`, `manifold`, `infer`, `gdelt-kg`, or `yfinance`. |
| `hypothesis_set` | list[string] | **Primary target, always binary**: `["Comply", "Surprise"]`. `Comply` means the realized class equals `prior_state`; `Surprise` means it differs. |
| `hypothesis_definitions` | dict[string, string] | Mapping from each label in `hypothesis_set` to a one-sentence natural-language definition. Enables self-contained evaluation without external codebooks. |
| `question` | string | Natural-language question text. Never contains raw codes. |
| `background` | string | Long-form resolution criteria, market description, or actor context. May be empty. |
| `forecast_point` | string (YYYY-MM-DD) | Date at which a model is asked to forecast. Articles with `publish_date >= forecast_point` are retrieval leakage. |
| `resolution_date` | string (YYYY-MM-DD) | Date of ground-truth resolution. May equal `forecast_point` for snapshot markets. |
| `ground_truth` | string | Realized binary label (`Comply` or `Surprise`), always a member of `hypothesis_set`. |
| `ground_truth_idx` | int | Index of `ground_truth` within `hypothesis_set` (0 for `Comply`, 1 for `Surprise`). |
| `prior_state` | string | Per-benchmark oracle baseline computed by `scripts/annotate_prior_state.py`. GDELT-CAMEO: modal Peace/Tension/Violence intensity over the prior 30 days. Earnings: mode of Beat/Meet/Miss across the prior 4 quarters. ForecastBench: crowd majority class on `freeze_datetime_value`. Drawn from `x_multiclass_hypothesis_set`. |
| `fd_type` | string | Partition tag: `stability` (realized class equals `prior_state`), `change` (realized class differs from `prior_state`), or `unknown` (oracle could not be computed; e.g. insufficient prior history). Headline metrics in the paper are reported on the `change` subset. |
| `default_horizon_days` | int | Per-FD default forecast horizon. 14 for ForecastBench and GDELT-CAMEO; source-specific for Earnings. The baselines runner overrides at experiment time via `apply_experiment_horizon()`; rebuild-time overrides are available through `EMRACH_FB_HORIZON_DAYS`, `EMRACH_GDELT_HORIZON_DAYS`, `EMRACH_EARNINGS_HORIZON_DAYS`. |
| `x_multiclass_ground_truth` | string | Original domain class label preserved for secondary analysis: `Yes`/`No` (ForecastBench), `Peace`/`Tension`/`Violence` (GDELT-CAMEO; CAMEO root codes 01-09 / 10-17 / 18-20), `Beat`/`Meet`/`Miss` (Earnings). |
| `x_multiclass_hypothesis_set` | list[string] | Original domain hypothesis set, e.g. `["Yes","No"]`, `["Peace","Tension","Violence"]`, `["Beat","Meet","Miss"]`. Source of `prior_state` and `x_multiclass_ground_truth`. |
| `crowd_probability` | float (0-1) or null | Crowd- or market-implied probability (of class 0 in the multiclass set for binary FB markets; `null` for GDELT-CAMEO and earnings tracks). |
| `lookback_days` | int | Retrieval window before `forecast_point`. Set per-source in `configs/default_config.yaml` (30d for ForecastBench sources, 90d for `gdelt-kg`, 14d for `yfinance`). |
| `article_ids` | list[string] | `art_*` IDs resolvable in the same folder's `articles.jsonl`. |

## Fields that do NOT ship

The quality filter and annotator together whitelist only the fields above when writing `forecasts.jsonl`. The following pipeline-internal fields exist in `data/unified/forecasts.jsonl` during the build but are stripped before publication and must not be expected in the shipped dataset:

- `metadata.*` (source-specific raw payloads)
- `_relevance_original_count`, `_relevance_added_count` (SBERT diagnostics)
- `_earnings_meta` (yfinance plus SEC EDGAR plus Finnhub surprise metadata)
- `_drop_reasons` (only present on dropped FDs in `forecasts_dropped.jsonl` under `data/unified/`)

## Example

```json
{
  "id": "fb_KtGRsTouYWiD",
  "benchmark": "forecastbench",
  "source": "manifold",
  "hypothesis_set": ["Comply", "Surprise"],
  "hypothesis_definitions": {
    "Comply": "The realized outcome matches the prior-state oracle for this question.",
    "Surprise": "The realized outcome differs from the prior-state oracle for this question."
  },
  "question": "Will an artificial intelligence system play a major artistic role in the 2024 Olympic opening ceremonies?",
  "background": "Olympic ceremonies have often featured technologies ... Otherwise, the market resolves to NO.",
  "forecast_point": "2024-07-27",
  "resolution_date": "2024-07-27",
  "ground_truth": "Comply",
  "ground_truth_idx": 0,
  "prior_state": "No",
  "fd_type": "stability",
  "default_horizon_days": 14,
  "x_multiclass_ground_truth": "No",
  "x_multiclass_hypothesis_set": ["Yes", "No"],
  "crowd_probability": 0.133,
  "lookback_days": 30,
  "article_ids": ["art_1e7cb2b2f8c8", "art_84abbb67c0b4", "art_b84361ef487e"]
}
```

## Leakage guarantees

1. **Retrieval leakage (article-level).** Every article referenced by `article_ids` satisfies `article.publish_date < forecast_point`. The quality filter prunes any violating article before applying the other accept-rule checks; the leakage-pruned count is logged in `meta/quality_meta.json`. Articles with `source_type: "filing"` are subject to an asymmetric pre-event filter in the Earnings track.
2. **Training leakage (model-level).** When `--model-cutoff` and `--cutoff-buffer-days` are forwarded by `scripts/build_benchmark.py` (always, by default), the quality filter drops any FD whose `resolution_date <= model_cutoff + cutoff_buffer_days`. The guard uses `resolution_date` (not `forecast_point`) so the ground truth lands strictly after the evaluator model's plausible knowledge horizon.
3. **Two-cutoff probe.** Builds run at the primary `2026-01-01` and leakage-probe `2024-04-01` cutoffs; comparing scores across cutoffs surfaces residual contamination. See `configs/leakage_probe_config.yaml`.

## Schema invariants enforced in CI

`tests/test_fd_schema_invariants.py` (14 tests) enforces, among other invariants: presence and type of `prior_state`, `fd_type`, `x_multiclass_ground_truth`, `x_multiclass_hypothesis_set`, and `default_horizon_days`; that `hypothesis_set == ["Comply", "Surprise"]`; that `ground_truth in hypothesis_set` and `ground_truth_idx` indexes correctly; and that `fd_type` is one of `stability`, `change`, `unknown`. See also `tests/test_pipeline_invariants.py` (16 tests) and `tests/test_article_schema.py` (8 tests) for cross-cutting and article-side invariants.
