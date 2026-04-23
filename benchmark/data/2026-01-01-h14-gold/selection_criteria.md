# EMR-ACH Gold Subset, selection criteria

This document records the exact filter cascade applied to produce the gold
subset from the parent v2.1 publish at `benchmark/data/2026-01-01-h14/`.
Every threshold is recorded here AND in `build_manifest.json` for
reproducibility.

## Cascade (applied in order; first failure drops the FD)

1. **Article count**: `len(article_ids) >= 3`
2. **Distinct article dates**: `>= 2` (no single-news-cycle FDs)
3. **Forecast horizon**: `resolution_date - forecast_point >= 14 days`
4. **Strict horizon enforcement**: at most 20% of an FD's articles may be
   dated after `forecast_point - 14 days` (no nowcasting)
5. **fd_type**: in `{stability, change}`, plus `unknown`
6. **Avg article text length**: `>= 300 chars` (no title-only stubs)
7. **Source diversity**: `>= 1` distinct hostnames per FD
8. **Per-benchmark predictability**:
   - `forecastbench`: drop FDs whose question or background matches a
     denylist of pure sports / lottery / pure-entertainment topics
     (8 regex patterns; see `scripts/build_gold_subset.py`).
   - `gdelt-cameo`: at least one of the two actors must be in
     `GEOPOLITICALLY_SIGNIFICANT_ACTORS` (G20 + NATO + ME principals +
     East Asia powers; ~70 country codes).
   - `earnings`: ticker must be in S&P 100.
9. **Stratified sampling** to per-(benchmark, fd_type) quotas; default
   targets: `{"forecastbench::stability": 60, "forecastbench::change": 40, "gdelt-cameo::stability": 100, "gdelt-cameo::change": 100, "earnings::stability": 60, "earnings::change": 60}`. The actual
   sample sizes per stratum are in `meta/distribution.md`.

## Why these thresholds

- **8 articles, 5 distinct days** filter out FDs with shallow evidence
  pools where forecasting reduces to luck.
- **14-day horizon** matches the v2.1 design (`docs/FORECAST_DOSSIER.md`)
  and rules out nowcasting confounds.
- **1500 chars / 3 distinct domains** filter out title-only stubs and
  echo-chamber FDs where multiple recordings are the same wire story.
- **S&P 100 only for earnings**: large caps have strong analyst coverage
  and well-defined consensus EPS expectations; mid-cap earnings have
  much sparser news coverage and noisier EPS estimates.

## Reproducibility

Re-run with the recorded thresholds in `build_manifest.json` to regenerate
this exact subset bytes-for-bytes. The selection script is at
`scripts/build_gold_subset.py` in the parent repository.
