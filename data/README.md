# `data/` — Pipeline Intermediate Storage

**This directory is NOT the shipped benchmark.** Everything here is scratch
space: raw downloads, intermediate transformations, cached embeddings, and
staged snapshots. Mutable, regenerated per build, mostly gitignored.

**The final, clean, shipped benchmark lives under `benchmark/data/{cutoff}/`.**

## Layout

| Dir / file | What it is | Mutable? |
|---|---|---|
| `forecastbench_geopolitics.jsonl` | Raw upstream FB clone | append |
| `gdelt_articles.jsonl` | FB GDELT-DOC title metadata | append |
| `fb_articles_full.jsonl` | FB trafilatura-extracted text | overwrite |
| `gdelt_cameo/kg_raw/` | 14 687 GDELT 15-min export CSVs | append |
| `gdelt_cameo/kg_tmp/` | streaming merge/filter staging | overwrite |
| `gdelt_cameo/data_{kg,news,news_full}.csv` | filtered event + article tables | overwrite |
| `gdelt_cameo/test/relation_query.csv` | per-query FDs | overwrite |
| `earnings/earnings_forecasts.jsonl` | yfinance earnings FDs | overwrite |
| `unified/` | pipeline STAGING (current live state) | overwrite |
| `staged/{run_id}/{step}/` | VERSIONED SNAPSHOTS of `unified/` between steps | **write-once** |
| `reference/gdelt_lookups/` | Static upstream lookups (CAMEO + ISO) | **never** |

## Invariants

1. `unified/` is mutable — overwritten during each build.
2. `staged/` is append-only, versioned by `{run_id}`, never modified.
3. `reference/` is bundled static data; no script writes to it.
4. **Nothing under `data/` is a deliverable.**

## For downstream consumers

- **Evaluating a forecasting model?** Read `benchmark/data/{cutoff}/forecasts.jsonl` + `articles.jsonl` only. Ignore this folder.
- **Debugging / post-processing the build?** `data/staged/{run_id}/` is your audit trail.
