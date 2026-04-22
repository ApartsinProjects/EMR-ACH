# `benchmark/data/` — Shipped Benchmark (deliverable ONLY)

**This folder contains only clean shipped data.** Audit material, diagnostic
reports, and build metadata live in the sibling `benchmark/audit/{cutoff}/`
folder so the deliverable tree is never polluted with intermediate info.

## Layout

```
benchmark/
├── data/                              ← SHIPPED — the benchmark itself
│   ├── reference/gdelt_lookups/         static upstream lookups (same across builds)
│   └── {cutoff}/                        one per model-cutoff build
│       ├── forecasts.jsonl                PRIMARY #1 — Forecast Dossiers
│       ├── articles.jsonl                 PRIMARY #2 — FD-referenced articles
│       ├── benchmark.yaml                 effective config (reproducibility)
│       └── build_manifest.json            timestamp, git sha, benchmarks included
│
└── audit/                             ← NOT SHIPPED — diagnostics for debugging only
    └── {cutoff}/
        ├── eda_report.html                human-readable EDA
        ├── diagnostic_report.md/.json     leakage audit + stats
        ├── quality_meta.json              filter drop reasons
        └── relevance_meta.json            SBERT scoring stats
```

## Guarantees (data/ only)

- **Self-contained**: every `article_id` in `forecasts.jsonl` is resolvable
  within the same folder's `articles.jsonl`.
- **Leakage-free** within the stated cutoff: `resolution_date > cutoff` for
  every FD; every article's `publish_date < forecast_point`.
- **Versioned, never overwritten**: a second build for the same cutoff
  creates a sibling folder; the original is preserved.
- **Clean schema**: only 14 shipped fields per FD, no pipeline-internal
  metadata (`_drop_reasons`, `metadata.*`, `_relevance_*` are stripped).

## What NOT to read

- `../audit/{cutoff}/` is debugging material, not evaluation input.
- `../../data/` (repo root) is pipeline intermediate storage — private.
