# EMR-ACH Forecasting Benchmark

A leakage-free, multi-domain benchmark of **Forecast Dossiers (FDs)** pairing resolved forecast questions with pre-event evidence articles. Designed for evaluating retrieval-augmented LLM forecasters under strict temporal cutoffs.

**FD** = Forecast Dossier: one resolved question plus its pre-event article bundle.

## Sources

Three tracks are combined into a single unified schema:

- **`forecastbench`** — resolved questions from the upstream ForecastBench project (`polymarket`, `metaculus`, `manifold`, `infer`).
- **`gdelt_cameo`** — GDELT-CAMEO event-forecasting queries over the quad-class CAMEO taxonomy (Verbal-Coop, Material-Coop, Verbal-Conflict, Material-Conflict). Methodology follows Ye et al., "MIRAI: Evaluating LLM Agents for Event Forecasting" ([arXiv:2407.01231](https://arxiv.org/abs/2407.01231)); we re-implement their pipeline against fresh GDELT data rather than redistributing their dataset.
- **`earnings`** — S&P earnings-surprise questions (Beat / Meet / Miss) built from `yfinance`.

## Folder layout

```
benchmark/
├── README.md                    (this file)
├── RECREATE.md                  (how to rebuild for any cutoff)
├── DATASET.md                   (schema + EDA stats + meta-file guide)
├── requirements.txt
├── build.py                     (SINGLE entry point)
├── configs/
│   └── default_config.yaml      (ALL pipeline params, including `relevance:`)
├── scripts/
│   ├── gdelt_cameo/             (GDELT-CAMEO build, text fetch, context relink)
│   ├── forecastbench/           (upstream download + trafilatura fetch)
│   ├── earnings/                (yfinance-based FD builder)
│   └── common/                  (unify, relevance, quality, diagnostics, EDA)
├── schema/
│   ├── forecast_dossier.md      (per-field FD reference)
│   └── article.md               (per-field Article reference)
└── data/
    ├── reference/gdelt_lookups/ (static lookups: cameo_codes.txt, iso_country_names.txt)
    └── {cutoff}/                (produced by `build.py`)
        ├── forecasts.jsonl      (PRIMARY DELIVERABLE #1)
        ├── articles.jsonl       (PRIMARY DELIVERABLE #2, referenced subset only)
        ├── benchmark.yaml       (effective config — drop-in reproducer)
        ├── build_manifest.json  (timestamp, git sha, benchmarks included)
        └── meta/                (diagnostics, NOT primary benchmark data)
            ├── eda_report.html
            ├── diagnostic_report.md
            ├── diagnostic_report.json
            ├── quality_meta.json
            └── relevance_meta.json
```

## Quickstart

```bash
pip install -r requirements.txt

# Build for a given LLM training cutoff (one command drives the full pipeline):
cd benchmark && python build.py --cutoff 2026-01-01
```

Common flags:

```bash
python build.py --cutoff 2026-01-01 --benchmarks forecastbench,earnings  # subset
python build.py --cutoff 2026-01-01 --skip-raw                           # reuse raw caches
python build.py --cutoff 2026-01-01 --fresh                              # wipe and rebuild
python build.py --cutoff 2026-01-01 --dry-run                            # preview only
```

## Start here

- Rebuilding for a new cutoff, different time window, or custom slice: see [RECREATE.md](RECREATE.md).
- Schema reference and EDA statistics: see [DATASET.md](DATASET.md).
- Per-field references: [schema/forecast_dossier.md](schema/forecast_dossier.md), [schema/article.md](schema/article.md).
- Running baseline methods against a built cutoff: see [evaluation/README.md](evaluation/README.md) and the per-method reference at [evaluation/BASELINES.md](evaluation/BASELINES.md).

## Running baselines

The baselines battery lives under `evaluation/baselines/` and is driven by `benchmark/configs/baselines.yaml`. A single CLI runs any of the nine methods (B1-B9) against a built cutoff's `forecasts.jsonl` + `articles.jsonl`. Every run is written to a versioned directory under `benchmark/results/{cutoff}/{method}/{run_id}/` so results are never overwritten.

Smoke check (no API calls):

```bash
cd benchmark
python -m evaluation.baselines.runner \
    --method b4_self_consistency \
    --fds data/2024-04-01/forecasts.jsonl \
    --articles data/2024-04-01/articles.jsonl \
    --config configs/baselines.yaml \
    --dry-run --smoke 3
```

See [evaluation/README.md](evaluation/README.md) for the three-stage debug flow (`--dry-run` -> `--smoke N --sync` -> Batch API production) and [evaluation/BASELINES.md](evaluation/BASELINES.md) for per-method descriptions, citations, and compute costs.

## Citation

_Placeholder: BibTeX entry to be added upon release._

GDELT-CAMEO track methodology: Ye et al., arXiv:2407.01231 (cited, not redistributed).

## License

_Placeholder: code license TBD._ Third-party data (GDELT, Yahoo Finance, Polymarket, Metaculus, Manifold, Infer, trafilatura-scraped articles) retains its original upstream license; users are responsible for respecting source terms.
