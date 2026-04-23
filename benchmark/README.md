# EMR-ACH Forecasting Benchmark

A leakage-free, multi-domain benchmark of **Forecast Dossiers (FDs)** pairing resolved forecast questions with pre-event evidence articles. Designed for evaluating retrieval-augmented LLM forecasters under strict temporal cutoffs.

**FD** = Forecast Dossier: one resolved question plus its pre-event article bundle.

## Sources

Three tracks are combined into a single unified schema:

- **`forecastbench`** — resolved questions from the upstream ForecastBench project (`polymarket`, `metaculus`, `manifold`, `infer`). v2.1 default includes **all subjects**; set `EMRACH_FB_SUBJECT_FILTER=geopolitics` to restrict to the legacy geopolitics-only subset.
- **`gdelt_cameo`** — GDELT-CAMEO Geopolitics: country-pair event-forecasting queries built from scratch on the public GDELT 2.0 Knowledge Graph. Primary target is the 3-class ordinal intensity (Peace / Tension / Violence, derived from CAMEO root codes 01–09 / 10–17 / 18–20). MIRAI (Ye et al. 2024, [arXiv:2407.01231](https://arxiv.org/abs/2407.01231)) is cited as prior work on the same data source; we share no code or methodology (different target, horizon, retrieval, evaluation protocol).
- **`earnings`** — S&P 500 earnings-surprise questions (Beat / Meet / Miss) built from `yfinance`.

**Primary task (v2.1).** Regardless of domain, every FD's primary prediction target is the binary **Comply vs Surprise** label: *will the status-quo expectation hold or break?* Domain-specific multi-class labels (Peace/Tension/Violence, Beat/Meet/Miss, Yes/No) are preserved as secondary ablation targets under `x_multiclass_*` fields. See [`docs/FORECAST_DOSSIER.md`](../docs/FORECAST_DOSSIER.md) for the full contract.

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

# Build for a given LLM training cutoff (one command drives the full pipeline).
# Run from the repo root:
python scripts/build_benchmark.py --cutoff 2026-01-01
```

Common flags:

```bash
python scripts/build_benchmark.py --cutoff 2026-01-01 --benchmarks forecastbench,earnings  # subset
python scripts/build_benchmark.py --cutoff 2026-01-01 --skip-raw                           # reuse raw caches
python scripts/build_benchmark.py --cutoff 2026-01-01 --fresh                              # wipe and rebuild
python scripts/build_benchmark.py --cutoff 2026-01-01 --dry-run                            # preview only

# Leakage-probe build (smaller 2024-Q1 window, GPT-4o-mini era):
python scripts/build_benchmark.py --cutoff 2024-04-01 --config configs/leakage_probe_config.yaml
```

### v2.2 flags (in flight)

The v2.2 rollout adds an OpenAI-embedder family and a news-fetch skip flag.
v2.1 today defaults to SBERT (`all-MiniLM-L6-v2`) and to the per-FD GDELT DOC
fetch; v2.2 defaults to the OpenAI embedder and prefers the bulk DOC archive
+ index when one is available (see `docs/V2_2_ARCHITECTURE.md`).

```bash
# v2.1 today (SBERT, per-FD GDELT DOC fetch):
python scripts/build_benchmark.py --cutoff 2026-01-01 --embedder sbert

# v2.2 planned (OpenAI embeddings, reuse pre-fetched news archives):
python scripts/build_benchmark.py --cutoff 2026-01-01 \
    --embedder openai \
    --openai-model text-embedding-3-small \
    --openai-mode batch \
    --skip-news-fetch
```

`--embedder {sbert,openai}` selects the embeddings backend.
`--openai-model` and `--openai-mode {sync,batch}` configure the OpenAI path
(see `src/common/openai_embeddings.py`). `--skip-news-fetch` reuses the
already-downloaded CC-News and GDELT DOC archives without re-hitting the
upstream APIs (companion to the existing `--skip-raw`).

Note: the orchestrator lives at `scripts/build_benchmark.py` (top-level). The
`benchmark/build.py` shim that previously sat here was deprecated in v2.1
because it pointed at a parallel `benchmark/scripts/` script tree that had
drifted from the canonical top-level scripts. Output still lands in
`benchmark/data/{cutoff}/` either way.

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
