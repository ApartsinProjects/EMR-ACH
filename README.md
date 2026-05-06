# EMR-ACH

**Evidence-Marshalling Retrieval with Analysis of Competing Hypotheses**
A leakage-free, multi-domain benchmark for LLM forecasting under realistic temporal constraints, plus a proposed analytical framework for hypothesis-aware reasoning over news evidence.

[![tag](https://img.shields.io/badge/tag-v2.1--data--ready-blue)](https://github.com/ApartsinProjects/EMR-ACH/releases/tag/v2.1-data-ready)
[![status](https://img.shields.io/badge/status-v2.2%20rebuild%20in%20progress-orange)](docs/PROJECT_SPEC.md)
[![license](https://img.shields.io/badge/code-MIT-green)](LICENSE)

---

## Why this exists

Most LLM forecasting evaluations leak. Either the question's resolution date sits inside the model's training window, or the "evidence" articles are pulled retroactively from the day the answer was already public. Either way, the model is grading itself with the answer key.

EMR-ACH closes that gap by enforcing two invariants per **Forecast Dossier (FD)**:

1. **Horizon ≥ 14 days** — the simulated forecaster sees the question 14 days before resolution, not on the day of.
2. **Strict leakage filter** — every retrieved article has `publish_date ≤ forecast_point`. Verified at publish time and again at evaluation.

The resulting benchmark covers three forecasting domains under one schema, three evaluation tracks, and ten reproducible baselines on a shared pick-only response contract.

---

## What's in the box

```
benchmark/data/{cutoff}/                 ← published benchmark
benchmark/data/{cutoff}-gold/            ← curated, self-contained gold subset
benchmark/data/{cutoff}-h14/             ← v2.2 horizon-14 reuse-first build
benchmark/data/{cutoff}-h14-ccnews/      ← v2.2 horizon-14 CC-News rebuild
benchmark/evaluation/baselines/          ← B1..B9 + majority-class reference
docs/PROJECT_SPEC.md                     ← single-source-of-truth spec
docs/V2_2_REFACTOR_BACKLOG.md            ← living backlog (80 items)
docs/V2_2_EVAL_PLAN.md                   ← ready-to-fire Batch API commands
paper/index.html                         ← HTML paper (Tables, Appendices A-G)
src/common/openai_embeddings.py          ← Batch API helper (auto-chunk, salvage, local-completed)
src/pipeline/                            ← (proposed) EMR-ACH method modules — see §method status
scripts/                                 ← every pipeline stage as a CLI
```

---

## Three tracks, one schema

Every Forecast Dossier (FD) carries the same fields regardless of source:

| Track | n FDs (v2.1) | Domain | Primary target | Secondary |
|---|---:|---|---|---|
| `forecastbench` | 134 | Public-interest forecasting markets | Comply / Surprise | resolved label |
| `gdelt-cameo` | 5,975 | Geopolitical event intensity | Comply / Surprise | Peace / Tension / Violence |
| `earnings` | 185 | S&P 500 earnings | Comply / Surprise | Beat / Meet / Miss |

The **Comply/Surprise** primary target unifies the three tracks: did the prior expectation hold, or did something break? `fd_type ∈ {stability, change, unknown}` stratifies further.

---

## The pipeline

```
       (1) ingest                 (2) unify                  (3) match
  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ forecastbench    │ ───▶ │  unify_articles  │ ───▶ │ compute_relevance│
  │ gdelt-cameo      │      │  unify_forecasts │      │  (OpenAI embed)  │
  │ earnings         │      │                  │      │                  │
  │ CC-News (v2.2)   │      └──────────────────┘      └──────────────────┘
  └──────────────────┘                                          │
                                                                ▼
                                                       ┌──────────────────┐
                                                       │  link_earnings   │ (ticker-date join)
                                                       │  relink_gdelt    │ (pre-event substitution)
                                                       │  prior_state     │ (status-quo annotation)
                                                       └──────────────────┘
                                                                │
       (5) ETD                   (4) publish                    │
  ┌──────────────────┐      ┌──────────────────┐                │
  │ articles_to_facts│ ◀─── │  step_publish    │ ◀──────────────┘
  │ etd_dedup (G8)   │      │  + leakage check │
  │ etd_link         │      │  + dangling-id   │
  │ etd_filter       │      │    integrity     │
  │ etd_audit        │      └──────────────────┘
  └──────────────────┘                │
           │                          ▼
           ▼                ┌──────────────────┐
  ┌──────────────────┐ ◀─── │  build_gold_     │
  │  gold subset     │      │    subset        │
  └──────────────────┘      └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ baselines runner │ ──▶ paper Tables
                            │  B1..B9 + maj    │
                            └──────────────────┘
```

Every stage is a self-contained CLI under `scripts/`. The orchestrator at `scripts/build_benchmark.py` runs the full chain.

---

## Forecast Dossier (FD)

A Forecast Dossier is one resolved forecasting question paired with its pre-event evidence bundle:

```jsonc
{
  "id": "earn_AAPL_2026-01-30",
  "benchmark": "earnings",
  "question": "Will Apple's Q1 FY26 EPS surprise be Beat / Meet / Miss?",
  "hypothesis_set": ["Comply", "Surprise"],
  "hypothesis_definitions": {"Comply": "...", "Surprise": "..."},
  "ground_truth": "Comply",
  "ground_truth_idx": 0,
  "forecast_point": "2026-01-16T00:00:00Z",       // resolution_date − 14d
  "resolution_date": "2026-01-30T20:30:00Z",
  "lookback_days": 30,                             // article window: [fp-30d, fp]
  "default_horizon_days": 14,
  "article_ids": ["art_a995…", "art_e394…", ...], // every article passes publish_date ≤ fp
  "prior_state_30d": "guidance reaffirmed; analyst consensus 2.10 EPS",
  "prior_state_stability": "stable",
  "fd_type": "stability",
  "metadata": {"ticker": "AAPL", "report_date": "2026-01-30", ...}
}
```

Full schema: [`docs/FORECAST_DOSSIER.md`](docs/FORECAST_DOSSIER.md).

---

## Event Timeline Dossier (ETD)

ETD is the evidence side: articles get distilled into atomic, dated facts that can be deduplicated, linked, and reasoned over independently of the article they came from.

| Stage | Script | What it does |
|---|---|---|
| 1 | `articles_to_facts.py` | LLM extraction of (subject, predicate, time, polarity) tuples |
| 2 | `etd_dedup.py` | Date-bucketed FAISS kNN canonicalisation (G8) |
| 3 | `etd_link.py` | Per-cutoff linkage to FDs via `primary_article_id` |
| filter | `etd_filter.py` | Production filter (blocklist + confidence + polarity + no-future) |
| audit | `etd_audit.py` | Schema, leakage, source distribution check |

Sample ETD fact:

```jsonc
{
  "id": "f_3e0613…",
  "fact": "Apple reaffirmed Q1 FY26 revenue guidance",
  "time": "2026-01-09",
  "primary_article_id": "art_a995…",
  "entities": [{"name": "Apple", "type": "organization"}],
  "polarity": "asserted",
  "extraction_confidence": "high",
  "canonical_id": "f_3e0613…",
  "linked_fd_ids": ["earn_AAPL_2026-01-30"]
}
```

---

## Baselines (pick-only, plurality-vote)

Every baseline returns one hypothesis label per FD. Multi-sample methods (B4-B7, B9) aggregate via plurality vote, ties broken by `hypothesis_set` order. No probability distributions. Reference: [`benchmark/evaluation/BASELINES.md`](benchmark/evaluation/BASELINES.md).

| ID | Method | Calls / FD | Reference |
|---|---|---|---|
| B1 | Direct prompting | 1 | Brown et al. 2020 |
| B2 | Chain-of-Thought (ACH-style) | 1 | Wei et al. 2022 |
| B3 | RAG-only | 1 | Lewis et al. 2020 |
| B4 | Self-Consistency | n_samples (4) | Wang et al. 2022 |
| B5 | Multi-Agent Debate | n_agents × n_rounds | Du et al. 2023 |
| B6 | Tree of Thoughts | breadth + breadth² | Yao et al. 2023 |
| B7 | Reflexion | 1 + 2(n_iter − 1) | Shinn et al. 2023 |
| B8 | Verbalized Confidence | 1 | Lin et al. 2022 (DEPRECATED under pick-only; majority-class reference replaces it) |
| B9 | Heterogeneous LLM Ensemble | len(configs) | Jiang et al. 2023 |

---

## Method status: EMR-ACH (proposed)

The repo is named for the proposed analytical framework — but the framework itself is **partially prototyped, not shipped**. `docs/EMRACH_IMPLEMENTATION_AUDIT.md` (2026-04-23) traced each component:

| Component | Status |
|---|---|
| Contrastive indicators | PARTIAL (MIRAI-locked to 4-CAMEO) |
| Diagnostic weighting (ACH analysis matrix A) | MISSING |
| Multi-agent debate | PARTIAL (single-round) |
| Hybrid retrieval (MMR + RRF + temporal decay) | MISSING |
| Live entry point `scripts/eval/emrach_on_gold.py` | STUB (raises `NotImplementedError`) |

Per a 2026-04-23 scope decision, the paper is reframed as a **benchmark + baselines contribution**; EMR-ACH method claims are deferred to v2.3. Tracking: backlog [H7](docs/V2_2_REFACTOR_BACKLOG.md).

---

## Versions

| Tag | Status | What it is |
|---|---|---|
| `v2.1-data-ready` | shipped | 6,294 FDs, 28,945 articles, 81-FD gold subset. Horizon=0 (retrospective evaluation). Tagged at `8ffba6f`. |
| `v2.2-h14` (in progress) | local | Horizon=14, lookback=30, fb+earnings. Two parallel builds: reuse-first (89 FDs, 291 articles) + CC-News rebuild (89 FDs, 1,209 articles, 5,956 ETD facts). |
| `v2.2-data-ready` (next) | TBD | After baselines run on v2.2 gold + paper reframe. |
| `v2.3` (future) | future | EMR-ACH method shipped; full paper with method claims. |

---

## Quickstart

```bash
# Build the benchmark for a given cutoff (one command, full pipeline)
python scripts/build_benchmark.py --cutoff 2026-01-01 \
    --benchmarks forecastbench,earnings \
    --horizon-days 14 --lookback-days 30 \
    --embedder openai --openai-mode batch

# Build a self-contained gold subset
python scripts/build_gold_subset.py --cutoff 2026-01-01 \
    --min-articles 5 --min-distinct-days 3 --min-source-diversity 2

# Smoke-test a baseline (pick-only, sync, 3 FDs)
cd benchmark && python -m evaluation.baselines.runner \
    --method b1_direct \
    --fds data/2026-01-01-gold/forecasts.jsonl \
    --articles data/2026-01-01-gold/articles.jsonl \
    --smoke 3 --sync

# Full Batch API run (~$0.30 for 10 baselines × ~80 FDs)
python -m evaluation.baselines.runner --method b1_direct --batch ...
```

For the v2.2 CC-News full-pool path, see [`docs/v2_2_ccnews_build.md`](docs/v2_2_ccnews_build.md).

---

## Reproducing the data

The pipeline is fully reproducible from the public sources:

- **forecastbench** ← upstream ForecastBench repo (resolved questions only).
- **gdelt-cameo** ← GDELT 2.0 KG (publicly hosted).
- **earnings** ← yfinance + Finnhub + EDGAR + GDELT-slug + NYT/Guardian editorial.
- **CC-News** ← Common Crawl `data.commoncrawl.org/crawl-data/CC-NEWS/{YYYY}/{MM}/`.

Every FD carries `metadata.source_*` fields documenting where each piece came from. Article-level provenance is in `articles.jsonl[*].provenance`.

The published `benchmark/data/2026-01-01/` is the canonical dataset for v2.1; the gold subset under it is self-contained and downstream-friendly (schema + examples + LICENSE + CITATION inline).

---

## Project layout

```
ACH/
├── benchmark/
│   ├── README.md            ← user-facing benchmark docs
│   ├── DATASET.md           ← schema + EDA + horizon/lookback config
│   ├── RECREATE.md          ← step-by-step rebuild instructions
│   ├── configs/             ← default_config.yaml, leakage_probe_config.yaml, baselines.yaml
│   ├── schema/              ← FD + article + ETD JSON Schemas
│   ├── evaluation/
│   │   ├── BASELINES.md     ← per-method reference
│   │   └── baselines/       ← B1..B9 implementations + runner
│   └── data/
│       ├── 2026-01-01/             ← v2.1 publish (tagged)
│       ├── 2026-01-01-gold/        ← v2.1 gold subset
│       ├── 2026-01-01-h14/         ← v2.2 reuse-first
│       └── 2026-01-01-h14-ccnews/  ← v2.2 CC-News rebuild
├── docs/
│   ├── PROJECT_SPEC.md             ← single source of truth
│   ├── V2_2_REFACTOR_BACKLOG.md    ← 80-item living backlog
│   ├── V2_2_ARCHITECTURE.md        ← pipeline contracts
│   ├── FORECAST_DOSSIER.md         ← FD schema + invariants
│   ├── PIPELINE.md                 ← stage-by-stage reference
│   ├── EMRACH_IMPLEMENTATION_AUDIT.md  ← method-side gap audit
│   └── V2_2_EVAL_PLAN.md           ← ready-to-fire Batch API commands
├── paper/
│   └── index.html                  ← HTML paper (7 tables, 7 appendices)
├── scripts/                        ← every pipeline stage as a CLI
├── src/
│   ├── common/openai_embeddings.py ← Batch API helper
│   └── pipeline/                   ← (in-progress) EMR-ACH method modules
└── tests/                          ← pytest suites for invariants + leakage
```

---

## Design choices worth flagging

- **No GPU required.** Every component routes through the OpenAI Batch API or CPU pipelines. The repo can run end-to-end on a laptop given disk + bandwidth for CC-News.
- **OpenAI embeddings, not SBERT.** `text-embedding-3-small` (1536-d, L2-normalized). Auto-chunked above 50k cap, parallel-polled across chunks, salvage-on-cancel + LOCAL-COMPLETED short-circuit for offline-synced shards. See [`src/common/openai_embeddings.py`](src/common/openai_embeddings.py).
- **G8 date-bucketed FAISS kNN.** ETD Stage-2 dedup runs in ~8 seconds for 78k facts vs ~15 minutes for the brute-force baseline. Exact recall within the date-window constraint; 6/6 parity tests pass.
- **Leakage is a hard test.** Per-FD assertions at publish time (`step_publish` dangling-ref dump), at gold build (re-asserted), and at evaluation (regression tests).
- **Pick-only contract.** Removes the calibration confound; B8 (verbalized confidence) is deprecated and replaced by a majority-class reference in the paper.

---

## Citation

Citation BibTeX will be added at v2.2 release. For now:

```
EMR-ACH Forecasting Benchmark.
ApartsinProjects, 2026.
https://github.com/ApartsinProjects/EMR-ACH
Tag: v2.1-data-ready
```

---

## License

Code: MIT. Third-party data (GDELT, Yahoo Finance, ForecastBench upstream, Common Crawl, NYT/Guardian editorial) retains its original upstream licence; users are responsible for respecting source terms.

---

## See also

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) — full project specification + open decisions for each session.
- [`docs/V2_2_REFACTOR_BACKLOG.md`](docs/V2_2_REFACTOR_BACKLOG.md) — 80-item living backlog with priorities P0/P1/P2.
- [`paper/index.html`](paper/index.html) — HTML paper (Tables, Appendices A-G; Appendix F for baseline reference, G for worked examples).
- [`benchmark/README.md`](benchmark/README.md) — user-facing benchmark documentation.
- [`benchmark/evaluation/BASELINES.md`](benchmark/evaluation/BASELINES.md) — per-method reference with citations.
