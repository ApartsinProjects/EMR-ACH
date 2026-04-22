# EMR-ACH Experiment Plan

## Overview

**Task framing (pick-only, 2026-04-21 revision).**
Every baseline and the EMR-ACH system solves the same task: **given a forecasting question, a candidate hypothesis set, and a set of news articles retrieved for that question, select the single most likely hypothesis.** No probability distributions, no calibration, no confidence scores. Models return exactly one label from the hypothesis set.

Headline metric: **selection accuracy**. Secondary metrics (to expose class-imbalance effects): **macro-F1**, **per-class recall**, and **confusion matrix**. Bootstrap 95% CIs on all point estimates. No Brier, no ECE, no KL, no Platt scaling.

Every experiment uses the **OpenAI Batch API** (50% cost discount, mandatory).
Every experiment group begins with a **sync smoke test** (5–10 queries, direct API) for prompt debugging.

**Model**: `gpt-4o-mini-2024-07-18` throughout (cost-optimized profile, 2026-04-21).

---

## Directory Layout

```
ACH/
├── EXPERIMENT_PLAN.md           ← this file
├── requirements.txt
├── config.yaml
├── benchmark/
│   ├── configs/baselines.yaml   ← shared defaults + per-baseline overrides
│   ├── data/{cutoff}/           ← published benchmark (forecasts + articles)
│   ├── audit/{cutoff}/          ← diagnostics, build snapshots
│   ├── evaluation/baselines/    ← B1–B9 methods (all pick-only)
│   └── results/{cutoff}/{method}/{run_id}/
├── src/
│   ├── batch_client.py          ← OpenAI Batch API wrapper (core)
│   └── pipeline/
│       ├── prompts.py
│       ├── indicators.py        ← Step 1: contrastive indicator generation
│       ├── influence.py         ← Step 2: influence matrix I
│       ├── retrieval.py         ← Step 3: SBERT + MMR retrieval
│       ├── presence.py          ← Step 4: analysis matrix A
│       ├── aggregation.py       ← Step 5: A·I → score vector → argmax
│       ├── deep_analysis.py     ← Step 7: evidence disambiguation
│       └── multi_agent.py       ← adversarial debate variant
├── analysis/
│   ├── compute_metrics.py       ← accuracy, macro-F1, per-class recall, bootstrap CIs
│   └── compare_systems.py       ← LaTeX tables
└── results/
    └── figures/                 ← bar charts, confusion matrices
```

---

## Phase 0: Environment & Benchmark Build

**Duration**: 1 day. No LLM calls for the build itself (SBERT embeddings only).

Benchmark sources (unified):
- **ForecastBench** — binary `{Yes, No}` forecasting questions
- **GDELT-CAMEO / MIRAI-2024** — 4-class event type `{Verbal Cooperation, Material Cooperation, Verbal Conflict, Material Conflict}`
- **Earnings** — 3-class `{Beat, Meet, Miss}`

Every Forecast Dossier (FD) exposes: `question, hypothesis_set, hypothesis_definitions, article_ids (leakage-filtered), forecast_point, resolution_date, ground_truth, benchmark`.

**Leakage guards** (both enforced at build time):
1. Article-level: `article.publish_date < fd.forecast_point`
2. Training-level: `fd.resolution_date > model_cutoff + buffer`

Run `scripts/build_benchmark.py --cutoff YYYY-MM-DD` to (re)build. Output goes to `benchmark/data/{cutoff}/`; snapshots and drop-reason audits go to `benchmark/audit/{cutoff}/`.

---

## Phase 1: Smoke Tests (Prompt Debugging)

**Duration**: 1–2 days.
**API mode**: sync (`--sync --smoke 10`) for fast iteration.
**Model**: `gpt-4o-mini`.

Every baseline and every EMR-ACH pipeline stage has a smoke test that:
1. Loads 10 FDs (mixed across all three benchmark sources).
2. Issues the stage's prompt to the LLM synchronously.
3. Validates response shape: must be JSON of the form `{"prediction": "<one of hypothesis_set>", "reasoning": "<short>"}`.
4. Logs prompt+response to `benchmark/audit/{cutoff}/smoke/{method}/`.
5. Reports parse failures and off-label predictions (prediction not in `hypothesis_set`).

Iterate on YAML prompt templates until **10/10 parse correctly and predictions are all in `hypothesis_set`** for each stage.

Per-stage smoke scripts:

| Script                      | Tests                                         | Success criterion                           |
|-----------------------------|-----------------------------------------------|---------------------------------------------|
| `smoke_indicators.py`       | Step 1: contrastive indicator generation      | 24 indicators, valid JSON, all hypotheses covered |
| `smoke_influence.py`        | Step 2: influence scoring                     | 24 × |H| matrix, values in {-1, 0, +1} or [-1, 1] |
| `smoke_presence.py`         | Step 4: per-article presence scoring          | n × 24 matrix, values in [0, 1]             |
| `smoke_deep_analysis.py`    | Step 7: evidence disambiguation               | classification in valid set, valid JSON     |
| `smoke_pipeline_e2e.py`     | Full pipeline → final pick                    | 10/10 produce a valid hypothesis label      |
| `smoke_multiagent.py`       | Debate + judge → final pick                   | judge outputs one of `hypothesis_set`       |

**Smoke cost**: < $0.10 total.

---

## Phase 2: Baselines (B1–B9, pick-only)

**Duration**: 1 day.
**API mode**: Batch.
**N FDs**: full benchmark (whatever the cutoff yields, typically 2–3k).
**Model**: `gpt-4o-mini`.

All nine baselines return **a single hypothesis pick** (no probabilities, no confidence). See `benchmark/configs/baselines.yaml` for exact parameters.

| ID  | Name                   | Mechanism                                                             | Calls / FD |
|-----|------------------------|-----------------------------------------------------------------------|-----------:|
| B1  | Direct                 | Single call, question + hypotheses, no articles                       | 1          |
| B2  | Chain-of-Thought       | Single call, CoT instructions, no articles                            | 1          |
| B3  | RAG                    | Single call, question + hypotheses + top-10 articles                  | 1          |
| B4  | Self-Consistency       | k=4 CoT samples at T=0.7 + articles, **majority vote** over picks     | 4          |
| B5  | Multi-Agent Debate     | n=2 agents, 1 round, articles, judge picks winner                     | 3          |
| B6  | Tree-of-Thoughts       | breadth=2, depth=2, articles, best-path argmax                        | ~6         |
| B7  | Reflexion              | n=2 iterations, articles, self-critique then revise                   | 4          |
| B8  | *(deprecated)*         | Verbalized Confidence no longer applicable — **dropped from the paper** under pick-only framing. Replaced with a trivial **majority-class** reference baseline reported in every results table. | — |
| B9  | LLM Ensemble           | 3 configs at T ∈ {0.0, 0.5, 1.0}, articles, **majority vote**        | 3          |

**Aggregation (B4, B9)**: simple plurality over `k` picks. Ties broken deterministically (alphabetical on hypothesis label).

**Estimated cost (Batch API, 2k FDs × 9 baselines)**: ~$45.

---

## Phase 3: EMR-ACH Staged Build-Up

**Duration**: 2–3 days.
**API mode**: Batch (one batch per pipeline stage, cached in `results/raw/`).
**Model**: `gpt-4o-mini`.

**Pick-only aggregation.** EMR-ACH's scoring step is:

```
score[h] = sum_j A[:, j] · I[j, h] · d_j        (analysis · influence · diagnosticity)
pick    = argmax_h score[h]
```

No calibration, no Platt, no softmax. The argmax is the final system output.

Staged batches (cache-keyed by `{cutoff}/{stage}/{config_hash}`):

| Stage             | Batch job            | N requests   | Purpose                                   |
|-------------------|----------------------|--------------|-------------------------------------------|
| 1. indicators     | `indicators_full`    | N_FDs        | Generate 24 contrastive indicators / FD   |
| 2. influence      | `influence_full`     | N_FDs        | Score each indicator's influence / hypo   |
| 4. presence       | `presence_full`      | N_FDs × 10   | Score presence in each retrieved article  |
| 7. deep_analysis  | `deep_analysis_full` | N_FDs × 10   | Disambiguate under-specified evidence     |

**No `background_prior` batch** (that step existed for absence-of-evidence in the probabilistic formulation; irrelevant under pick-only).

Table 1 in the paper is built by toggling components — all rows reuse the same cached stage outputs:

| Row                              | Active components                                   | New batches? |
|----------------------------------|-----------------------------------------------------|:------------:|
| EMR-ACH (base)                   | generic indicators + influence + presence + argmax  | Steps 1–4    |
| + Contrastive indicators         | contrastive prompt + steps 2–4                      | New S1+S2    |
| + Diagnostic weighting (d_j)     | same + d_j in score                                 | None (CPU)   |
| + Multi-agent ACH                | same + debate + judge                               | New debate   |
| + Deep Analysis (Full)           | all above + Step 7                                  | New S7       |

**Estimated cost (2k FDs full pipeline)**: ~$18.

---

## Phase 4: Ablation Study

**Duration**: 2 days. Reuses cached matrices wherever possible.

| Ablation                    | New batch? | Rationale                                         |
|-----------------------------|:----------:|---------------------------------------------------|
| w/o Multi-agent             | No         | Skip debate, reuse A, I                           |
| w/o Deep Analysis           | No         | Skip Step 7, reuse A, I                           |
| w/o Diag. weighting         | No         | Set d_j = 1, reuse A, I                           |
| w/o Contrastive indicators  | **Yes**    | New S1+S2 with generic indicator prompt           |
| w/o MMR / RRF               | **Yes**    | BM25-only retrieval → new A matrix                |
| w/o Temporal decay          | **Yes**    | Retrieval without time weighting → new A matrix   |

**Additional Phase 4 cost**: ~$10.

---

## Phase 5: Multi-Agent Scaling

**Duration**: 2 days.
**N_agents ∈ {1, 2, 3, 4, 5, 6}**. Each agent argues for one hypothesis; a judge picks the final hypothesis.

Per-FD cost grows linearly in `N_agents` (1 advocate call each + 1 judge call). All agents and the judge return a single hypothesis label — the judge's pick is the final answer.

Estimated Phase 5 cost: ~$15 across the scan.

---

## Phase 6: Cross-Benchmark Generalization

**Duration**: 1 day.
**API mode**: Batch.

The unified benchmark already mixes ForecastBench, GDELT-CAMEO, and Earnings. Phase 6 reports **per-source accuracy and macro-F1** for every system in Phase 2 and Phase 3, plus a **confusion matrix per source** to expose systematic mispicks (e.g. always predicting "No" on ForecastBench, always "Material Cooperation" on GDELT).

---

## Phase 7: Analysis and Figure Generation

**Duration**: 1 day. No LLM calls.

Scripts:
1. `analysis/compute_metrics.py` — accuracy, macro-F1, per-class recall, confusion matrix, bootstrap 95% CIs, per-source breakdown.
2. `analysis/compare_systems.py` — LaTeX tables (Table 1: incremental build-up; Table 2: baselines vs EMR-ACH; Table 3: per-source; Table 4: ablation).
3. `analysis/generate_paper_figures.py` — bar charts of accuracy and macro-F1 with CIs; confusion matrices.

**No reliability diagrams. No calibration plots.** Those belong to the probabilistic formulation that the paper no longer makes.

---

## Metrics (pick-only)

For every system × benchmark-source cell:

- **Accuracy** = `1/N · Σ 1[pick_i == ground_truth_i]`
- **Macro-F1** = mean of per-class F1 over `|H|` classes (robust to imbalance)
- **Per-class recall** = `1/N_c · Σ 1[pick_i == c | gt_i == c]`
- **Confusion matrix** = `C[gt, pick]` normalized per row
- **Bootstrap CI** = 1000 resamples at FD level, 95% percentile

**Reference baselines always reported alongside the LLM systems:**
- *Majority-class*: always predict the most frequent class in the benchmark
- *Uniform-random*: sample uniformly from `hypothesis_set` (seeded, same across systems)

A system that does not beat *majority-class accuracy* has **no selection skill** on that benchmark and must be reported as such.

---

## Total Cost Estimate (updated, gpt-4o-mini, ~2k FDs)

| Phase | Purpose                                | Cost (Batch API) |
|-------|----------------------------------------|-----------------:|
| 1     | Smoke tests (sync, mini)               |           < $0.10 |
| 2     | Baselines B1–B9 (pick-only)            |              ~$45 |
| 3     | EMR-ACH staged build-up                |              ~$18 |
| 4     | Ablation study (new batches only)      |              ~$10 |
| 5     | Multi-agent scaling                    |              ~$15 |
| 6     | Per-source cross-benchmark analysis    |           (reuses Ph.2–3 results) |
| 7     | Figures + tables                       |                 $0 |
| **Total** |                                    |          **~$88** |

---

## Batch API Workflow

```
1. Prepare requests → write batch_input_{job_name}.jsonl
2. Upload file → get file_id
3. Create batch → get batch_id (saved to results/batch_jobs.json)
4. Poll every 60s until status = "completed" | "failed"
5. Download output file → parse results
6. Save parsed results to results/processed/{job_name}.jsonl
```

**Resumability**: `batch_jobs.json` maps `job_name → batch_id`. On re-run, a completed batch is skipped and its cached output is reused.

---

## Prompt Contract (every stage that produces a pick)

Every pick-producing prompt ends with:

```
Return JSON only, no prose, no code fences:
{
  "prediction": "<exactly one of: hypothesis_set>",
  "reasoning":  "<one or two concise sentences citing article numbers>"
}
```

Parser validates `prediction ∈ hypothesis_set`. On parse failure or off-label prediction, the FD is marked `parse_failed` and counted against the system's accuracy as **wrong** (no silent fallbacks to uniform or majority).

Typical prompt-debugging issues (addressed during Phase 1):
- Markdown code fences around JSON → prompt explicitly forbids code fences.
- Model outputs an unlisted hypothesis (e.g. "Maybe") → prompt enumerates the exact set and the validator rejects anything else.
- Model outputs multiple hypotheses (e.g. `"prediction": "Yes or No"`) → prompt enforces exactly one.

---

## Running Order

```bash
# Phase 0: build benchmark
python scripts/build_benchmark.py --cutoff 2026-01-01

# Phase 1: smoke tests (iterate until 10/10 pass per stage)
python experiments/00_smoke/smoke_indicators.py
python experiments/00_smoke/smoke_influence.py
python experiments/00_smoke/smoke_presence.py
python experiments/00_smoke/smoke_deep_analysis.py
python experiments/00_smoke/smoke_multiagent.py
python experiments/00_smoke/smoke_pipeline_e2e.py

# Phase 2: baselines (pick-only)
cd benchmark
python -m evaluation.baselines.runner --method all --sync --smoke 10   # smoke
python -m evaluation.baselines.runner --method all                     # full, batch

# Phase 3: EMR-ACH build-up
python experiments/02_emrach/run_emrach.py --mode smoke
python experiments/02_emrach/run_emrach.py

# Phase 4: ablation
python experiments/03_ablation/run_ablation.py

# Phase 5: multi-agent scaling
python experiments/04_multiagent/run_multiagent.py

# Phase 6/7: analysis
python analysis/compute_metrics.py
python analysis/compare_systems.py
python analysis/generate_paper_figures.py
```
