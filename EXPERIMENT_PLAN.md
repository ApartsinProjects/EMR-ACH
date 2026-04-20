# EMR-ACH Experiment Plan

## Overview

Full implementation plan for reproducing and extending the EMR-ACH pipeline for publication.
Every experiment uses the **OpenAI Batch API** (50% cost discount, mandatory).
Every experiment group begins with a **smoke test** (5 queries, direct API) for prompt debugging.

---

## Directory Layout

```
ACH/
├── EXPERIMENT_PLAN.md        ← this file
├── requirements.txt
├── config.yaml               ← defaults; overridden per experiment
├── .env.example
├── src/
│   ├── config.py
│   ├── batch_client.py       ← OpenAI Batch API wrapper (core)
│   ├── data/
│   │   ├── mirai.py
│   │   └── forecastbench.py
│   ├── pipeline/
│   │   ├── prompts.py        ← load YAML templates
│   │   ├── indicators.py     ← Step 1: contrastive indicator generation
│   │   ├── influence.py      ← Step 2: influence matrix I
│   │   ├── retrieval.py      ← Step 3: RAG (Weaviate or mock)
│   │   ├── presence.py       ← Step 4: analysis matrix A + background row
│   │   ├── aggregation.py    ← Step 5: diagnosticity-weighted aggregation
│   │   ├── calibration.py    ← Platt scaling (fit + apply)
│   │   ├── deep_analysis.py  ← Step 7: V/M disambiguation
│   │   └── multi_agent.py    ← adversarial debate variant
│   ├── eval/
│   │   └── metrics.py        ← F1, ECE, KL, accuracy, bootstrap CIs
│   └── baselines/
│       ├── direct_prompting.py
│       ├── cot.py
│       └── rag_only.py
├── prompts/                  ← YAML prompt templates (versioned)
│   ├── indicators.yaml
│   ├── influence.yaml
│   ├── presence.yaml
│   ├── background_prior.yaml
│   ├── deep_analysis.yaml
│   ├── multiagent_advocate.yaml
│   ├── multiagent_judge.yaml
│   ├── direct.yaml
│   └── cot.yaml
├── experiments/
│   ├── runner.py             ← shared utilities
│   ├── 00_smoke/             ← DEBUG: 5 queries, direct API
│   │   ├── smoke_indicators.py
│   │   ├── smoke_influence.py
│   │   ├── smoke_presence.py
│   │   ├── smoke_deep_analysis.py
│   │   ├── smoke_pipeline_e2e.py
│   │   └── smoke_multiagent.py
│   ├── 01_baselines/
│   │   └── run_baselines.py
│   ├── 02_emrach/
│   │   └── run_emrach.py     ← staged build-up (base → full)
│   ├── 03_ablation/
│   │   └── run_ablation.py
│   ├── 04_multiagent/
│   │   └── run_multiagent.py
│   └── 05_forecastbench/
│       └── run_forecastbench.py
├── analysis/
│   ├── compute_metrics.py
│   ├── compare_systems.py
│   └── generate_paper_figures.py
├── data/
│   └── README.md             ← instructions for obtaining MIRAI + ForecastBench
└── results/                  ← auto-created; gitignored
    ├── raw/                  ← raw batch outputs (.jsonl)
    ├── processed/            ← parsed results per experiment
    └── figures/              ← updated paper figures from real data
```

---

## Phase 0: Environment Setup

**Duration:** 1 day  
**Goal:** Working data pipeline, no LLM calls yet.

### Steps
1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env`, fill in OPENAI_API_KEY
3. Download MIRAI benchmark (see `data/README.md`)
4. Index MIRAI news corpus into Weaviate (or use mock retrieval during smoke tests)
5. Download ForecastBench geopolitics/conflict subset (300 questions)
6. Run `python -c "from src.data.mirai import MiraiDataset; d=MiraiDataset(); print(len(d))"` to verify

### Data formats assumed
- MIRAI queries: JSONL with fields: `id, timestamp, subject, relation, object, label, doc_ids`
- MIRAI articles: JSONL with fields: `id, title, text, abstract, date, source`
- ForecastBench: JSONL with fields: `id, question, resolution_date, ground_truth, crowd_probability, category`

---

## Phase 1: Smoke Tests (Prompt Debugging)

**Duration:** 2-3 days  
**Goal:** Each pipeline prompt produces valid, parseable output before batching 100 queries.
**API mode:** DIRECT (no batch — need fast iteration)  
**N queries:** 5  
**Model:** gpt-4o-mini (cheap for debugging)

### Smoke test protocol (each stage)
Each smoke script:
1. Loads 5 MIRAI queries
2. Calls the LLM 5× with the stage prompt
3. Pretty-prints inputs and outputs
4. Validates JSON structure
5. Reports any parse failures
6. Reports cost estimate

**Iterate:** Fix prompt YAML, re-run, until 5/5 parse correctly and outputs look sensible.

### Stage-specific smoke scripts

| Script | Tests | Success criterion |
|--------|-------|-------------------|
| `smoke_indicators.py` | Step 1: indicator generation | 24 indicators, valid JSON, all 4 hypotheses covered |
| `smoke_influence.py` | Step 2: influence scoring | 24×4 matrix, values in [0,1] |
| `smoke_presence.py` | Step 4: presence scoring | n×24 matrix per query, values in [0,1] |
| `smoke_deep_analysis.py` | Step 7: V/M classification | classification in valid set, valid JSON |
| `smoke_pipeline_e2e.py` | Full pipeline (steps 1-7) | ranked output, KL < 1.0 for at least 3/5 queries |
| `smoke_multiagent.py` | Multi-agent debate | advocate + judge output parseable |

**Estimated cost (smoke phase):** ~$0.50 total (gpt-4o-mini, 5 queries × 6 stages)

---

## Phase 2: Baseline Experiments (B1-B3)

**Duration:** 1 day  
**API mode:** BATCH  
**N queries:** 100 (full MIRAI test set)  
**Model:** gpt-4o  

### B1: Direct Prompting
- 1 LLM call per query
- Prompt: query context only, no retrieval, no reasoning
- Output: ranked 4-class prediction + probability distribution
- **Batch size:** 100 requests
- **Estimated tokens:** 100 × 1,200 = 120k tokens
- **Estimated cost (batch):** $0.30

### B2: Chain-of-Thought
- 1 LLM call per query, system prompt instructs step-by-step reasoning
- Output: CoT text + final ranked prediction
- **Batch size:** 100 requests
- **Estimated tokens:** 100 × 2,500 = 250k tokens
- **Estimated cost (batch):** $0.63

### B3: RAG-Only
- Retrieve n=10 articles, concatenate abstracts into context
- 1 LLM call per query, no ACH structure
- **Batch size:** 100 requests
- **Estimated tokens:** 100 × 4,000 = 400k tokens
- **Estimated cost (batch):** $1.00

---

## Phase 3: EMR-ACH Staged Build-Up

**Duration:** 2-3 days  
**API mode:** BATCH (staged — one batch per pipeline stage)  
**N queries:** 100  
**Model:** gpt-4o  

### Staged batch protocol
Each stage submits one batch job and writes results to `results/raw/`.
Results are cached — if `results/raw/{stage}.jsonl` exists, the stage is skipped.

| Stage | Batch job name | N requests | Tokens/req | Total tokens | Cost (batch) |
|-------|---------------|------------|------------|--------------|--------------|
| Step 1: indicators | `indicators_full` | 100 | ~1,400 | 140k | $0.35 |
| Step 2: influence | `influence_full` | 100 | ~2,800 | 280k | $0.70 |
| Step 4: presence | `presence_full` | 1,000 (100×10) | ~1,900 | 1,900k | $4.75 |
| Step 4b: background | `background_full` | 100 | ~1,900 | 190k | $0.48 |
| Step 7: deep analysis | `deep_analysis_full` | 1,000 (100×10) | ~1,000 | 1,000k | $2.50 |
| **Total** | | **2,300** | | **3,510k** | **$8.78** |

*Batch API pricing: $2.50/1M input + $10/1M output tokens for gpt-4o. Estimate: 70% input, 30% output.*

### Incremental build-up (Table 1 in paper)
Each row of Table 1 is computed from cached intermediate results by varying which components are active:

| Row | Components active | Requires new batches? |
|-----|------------------|----------------------|
| EMR-ACH (base) | indicators (generic) + influence + presence + aggregate | Steps 1-5 only |
| + Calibrated mapping | same + Platt scaling | No new batches (CPU only) |
| + Contrastive indicators | contrastive indicator prompt + steps 2-5 | New Step 1+2 batch |
| + Diagnostic weighting | same + d_j weights | No new batches (CPU only) |
| + Absence-of-evidence | same + background prior | New Step 4b batch |
| + Multi-agent ACH | same + debate | New debate batch |
| + Deep Analysis (Full) | all above + Step 7 | New Step 7 batch |

---

## Phase 4: Ablation Study

**Duration:** 2 days  
**API mode:** BATCH  
**N queries:** 100  
**Model:** gpt-4o  

### Cache reuse strategy
Most ablations reuse intermediate results. Only 3 ablations require new batches:

| Ablation | New batch needed | Why |
|----------|-----------------|-----|
| w/o Multi-agent | No | Skip debate step, reuse A,I matrices |
| w/o Deep Analysis | No | Skip Step 7, reuse A,I matrices |
| w/o Diag. Weighting | No | Set d_j=1, reuse A,I matrices |
| w/o Absence-of-evidence | No | Skip background row, reuse A,I matrices |
| w/o Calibrated mapping | No | Use heuristic map, reuse A,I matrices |
| w/o Contrastive indicators | **Yes** | New indicator prompt → new Step 1+2 |
| w/o MMR/RRF | **Yes** | BM25-only retrieval → new A matrix |
| w/o Temporal decay | **Yes** | No time decay in retrieval → new A matrix |

**Additional batches needed:**
- `indicators_generic` (non-contrastive): 100 × 1,400 = 140k tokens → $0.35
- `presence_bm25only` (BM25 retrieval): 1,000 × 1,900 = 1,900k → $4.75
- `presence_nodecay` (no temporal decay): 1,000 × 1,900 = 1,900k → $4.75

**Phase 4 additional cost:** ~$9.85

---

## Phase 5: Multi-Agent Scaling

**Duration:** 2 days  
**API mode:** BATCH  
**N queries:** 100  
**Model:** gpt-4o  

### Protocol
Run with N_agents ∈ {1, 2, 3, 4, 5, 6}. N_agents=1 reuses the single-agent ablation result.

Per agent per query:
- 1 advocate call: ~800 tokens
- (Optional) 1 challenge call against top competitor: ~1,200 tokens
- Judge call: ~2,000 tokens

| N_agents | Calls/query | Total calls | Tokens | Cost (batch) |
|----------|------------|-------------|--------|--------------|
| 2 | 3 | 300 | 900k | $2.25 |
| 3 | 4 | 400 | 1,200k | $3.00 |
| 4 | 6 | 600 | 1,800k | $4.50 |
| 5 | 8 | 800 | 2,400k | $6.00 |
| 6 | 10 | 1,000 | 3,000k | $7.50 |

**Phase 5 total:** ~$23.25

---

## Phase 6: ForecastBench Cross-Domain

**Duration:** 1 day  
**API mode:** BATCH  
**N queries:** 300  
**Model:** gpt-4o  

### Adaptation
ForecastBench questions are binary (Yes/No) with crowd probability. We adapt:
- Hypotheses H = {Yes, No} instead of {VC, MC, VK, MK}
- Indicators are general forecasting signals, not geopolitical-specific
- Evaluation: Brier score, ECE (crowd probability as calibration target)

**Estimated tokens:** 300 × 5,000 = 1,500k tokens → **$3.75**

---

## Phase 7: Analysis and Figure Generation

**Duration:** 1 day  
**No LLM calls.**

### Scripts
1. `analysis/compute_metrics.py` — reads all results, computes F1/ECE/KL/accuracy + bootstrap CIs
2. `analysis/compare_systems.py` — generates LaTeX tables (Table 1, Table 2, Table 3)
3. `analysis/generate_paper_figures.py` — generates updated figures with real data

---

## Total Cost Estimate

| Phase | Queries | Model | Batch tokens | Cost |
|-------|---------|-------|-------------|------|
| 0: Smoke tests | 5 | gpt-4o-mini | ~180k | $0.05 |
| 1: Smoke tests (prompt debug) | 5×6 | gpt-4o-mini | ~200k | $0.10 |
| 2: Baselines B1-B3 | 100×3 | gpt-4o | 770k | $1.93 |
| 3: EMR-ACH staged build-up | 100 | gpt-4o | 3,510k | $8.78 |
| 4: Ablation study (new batches only) | 100×3 | gpt-4o | 3,940k | $9.85 |
| 5: Multi-agent scaling (2-6 agents) | 100×5 | gpt-4o | 9,300k | $23.25 |
| 6: ForecastBench | 300 | gpt-4o | 1,500k | $3.75 |
| **Total** | | | **~19.4M** | **~$47.71** |

*All LLM costs use OpenAI Batch API at 50% discount over standard pricing.*
*gpt-4o pricing: $2.50/1M input, $10/1M output (batch). Estimate: 70/30 split.*

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

**Resumability:** `batch_jobs.json` maps job_name → batch_id. On re-run, if batch_id exists and is completed, skip to step 5.

---

## Prompt Debugging Workflow (Smoke Tests)

```
1. Run smoke_<stage>.py
2. Inspect printed output — does JSON parse? Are values in range? Content sensible?
3. If not: edit prompts/<stage>.yaml
4. Re-run smoke test
5. Repeat until 5/5 queries pass validation
6. Run smoke_pipeline_e2e.py for full pipeline check
```

**Typical issues to fix in prompts:**
- Model outputs markdown code fences (```json ... ```) around JSON → add "Do not wrap in code blocks"
- Scores outside [0,1] → add explicit range constraint
- Missing fields in JSON → add required fields example in prompt
- Hypotheses abbreviated inconsistently → enumerate exact strings: "VC", "MC", "VK", "MK"

---

## Running Order

```bash
# Phase 0: verify data
python -m src.data.mirai

# Phase 1: smoke tests (iterate until all pass)
python experiments/00_smoke/smoke_indicators.py
python experiments/00_smoke/smoke_influence.py
python experiments/00_smoke/smoke_presence.py
python experiments/00_smoke/smoke_deep_analysis.py
python experiments/00_smoke/smoke_multiagent.py
python experiments/00_smoke/smoke_pipeline_e2e.py

# Phase 2: baselines
python experiments/01_baselines/run_baselines.py --mode smoke  # 5 queries, verify
python experiments/01_baselines/run_baselines.py               # full run, batch API

# Phase 3: EMR-ACH build-up
python experiments/02_emrach/run_emrach.py --mode smoke
python experiments/02_emrach/run_emrach.py

# Phase 4: ablation
python experiments/03_ablation/run_ablation.py --mode smoke
python experiments/03_ablation/run_ablation.py

# Phase 5: multi-agent
python experiments/04_multiagent/run_multiagent.py --mode smoke
python experiments/04_multiagent/run_multiagent.py

# Phase 6: ForecastBench
python experiments/05_forecastbench/run_forecastbench.py --mode smoke
python experiments/05_forecastbench/run_forecastbench.py

# Phase 7: analysis
python analysis/compute_metrics.py
python analysis/compare_systems.py
python analysis/generate_paper_figures.py
```
