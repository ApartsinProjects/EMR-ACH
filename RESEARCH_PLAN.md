# Research Plan: ACH-LLM Geopolitical Forecasting (Publication Track)

**Target venues**: ACL / EMNLP / NAACL (NLP focus) or NeurIPS / ICLR (reasoning/decision focus) or AAAI (AI focus)

---

## 1. Motivation and Gap Analysis

The existing work (Remez 2025) establishes a functional ACH-LLM pipeline and demonstrates it can beat the MIRAI baseline. However, several critical weaknesses prevent it from being publication-ready at top venues:

| Gap | Impact on quality |
|---|---|
| Heuristic qualitative-to-numeric mapping (hardcoded 0.9/0.66/...) | No principled uncertainty; uncalibrated probabilities |
| Naive Bayes aggregation assumes indicator independence | Indicators are correlated in practice; model misspecified |
| Evaluation on small subset; no significance tests | Claims may not generalize; no statistical rigor |
| No comparison to strong LLM baselines (CoT, ReAct, direct prompting) | Hard to isolate what ACH structure contributes |
| Deep Analysis is a heuristic post-process, not integrated | Sub-optimal; doesn't generalize naturally |
| Abstract-only context | Information bottleneck limiting a fair comparison |
| No ablation study on each pipeline component | Unclear which components drive performance |

---

## 2. Proposed Research Direction

**Title (draft)**: "Structured Evidence Aggregation for LLM-Based Geopolitical Forecasting via Analysis of Competing Hypotheses"

**One-sentence thesis**: Encoding the ACH framework as a calibrated probabilistic inference problem — rather than a heuristic pipeline — yields better-calibrated, more accurate, and interpretable event forecasts than chain-of-thought and retrieval-augmented baselines.

### Core Contributions (planned)

**C1. Principled probabilistic formulation of ACH**
- Replace the hardcoded qualitative-to-numeric mapping with a **learned calibration layer**: train a small model (logistic regression or MLP) on a calibration set to map LLM verbalized uncertainty to probabilities
- Model indicator correlations via a simple **latent factor model** (or Markov blanket structure) rather than flat Naive Bayes
- Provide confidence intervals on forecasts via bootstrapping over retrieved articles

**C2. Multi-agent adversarial ACH (Agent ACH)**
- Assign one LLM agent per hypothesis; each agent selectively advocates for its hypothesis, searching for supporting evidence and challenging opponents
- A "judge" agent scores arguments and produces final probability estimates
- This operationalizes ACH's original spirit (competing hypotheses, not just passive scoring) and aligns with the multi-agent LLM literature

**C3. Retrieval-Integrated Indicator Learning**
- Instead of using fixed LLM-generated indicators for all queries, learn which indicator types correlate with correct forecasts from the training portion of MIRAI
- Use this signal to **re-weight** or **re-rank** indicators at inference time (lightweight, no fine-tuning of the LLM required)

**C4. Rigorous benchmarking and ablation**
- Full MIRAI test set (not a subset); binary + quad classification
- Statistical significance testing (bootstrap confidence intervals, McNemar's test)
- Ablation: indicator generation, retrieval method, aggregation scheme, Deep Analysis — each removed independently
- Baselines: GPT-4 direct prompting, CoT, ReAct (from MIRAI paper), RAG-only without ACH structure

**C5. Calibration analysis**
- Primary metric: Expected Calibration Error (ECE) alongside KL divergence
- Reliability diagrams comparing ACH-LLM vs direct prompting baselines
- Show that structured ACH reasoning produces better-calibrated probability estimates — this is the key scientific claim

---

## 3. Research Questions

1. Does ACH structure improve **calibration** (not just accuracy) compared to unstructured LLM forecasting?
2. Which ACH component contributes the most: indicator generation, influence scoring, retrieval, or aggregation?
3. Does multi-agent adversarial ACH outperform single-agent ACH, and at what additional compute cost?
4. How does the pipeline generalize beyond geopolitics (e.g., financial event forecasting, epidemiological forecasting)?

---

## 4. Experiment Plan

### 4.1 Baseline Experiments (Reproduce + Extend)

**Goal**: Replicate Remez 2025 and establish a clean, reproducible baseline.

| Exp ID | Name | Description |
|---|---|---|
| B1 | Direct Prompting | GPT-4 with no retrieval, no ACH structure; just "predict the event type" |
| B2 | CoT Prompting | Chain-of-thought: ask GPT-4 to reason step-by-step before predicting |
| B3 | RAG-only | Retrieve articles, concatenate into context, ask GPT-4 to predict |
| B4 | ReAct (MIRAI) | Re-run MIRAI's best ReAct agent on same queries for direct comparison |
| B5 | ACH-basic | Replicate Remez's Exp 1 exactly (manual doc IDs + simple RAG) |
| B6 | ACH-improved | Replicate Remez's Exp 2-3 (enhanced RAG + fine-tuned indicators + Deep Analysis) |

**Metrics**: Precision, Recall, F1, Accuracy, KL Divergence, ECE (new), runtime

### 4.2 Calibration Experiments

**Goal**: Test C1 — does principled calibration improve probability quality?

| Exp ID | Name | Description |
|---|---|---|
| C1a | Calibration mapping | Replace hardcoded values with Platt scaling on a held-out calibration split |
| C1b | Temperature scaling | Apply temperature scaling to LLM log-probs before mapping to probabilities |
| C1c | Isotonic regression | Non-parametric calibration on the qualitative→numeric mapping |
| C1d | Ensemble calibration | Average predictions from K=5 indicator sets to reduce variance |

**Metrics**: ECE, KL Divergence, reliability diagram, sharpness

### 4.3 Multi-Agent ACH Experiments

**Goal**: Test C2 — does adversarial debate improve accuracy?

| Exp ID | Name | Description |
|---|---|---|
| M1 | Single-agent ACH | Baseline: one LLM handles all hypotheses (current system) |
| M2 | Agent-per-hypothesis | Assign one GPT-4 instance per hypothesis; each retrieves supporting evidence |
| M3 | Adversarial debate (1 round) | Each agent also challenges the top competitor; judge aggregates |
| M4 | Adversarial debate (2 rounds) | Two rounds of debate; tests diminishing returns vs. cost |
| M5 | Agent diversity | Use different LLMs for different roles (GPT-4 agents, Claude judge) |

**Metrics**: Accuracy, F1, cost (number of API calls), latency

### 4.4 Indicator Learning Experiments

**Goal**: Test C3 — can we learn which indicator types are predictive?

| Exp ID | Name | Description |
|---|---|---|
| I1 | Fixed indicators (baseline) | Current approach: same LLM prompt for all queries |
| I2 | Indicator type clustering | Cluster indicators across training queries; keep cluster centroids |
| I3 | Indicator re-weighting | Learn scalar weights per indicator cluster from training accuracy |
| I4 | Dynamic indicator selection | At inference time, use training-derived indicator-type priors to bias generation |

**Metrics**: F1, accuracy, indicator diversity, retrieval relevance

### 4.5 Ablation Study

**Goal**: Test C4 — quantify each component's contribution.

Run the full best system and remove one component at a time:

| Ablation | Removed component | Expected effect |
|---|---|---|
| A1 | No indicators (random retrieval) | Large drop in retrieval relevance |
| A2 | No influence matrix (uniform I) | Moderate drop; tests if scoring matters |
| A3 | No Deep Analysis | Some drop in Verbal/Material disambiguation |
| A4 | No MMR/RRF | Drop in retrieval diversity/quality |
| A5 | No time decay | Drop when temporal proximity is signal |
| A6 | No article retrieval (indicators only) | Tests if retrieval adds beyond prior knowledge |

### 4.6 Generalization Experiments (Stretch Goal)

**Goal**: Show the framework is domain-agnostic.

| Domain | Dataset | Task |
|---|---|---|
| Financial | ACL18/BigData22 stock news | Predict next-day price direction |
| Medical | CDC weekly reports | Forecast outbreak escalation/decline |
| Conflict (fine-grained) | MIRAI level-2 codes (38 classes) | Full CAMEO relation prediction |

---

## 5. Technical Improvements to Implement

### 5.1 Replace abstract-only with full article chunks
- Chunk full articles into 512-token segments
- Retrieve top-3 chunks per article rather than the abstract
- Expected impact: +3-5% F1 based on the limitation analysis

### 5.2 Structured output and parsing
- Use GPT-4 JSON mode / function calling for all LLM prompts to avoid parsing failures
- Log and analyze parsing failure rates as a reliability metric

### 5.3 Indicator balance enforcement
- Current fine-tuned prompt asks for balance but doesn't enforce it
- Post-process: if any hypothesis has 0 indicators, resample
- Analyze correlation between indicator balance and accuracy

### 5.4 Reproducibility infrastructure
- Fix random seed for indicator generation (temperature=0 or low temp)
- Cache all LLM calls to enable reruns without API cost
- Open-source the full pipeline on GitHub with Docker

---

## 6. Evaluation Protocol

### Dataset Split
- MIRAI provides a fixed test set; use it entirely (do not subsample)
- Use first 20% of test queries as calibration set for C1 experiments
- Report both binary and quad classification

### Statistical Rigor
- Bootstrap confidence intervals (N=1000) on all metrics
- McNemar's test for pairwise system comparisons
- Bonferroni correction for multiple comparisons across ablations

### Compute Budget Estimate
| Experiment Group | Queries | LLM calls/query (est.) | Total calls | Cost est. (GPT-4) |
|---|---|---|---|---|
| Baselines (B1-B6) | 100 | 5-30 | ~3,000 | ~$15 |
| Calibration (C1a-d) | 100 | 30 | ~3,000 | ~$15 |
| Multi-agent (M1-M5) | 100 | 50-150 | ~10,000 | ~$50 |
| Ablations (A1-A6) | 100 | 30 | ~18,000 | ~$90 |
| **Total estimate** | | | **~35,000** | **~$170** |

Cost can be reduced by using GPT-3.5-turbo for non-critical steps, or Claude Haiku for initial experiments.

---

## 7. Narrative and Claims for Paper

**Claim 1 (main)**: Structuring LLM reasoning via ACH evidence matrices significantly improves calibration (ECE) on geopolitical forecasting compared to CoT and RAG baselines, while remaining competitive or superior on accuracy.

**Claim 2**: Calibrating the qualitative-to-numeric mapping with a held-out set reduces KL divergence by X% over the heuristic mapping.

**Claim 3**: Multi-agent adversarial ACH improves accuracy on ambiguous queries (those where the top-2 hypotheses are closely scored) with manageable compute overhead.

**Claim 4**: The pipeline is interpretable by design — the indicator/influence matrices provide human-readable audit trails of how each article contributed to the forecast.

---

## 8. Recommended Timeline

| Phase | Duration | Milestones |
|---|---|---|
| Setup and replication | 2 weeks | Reproduce Remez results; full MIRAI eval; open-source code |
| Baselines B1-B4 | 1 week | Clean comparison with direct prompting / CoT / RAG |
| Calibration experiments C1 | 2 weeks | Calibrated mapping; ECE analysis |
| Ablation study | 1 week | Identify key components |
| Multi-agent experiments | 3 weeks | M1-M4; cost-accuracy tradeoff analysis |
| Indicator learning | 2 weeks | I1-I4 |
| Generalization experiments | 2 weeks | Finance / full CAMEO if compute allows |
| Writing and submission | 4 weeks | Target: ACL 2026 (Feb deadline) or EMNLP 2026 |

---

## 9. Positioning vs. Related Work

| Related work | How we differ |
|---|---|
| MIRAI benchmark (Qi et al. 2024) | We propose a new pipeline architecture, not just another agent benchmark entry |
| ReAct (Yao et al. 2022) | ACH provides structure over what evidence to collect and how to aggregate it |
| Multi-agent debate (Du et al. 2023) | We apply adversarial debate specifically to structured hypothesis competition |
| ForecastBench (Zou et al. 2024) | MIRAI focuses on geopolitics with CAMEO ontology; different forecasting regime |
| LLM calibration (Kadavath et al. 2022) | We study calibration in structured multi-hypothesis forecasting, not single QA |
