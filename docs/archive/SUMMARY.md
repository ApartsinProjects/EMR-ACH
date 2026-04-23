# ACH-LLM Geopolitical Event Forecasting: Project Summary

**Source**: Ben Remez, MSc Final Report — "Automating Structural Analytics Techniques for Event Forecasting using Large Language Models"

---

## Core Idea

Automate the **Analysis of Competing Hypotheses (ACH)** framework using LLMs to forecast geopolitical events from news articles. ACH is a CIA-developed structured reasoning technique that reduces cognitive bias by systematically evaluating evidence against multiple competing hypotheses. The original method is manual and error-prone; this work replaces the human analyst with an LLM-driven pipeline.

---

## Task Formulation

- **Dataset**: MIRAI benchmark — geopolitical events represented as quadruples: `(timestamp, subject_country, relation, object_country)`
- **Classification target**: Quad-class relation type from the CAMEO ontology:
  - Verbal Cooperation (codes 01-04)
  - Material Cooperation (codes 05-08)
  - Verbal Conflict (codes 09-16)
  - Material Conflict (codes 17-20)
- **Input**: A forecasting query (e.g., `"2023-11-04, Israel, Accuse, Palestine"`) + news corpus
- **Output**: Ranked list of the 4 hypotheses by predicted likelihood

---

## Pipeline Architecture (6 Steps)

1. **Generate indicators**: Prompt GPT-4 to produce K "early-sign" indicators per hypothesis (balanced: at least one per outcome, no overlap)
2. **Score indicator-hypothesis influence**: For each (indicator `f`, hypothesis `h`) pair, LLM rates likelihood qualitatively; mapped to numeric probabilities:
   - `"highly likely"` → 0.9, `"likely"` → 0.66, `"possible"` → 0.33, `"unlikely"` → 0.0, `"highly unlikely"` → 0.1
   - Result: **Influence matrix** `I ∈ R^{m×h}`
3. **Retrieve relevant articles**: Either manually (from MIRAI-provided doc IDs) or via RAG (Weaviate vector DB)
4. **Estimate indicator presence in each article**: LLM reads each article abstract and rates whether each indicator is present (same qualitative scale → numeric)
   - Result: **Analysis matrix** `A ∈ R^{n×m}`
5. **Aggregate scores**:

   ```
   P = A · I        (prediction matrix, n×h)
   Score(h) = (1/n) * Σ_i Σ_j  A_ij · I_jh
   ```

6. **Rank hypotheses** by score; top-ranked = predicted outcome

### RAG Infrastructure

- **Weaviate** vector database with hybrid retrieval
- **LangChain** orchestration with:
  - Metadata filtering (country names, date range)
  - **MMR** (Maximal Marginal Relevance) re-ranking for diversity
  - **RRF** (Reciprocal Rank Fusion) to combine retrieval signals
  - Exponential time-decay weighting (α = 0.015, 30-day window)

---

## Deep Analysis (Second-Stage Reasoning)

**Problem**: The pipeline struggled to distinguish Verbal vs Material categories when articles had mixed or ambiguous signals.

**Solution**: After the main pipeline ranks hypotheses, a second LLM pass reads each retrieved article and classifies it as:
- `"Verbal"` (High or Low certainty)
- `"Material"` (High or Low certainty)
- `"Uncertain"` (ignored)

**Scoring**: High → 1 pt, Low → 0.5 pt. Category with higher total score wins. Ties retain the main pipeline's top result.

---

## Experiments (Three Iterations)

| Experiment | Retrieval | Indicators | Deep Analysis |
|---|---|---|---|
| 1. Baseline | Simple Weaviate / Manual doc IDs | Basic (unbalanced) | No |
| 2. Improved | LangChain + Weaviate + MMR + RRF + time decay | Fine-tuned (balanced) | No |
| 3. Deep Analysis | Same as Exp 2 | Fine-tuned | Yes |

---

## Results (Quad Classification)

| Method | Precision (↑) | Recall (↑) | F1 (↑) | KL Div (↓) | Accuracy (↑) |
|---|---|---|---|---|---|
| **RAG Improved + DA** | 50.4% | 51.1% | **50.5%** | **0.014** | 54% |
| RAG Improved | 51.4% | 43.2% | 34.7% | 5.209 | 53% |
| RAG (baseline) | 22.1% | 37.0% | 27.5% | 10.643 | 43% |
| **Manual + DA** | **66.7%** | **54.2%** | 49.5% | 0.312 | **56%** |
| Manual FT | 50.4% | 41.5% | 33.3% | 0.880 | 50% |
| Manual (baseline) | 23.2% | 38.8% | 28.8% | 10.645 | 45% |
| **MIRAI baseline** | 47.6% | 58.3% | 44.2% | 5.9 | 42% |

**Key findings**:
- Deep Analysis is critical: it collapsed KL divergence from ~10.6 to 0.014 for RAG, and from 10.6 to 0.312 for Manual
- RAG Improved DA and Manual DA both **surpass MIRAI** on precision, F1, and accuracy
- Manual DA leads overall but requires curated document selection (not scalable)
- RAG Improved DA is the best practical system — fully automated, competitive performance

---

## Limitations

- Only **article abstracts** used (not full text) due to token/cost constraints on Azure Student OpenAI API
- Experiments limited to the **quad classification** (4 classes); the full 38 CAMEO codes not evaluated
- Threshold-based selective Deep Analysis was attempted (apply DA only when top-2 scores are close) but **unexpectedly failed**: lower deltas correlated with *higher* accuracy, making the threshold approach ineffective

---

## Future Work (Suggested in Report)

1. Benchmark on larger and more diverse datasets
2. Refine RAG retrieval further (full articles, better chunking)
3. Explore **multi-agent LLM architectures** for collaborative forecasting
4. Full article analysis instead of abstract-only
5. Extend to all 38 CAMEO second-level relations
6. Optimize prompt engineering across all pipeline steps

---

## Reproduction Checklist

- [ ] Obtain MIRAI benchmark dataset (test queries + `relation_query.csv` doc IDs)
- [ ] Build Weaviate news corpus (indexed article abstracts with metadata)
- [ ] Implement 6-step pipeline with LangChain + Weaviate + GPT-4
- [ ] Run Experiment 1 (baseline) to verify reported numbers
- [ ] Run Experiment 2 (improved RAG + fine-tuned indicators)
- [ ] Run Experiment 3 (add Deep Analysis step)
- [ ] Compare against MIRAI baseline numbers

---

## Tech Stack

- **LLM**: GPT-4 via Azure OpenAI
- **Orchestration**: LangChain
- **Vector DB**: Weaviate (hybrid retrieval)
- **Retrieval techniques**: MMR, RRF, exponential time decay
- **Evaluation metrics**: Precision, Recall, F1, Accuracy, KL Divergence
- **Benchmark**: MIRAI (quad-class relation forecasting)
