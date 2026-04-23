# Contribution and Dataset Analysis

## A. What Can We Invent / Contribute for Greater Novelty and Better Results

---

### A.1 Diagnostic Evidence Weighting (Highest Priority — Novel Theory)

**The idea**: ACH's original theoretical insight is that "diagnostic evidence" — evidence that supports one hypothesis while *disconfirming* others — is more valuable than evidence that is consistent with all hypotheses. The current A·I aggregation ignores this entirely: an indicator that scores 0.8 for all four hypotheses is treated the same as one that scores 0.9 for VK and 0.1 for all others.

**Formalization**:

Define the *diagnosticity* of indicator $j$ as the information gain it provides:
```
d_j = H(H) - H(H | f_j)   [reduction in entropy over hypotheses]
     ≈ max_k I_jk - mean_k I_jk   [simple proxy]
```

Replace the aggregation with diagnosticity-weighted scoring:
```
Score(h_k) = (1/n) Σ_i Σ_j  d_j · A_ij · I_jk
```

**Why novel**: No current paper formalizes diagnostic evidence weighting in LLM forecasting. AgentCDM doesn't do it. ThinkTank-ME doesn't do it. This connects to information-theoretic fundamentals (mutual information), maps directly to ACH's theoretical foundations, and gives us a clean theoretical contribution alongside the empirical one.

**Expected gain**: Diagnostically-weighted indicators should cut noise from uninformative indicators. Estimate +3-5% F1, and a cleaner ablation story.

---

### A.2 Absence-of-Evidence Modeling (Novel Mechanism)

**The idea**: ACH explicitly treats the *absence* of expected evidence as informative — a key principle missing from all current LLM forecasting work. If hypothesis H predicts that indicator $f$ should be observable, and $f$ does not appear in any retrieved article, that is evidence *against* H.

**Formalization**: Extend the analysis matrix to include a "background absence" probability:

```
Ã_ij = A_ij                      if article i is retrieved
Ã_ij = 1 - P(f_j | background)   if no article mentions f_j at all
```

Where $P(f_j | \text{background})$ is a prior estimated by asking the LLM: "In the current geopolitical context, how likely is indicator $f_j$ to be observable in news if it were happening?"

**Why novel**: No LLM forecasting system models absence of evidence. The closest is confidence-calibrated retrieval but it doesn't connect absence to hypothesis disconfirmation. This is a direct translation of ACH theory into math.

**Expected gain**: Particularly useful for Material Conflict queries where the absence of military movement news is informative. Estimate +2-4% F1 on conflict sub-classes.

---

### A.3 Contrastive Indicator Generation (Improved Prompting)

**The idea**: Replace generic "generate indicators for this outcome" with "generate indicators that most distinguish outcome A from outcome B" — directly targeting diagnostic evidence at the prompt level.

**Implementation**: For each pair of hypotheses $(h_a, h_b)$, generate $K/\binom{|\mathcal{H}|}{2}$ contrastive indicators:
> "Generate 2 observable indicators that would strongly distinguish between {VC} and {MC}. The indicator should have high probability under one and low probability under the other."

This produces a richer, more diverse indicator set with guaranteed per-pair coverage.

**Why novel**: Closest work (AutoCast++) uses generic retrieval queries. No paper applies contrastive prompting to indicator design. The combinatorial pair coverage connects to tournament-style hypothesis elimination.

**Expected gain**: Better indicator diagnosticity. Reduced indicator redundancy. Estimate +3-5% F1.

---

### A.4 Theoretical Calibration Analysis (Novel Theory Contribution)

**The idea**: Prove (analytically, under reasonable assumptions) *why* ACH structure improves calibration compared to free-form LLM reasoning.

**Sketch of argument**: Under the assumption that each LLM sub-query (indicator scoring, article analysis) has independent, well-distributed noise with variance $\sigma^2$, the matrix aggregation averages $n \times m$ independent estimates. By the law of large numbers, the final score variance decreases as $\sigma^2 / (nm)$. In contrast, a single CoT query has variance $\sigma^2$ — effectively one sample.

This means ACH is a *variance reduction* mechanism for LLM uncertainty. Better calibration follows from lower variance around the correct probability.

**Formalization**: Compute the theoretical ECE upper bound for ACH vs. CoT as a function of $n$, $m$, and $\sigma$. Validate empirically by sweeping $n$ (number of articles) and $m$ (number of indicators) and showing ECE follows the predicted $1/\sqrt{nm}$ curve.

**Why novel**: No paper has characterized the calibration properties of structured reasoning analytically. This would be the first theoretical result linking reasoning structure to calibration, and it would be broadly applicable beyond our specific application.

---

### A.5 Multi-Domain Generalization (Breadth Contribution)

**The idea**: Apply the exact same pipeline — with domain-specific indicator prompts — to forecasting tasks outside geopolitics.

**Target domains**:
- **Financial**: Predict earnings surprise direction (beat/miss/in-line) given news articles preceding earnings release
- **Epidemiological**: Predict outbreak trajectory (escalating/stable/declining) given WHO/CDC reports
- **Supply chain**: Predict disruption event type given logistics news

**Why novel**: ThinkTank-ME is domain-specific (Middle East). AgentCDM is domain-agnostic but doesn't do forecasting. We would be the first to show ACH structure works across forecasting domains, which is a strong generalization claim for top venues (NeurIPS/ICLR love domain-agnostic methods).

**Expected outcome**: ACH-LLM should transfer cleanly because the pipeline is domain-parameterized only by the hypothesis set $\mathcal{H}$ and indicator generation prompt.

---

### A.6 Cognitive Bias Quantification Experiments (Novel Empirical Claim)

**The idea**: ACH was invented to reduce cognitive biases — specifically confirmation bias, anchoring, and availability bias. We can design controlled experiments to measure whether LLM-ACH actually reduces these biases compared to CoT.

**Protocol**:
- **Anchoring bias test**: Present the LLM with the same query but prime it with the "wrong" hypothesis as a system message. Measure how much the prediction shifts vs. ACH (which should be more robust because it evaluates all hypotheses from scratch).
- **Confirmation bias test**: Provide articles that predominantly support hypothesis A, but where the ground truth is B. Measure over-prediction of A in CoT vs. ACH.
- **Availability bias test**: Weight article retrieval toward more salient/sensational events. Measure prediction shift.

**Why novel**: No LLM forecasting paper measures cognitive bias reduction. This connects directly to the ACH motivation and would be a compelling human-analogous result. Very attractive to venues like NeurIPS that care about AI behavior and safety.

---

### A.7 Temporal Consistency Metric (New Evaluation Dimension)

**The idea**: For the same country pair over multiple time periods, do our probability estimates form a coherent temporal narrative? Good forecasting systems should show smooth, causally-interpretable probability evolution, not random jumps.

**Metric**: Define *temporal consistency* as the auto-correlation of predicted probability distributions over time:
```
TC = (1/T) Σ_t  KL(p_t || p_{t-1})   [should be low for smooth evolution]
```

**Why novel**: No current benchmark measures temporal consistency. MIRAI treats each query independently. This is a new dimension that ACH-LLM should excel at because the indicator structure provides continuity across time.

---

## B. Dataset Recommendations

---

### B.1 Primary: MIRAI (Already Planned)

**Why**: Structured CAMEO ontology, ground-truth from GDELT, 100+ test queries, established baseline comparisons (ReAct etc.), already used by related work so comparisons are direct.

**Best for proving**: Structured reasoning beats free-form reasoning on multi-class geopolitical forecasting.

**Limitation**: Small test set (~100 queries), single domain.

---

### B.2 ForecastBench (ICLR 2025) — Calibration Champion

**What it is**: Dynamic benchmark of 1,000 continuously-updated forecasting questions from prediction markets (Metaculus, GJOpen), ACLED conflict data, economics (FRED, DBnomics), and Wikipedia. The best LLM (GPT-4.5) achieves Brier score 0.101 vs. superforecasters' 0.081.

**Why ideal for us**: 
- Ground-truth probability estimates are available (from crowd forecasters), making ECE computation meaningful
- Diverse domains — proves generalization
- Binary + MCQ formats
- Large enough for statistical significance
- Publicly available at [forecastbench.org](https://www.forecastbench.org/)

**Best for proving**: ACH improves calibration (our central novel claim). The crowd-forecaster baseline gives us a human calibration target to compare against.

**Proposed experiment**: Run ACH-LLM on a 200-question subset of ForecastBench (geopolitics + conflict categories) and compare Brier score and ECE vs. direct prompting and CoT. Show ACH closes the LLM-to-superforecaster gap.

---

### B.3 AutoCast / AutoCast++ — Generalization Benchmark

**What it is**: Static benchmark of forecasting questions from Metaculus, INFER, GJOpen. MCQ and True/False formats. Existing results from many systems published.

**Why useful**: 
- Existing competitive baselines already published (AutoCast++ improves MCQ by 48%)
- Well-understood difficulty
- Can compare directly to published numbers without needing to run their systems

**Best for proving**: ACH generalizes to general (non-geopolitical) forecasting questions beyond MIRAI.

**Limitation**: Not calibration-focused, smaller than ForecastBench.

---

### B.4 POLECAT / POLECAT-FOR-ME — Head-to-Head with ThinkTank-ME

**What it is**: Political Event Coding Across Topics dataset, with a Middle East subset (POLECAT-FOR-ME) created by ThinkTank-ME.

**Why useful**: Running on POLECAT-FOR-ME allows a direct comparison with ThinkTank-ME (which claims to be first on their own dataset). If we outperform their specialized multi-expert approach without any domain-specific fine-tuning, that's a strong result.

**Best for proving**: ACH-LLM is competitive with or better than specialized domain-tuned systems.

---

### B.5 Custom Binary Calibration Dataset (Construct from GDELT)

**What it is**: Extract 500+ binary geopolitical predictions from GDELT/ACLED that have resolved outcomes. For each: "Will {country A} take military action against {country B} in the next 30 days?" (yes/no with probability estimate).

**Why useful**: Binary probability calibration is much easier to measure and visualize (reliability diagrams). Gives the cleanest demonstration of our calibration claim.

**Effort**: Moderate — GDELT is freely available, event extraction is automated.

---

### B.6 Dataset Summary Table

| Dataset | Domain | Size | Multi-class | Calibration GT | Novelty value |
|---|---|---|---|---|---|
| MIRAI | Geopolitics | ~100 | Yes (4-class) | No | Primary — accuracy |
| ForecastBench | General | 1000+ | Yes | Yes (crowd) | Primary — calibration |
| AutoCast | General | 2000+ | MCQ + TF | Partial | Generalization |
| POLECAT-FOR-ME | Middle East | ~500 | Yes | No | Comparison with ThinkTank-ME |
| GDELT binary | Geopolitics | 500+ | No (binary) | Constructable | Calibration visualization |

**Recommended minimum for top venue**: MIRAI (primary) + ForecastBench (calibration) + one of AutoCast or POLECAT.

---

## C. Summary: Priority Ranking of New Contributions

| Priority | Contribution | Novelty | Expected gain | Effort |
|---|---|---|---|---|
| 1 | Diagnostic evidence weighting (A.1) | High — novel theory | +3-5% F1 | Low |
| 2 | Theoretical calibration analysis (A.4) | Very high — first theory | ECE proof | Medium |
| 3 | Absence-of-evidence modeling (A.2) | High — novel mechanism | +2-4% F1 | Medium |
| 4 | ForecastBench calibration experiments (B.2) | High — new benchmark | Calibration story | Medium |
| 5 | Contrastive indicator generation (A.3) | Medium-high | +3-5% F1 | Low |
| 6 | Cognitive bias quantification (A.6) | High — novel eval | Compelling story | Medium |
| 7 | Multi-domain generalization (A.5) | Medium | Breadth claim | High |
| 8 | Temporal consistency metric (A.7) | Medium | New eval axis | Low |

**Minimum viable top-venue package**: Items 1 + 2 + 4 + 5, evaluated on MIRAI + ForecastBench. This gives: (a) a novel theoretical contribution, (b) two novel mechanisms, (c) rigorous calibration evaluation on a second dataset, and (d) state-of-the-art numbers.
