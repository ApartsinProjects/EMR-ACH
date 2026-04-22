# Baselines: Methods Reference

This document describes each of the nine baseline methods in the battery: their provenance, how they are implemented here, their configuration knobs, their compute cost per forecast-document (FD), and their known limitations.

## Summary table

| ID | Method | Calls per FD | Main config knob | Citation |
| --- | --- | --- | --- | --- |
| B1 | Direct prompting | 1 | (none) | Brown et al. 2020 (GPT-3) |
| B2 | Chain-of-Thought | 1 | (none) | Wei et al. 2022 (arXiv:2201.11903) |
| B3 | RAG-only | 1 | `max_articles` | Lewis et al. 2020 (arXiv:2005.11401) |
| B4 | Self-Consistency | `n_samples` | `n_samples` (default 8) | Wang et al. 2022 (arXiv:2203.11171) |
| B5 | Multi-Agent Debate | `n_agents × n_rounds` | `n_agents`, `n_rounds` | Du et al. 2023 (arXiv:2305.14325) |
| B6 | Tree of Thoughts | up to `breadth^depth` | `breadth`, `depth` | Yao et al. 2023 (arXiv:2305.10601) |
| B7 | Reflexion | `2·n_iterations − 1` | `n_iterations` | Shinn et al. 2023 (arXiv:2303.11366) |
| B8 | Verbalized Confidence | 1 | (none, RAG context) | Lin et al. 2022 (arXiv:2205.14334); Tian et al. 2023 (arXiv:2305.14975) |
| B9 | Heterogeneous LLM Ensemble | `len(configs)` (default 6) | `configs` list | Jiang et al. 2023 (LLM-Blender, arXiv:2306.02561) |

All baselines share the same FD schema, JSON response contract (`probabilities`, `prediction`, `reasoning`), and fairness-enforced defaults (same model, same temperature unless explicitly configured). Only B9 is permitted to deviate from the default model, and it does so explicitly via the `configs` list.

---

## B1: Direct Prompting

**One-line summary.** Zero-shot direct prompting: model is asked to produce a probability distribution from the question and hypothesis set alone, with no articles and no reasoning scaffold.

**Citation.** Brown et al. 2020, "Language Models are Few-Shot Learners" (arXiv:2005.14165). Baseline representative of the simplest possible prompting strategy.

**Method description.** For each FD we build a single prompt containing the question, background, hypothesis set, and the fixed JSON response schema. No articles are included ("(not provided; use general knowledge)"). The model is expected to internalize world knowledge from pre-training and produce a calibrated distribution.

This baseline is useful as a lower bound: if B1 matches a more complex method, the complexity is not buying anything. It is also the cheapest baseline per FD.

**Config parameters.** None beyond shared defaults (`model`, `temperature`, `max_tokens`).

**Compute cost.** 1 call per FD.

**Known limitations.** No access to news context, so any recent-event benchmark (ForecastBench, GDELT-CAMEO near cutoff) will be systematically mis-calibrated for events after the model's pre-training cutoff.

**Expected output.** One prediction row per FD with a single probability distribution.

---

## B2: Chain-of-Thought

**One-line summary.** Same inputs as B1, but the instruction block asks the model to think step-by-step before emitting the final probabilities.

**Citation.** Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (arXiv:2201.11903).

**Method description.** Identical request structure to B1 with the CoT instruction block substituted: the model is directed to consider base rates, strongest evidence for each hypothesis, disconfirming evidence, and recency before outputting probabilities. The reasoning is kept in the `reasoning` field of the JSON response; final probabilities are still produced in one shot.

**Config parameters.** None beyond shared defaults.

**Compute cost.** 1 call per FD.

**Known limitations.** At temperature 0, CoT can produce deterministic but biased reasoning; any single-sample CoT inherits the same uncertainty as B1. Also no articles, so same recency limitation as B1.

**Expected output.** Single probability distribution per FD.

---

## B3: RAG-only

**One-line summary.** Concatenate the FD's retrieved articles into the prompt and make a single prediction call. No reasoning scaffold, no multi-sample aggregation.

**Citation.** Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (arXiv:2005.11401). Our formulation uses frozen retrieval (article_ids pre-computed by the benchmark's retriever), not a learned retriever.

**Method description.** For each FD we build the articles block (up to `max_articles`, each truncated to ~900 chars) and render it into the shared prompt template along with the question and hypothesis set. The instruction block directs the model to use only the provided articles and background. One call per FD.

**Config parameters.**
- `max_articles` (default 10). Number of top-ranked articles to include from `fd.article_ids`.

**Compute cost.** 1 call per FD.

**Known limitations.** Pure RAG without reasoning is sensitive to retrieval quality; irrelevant articles can crowd out the signal. No uncertainty quantification.

**Expected output.** Single probability distribution per FD.

---

## B4: Self-Consistency

**One-line summary.** Draw `n_samples` CoT completions at temperature > 0, parse each to a probability distribution, average over samples.

**Citation.** Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (arXiv:2203.11171).

**Method description.** We issue `n_samples` independent CoT calls per FD, each at the configured temperature (default 0.7). Every sample is parsed independently via the shared probability parser; the final distribution is the arithmetic mean of the sample distributions, re-normalized to sum to 1.

This turns CoT's variance into a calibration signal: if the model is consistent, the average collapses toward a confident distribution; if it is uncertain, the average spreads.

**Config parameters.**
- `n_samples` (default 8). Number of independent samples per FD.
- `temperature` (default 0.7). Must be > 0 for sampling to be meaningful.
- `max_articles` (default 10). Articles included in each sample.

**Compute cost.** `n_samples` calls per FD. At `n_samples=8` this is 8× B3.

**Known limitations.** Cost scales linearly with samples. Averaging probabilities is only one aggregation strategy; majority-vote on argmax classes is an alternative not used here.

**Expected output.** Probability distribution averaged across `n_samples` CoT samples; extras include `n_samples`.

---

## B5: Multi-Agent Debate

**One-line summary.** `n_agents` agents each produce an independent CoT forecast, then revise over `n_rounds − 1` subsequent rounds given the prior round's peer responses. Final distribution is the mean over agents in the last round.

**Citation.** Du et al. 2023, "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (arXiv:2305.14325).

**Method description.** Round 0: each of `n_agents` agents receives the same prompt at temperature ≥ 0.4 and independently produces a forecast. Round `r > 0`: each agent sees the truncated content of every agent's round-`r-1` response embedded in the instruction block, and is asked to revise. The runner dispatches each round as its own Batch API submission (one batch per round) because later rounds depend on earlier results.

**Config parameters.**
- `n_agents` (default 3). Number of debating agents per FD.
- `n_rounds` (default 2). Total rounds including round 0.
- `max_articles` (default 10).

**Compute cost.** `n_agents × n_rounds` calls per FD. Default config = 6 calls.

**Known limitations.** Agents are not heterogeneous (same model, same temperature) in this implementation, so "debate" is really sampling-with-revision. True heterogeneous debate is left to B9. Context grows each round because prior responses are embedded.

**Expected output.** Probability distribution averaged across agents in the final round; extras include `n_agents` and `n_rounds`.

---

## B6: Tree of Thoughts

**One-line summary.** At depth 0 generate `breadth` diverse reasoning seeds; at each subsequent depth expand each seed into `breadth` children terminating with a probability distribution. Final distribution is the mean over all leaves.

**Citation.** Yao et al. 2023, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (arXiv:2305.10601). We use a simplified batch-friendly variant without explicit pruning.

**Method description.** Depth 0: for each FD, issue `breadth` seed-thought calls at high temperature. Depth `d > 0`: for each parent seed, issue `breadth` expansion calls that build on the parent's reasoning and terminate with a JSON distribution. At `depth = 2` this produces `breadth × breadth` leaves per FD. The final distribution is the arithmetic mean of leaf distributions.

**Config parameters.**
- `breadth` (default 3). Fan-out per node.
- `depth` (default 2). Tree depth. `depth=1` is equivalent to B4 with `n_samples=breadth`.
- `max_articles` (default 10).

**Compute cost.** `breadth + breadth^2` calls per FD at `depth=2`. Default config = 3 + 9 = 12 calls.

**Known limitations.** No pruning: all leaves contribute equally to the average, which differs from the original ToT which prunes via a value function. Cost grows quickly with depth.

**Expected output.** Probability distribution averaged over all leaves; extras include `breadth` and `depth`.

---

## B7: Reflexion

**One-line summary.** Self-critique loop over `n_iterations` iterations: iteration 0 is an initial forecast; each subsequent iteration issues a critique call followed by a revise call.

**Citation.** Shinn et al. 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning" (arXiv:2303.11366).

**Method description.** Iteration 0: CoT forecast. Iteration `i > 0`: a critic (system prompt `SYSTEM_CRITIC`) is asked to list at most three concrete weaknesses in the previous forecast; then a revise call is issued that has access to both the prior forecast text and the critique framing. The final distribution is parsed from the last revise call.

**Config parameters.**
- `n_iterations` (default 3). Total iterations including iteration 0.
- `max_articles` (default 10).

**Compute cost.** `1 + 2 · (n_iterations − 1)` calls per FD. At default = 1 + 4 = 5 calls.

**Known limitations.** The critic and the forecaster are the same underlying model, so the critic's biases mirror the forecaster's. No explicit memory buffer across FDs (unlike the original Reflexion paper's use on agent tasks).

**Expected output.** Probability distribution from the final revise iteration; extras include `n_iterations`.

---

## B8: Verbalized Confidence

**One-line summary.** B3 (RAG) with an explicit verbalized-confidence calibration framing in the instruction block.

**Citation.** Lin et al. 2022, "Teaching Models to Express Their Uncertainty in Words" (arXiv:2205.14334); Tian et al. 2023, "Just Ask for Calibration" (arXiv:2305.14975). These papers establish that asking the model to state its confidence explicitly, and framing it as calibrated probability, improves ECE relative to implicit logit-based signals.

**Method description.** Identical to B3 but with an additional instruction asking the model to treat each probability as its verbalized confidence the hypothesis is true, to avoid 0.5 defaults and to avoid extreme values unless the evidence is overwhelming.

**Config parameters.**
- `max_articles` (default 10).

**Compute cost.** 1 call per FD.

**Known limitations.** Framing alone does not guarantee calibration; on binary benchmarks (ForecastBench) with a strong class imbalance, the model can still cluster around 0.5. Combining with B4-style sampling is left as future work.

**Expected output.** Single probability distribution per FD.

---

## B9: Heterogeneous LLM Ensemble

**One-line summary.** Issue one RAG call per `(model, temperature)` configuration in the `configs` list; average the resulting distributions.

**Citation.** Jiang et al. 2023, "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion" (arXiv:2306.02561). We use simple mean-of-probabilities rather than a learned fuser.

**Method description.** Unique among the baselines, B9 is allowed to deviate from the default model (`allow_model_override = True`). The default `configs` list has six members spanning two models (gpt-4o, gpt-4o-mini) × three temperatures (0.0, 0.5, 1.0). For each FD we issue one RAG call per member, parse each to a distribution, and average.

**Config parameters.**
- `configs`: list of `{model, temperature}` dicts. Default has 6 members.
- `max_articles` (default 10).

**Compute cost.** `len(configs)` calls per FD. Default = 6 calls.

**Known limitations.** The ensemble is cost-heterogeneous (gpt-4o and gpt-4o-mini are priced differently), so `configs` is not a clean single-knob experiment. Simple averaging ignores systematic biases that could be learned by a blender.

**Expected output.** Probability distribution averaged across ensemble members; extras include `n_members`.

---

## Cross-cutting notes

- **Fairness.** Except for B9, the runner enforces that `baseline.model == defaults.model`. This prevents accidental model drift across methods.
- **Parsing.** All methods share the `Baseline.parse_probabilities` helper, which first tries JSON parsing, then a regex fallback per hypothesis, then falls back to a uniform distribution. A uniform fallback is counted as a parse failure by the smoke summary.
- **Article budget.** All article-using methods cap at `max_articles` (default 10) and truncate each article body to ~900 chars. This is shared at the `prompts.build_articles_block` layer, not redefined per method.
- **Benchmark shape.** Prompts and parsing are hypothesis-set agnostic, so the same method file works for binary (ForecastBench Yes/No), 3-class (Earnings Beat/Meet/Miss), and 4-class (GDELT-CAMEO) FDs without modification.
