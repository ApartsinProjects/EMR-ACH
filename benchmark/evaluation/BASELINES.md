# Baselines: Methods Reference

This document describes each of the nine baseline methods in the battery: their provenance, how they are implemented here, their configuration knobs, their compute cost per forecast-document (FD), and their known limitations.

**v2.1 framing (2026-04-22).** All baselines are **pick-only**: every method returns one hypothesis label per FD (no probabilities, no confidence vector, no calibration). The primary target is binary `["Comply", "Surprise"]`; the legacy domain-specific multiclass label is preserved as `x_multiclass_*` for the ablation. The shared user prompt injects a `prior_expectation_block` so the model knows the status-quo expectation before being asked to confirm or contradict it. Multi-sample baselines (B4, B5, B6, B9) aggregate picks via plurality vote over `hypothesis_set`. B8 (Verbalized Confidence) is deprecated under pick-only and is replaced in the paper by a majority-class reference baseline.

## Summary table

| ID | Method | Calls per FD | Main config knob | Citation |
| --- | --- | --- | --- | --- |
| B1 | Direct prompting | 1 | (none) | Brown et al. 2020 (GPT-3) |
| B2 | Chain-of-Thought | 1 | (none) | Wei et al. 2022 (arXiv:2201.11903) |
| B3 | RAG-only | 1 | `max_articles` | Lewis et al. 2020 (arXiv:2005.11401) |
| B4 | Self-Consistency | `n_samples` | `n_samples` (default 4) | Wang et al. 2022 (arXiv:2203.11171) |
| B5 | Multi-Agent Debate | `n_agents × n_rounds` | `n_agents`, `n_rounds` | Du et al. 2023 (arXiv:2305.14325) |
| B6 | Tree of Thoughts | up to `breadth^depth` | `breadth`, `depth` | Yao et al. 2023 (arXiv:2305.10601) |
| B7 | Reflexion | `2·n_iterations − 1` | `n_iterations` | Shinn et al. 2023 (arXiv:2303.11366) |
| B8 | Verbalized Confidence (DEPRECATED) | 1 | (none, RAG context) | Lin et al. 2022 (arXiv:2205.14334); Tian et al. 2023 (arXiv:2305.14975) |
| B9 | Heterogeneous LLM Ensemble | `len(configs)` (default 3) | `configs` list | Jiang et al. 2023 (LLM-Blender, arXiv:2306.02561) |

All baselines share the same FD schema, the pick-only JSON response contract (`{"prediction": "<one of hypothesis_set>", "reasoning": "..."}`), and fairness-enforced defaults (same model, same temperature unless explicitly configured). Only B9 is permitted to deviate from the default model, and it does so explicitly via the `configs` list.

---

## B1: Direct Prompting

**One-line summary.** Zero-shot direct prompting: model is asked to pick one hypothesis from the question and hypothesis set alone, with no articles and no reasoning scaffold.

**Citation.** Brown et al. 2020, "Language Models are Few-Shot Learners" (arXiv:2005.14165). Baseline representative of the simplest possible prompting strategy.

**Method description.** For each FD we build a single prompt containing the question, background, hypothesis set, the prior-state expectation block, and the pick-only JSON response schema. No articles are included ("(not provided; use general knowledge)"). The model is expected to lean on pre-training knowledge and the status-quo expectation to pick Comply or Surprise.

This baseline is useful as a lower bound: if B1 matches a more complex method, the complexity is not buying anything. It is also the cheapest baseline per FD.

**Config parameters.** None beyond shared defaults (`model`, `temperature`, `max_tokens`).

**Compute cost.** 1 call per FD.

**Known limitations.** No access to news context, so on the change subset (where the status quo breaks) B1 has no signal beyond pre-training priors.

**Expected output.** One prediction row per FD with a single picked hypothesis label.

---

## B2: Chain-of-Thought

**One-line summary.** Same inputs as B1, but the instruction block asks the model to walk a structured ACH-style hypothesis comparison (support/disconfirm per hypothesis) before emitting the final pick.

**Citation.** Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (arXiv:2201.11903).

**Method description.** Identical request structure to B1 with the CoT instruction block substituted: the model lists supporting and disconfirming evidence for each candidate hypothesis, then picks the one whose evidence profile is most asymmetrically consistent. The reasoning is kept in the `reasoning` field of the JSON response; the final pick is still produced in one shot.

**Config parameters.** None beyond shared defaults.

**Compute cost.** 1 call per FD.

**Known limitations.** At temperature 0, CoT can produce deterministic but biased reasoning; any single-sample CoT inherits the same uncertainty as B1. Also no articles, so same recency limitation as B1.

**Expected output.** Single picked hypothesis label per FD.

---

## B3: RAG-only

**One-line summary.** Concatenate the FD's retrieved articles into the prompt and make a single prediction call. No reasoning scaffold, no multi-sample aggregation.

**Citation.** Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (arXiv:2005.11401). Our formulation uses frozen retrieval (article_ids pre-computed by the benchmark's retriever), not a learned retriever.

**Method description.** For each FD we build the articles block (up to `max_articles`, each truncated to ~900 chars) and render it into the shared prompt template along with the question and hypothesis set. The instruction block directs the model to use only the provided articles and background. One call per FD.

**Config parameters.**
- `max_articles` (default 10). Number of top-ranked articles to include from `fd.article_ids`.

**Compute cost.** 1 call per FD.

**Known limitations.** Pure RAG without reasoning is sensitive to retrieval quality; irrelevant articles can crowd out the signal.

**Expected output.** Single picked hypothesis label per FD.

---

## B4: Self-Consistency

**One-line summary.** Draw `n_samples` CoT completions at temperature > 0, parse each to a single picked hypothesis, return the plurality vote.

**Citation.** Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (arXiv:2203.11171).

**Method description.** We issue `n_samples` independent CoT calls per FD, each at the configured temperature (default 0.7) and with a per-sample seed so the Batch API does not deduplicate identical requests. Every sample is parsed independently to one hypothesis label; the final pick is the plurality vote over the `n_samples` picks (ties broken by `hypothesis_set` order).

**Config parameters.**
- `n_samples` (default 4). Number of independent samples per FD. Wang et al. use k=5; anything ≥3 is sufficient. Reduced from 8 for cost.
- `temperature` (default 0.7). Must be > 0 for sampling to be meaningful.
- `max_articles` (default 10). Articles included in each sample.

**Compute cost.** `n_samples` calls per FD. At `n_samples=4` this is 4× B3.

**Known limitations.** Cost scales linearly with samples. Plurality vote with K=2 (Comply/Surprise) means a 2-2 split falls back to the canonical hypothesis_set ordering.

**Expected output.** Plurality-vote pick across `n_samples` CoT samples; extras include `n_samples` and the per-sample picks.

---

## B5: Multi-Agent Debate

**One-line summary.** `n_agents` agents each produce an independent CoT pick, then revise over `n_rounds − 1` subsequent rounds given the prior round's peer responses. Final pick is the plurality vote over the agents' final-round picks.

**Citation.** Du et al. 2023, "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (arXiv:2305.14325).

**Method description.** Round 0: each of `n_agents` agents receives the same prompt at temperature ≥ 0.4 and independently produces a pick. Round `r > 0`: each agent sees the truncated content of every agent's round-`r-1` response embedded in the instruction block, and is asked to revise. The runner dispatches each round as its own Batch API submission (one batch per round) because later rounds depend on earlier results.

**Config parameters.**
- `n_agents` (default 2). Number of debating agents per FD.
- `n_rounds` (default 1). Total rounds including round 0; one round captures the debate mechanism at minimum cost.
- `max_articles` (default 10).

**Compute cost.** `n_agents × n_rounds` calls per FD. Default config = 2 calls.

**Known limitations.** Agents are not heterogeneous (same model, same temperature) in this implementation, so "debate" is really sampling-with-revision. True heterogeneous debate is left to B9. Context grows each round because prior responses are embedded.

**Expected output.** Plurality-vote pick over agents' final-round picks; extras include `n_agents`, `n_rounds`, and the per-agent picks.

---

## B6: Tree of Thoughts

**One-line summary.** At depth 0 generate `breadth` diverse reasoning seeds; at each subsequent depth expand each seed into `breadth` children terminating with a single pick. Final pick is the plurality vote over all leaves.

**Citation.** Yao et al. 2023, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (arXiv:2305.10601). We use a simplified batch-friendly variant without explicit pruning.

**Method description.** Depth 0: for each FD, issue `breadth` seed-thought calls at high temperature. Depth `d > 0`: for each parent seed, issue `breadth` expansion calls that build on the parent's reasoning and terminate with a single pick. At `depth = 2` this produces `breadth × breadth` leaves per FD. The final pick is the plurality vote over leaf picks.

**Config parameters.**
- `breadth` (default 2). Fan-out per node; canonical ToT demos use B=2-3.
- `depth` (default 2). Tree depth. `depth=1` is equivalent to B4 with `n_samples=breadth`.
- `max_articles` (default 10).

**Compute cost.** `breadth + breadth^2` calls per FD at `depth=2`. Default config = 2 + 4 = 6 calls.

**Known limitations.** No pruning: all leaves vote equally, which differs from the original ToT which prunes via a value function. Cost grows quickly with depth.

**Expected output.** Plurality-vote pick over all leaves; extras include `breadth`, `depth`, and the per-leaf picks.

---

## B7: Reflexion

**One-line summary.** Self-critique loop over `n_iterations` iterations: iteration 0 is an initial forecast; each subsequent iteration issues a critique call followed by a revise call.

**Citation.** Shinn et al. 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning" (arXiv:2303.11366).

**Method description.** Iteration 0: CoT pick. Iteration `i > 0`: a critic (system prompt `SYSTEM_CRITIC`) is asked to list at most three concrete weaknesses in the previous forecast; then a revise call is issued that has access to both the prior forecast text and the critique framing. The final pick is parsed from the last revise call.

**Config parameters.**
- `n_iterations` (default 2). Total iterations including iteration 0; 2 is sufficient for convergence on binary picks.
- `max_articles` (default 10).

**Compute cost.** `1 + 2 · (n_iterations − 1)` calls per FD. At default = 1 + 2 = 3 calls.

**Known limitations.** The critic and the forecaster are the same underlying model, so the critic's biases mirror the forecaster's. No explicit memory buffer across FDs (unlike the original Reflexion paper's use on agent tasks).

**Expected output.** Pick from the final revise iteration; extras include `n_iterations`.

---

## B8: Verbalized Confidence (DEPRECATED under pick-only)

**One-line summary.** Verbalized Confidence no longer applies under the pick-only response contract; the implementation now degenerates to a B3-equivalent RAG pick. Kept in the registry for backward compatibility but the paper replaces it with a majority-class reference.

**Citation.** Lin et al. 2022, "Teaching Models to Express Their Uncertainty in Words" (arXiv:2205.14334); Tian et al. 2023, "Just Ask for Calibration" (arXiv:2305.14975). These papers established that asking the model to state its confidence explicitly improves ECE; ECE is no longer a metric in this benchmark (every method emits a single label), so the technique has nothing to calibrate.

**Method description.** Implementation runs the same RAG prompt as B3 and returns the same single picked hypothesis. Results should be read as identical in spirit to B3.

**Config parameters.**
- `max_articles` (default 10).

**Compute cost.** 1 call per FD.

**Known limitations.** No longer a distinct baseline; the v2.1 paper substitutes the always-predict-majority-class reference baseline for headline comparisons.

**Expected output.** Single picked hypothesis label per FD.

---

## B9: Heterogeneous LLM Ensemble

**One-line summary.** Issue one RAG call per `(model, temperature)` configuration in the `configs` list; plurality-vote over the resulting picks.

**Citation.** Jiang et al. 2023, "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion" (arXiv:2306.02561). We use plurality vote rather than a learned fuser.

**Method description.** Unique among the baselines, B9 is allowed to deviate from the default model (`allow_model_override = True`). The default `configs` list has three members of the same model (gpt-4o-mini) at three temperatures (0.0, 0.5, 1.0); this isolates the temperature-diversity effect at minimum cost. For each FD we issue one RAG call per member, parse each to a single pick, and return the plurality vote.

**Config parameters.**
- `configs`: list of `{model, temperature}` dicts. Default has 3 members.
- `max_articles` (default 10).

**Compute cost.** `len(configs)` calls per FD. Default = 3 calls.

**Known limitations.** The default config is temperature-only; cross-model heterogeneity (e.g., adding gpt-4o variants) is omitted for cost. Plurality vote with K=2 picks falls back to `hypothesis_set` ordering on ties.

**Expected output.** Plurality-vote pick across ensemble members; extras include `n_members` and the per-member picks.

---

## Cross-cutting notes

- **Fairness.** Except for B9, the runner enforces that `baseline.model == defaults.model`. This prevents accidental model drift across methods.
- **Parsing.** All methods share the `Baseline.parse_pick` helper, which extracts a single hypothesis label from the JSON `prediction` field. Parse failures are flagged on the prediction row and counted as wrong by the metrics layer (no silent fallback to `hypothesis_set[0]`).
- **Article budget.** All article-using methods cap at `max_articles` (default 10) and truncate each article body to ~900 chars. This is shared at the `prompts.build_articles_block` layer, not redefined per method.
- **Forecast horizon.** The runner's `apply_experiment_horizon` step filters each FD's `article_ids` to those with `publish_date < (resolution_date - horizon_days)` BEFORE prompt rendering; default horizon is 14 days. Horizon is an experiment-time knob (`--horizon`), not a build-time one.
- **Primary target.** Every FD's primary `hypothesis_set` is `["Comply", "Surprise"]`. The legacy domain-specific multiclass label is preserved as `x_multiclass_*` and reachable only via `--multiclass`.
- **Headline metric.** The per-fd_type breakdown in `metrics.json` reports the `change` subset (FDs where the status quo broke) as the headline number; that is where forecasting skill above the majority-class reference is actually demonstrated.
