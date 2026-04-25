# v2.2 Baselines Evaluation Plan (refresh 2026-04-25)

This document was refreshed for the v2.2 strategy 6.1B (CC-News rebuild)
gold subset at `benchmark/data/2026-01-01-h14-ccnews-gold/`. The previous
plan against the reuse-first h14 gold (22 FDs) is preserved at the bottom
for reference.

This refresh is the result of B6.1B-6 (dry-run only). No real Batch
API calls have been fired.

## Active gold subset

`benchmark/data/2026-01-01-h14-ccnews-gold/`: 36 FDs, 705 articles,
2,018 ETD facts.

Composition:

| benchmark | fd_type | n |
| --- | --- | ---: |
| forecastbench | stability | 24 |
| forecastbench | change | 1 |
| earnings | stability | 6 |
| earnings | change | 5 |
| **total** | | **36** |

## Per-baseline dry-run verdict

All 10 baselines plumb cleanly against the v2.2 h14-ccnews gold subset.
Each was exercised with `--dry-run` against all 36 FDs.

| Method                     | Description                                  | Dry-run OK | Full requests |
|----------------------------|----------------------------------------------|------------|---------------|
| `b1_direct`                | No-context direct pick                       | yes        | 36            |
| `b2_cot`                   | No-context chain-of-thought                  | yes        | 36            |
| `b3_rag`                   | Articles in context, direct pick             | yes        | 36            |
| `b3b_rag_claims`           | Per-article claim extraction then aggregate  | yes        | 305           |
| `b4_self_consistency`      | 4 samples, plurality vote                    | yes        | 144           |
| `b5_multi_agent_debate`    | 2 agents, 1 debate round                     | yes        | 72            |
| `b6_tree_of_thoughts`      | 2 thought branches, then aggregate           | yes        | 72            |
| `b7_reflexion`             | Self-critique pass                           | yes        | 36            |
| `b8_verbalized_confidence` | Direct pick + self-rated confidence          | yes        | 36            |
| `b9_llm_ensemble`          | 3 model variants, vote                       | yes        | 108           |
| **total**                  |                                              |            | **881**       |

All requests target `gpt-4o-mini-2024-07-18` (per
`benchmark/configs/baselines.yaml` defaults; b9 may swap models
internally). Temperature 0.0, max_tokens 512, response_format
json_object, batch_api true.

## Cost estimate (Batch API rates, gpt-4o-mini)

Batch API pricing (as of 2026-04-25):
- Input:  `$0.075 / 1M tokens`
- Output: `$0.30  / 1M tokens`

Conservative per-request token budget. The h14-ccnews gold articles are
fuller-bodied than the reuse-first h14 gold (mean ~6.4 facts/article,
705 articles vs. 98), so the average b3 RAG context is larger:
- Input avg:  ~3,500 tokens (system + question + background + up to 10
  articles snippeted; CC-News articles run 2-8k chars each)
- Output avg: ~256 tokens (json_object reply, well under the 512 cap)

For 881 requests:
- Input  cost: 881 x 3,500  / 1e6 x 0.075 = **$0.231**
- Output cost: 881 x 256    / 1e6 x 0.30  = **$0.068**
- **Total: ~$0.30** for the full B1 to B9 battery on h14-ccnews gold.

Two orders of magnitude under the typical $15 budget cap. b3b_rag_claims
dominates at 305 requests; if its per-request input runs to 5k tokens
the total still fits under $0.50.

## Wall-clock estimate

Batch API SLA is 24h max but typically returns within 1-2 hours for
jobs of this size. With 10 baselines submitted in parallel, expect 1-3h
end-to-end.

Per-method synchronous fallback (`--sync`, no Batch) is ~3-10 min per
baseline at 36 FDs; useful for `b1`/`b2` smoke if avoiding Batch
latency for the trivial cases.

## Proposed submission order

1. `b1_direct` and `b2_cot` first: cheapest sanity check; if their JSON
   format breaks, fix before batching the heavier methods.
2. `b3_rag`, `b7_reflexion`, `b8_verbalized_confidence`: single-call
   methods that exercise the article-context path.
3. `b3b_rag_claims`, `b4_self_consistency`, `b9_llm_ensemble`: multi-call
   methods. b9 may incur per-model auth checks (GPT-4o, Claude, etc.).
4. `b5_multi_agent_debate`, `b6_tree_of_thoughts`: multi-round; submit
   last so any prompt-template edits caught by 1 to 3 propagate first.

## Budget cap

Hard cap **$15** per upstream task spec. Soft cap **$1** more than
sufficient given the $0.30 estimate. If costs exceed $0.50, halt and
investigate (likely a runaway loop in a multi-call method).

## Ready-to-fire commands

Run from `E:\Projects\ACH\benchmark` so the
`evaluation.baselines.runner` module resolves:

```
cd E:/Projects/ACH/benchmark

# Pre-flight: confirm OpenAI key has quota.
/c/Python314/python -m evaluation.baselines.runner --method b1 \
    --fds data/2026-01-01-h14-ccnews-gold/forecasts.jsonl \
    --articles data/2026-01-01-h14-ccnews-gold/articles.jsonl \
    --smoke 3 --sync   # 3 sync calls, costs ~$0.001, validates auth + JSON

# Then fire the full battery via Batch API (independent submissions):
for M in b1_direct b2_cot b3_rag b7_reflexion b8_verbalized_confidence \
         b3b_rag_claims b4_self_consistency b9_llm_ensemble \
         b5_multi_agent_debate b6_tree_of_thoughts; do
  /c/Python314/python -m evaluation.baselines.runner --method "$M" \
      --fds data/2026-01-01-h14-ccnews-gold/forecasts.jsonl \
      --articles data/2026-01-01-h14-ccnews-gold/articles.jsonl
done

# Results land at benchmark/results/2026-01-01/{method}/{run_id}/
# (the runner infers cutoff from the FD path; "2026-01-01" is the parent
# cutoff embedded in the publish year, not the h14-ccnews subdir).
```

Note: `runner.py:infer_cutoff` strips the `-h14-ccnews` and `-gold`
suffixes, so results land under `2026-01-01`. If a separate results
tree for v2.2 6.1B specifically is desired, pass
`--results-dir benchmark/results-h14-ccnews/`.

## Open issues / follow-ups for the eval session

1. **Gold pool composition**: 36 FDs is below the 50-FD soft floor. The
   forecastbench-stability cell carries 24 of 36. Macro accuracy will be
   noisy (each FD = 2.8 percentage points). Consider broadening the
   CC-News fetch (more hosts, more months) before tagging v2.2-data-ready
   for paper.
2. **Class skew**: 30 of 36 are stability, 6 are change. Majority-class
   baseline hits 83.3%. That should be the headline reference.
3. **OpenAI quota**: confirm Batch API quota on the .env key before
   firing (Batch quota is separate from sync quota; the ETD compare
   step in 2026-04-23 hit 429 on sync calls).
4. After the batch returns, regenerate paper Table 1 + Table 4 cells
   for the v2.2-h14-ccnews row, then decide whether to tag
   `v2.2-data-ready` (or hold for a broadened CC-News rebuild).

---

## Archived: v2.2 reuse-first h14 plan (2026-04-23)

The original plan against `benchmark/data/2026-01-01-h14-gold/`
(22 FDs, 98 articles, 269 facts) is preserved here for reference.

| Method                     | Smoke OK | Full requests |
|----------------------------|----------|---------------|
| `b1_direct`                | yes      | 22            |
| `b2_cot`                   | yes      | 22            |
| `b3_rag`                   | yes      | 22            |
| `b3b_rag_claims`           | yes      | 98            |
| `b4_self_consistency`      | yes      | 88            |
| `b5_multi_agent_debate`    | yes      | 44            |
| `b6_tree_of_thoughts`      | yes      | 44            |
| `b7_reflexion`             | yes      | 22            |
| `b8_verbalized_confidence` | yes      | 22            |
| `b9_llm_ensemble`          | yes      | 66            |
| **total**                  |          | **472**       |

Estimated cost: $0.11. Composition: 21 fb-stability + 1 fb-change.
