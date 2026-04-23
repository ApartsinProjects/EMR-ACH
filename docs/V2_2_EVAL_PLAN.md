# v2.2 Baselines Evaluation Plan (2026-04-23)

Plan for the next session's Batch API run against the v2.2 h14 gold subset
at `benchmark/data/2026-01-01-h14-gold/` (22 FDs, 98 articles, 269 facts).

This document is the result of B4 (dry-run only). No real Batch API calls
have been fired.

## Per-baseline dry-run verdict

All 10 baselines plumb cleanly against the v2.2 gold subset. Each was
exercised with `--dry-run --smoke 3` and a full-scale `--dry-run` against
all 22 FDs to capture request counts.

| Method                     | Description                                  | Smoke OK | Full requests |
|----------------------------|----------------------------------------------|----------|---------------|
| `b1_direct`                | No-context direct pick                       | yes      | 22            |
| `b2_cot`                   | No-context chain-of-thought                  | yes      | 22            |
| `b3_rag`                   | Articles in context, direct pick             | yes      | 22            |
| `b3b_rag_claims`           | Per-article claim extraction then aggregate  | yes      | 98            |
| `b4_self_consistency`      | 4 samples, plurality vote                    | yes      | 88            |
| `b5_multi_agent_debate`    | 2 agents, 1 debate round                     | yes      | 44            |
| `b6_tree_of_thoughts`      | 2 thought branches, then aggregate           | yes      | 44            |
| `b7_reflexion`             | Self-critique pass                           | yes      | 22            |
| `b8_verbalized_confidence` | Direct pick + self-rated confidence          | yes      | 22            |
| `b9_llm_ensemble`          | 3 model variants, vote                       | yes      | 66            |
| **total**                  |                                              |          | **472**       |

All requests target `gpt-4o-mini-2024-07-18` (per `benchmark/configs/baselines.yaml` defaults; b9 may swap models internally). Temperature 0.0, max_tokens 512, response_format json_object, batch_api true.

## Cost estimate (Batch API rates, gpt-4o-mini)

Batch API pricing (as of 2026-04-23):
- Input:  `$0.075 / 1M tokens`
- Output: `$0.30  / 1M tokens`

Conservative per-request token budget (b3 RAG dominates the average; b1/b2 are smaller):
- Input avg:  ~2,000 tokens (system + question + background + up to 10 articles snippeted)
- Output avg: ~256 tokens (json_object reply, well under the 512 cap)

For 472 requests:
- Input  cost: 472 × 2,000  / 1e6 × 0.075 = **$0.071**
- Output cost: 472 × 256    / 1e6 × 0.30  = **$0.036**
- **Total: ~$0.11** for the full B1-B9 battery on v2.2 gold.

This is two orders of magnitude under the $15 budget cap. Even tripling the
input estimate (to account for b3b_rag_claims pulling per-article context)
keeps the total under $0.50.

## Wall-clock estimate

Batch API SLA is 24h max but typically returns within 1-2 hours for jobs of
this size. With 10 baselines submitted in parallel, expect 1-3h end-to-end.

Per-method synchronous fallback (`--sync`, no Batch) would be ~3-10 min per
baseline at this scale; useful for `b1`/`b2` smoke if the user prefers to
avoid Batch latency for the trivial cases.

## Proposed submission order

1. `b1_direct` and `b2_cot` first: cheapest sanity check; if their JSON
   format breaks, fix before batching the heavier methods.
2. `b3_rag`, `b7_reflexion`, `b8_verbalized_confidence`: single-call
   methods that exercise the article-context path.
3. `b3b_rag_claims`, `b4_self_consistency`, `b9_llm_ensemble`: multi-call
   methods. b9 may incur per-model auth checks (GPT-4o, Claude, etc.).
4. `b5_multi_agent_debate`, `b6_tree_of_thoughts`: multi-round; submit
   last so any prompt-template edits caught by 1-3 propagate first.

## Budget cap

Hard cap **$15** per task spec. Soft cap **$1** more than sufficient given
the $0.11 estimate. If costs exceed $0.50, halt and investigate (likely a
runaway loop in a multi-call method).

## Ready-to-fire commands

Run from `E:\Projects\ACH\benchmark` so the `evaluation.baselines.runner`
module resolves:

```
cd E:/Projects/ACH/benchmark

# Pre-flight: confirm OpenAI key has quota (one of the dry-runs hit 429
# during the ETD compare step on 2026-04-23; etd_compare runs synchronous,
# baselines run via Batch API with separate quota).
python -m evaluation.baselines.runner --method b1 \
    --fds data/2026-01-01-h14-gold/forecasts.jsonl \
    --articles data/2026-01-01-h14-gold/articles.jsonl \
    --smoke 3 --sync   # 3 sync calls, costs ~$0.001, validates auth + JSON

# Then fire the full battery via Batch API (independent submissions):
for M in b1_direct b2_cot b3_rag b7_reflexion b8_verbalized_confidence \
         b3b_rag_claims b4_self_consistency b9_llm_ensemble \
         b5_multi_agent_debate b6_tree_of_thoughts; do
  python -m evaluation.baselines.runner --method "$M" \
      --fds data/2026-01-01-h14-gold/forecasts.jsonl \
      --articles data/2026-01-01-h14-gold/articles.jsonl
done

# Results land at benchmark/results/2026-01-01/{method}/{run_id}/
# (the runner infers cutoff from the FD path; "2026-01-01" is the parent
# cutoff embedded in the publish year, not the h14 subdir).
```

Note: `runner.py:infer_cutoff` strips the `-h14` and `-gold` suffixes, so
results land under `2026-01-01`. If we want a separate results tree for
v2.2 specifically, pass `--results-dir benchmark/results-h14/`.

## Open issues / follow-ups for the eval session

1. **Gold pool is fb-only**: 21 fb-stability + 1 fb-change. Earnings and
   gdelt are not represented. Macro accuracy will be entirely a
   forecastbench measure; the paper Table 1 row for h14 should be
   labeled "h14 fb-only" or the pool needs broadening (strategy 6.1B).
2. **Class skew**: 21 of 22 are stability. A majority-class baseline will
   hit 95.5% on this subset; that should be the headline reference.
3. **OpenAI quota**: the .env key hit 429 during the ETD compare audit
   step in B2. Confirm Batch API quota is on the same key before firing.
4. After the batch returns, regenerate paper Table 1 + Table 4 cells for
   the v2.2 row, then tag `v2.2-data-ready`.
