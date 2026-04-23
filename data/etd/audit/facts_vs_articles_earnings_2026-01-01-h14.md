# Facts vs Articles head-to-head (sample N=49)

- Cutoff: `2026-01-01-h14` | Bench: `earnings` | Seed: 42
- Model: `gpt-4o-mini` | Mode: live
- Facts source: `facts.v1_linked.jsonl` (FD-linked: True)
- Generated: 2026-04-23T20:03:22.954416Z

## Headline
| metric | articles | facts | delta |
|---|---:|---:|---:|
| accuracy   | 0.0% | 0.0% | +0.0pp |
| parsed     | 0/49 | 0/49 | |
| agreement  | colspan=3 | 0/49 (0.0%) | |
| input tokens (mean)   | 206 | 208 | +1% |
| input tokens (median) | 168 | 166 | -1% |
| input tokens (p95)    | 331 | 435 | +31% |

## Per-(benchmark, fd_type) accuracy
| benchmark | fd_type | n | acc(articles) | acc(facts) | delta |
|---|---|---:|---:|---:|---:|
| `earnings` | `change` | 16 | 0.0% | 0.0% | +0.0pp |
| `earnings` | `stability` | 33 | 0.0% | 0.0% | +0.0pp |

## Per-FD diffs
Full per-FD records (with reasoning) at `E:\Projects\ACH\data\etd\audit\facts_vs_articles_earnings_2026-01-01-h14_diffs.jsonl`.

## How to read this
- A negative accuracy delta on the **change** subset is the canonical signal that ETD is dropping causal evidence the model needs at the hard cases. Inspect those rows in the diff file first.
- If facts beat articles on **stability** but lose on **change**, ETD is stripping noise (good) but also stripping the leading indicator of regime shift (bad). Tighten Stage-1 to require entity + numeric context on flagged-as-change facts.
- A token-count ratio worse than ~30% (facts comparable to articles) means Stage-2 dedup or summarization isn't compressing as expected; re-check `--max-facts` and Stage-2 cluster-size threshold.
