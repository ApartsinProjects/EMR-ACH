# Facts vs Articles head-to-head (sample N=50)

- Cutoff: `2026-01-01` | Bench: `earnings` | Seed: 42
- Model: `gpt-4o-mini` | Mode: live
- Facts source: `facts.v1_linked.jsonl` (FD-linked: True)
- Generated: 2026-04-23T14:48:07.579757Z

## Headline
| metric | articles | facts | delta |
|---|---:|---:|---:|
| accuracy   | 50.0% | 52.0% | +2.0pp |
| parsed     | 50/50 | 50/50 | |
| agreement  | colspan=3 | 45/50 (90.0%) | |
| input tokens (mean)   | 216 | 214 | -1% |
| input tokens (median) | 193 | 196 | +2% |
| input tokens (p95)    | 283 | 407 | +44% |

## Per-(benchmark, fd_type) accuracy
| benchmark | fd_type | n | acc(articles) | acc(facts) | delta |
|---|---|---:|---:|---:|---:|
| `earnings` | `change` | 24 | 0.0% | 8.3% | +8.3pp |
| `earnings` | `stability` | 26 | 96.2% | 92.3% | -3.8pp |

## Per-FD diffs
Full per-FD records (with reasoning) at `E:\Projects\ACH\data\etd\audit\facts_vs_articles_earnings_2026-01-01_diffs.jsonl`.

## How to read this
- A negative accuracy delta on the **change** subset is the canonical signal that ETD is dropping causal evidence the model needs at the hard cases. Inspect those rows in the diff file first.
- If facts beat articles on **stability** but lose on **change**, ETD is stripping noise (good) but also stripping the leading indicator of regime shift (bad). Tighten Stage-1 to require entity + numeric context on flagged-as-change facts.
- A token-count ratio worse than ~30% (facts comparable to articles) means Stage-2 dedup or summarization isn't compressing as expected; re-check `--max-facts` and Stage-2 cluster-size threshold.
