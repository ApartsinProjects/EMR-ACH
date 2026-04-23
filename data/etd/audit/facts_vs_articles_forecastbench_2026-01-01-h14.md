# Facts vs Articles head-to-head (sample N=40)

- Cutoff: `2026-01-01-h14` | Bench: `forecastbench` | Seed: 42
- Model: `gpt-4o-mini` | Mode: live
- Facts source: `facts.v1_linked.jsonl` (FD-linked: True)
- Generated: 2026-04-23T20:00:05.217250Z

## Headline
| metric | articles | facts | delta |
|---|---:|---:|---:|
| accuracy   | 0.0% | 0.0% | +0.0pp |
| parsed     | 0/40 | 0/40 | |
| agreement  | colspan=3 | 0/40 (0.0%) | |
| input tokens (mean)   | 714 | 580 | -19% |
| input tokens (median) | 599 | 534 | -11% |
| input tokens (p95)    | 1790 | 1145 | -36% |

## Per-(benchmark, fd_type) accuracy
| benchmark | fd_type | n | acc(articles) | acc(facts) | delta |
|---|---|---:|---:|---:|---:|
| `forecastbench` | `change` | 3 | 0.0% | 0.0% | +0.0pp |
| `forecastbench` | `stability` | 37 | 0.0% | 0.0% | +0.0pp |

## Per-FD diffs
Full per-FD records (with reasoning) at `E:\Projects\ACH\data\etd\audit\facts_vs_articles_forecastbench_2026-01-01-h14_diffs.jsonl`.

## How to read this
- A negative accuracy delta on the **change** subset is the canonical signal that ETD is dropping causal evidence the model needs at the hard cases. Inspect those rows in the diff file first.
- If facts beat articles on **stability** but lose on **change**, ETD is stripping noise (good) but also stripping the leading indicator of regime shift (bad). Tighten Stage-1 to require entity + numeric context on flagged-as-change facts.
- A token-count ratio worse than ~30% (facts comparable to articles) means Stage-2 dedup or summarization isn't compressing as expected; re-check `--max-facts` and Stage-2 cluster-size threshold.
