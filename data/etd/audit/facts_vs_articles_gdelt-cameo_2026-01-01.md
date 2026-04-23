# Facts vs Articles head-to-head (sample N=50)

- Cutoff: `2026-01-01` | Bench: `gdelt-cameo` | Seed: 42
- Model: `gpt-4o-mini` | Mode: live
- Facts source: `facts.v1_linked.jsonl` (FD-linked: True)
- Generated: 2026-04-23T14:46:05.423596Z

## Headline
| metric | articles | facts | delta |
|---|---:|---:|---:|
| accuracy   | 30.0% | 42.0% | +12.0pp |
| parsed     | 50/50 | 50/50 | |
| agreement  | colspan=3 | 29/50 (58.0%) | |
| input tokens (mean)   | 242 | 562 | +132% |
| input tokens (median) | 236 | 380 | +61% |
| input tokens (p95)    | 267 | 1245 | +366% |

## Per-(benchmark, fd_type) accuracy
| benchmark | fd_type | n | acc(articles) | acc(facts) | delta |
|---|---|---:|---:|---:|---:|
| `gdelt-cameo` | `change` | 6 | 0.0% | 66.7% | +66.7pp |
| `gdelt-cameo` | `stability` | 42 | 35.7% | 40.5% | +4.8pp |
| `gdelt-cameo` | `unknown` | 2 | 0.0% | 0.0% | +0.0pp |

## Per-FD diffs
Full per-FD records (with reasoning) at `E:\Projects\ACH\data\etd\audit\facts_vs_articles_gdelt-cameo_2026-01-01_diffs.jsonl`.

## How to read this
- A negative accuracy delta on the **change** subset is the canonical signal that ETD is dropping causal evidence the model needs at the hard cases. Inspect those rows in the diff file first.
- If facts beat articles on **stability** but lose on **change**, ETD is stripping noise (good) but also stripping the leading indicator of regime shift (bad). Tighten Stage-1 to require entity + numeric context on flagged-as-change facts.
- A token-count ratio worse than ~30% (facts comparable to articles) means Stage-2 dedup or summarization isn't compressing as expected; re-check `--max-facts` and Stage-2 cluster-size threshold.
