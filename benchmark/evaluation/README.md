# Baselines Battery

Unified baselines battery for evaluating forecasting methods on the GDELT-CAMEO, ForecastBench, and Earnings Forecast-Document (FD) benchmarks. Every baseline is declared via YAML, built against a shared `Baseline` abstract class, and dispatched through the same OpenAI Batch API client so that cost, prompt length, and latency are directly comparable across methods.

**v2.1 framing (2026-04-22).** Every FD's primary target is the binary `["Comply", "Surprise"]` hypothesis set: does the outcome match the prior-state status-quo expectation (Comply) or break it (Surprise)? The legacy domain-specific multiclass label (Yes/No, Beat/Meet/Miss, Peace/Tension/Violence) is preserved on the FD as `x_multiclass_hypothesis_set` / `x_multiclass_ground_truth` for the multiclass ablation; pass `--multiclass` to evaluate against it instead.

Each baseline consumes the unified FD schema (`question`, `background`, `hypothesis_set`, `hypothesis_definitions`, `article_ids`, `forecast_point`, `resolution_date`, `ground_truth`, `prior_state_30d`, `fd_type`) and emits **one prediction row per FD with a single picked hypothesis label** (no probabilities, no calibration). The shared user prompt injects a natural-language `prior_expectation_block` derived from the FD's prior-state annotation so the model is told what the status quo IS before being asked to pick Comply or Surprise.

## Folder layout

```
benchmark/
├── evaluation/
│   ├── README.md              (this file)
│   ├── BASELINES.md           (per-method reference, citations, costs)
│   └── baselines/
│       ├── __init__.py
│       ├── _shim.py           (re-exports src.batch_client, src.config, src.eval.metrics)
│       ├── base.py            (Baseline ABC + shared parsing helpers)
│       ├── prompts.py         (shared prompt templates and instruction blocks)
│       ├── runner.py          (single CLI entry point)
│       └── methods/
│           ├── __init__.py
│           ├── b1_direct.py
│           ├── b2_cot.py
│           ├── b3_rag.py
│           ├── b4_self_consistency.py
│           ├── b5_multi_agent_debate.py
│           ├── b6_tree_of_thoughts.py
│           ├── b7_reflexion.py
│           ├── b8_verbalized_confidence.py
│           └── b9_llm_ensemble.py
├── configs/
│   └── baselines.yaml         (defaults + per-method config)
└── results/
    └── {cutoff}/{method}/{run_id}/...
```

## Shim choice

The baselines depend on three modules under `src/` (`batch_client`, `config`, `eval.metrics`). Rather than inlining copies (~680 LOC total) and risking drift from the canonical paper pipeline, we introduced a thin shim at `evaluation/baselines/_shim.py` that (a) prepends the repository root to `sys.path` at import time, and (b) re-exports exactly the symbols the baselines need. Every baseline imports through the shim (`from .._shim import BatchResult` etc.), so if `src/` is ever relocated, the shim is the single place to update.

## Quickstart

Install dependencies (shared with the rest of the benchmark; see `benchmark/requirements.txt`) and set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

Smoke run (3 FDs, no API calls):

```bash
cd benchmark
python -m evaluation.baselines.runner \
    --method b4_self_consistency \
    --fds data/2026-01-01/forecasts.jsonl \
    --articles data/2026-01-01/articles.jsonl \
    --config configs/baselines.yaml \
    --dry-run --smoke 3
```

Production run (all FDs, Batch API):

```bash
python -m evaluation.baselines.runner \
    --method b3_rag \
    --fds data/2026-01-01/forecasts.jsonl \
    --articles data/2026-01-01/articles.jsonl \
    --config configs/baselines.yaml
```

## Command reference

| Flag | Description |
| --- | --- |
| `--method` | Baseline key. Accepts short aliases (`b4` resolves to `b4_self_consistency`). Required. |
| `--fds` | Path to `forecasts.jsonl` (unified FD format). Required. |
| `--articles` | Path to `articles.jsonl`. Required. |
| `--config` | Path to `baselines.yaml`. Default: `benchmark/configs/baselines.yaml`. |
| `--dry-run` | Build requests, print the first 3, exit. No API calls, no results written. |
| `--smoke [N]` | Smoke mode. Limits to first N FDs (default N=5) and dumps per-FD debug JSON under `debug/`. |
| `--sync` | Use synchronous Chat Completions instead of the Batch API. Intended for small smoke runs only. |
| `--limit N` | Limit number of FDs (overridden by `--smoke`). |
| `--results-dir PATH` | Override base results directory. |
| `--horizon DAYS` | Experiment-time forecast horizon. Filters each FD's `article_ids` to `publish_date < (resolution_date - horizon)`. Defaults to `fd.default_horizon_days` then 14. |
| `--multiclass` | Opt-in ablation: evaluate against the legacy `x_multiclass_*` target instead of the primary binary Comply/Surprise target. |

## Output layout

Every run creates a fresh, timestamped directory. Existing results are never overwritten.

```
benchmark/results/{cutoff}/{method}/
├── latest.txt                          (pointer: most recent run_id)
└── {run_id}/                           (run_id = YYYYMMDD_HHMMSS_{git_sha8})
    ├── run_manifest.json               (timestamp, model, temp, git_sha, horizon, target, config snapshot, FD ids)
    ├── predictions_{method}.jsonl      (one row per FD: predicted_class, ground_truth, fd_type, parse_failed)
    ├── metrics_{method}.json           (accuracy, balanced accuracy, NBA, MCC, macro-F1, per-class P/R/F1, confusion matrix, by_fd_type breakdown, 95% bootstrap CIs)
    └── debug/                          (only in --smoke mode)
        └── {fd_id}.json                (rendered prompt, raw response, parsed pick, latency)
```

## Three-stage debug flow

1. `--dry-run` alone. Builds requests and prints the first 3 as JSON. No API calls. Use this to debug request construction and prompt rendering.
2. `--smoke 5 --sync`. Five FDs, synchronous one-at-a-time Chat Completions. Use this to iterate on prompt wording and JSON parsing with minimal cost.
3. `--smoke 20 --sync`. Same as above but with enough FDs to get a first calibration / accuracy signal.
4. No debug flags. Full Batch API flow on all FDs. Production.

Each stage produces its own versioned run directory, so you can diff runs across code changes.

## Adding a new baseline

1. Add a file `evaluation/baselines/methods/b{N}_{short_name}.py`.
2. Subclass `Baseline` (from `..base`). Implement `build_requests(fds, articles) -> list[BatchRequest]` and `parse_responses(results, fds) -> list[dict]`. For multi-round methods, set `multi_round = True` on the class and also implement `build_requests_round(r, fds, articles, prior)`.
3. Use `self.make_request(custom_id=...)`, `self.render_user(fd, articles, ...)`, and `self.parse_pick(content, hs)` from the base class (every baseline returns one pick, never a probability vector). For multi-sample methods aggregate via `self.plurality(picks, hs)`. Do not call the OpenAI SDK directly; everything goes through `BatchClient`.
4. Register the class in `benchmark/configs/baselines.yaml` under `baselines:` with `class: "methods.b{N}_{short_name}.B{N}ClassName"` plus any per-method config knobs.
5. Document the method in `BASELINES.md` (citation, description, config params, compute cost, known limitations).

The runner is configuration-driven; no change to `runner.py` is required to add a baseline.

## See also

- [`BASELINES.md`](./BASELINES.md) — per-method reference (B1-B9 descriptions, citations, config knobs, compute cost).
- `benchmark/configs/baselines.yaml` — the source of truth for defaults and per-method configuration.
