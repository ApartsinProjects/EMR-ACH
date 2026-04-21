"""
B9: Heterogeneous LLM Ensemble.
9 configs (3 models x 3 temperatures). For each FD, query all 9; mean probabilities.

Models and temperatures chosen to match paper Section 6.2. Temperatures are standard
low/med/high; models are three distinct providers that would be used in a real run
(simulated in --dry-run).

Usage:
  /c/Python314/python experiments/06_baselines/run_b9_llm_ensemble.py \
      --fds data/unified/forecasts_filtered.jsonl \
      --articles data/unified/articles.jsonl \
      --mode smoke --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))

from _shared import (  # noqa: E402
    BASE_SYSTEM, BASE_USER_YESNO, common_argparser, concat_articles,
    evaluate_binary, extract_prob_yes, gt_to_int, load_articles, load_fds,
    n_for_mode, save_json, save_jsonl, synthetic_response,
)

MODELS = ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash"]
TEMPERATURES = [0.0, 0.5, 1.0]
CONFIGS = [(m, t) for m in MODELS for t in TEMPERATURES]


def run(args) -> None:
    fds = load_fds(args.fds, n_for_mode(args.mode, args.n_queries))
    articles = load_articles(args.articles)
    print(f"[b9] Loaded {len(fds)} FDs; ensemble size={len(CONFIGS)} "
          f"({len(MODELS)} models x {len(TEMPERATURES)} temps)")

    preds: list[dict] = []
    for fd in fds:
        ctx = concat_articles(fd.get("article_ids", []), articles)
        user = BASE_USER_YESNO.format(
            question=fd["question"],
            resolution_date=fd.get("resolution_date", "N/A"),
            context=ctx,
        )
        config_probs: list[tuple[str, float, float]] = []
        for (model, temp) in CONFIGS:
            tag = f"b9_{model}_T{temp}"
            if args.dry_run:
                resp = synthetic_response(fd["id"], variant=tag)
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            config_probs.append((model, temp, extract_prob_yes(resp)))

        prob_yes = float(np.mean([p for _, _, p in config_probs]))
        preds.append({
            "query_id": fd["id"],
            "prob_yes": round(prob_yes, 4),
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": gt_to_int(fd["ground_truth"]),
            "per_config": [
                {"model": m, "temperature": t, "prob_yes": round(p, 4)}
                for (m, t, p) in config_probs
            ],
        })
        if args.dry_run and len(preds) == 1:
            print(f"[b9] Shared prompt length (first FD): {len(user)} chars")

    out_pred = Path("results/processed/predictions_b9.jsonl")
    out_met = Path("results/processed/metrics_b9.json")
    save_jsonl(preds, out_pred)
    metrics = evaluate_binary(preds)
    metrics["baseline"] = "b9_llm_ensemble"
    metrics["n_configs"] = len(CONFIGS)
    metrics["models"] = MODELS
    metrics["temperatures"] = TEMPERATURES
    metrics["dry_run"] = bool(args.dry_run)
    save_json(metrics, out_met)
    print(f"[b9] {'DRY-RUN ' if args.dry_run else ''}complete. "
          f"N={metrics['n']} Brier={metrics['brier_score']} ECE={metrics['ece']} "
          f"Acc={metrics['accuracy']}% -> {out_pred}")


def main():
    run(common_argparser("B9 LLM Ensemble").parse_args())


if __name__ == "__main__":
    main()
