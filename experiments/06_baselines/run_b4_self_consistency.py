"""
B4: Self-Consistency (Wang et al., 2022).
Sample k=8 CoT chains at T=0.7; extract probability from each; mean.

Usage:
  /c/Python314/python experiments/06_baselines/run_b4_self_consistency.py \
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

K_SAMPLES = 8
TEMPERATURE = 0.7


def run(args) -> None:
    fds = load_fds(args.fds, n_for_mode(args.mode, args.n_queries))
    articles = load_articles(args.articles)
    print(f"[b4] Loaded {len(fds)} FDs, {len(articles)} articles")
    print(f"[b4] k={K_SAMPLES} samples @ T={TEMPERATURE}")

    preds: list[dict] = []
    for fd in fds:
        ctx = concat_articles(fd.get("article_ids", []), articles)
        user = BASE_USER_YESNO.format(
            question=fd["question"],
            resolution_date=fd.get("resolution_date", "N/A"),
            context=ctx,
        )
        sample_probs: list[float] = []
        for k in range(K_SAMPLES):
            if args.dry_run:
                resp = synthetic_response(fd["id"], variant=f"b4_s{k}")
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            sample_probs.append(extract_prob_yes(resp))
        prob_yes = float(np.mean(sample_probs))
        preds.append({
            "query_id": fd["id"],
            "prob_yes": round(prob_yes, 4),
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": gt_to_int(fd["ground_truth"]),
            "samples": [round(p, 4) for p in sample_probs],
        })
        if args.dry_run and len(preds) == 1:
            snippet = (user[:400] + ("..." if len(user) > 400 else "")).encode("ascii", "replace").decode("ascii")
            print(f"[b4] Prompt sample (first FD, {len(user)} chars):")
            print(snippet)

    out_pred = Path("results/processed/predictions_b4.jsonl")
    out_met = Path("results/processed/metrics_b4.json")
    save_jsonl(preds, out_pred)
    metrics = evaluate_binary(preds)
    metrics["baseline"] = "b4_self_consistency"
    metrics["k"] = K_SAMPLES
    metrics["temperature"] = TEMPERATURE
    metrics["dry_run"] = bool(args.dry_run)
    save_json(metrics, out_met)
    print(f"[b4] {'DRY-RUN ' if args.dry_run else ''}complete. "
          f"N={metrics['n']} Brier={metrics['brier_score']} ECE={metrics['ece']} "
          f"Acc={metrics['accuracy']}% -> {out_pred}")


def main():
    p = common_argparser("B4 Self-Consistency")
    run(p.parse_args())


if __name__ == "__main__":
    main()
