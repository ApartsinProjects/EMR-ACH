"""
B7: Reflexion (Shinn et al., 2023).
Single-agent self-critique loop: initial CoT -> critique -> revision. 3 iterations.
Final prediction = probability from the last revision.

Usage:
  /c/Python314/python experiments/06_baselines/run_b7_reflexion.py \
      --fds data/unified/forecasts_filtered.jsonl \
      --articles data/unified/articles.jsonl \
      --mode smoke --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))

from _shared import (  # noqa: E402
    BASE_SYSTEM, BASE_USER_YESNO, common_argparser, concat_articles,
    evaluate_binary, extract_prob_yes, gt_to_int, load_articles, load_fds,
    n_for_mode, save_json, save_jsonl, synthetic_response,
)

N_ITERATIONS = 3


def run(args) -> None:
    fds = load_fds(args.fds, n_for_mode(args.mode, args.n_queries))
    articles = load_articles(args.articles)
    print(f"[b7] Loaded {len(fds)} FDs; iterations={N_ITERATIONS}")

    preds: list[dict] = []
    for fd in fds:
        ctx = concat_articles(fd.get("article_ids", []), articles)
        base_user = BASE_USER_YESNO.format(
            question=fd["question"],
            resolution_date=fd.get("resolution_date", "N/A"),
            context=ctx,
        )

        trajectory: list[float] = []
        last_prob = 0.5
        for it in range(N_ITERATIONS):
            if it == 0:
                prompt = base_user
                tag = f"b7_init"
            else:
                prompt = (
                    f"{base_user}\n\nPrior attempt Yes={last_prob:.3f}. "
                    f"Critique your prior reasoning: list at least two weaknesses, "
                    f"then produce a revised JSON forecast."
                )
                tag = f"b7_rev{it}"
            if args.dry_run:
                resp = synthetic_response(fd["id"], variant=tag)
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            last_prob = extract_prob_yes(resp)
            trajectory.append(last_prob)

        prob_yes = float(last_prob)
        preds.append({
            "query_id": fd["id"],
            "prob_yes": round(prob_yes, 4),
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": gt_to_int(fd["ground_truth"]),
            "trajectory": [round(p, 4) for p in trajectory],
        })

    out_pred = Path("results/processed/predictions_b7.jsonl")
    out_met = Path("results/processed/metrics_b7.json")
    save_jsonl(preds, out_pred)
    metrics = evaluate_binary(preds)
    metrics["baseline"] = "b7_reflexion"
    metrics["iterations"] = N_ITERATIONS
    metrics["dry_run"] = bool(args.dry_run)
    save_json(metrics, out_met)
    print(f"[b7] {'DRY-RUN ' if args.dry_run else ''}complete. "
          f"N={metrics['n']} Brier={metrics['brier_score']} ECE={metrics['ece']} "
          f"Acc={metrics['accuracy']}% -> {out_pred}")


def main():
    run(common_argparser("B7 Reflexion").parse_args())


if __name__ == "__main__":
    main()
