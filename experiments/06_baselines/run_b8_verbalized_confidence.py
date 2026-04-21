"""
B8: Verbalized Confidence (Tian et al., 2023 / Lin et al., 2022).
RAG-only prompt variant that explicitly asks the model to verbalize a calibrated
probability (the Yes probability itself is the verbalized confidence).

Usage:
  /c/Python314/python experiments/06_baselines/run_b8_verbalized_confidence.py \
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
    BASE_SYSTEM, common_argparser, concat_articles, evaluate_binary,
    extract_prob_yes, gt_to_int, load_articles, load_fds, n_for_mode,
    save_json, save_jsonl, synthetic_response,
)

VERBALIZED_USER = """Forecasting question: {question}
Resolution date: {resolution_date}

Relevant articles:
{context}

State your best calibrated probability that the answer is Yes as a number in [0, 1].
Use the full range. A value of 0.5 means you are genuinely uncertain. Values near 0
or 1 must be justified by the evidence. Only after stating the probability, provide
a one-sentence justification.

Return JSON only:
{{
  "verbalized_probability_yes": <float in [0,1]>,
  "confidence_rationale": "<one sentence>",
  "probabilities": {{"Yes": <same float>, "No": <1 - float>}},
  "prediction": "<Yes_or_No>"
}}"""


def run(args) -> None:
    fds = load_fds(args.fds, n_for_mode(args.mode, args.n_queries))
    articles = load_articles(args.articles)
    print(f"[b8] Loaded {len(fds)} FDs (verbalized-confidence RAG)")

    preds: list[dict] = []
    for fd in fds:
        ctx = concat_articles(fd.get("article_ids", []), articles)
        user = VERBALIZED_USER.format(
            question=fd["question"],
            resolution_date=fd.get("resolution_date", "N/A"),
            context=ctx,
        )
        if args.dry_run:
            resp = synthetic_response(fd["id"], variant="b8_verbal")
        else:
            raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
        prob_yes = extract_prob_yes(resp)
        preds.append({
            "query_id": fd["id"],
            "prob_yes": round(prob_yes, 4),
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": gt_to_int(fd["ground_truth"]),
        })
        if args.dry_run and len(preds) == 1:
            print(f"[b8] Prompt length (first FD): {len(user)} chars")

    out_pred = Path("results/processed/predictions_b8.jsonl")
    out_met = Path("results/processed/metrics_b8.json")
    save_jsonl(preds, out_pred)
    metrics = evaluate_binary(preds)
    metrics["baseline"] = "b8_verbalized_confidence"
    metrics["dry_run"] = bool(args.dry_run)
    save_json(metrics, out_met)
    print(f"[b8] {'DRY-RUN ' if args.dry_run else ''}complete. "
          f"N={metrics['n']} Brier={metrics['brier_score']} ECE={metrics['ece']} "
          f"Acc={metrics['accuracy']}% -> {out_pred}")


def main():
    run(common_argparser("B8 Verbalized Confidence").parse_args())


if __name__ == "__main__":
    main()
