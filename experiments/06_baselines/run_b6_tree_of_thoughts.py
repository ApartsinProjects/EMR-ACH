"""
B6: Tree of Thoughts (Yao et al., 2023).
Breadth=3, depth=2. At each node generate 3 candidate next-step thoughts, score each
via self-critique (0-10), expand the best-scoring thought to depth 2, then sample a
terminal probability from the highest-scoring leaf.

Usage:
  /c/Python314/python experiments/06_baselines/run_b6_tree_of_thoughts.py \
      --fds data/unified/forecasts_filtered.jsonl \
      --articles data/unified/articles.jsonl \
      --mode smoke --dry-run
"""

from __future__ import annotations

import hashlib
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

BREADTH = 3
DEPTH = 2


def _pseudo_score(fd_id: str, path_tag: str) -> float:
    """Deterministic self-critique score in [0,10] for dry-run."""
    h = int(hashlib.md5(f"{fd_id}|{path_tag}".encode()).hexdigest(), 16)
    return round((h % 1000) / 100.0, 2)  # 0.00-9.99


def run(args) -> None:
    fds = load_fds(args.fds, n_for_mode(args.mode, args.n_queries))
    articles = load_articles(args.articles)
    print(f"[b6] Loaded {len(fds)} FDs; breadth={BREADTH} depth={DEPTH}")

    preds: list[dict] = []
    for fd in fds:
        ctx = concat_articles(fd.get("article_ids", []), articles)
        root_user = BASE_USER_YESNO.format(
            question=fd["question"],
            resolution_date=fd.get("resolution_date", "N/A"),
            context=ctx,
        )

        # Depth 1: generate + score BREADTH candidate thoughts
        d1_scores: list[tuple[int, float]] = []
        for b in range(BREADTH):
            if args.dry_run:
                _ = synthetic_response(fd["id"], variant=f"b6_d1_b{b}")
                score = _pseudo_score(fd["id"], f"d1_b{b}")
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            d1_scores.append((b, score))
        best_d1 = max(d1_scores, key=lambda x: x[1])[0]

        # Depth 2: expand best thought into BREADTH leaves, score each, take best
        d2_leaves: list[tuple[int, float, float]] = []
        for b in range(BREADTH):
            if args.dry_run:
                resp = synthetic_response(fd["id"], variant=f"b6_d2_parent{best_d1}_b{b}")
                score = _pseudo_score(fd["id"], f"d2_p{best_d1}_b{b}")
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            d2_leaves.append((b, score, extract_prob_yes(resp)))
        best_leaf = max(d2_leaves, key=lambda x: x[1])
        prob_yes = float(best_leaf[2])

        preds.append({
            "query_id": fd["id"],
            "prob_yes": round(prob_yes, 4),
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": gt_to_int(fd["ground_truth"]),
            "best_d1_branch": best_d1,
            "best_leaf_score": best_leaf[1],
        })
        if args.dry_run and len(preds) == 1:
            print(f"[b6] Root prompt length (first FD): {len(root_user)} chars")

    out_pred = Path("results/processed/predictions_b6.jsonl")
    out_met = Path("results/processed/metrics_b6.json")
    save_jsonl(preds, out_pred)
    metrics = evaluate_binary(preds)
    metrics["baseline"] = "b6_tree_of_thoughts"
    metrics["breadth"] = BREADTH
    metrics["depth"] = DEPTH
    metrics["dry_run"] = bool(args.dry_run)
    save_json(metrics, out_met)
    print(f"[b6] {'DRY-RUN ' if args.dry_run else ''}complete. "
          f"N={metrics['n']} Brier={metrics['brier_score']} ECE={metrics['ece']} "
          f"Acc={metrics['accuracy']}% -> {out_pred}")


def main():
    run(common_argparser("B6 Tree of Thoughts").parse_args())


if __name__ == "__main__":
    main()
