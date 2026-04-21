"""
B5: Multi-Agent Debate (Du et al., 2023).
3 agents x 2 rounds. Round 1: each agent independently produces a CoT prediction.
Round 2: each agent sees the other two answers and revises. Final = mean over agents
(using round-2 probabilities).

Usage:
  /c/Python314/python experiments/06_baselines/run_b5_multi_agent_debate.py \
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

N_AGENTS = 3
N_ROUNDS = 2


def run(args) -> None:
    fds = load_fds(args.fds, n_for_mode(args.mode, args.n_queries))
    articles = load_articles(args.articles)
    print(f"[b5] Loaded {len(fds)} FDs; agents={N_AGENTS}, rounds={N_ROUNDS}")

    preds: list[dict] = []
    for fd in fds:
        ctx = concat_articles(fd.get("article_ids", []), articles)
        base_user = BASE_USER_YESNO.format(
            question=fd["question"],
            resolution_date=fd.get("resolution_date", "N/A"),
            context=ctx,
        )

        # Round 1: independent agents
        round1_probs: list[float] = []
        for a in range(N_AGENTS):
            if args.dry_run:
                resp = synthetic_response(fd["id"], variant=f"b5_r1_a{a}")
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            round1_probs.append(extract_prob_yes(resp))

        # Round 2: each agent revises given peers' answers
        round2_probs: list[float] = []
        for a in range(N_AGENTS):
            peer_probs = [round1_probs[j] for j in range(N_AGENTS) if j != a]
            peer_block = "\n".join(
                f"- Peer agent {j+1}: Yes probability = {p:.3f}"
                for j, p in enumerate(peer_probs)
            )
            revised_user = (
                f"{base_user}\n\nYou previously estimated Yes={round1_probs[a]:.3f}. "
                f"Other agents reported:\n{peer_block}\n"
                f"Reconsider and return a revised JSON forecast."
            )
            if args.dry_run:
                resp = synthetic_response(fd["id"], variant=f"b5_r2_a{a}")
            else:
                raise RuntimeError("Non-dry-run disabled: OpenAI quota exhausted.")
            round2_probs.append(extract_prob_yes(resp))
            if args.dry_run and len(preds) == 0 and a == 0:
                print(f"[b5] Round-2 prompt length (first FD, agent 0): {len(revised_user)} chars")

        prob_yes = float(np.mean(round2_probs))
        preds.append({
            "query_id": fd["id"],
            "prob_yes": round(prob_yes, 4),
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": gt_to_int(fd["ground_truth"]),
            "round1": [round(p, 4) for p in round1_probs],
            "round2": [round(p, 4) for p in round2_probs],
        })

    out_pred = Path("results/processed/predictions_b5.jsonl")
    out_met = Path("results/processed/metrics_b5.json")
    save_jsonl(preds, out_pred)
    metrics = evaluate_binary(preds)
    metrics["baseline"] = "b5_multi_agent_debate"
    metrics["n_agents"] = N_AGENTS
    metrics["n_rounds"] = N_ROUNDS
    metrics["dry_run"] = bool(args.dry_run)
    save_json(metrics, out_met)
    print(f"[b5] {'DRY-RUN ' if args.dry_run else ''}complete. "
          f"N={metrics['n']} Brier={metrics['brier_score']} ECE={metrics['ece']} "
          f"Acc={metrics['accuracy']}% -> {out_pred}")


def main():
    run(common_argparser("B5 Multi-Agent Debate").parse_args())


if __name__ == "__main__":
    main()
