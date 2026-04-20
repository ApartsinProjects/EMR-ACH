"""
Experiment group 4: Multi-agent scaling (Figure 5 in paper).

Runs with N_agents ∈ {1, 2, 3, 4, 5, 6}.
Tracks F1 score and API calls per query vs. number of agents.

Usage:
  python experiments/04_multiagent/run_multiagent.py --mode smoke
  python experiments/04_multiagent/run_multiagent.py
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.data.mirai import MiraiDataset, make_mock_queries, HYPOTHESES
from src.eval.metrics import evaluate
from src.pipeline.aggregation import aggregate
from src.pipeline.calibration import PlattCalibration
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses
from src.pipeline.influence import build_influence_requests, parse_influence_responses
from src.pipeline.multi_agent import (
    build_advocate_requests, parse_advocate_responses,
    build_judge_requests, parse_judge_responses,
)
from src.pipeline.presence import (
    build_presence_requests, parse_presence_responses,
    build_background_prior_requests, parse_background_prior_responses,
    build_augmented_A,
)
from src.pipeline.retrieval import get_retriever
from experiments.runner import (
    parse_args, get_client, get_n_queries, save_predictions, save_metrics,
    print_cost_estimate, timing,
)


def main():
    args = parse_args("Multi-agent scaling")
    cfg = get_config(args.config)
    mode = args.mode
    n = get_n_queries(mode, args)
    client = get_client(mode, cfg)
    model = cfg.smoke_model if mode == "smoke" else cfg.experiment_model
    cal = PlattCalibration.default(cfg)
    results_dir = cfg.results_dir / "processed"

    try:
        ds = MiraiDataset(cfg)
        queries = ds.queries(n)
    except FileNotFoundError:
        queries = make_mock_queries(n or 5)
        print("[WARN] Using mock queries")

    labels = [q.label for q in queries]
    retriever = get_retriever(cfg, retrieval_type="mock" if mode == "smoke" else None)

    # Shared: indicators, influence, articles, presence
    print("Computing shared intermediates...")
    ind_reqs = build_indicator_requests(queries, model=model, config=cfg)
    ind_results = client.run(ind_reqs, job_name=f"ma_{mode}_indicators")
    indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)

    inf_reqs = build_influence_requests(queries, indicators_by_query, model=model, config=cfg)
    inf_results = client.run(inf_reqs, job_name=f"ma_{mode}_influence")
    I_by_query = parse_influence_responses(inf_results, queries, indicators_by_query, cfg)

    articles_by_query = retriever.retrieve_batch(queries)
    pres_reqs = build_presence_requests(queries, articles_by_query, indicators_by_query, model=model, config=cfg)
    pres_results = client.run(pres_reqs, job_name=f"ma_{mode}_presence")
    A_by_query = parse_presence_responses(pres_results, queries, articles_by_query, indicators_by_query, cfg)

    bg_reqs = build_background_prior_requests(queries, indicators_by_query, model=model, config=cfg)
    bg_results = client.run(bg_reqs, job_name=f"ma_{mode}_background")
    phi_by_query = parse_background_prior_responses(bg_results, queries, indicators_by_query)

    # N_agents = 1: single-agent baseline (just aggregation, no debate)
    agent_configs = [1, 2, 3, 4, 5, 6] if mode != "smoke" else [1, 2, 4]

    scaling_results = {}
    for n_agents in agent_configs:
        print(f"\n--- N_agents = {n_agents} ---")

        if n_agents == 1:
            # Single agent: use matrix aggregation only
            preds = []
            for q in queries:
                A = cal(A_by_query.get(q.id, np.zeros((5, len(indicators_by_query[q.id])))))
                I = cal(I_by_query.get(q.id, np.full((len(indicators_by_query[q.id]), 4), 0.25)))
                phi = phi_by_query.get(q.id, np.full(len(indicators_by_query[q.id]), 0.5))
                A_tilde = build_augmented_A(A, phi, cal)
                agg = aggregate(A_tilde, I, use_diagnostic_weighting=True)
                preds.append({
                    "query_id": q.id,
                    "prediction": agg["prediction"],
                    "probabilities": {h: float(agg["probs"][i]) for i, h in enumerate(HYPOTHESES)},
                    "ranking": agg["ranking"],
                })
            api_calls_per_query = 4  # indicators + influence + presence + background (all batched)
        else:
            # Multi-agent
            adv_reqs = build_advocate_requests(
                queries, articles_by_query, indicators_by_query,
                n_agents=n_agents, model=model, config=cfg,
            )
            adv_results = client.run(adv_reqs, job_name=f"ma_{mode}_advocates_{n_agents}")
            advocates_by_query = parse_advocate_responses(adv_results, queries, n_agents)

            judge_reqs = build_judge_requests(queries, advocates_by_query, model=model, config=cfg)
            judge_results = client.run(judge_reqs, job_name=f"ma_{mode}_judge_{n_agents}")
            judge_by_query = parse_judge_responses(judge_results, queries)

            preds = [
                {
                    "query_id": q.id,
                    "prediction": judge_by_query[q.id]["ranking"][0],
                    "probabilities": judge_by_query[q.id]["probabilities"],
                    "ranking": judge_by_query[q.id]["ranking"],
                }
                for q in queries
            ]
            api_calls_per_query = 4 + n_agents + 1  # shared + advocates + judge

        save_predictions(preds, results_dir / f"predictions_ma_agents{n_agents}.jsonl")
        if len(set(labels)) > 1:
            result = evaluate(preds, labels, bootstrap_n=0)
            scaling_results[n_agents] = {"result": result, "api_calls": api_calls_per_query}
            print(f"  {result}  API calls/query: {api_calls_per_query}")
            save_metrics(
                {**result.to_dict(), "n_agents": n_agents, "api_calls_per_query": api_calls_per_query},
                results_dir / f"metrics_ma_agents{n_agents}.json",
            )

    # Summary table
    print("\n=== Multi-Agent Scaling Summary ===")
    print(f"{'N_agents':>8}  {'F1%':>6}  {'Accuracy%':>10}  {'API calls/q':>12}")
    for na, d in scaling_results.items():
        r = d["result"]
        print(f"{na:>8}  {r.f1*100:>6.1f}  {r.accuracy*100:>10.1f}  {d['api_calls']:>12}")


if __name__ == "__main__":
    main()
