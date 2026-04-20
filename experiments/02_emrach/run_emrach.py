"""
Experiment group 2: EMR-ACH staged build-up (reproduces Table 1 in paper).

Computes ALL rows of Table 1 from cached intermediate results where possible.
Each new row reuses batch outputs from prior rows.

Rows:
  emrach_base     — indicators + influence + presence + aggregate (no extras)
  +calib          — + Platt calibration (CPU only, no new batch)
  +contrastive    — + contrastive indicator prompt (new Step 1+2 batch)
  +diag           — + diagnostic weighting (CPU only)
  +absence        — + absence-of-evidence row (new Step 4b batch)
  +multiagent     — + multi-agent debate (new advocate+judge batches)
  +deepanalysis   — + Deep Analysis (new Step 7 batch) = Full System

Usage:
  python experiments/02_emrach/run_emrach.py --mode smoke
  python experiments/02_emrach/run_emrach.py
"""

import json
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import MiraiDataset, make_mock_queries, HYPOTHESES
from src.eval.metrics import evaluate
from src.pipeline.aggregation import aggregate
from src.pipeline.calibration import PlattCalibration
from src.pipeline.deep_analysis import (
    build_deep_analysis_requests, parse_deep_analysis_responses, apply_deep_analysis,
)
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
    args = parse_args("EMR-ACH staged build-up")
    cfg = get_config(args.config)
    mode = args.mode
    n = get_n_queries(mode, args)

    # In smoke mode, use direct API + mock retrieval
    client = get_client(mode, cfg)
    retriever = get_retriever(cfg, retrieval_type="mock" if mode == "smoke" else None)
    model = cfg.smoke_model if mode == "smoke" else cfg.experiment_model
    cal = PlattCalibration.default(cfg)

    try:
        ds = MiraiDataset(cfg)
        queries = ds.queries(n)
    except FileNotFoundError:
        print("[WARN] Real data not found. Using mock queries.")
        queries = make_mock_queries(n or 5)

    labels = [q.label for q in queries]
    results_dir = cfg.results_dir / "processed"
    print(f"\nMode={mode}  N={len(queries)}  Model={model}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Stage 1: Generate indicators (contrastive prompt)                   #
    # ------------------------------------------------------------------ #
    print("\n[Stage 1] Generating contrastive indicators...")
    with timing("indicators"):
        ind_reqs = build_indicator_requests(queries, model=model, contrastive=True, config=cfg)
        if not args.dry_run:
            ind_results = client.run(ind_reqs, job_name=f"emrach_{mode}_indicators_contrastive")
            indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)
        else:
            print_cost_estimate(client, ind_reqs)
            return

    # ------------------------------------------------------------------ #
    # Stage 2: Score influence matrix                                     #
    # ------------------------------------------------------------------ #
    print("\n[Stage 2] Scoring influence matrix...")
    with timing("influence"):
        inf_reqs = build_influence_requests(queries, indicators_by_query, model=model, config=cfg)
        print_cost_estimate(client, inf_reqs)
        inf_results = client.run(inf_reqs, job_name=f"emrach_{mode}_influence")
        I_by_query = parse_influence_responses(inf_results, queries, indicators_by_query, cfg)

    # ------------------------------------------------------------------ #
    # Stage 3: Retrieve articles                                          #
    # ------------------------------------------------------------------ #
    print("\n[Stage 3] Retrieving articles...")
    with timing("retrieval"):
        n_articles = cfg.get("pipeline", "n", default=10)
        articles_by_query = retriever.retrieve_batch(queries, n=n_articles)

    # ------------------------------------------------------------------ #
    # Stage 4: Score presence matrix                                      #
    # ------------------------------------------------------------------ #
    print("\n[Stage 4] Scoring presence matrix...")
    with timing("presence"):
        pres_reqs = build_presence_requests(
            queries, articles_by_query, indicators_by_query, model=model, config=cfg)
        print_cost_estimate(client, pres_reqs)
        pres_results = client.run(pres_reqs, job_name=f"emrach_{mode}_presence")
        A_by_query = parse_presence_responses(
            pres_results, queries, articles_by_query, indicators_by_query, cfg)

    # ------------------------------------------------------------------ #
    # Stage 4b: Background priors                                         #
    # ------------------------------------------------------------------ #
    print("\n[Stage 4b] Scoring background priors...")
    with timing("background"):
        bg_reqs = build_background_prior_requests(queries, indicators_by_query, model=model, config=cfg)
        print_cost_estimate(client, bg_reqs)
        bg_results = client.run(bg_reqs, job_name=f"emrach_{mode}_background")
        phi_by_query = parse_background_prior_responses(bg_results, queries, indicators_by_query)

    # ------------------------------------------------------------------ #
    # Build-up: compute each row of Table 1                               #
    # ------------------------------------------------------------------ #
    rows = [
        # (name, use_cal, use_contrastive, use_diag, use_absence, use_da)
        ("emrach_base",      False, False, False, False, False),
        ("emrach_calib",     True,  False, False, False, False),
        ("emrach_contrast",  True,  True,  False, False, False),
        ("emrach_diag",      True,  True,  True,  False, False),
        ("emrach_absence",   True,  True,  True,  True,  False),
    ]

    all_predictions = {}
    for name, use_cal, use_contrastive, use_diag, use_absence, use_da in rows:
        print(f"\n--- {name} ---")
        preds = _aggregate_all(
            queries, indicators_by_query, I_by_query, A_by_query, phi_by_query,
            use_cal=use_cal, use_diag=use_diag, use_absence=use_absence,
            cal=cal,
        )
        all_predictions[name] = preds
        save_predictions(preds, results_dir / f"predictions_{name}.jsonl")
        if len(set(labels)) > 1:
            result = evaluate(preds, labels, bootstrap_n=0)
            print(f"  {result}")
            save_metrics(result.to_dict(), results_dir / f"metrics_{name}.json")

    # ------------------------------------------------------------------ #
    # Stage: Multi-agent debate (+multiagent row)                         #
    # ------------------------------------------------------------------ #
    print("\n[Stage: Multi-agent] Running advocate + judge...")
    n_agents = cfg.get("pipeline", "multi_agent", "n_agents", default=4)
    with timing("advocates"):
        adv_reqs = build_advocate_requests(
            queries, articles_by_query, indicators_by_query,
            n_agents=n_agents, model=model, config=cfg,
        )
        print_cost_estimate(client, adv_reqs)
        adv_results = client.run(adv_reqs, job_name=f"emrach_{mode}_advocates")
        advocates_by_query = parse_advocate_responses(adv_results, queries, n_agents)

    with timing("judge"):
        judge_reqs = build_judge_requests(queries, advocates_by_query, model=model, config=cfg)
        print_cost_estimate(client, judge_reqs)
        judge_results = client.run(judge_reqs, job_name=f"emrach_{mode}_judge")
        judge_by_query = parse_judge_responses(judge_results, queries)

    ma_preds = [
        {
            "query_id": q.id,
            "prediction": judge_by_query[q.id]["ranking"][0],
            "probabilities": judge_by_query[q.id]["probabilities"],
            "ranking": judge_by_query[q.id]["ranking"],
        }
        for q in queries
    ]
    save_predictions(ma_preds, results_dir / "predictions_emrach_multiagent.jsonl")
    if len(set(labels)) > 1:
        result = evaluate(ma_preds, labels, bootstrap_n=0)
        print(f"  +multiagent: {result}")
        save_metrics(result.to_dict(), results_dir / "metrics_emrach_multiagent.json")

    # ------------------------------------------------------------------ #
    # Stage: Deep Analysis (+deepanalysis = Full System)                  #
    # ------------------------------------------------------------------ #
    print("\n[Stage: Deep Analysis]...")
    with timing("deep analysis"):
        da_reqs = build_deep_analysis_requests(
            queries, articles_by_query, model=model, config=cfg)
        print_cost_estimate(client, da_reqs)
        da_results = client.run(da_reqs, job_name=f"emrach_{mode}_deepanalysis")
        da_by_query = parse_deep_analysis_responses(da_results, queries, articles_by_query)

    # Apply DA refinement on top of multi-agent predictions
    full_preds = []
    for pred, q in zip(ma_preds, queries):
        da = da_by_query.get(q.id)
        refined = apply_deep_analysis(pred["ranking"], da)
        full_preds.append({**pred, "prediction": refined[0], "ranking": refined})

    save_predictions(full_preds, results_dir / "predictions_emrach_full.jsonl")
    if len(set(labels)) > 1:
        result = evaluate(full_preds, labels, bootstrap_n=(0 if mode == "smoke" else 1000))
        print(f"  Full system: {result}")
        save_metrics(result.to_dict(), results_dir / "metrics_emrach_full.json")

    print("\nEMR-ACH build-up complete. Results in", results_dir)


def _aggregate_all(
    queries, indicators_by_query, I_by_query, A_by_query, phi_by_query,
    use_cal, use_diag, use_absence, cal,
) -> list[dict]:
    preds = []
    for q in queries:
        A = A_by_query.get(q.id, np.zeros((5, len(indicators_by_query.get(q.id, [])))))
        I = I_by_query.get(q.id, np.full((len(indicators_by_query.get(q.id, [])), 4), 0.25))
        phi = phi_by_query.get(q.id, np.full(len(indicators_by_query.get(q.id, [])), 0.5))

        if use_cal:
            A = cal(A)
            I = cal(I)

        if use_absence:
            A_tilde = build_augmented_A(A, phi, cal if use_cal else None)
        else:
            A_tilde = A

        agg = aggregate(A_tilde, I, use_diagnostic_weighting=use_diag)
        preds.append({
            "query_id": q.id,
            "prediction": agg["prediction"],
            "probabilities": {h: float(agg["probs"][i]) for i, h in enumerate(HYPOTHESES)},
            "ranking": agg["ranking"],
        })
    return preds


if __name__ == "__main__":
    main()
