"""
Smoke test: Full end-to-end pipeline (all 7 steps).

This is the final smoke test — run it after all individual step tests pass.
Success criterion: at least 3/5 queries have KL < 1.0 (not degenerate output).

Run: python experiments/00_smoke/smoke_pipeline_e2e.py
"""

import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import make_mock_queries, HYPOTHESES
from src.eval.metrics import evaluate
from src.pipeline.aggregation import aggregate
from src.pipeline.calibration import PlattCalibration
from src.pipeline.deep_analysis import (
    build_deep_analysis_requests,
    parse_deep_analysis_responses,
    apply_deep_analysis,
)
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses
from src.pipeline.influence import build_influence_requests, parse_influence_responses
from src.pipeline.presence import (
    build_presence_requests,
    parse_presence_responses,
    build_background_prior_requests,
    parse_background_prior_responses,
    build_augmented_A,
)
from src.pipeline.retrieval import MockRetriever

N = 5


def main():
    cfg = get_config()
    client = BatchClient(mode="direct", config=cfg)
    retriever = MockRetriever()
    cal = PlattCalibration.default(cfg)

    try:
        from src.data.mirai import MiraiDataset
        ds = MiraiDataset(cfg)
        queries = ds.queries(N)
        print(f"Using real MIRAI data ({len(queries)} queries)")
    except FileNotFoundError:
        queries = make_mock_queries(N)
        print(f"Using {len(queries)} mock queries")

    # Step 1: indicators
    print("\n--- Step 1: Indicator generation ---")
    ind_reqs = build_indicator_requests(queries, model=cfg.smoke_model, config=cfg)
    ind_results = client.run(ind_reqs, job_name="smoke_e2e_step1")
    indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)

    # Step 2: influence matrix
    print("\n--- Step 2: Influence scoring ---")
    inf_reqs = build_influence_requests(queries, indicators_by_query, model=cfg.smoke_model, config=cfg)
    inf_results = client.run(inf_reqs, job_name="smoke_e2e_step2")
    I_by_query = parse_influence_responses(inf_results, queries, indicators_by_query, cfg)

    # Step 3: retrieve articles
    print("\n--- Step 3: Article retrieval (mock) ---")
    articles_by_query = retriever.retrieve_batch(queries, n=cfg.get("pipeline", "n", default=10))

    # Step 4: presence matrix
    print("\n--- Step 4: Presence scoring ---")
    pres_reqs = build_presence_requests(queries, articles_by_query, indicators_by_query, model=cfg.smoke_model, config=cfg)
    pres_results = client.run(pres_reqs, job_name="smoke_e2e_step4")
    A_by_query = parse_presence_responses(pres_results, queries, articles_by_query, indicators_by_query, cfg)

    # Step 4b: background prior
    print("\n--- Step 4b: Background prior ---")
    bg_reqs = build_background_prior_requests(queries, indicators_by_query, model=cfg.smoke_model, config=cfg)
    bg_results = client.run(bg_reqs, job_name="smoke_e2e_step4b")
    phi_by_query = parse_background_prior_responses(bg_results, queries, indicators_by_query)

    # Step 5: aggregate
    print("\n--- Step 5: Aggregation ---")
    predictions_pre_da = []
    for q in queries:
        A = A_by_query.get(q.id, np.zeros((5, len(indicators_by_query[q.id]))))
        I = I_by_query.get(q.id, np.full((len(indicators_by_query[q.id]), 4), 0.5))
        phi = phi_by_query.get(q.id, np.full(len(indicators_by_query[q.id]), 0.5))
        A_tilde = build_augmented_A(A, phi, cal)
        agg = aggregate(A_tilde, I, use_diagnostic_weighting=True)
        predictions_pre_da.append({
            "query_id": q.id,
            "prediction": agg["prediction"],
            "probabilities": {h: float(agg["probs"][i]) for i, h in enumerate(HYPOTHESES)},
            "ranking": agg["ranking"],
        })

    # Step 7: deep analysis
    print("\n--- Step 7: Deep Analysis ---")
    da_reqs = build_deep_analysis_requests(queries, articles_by_query, model=cfg.smoke_model, config=cfg)
    da_results = client.run(da_reqs, job_name="smoke_e2e_step7")
    da_by_query = parse_deep_analysis_responses(da_results, queries, articles_by_query)

    # Apply deep analysis refinement
    final_predictions = []
    for pred, q in zip(predictions_pre_da, queries):
        da = da_by_query.get(q.id)
        refined_ranking = apply_deep_analysis(pred["ranking"], da)
        final_pred = dict(pred)
        final_pred["prediction"] = refined_ranking[0]
        final_pred["ranking"] = refined_ranking
        final_predictions.append(final_pred)

    # Evaluate
    labels = [q.label for q in queries]
    print("\n--- Results ---")
    for pred, q in zip(final_predictions, queries):
        p = pred["prediction"]
        probs = pred["probabilities"]
        correct = "CORRECT" if p == q.label else "WRONG"
        print(f"  {q.query_text}")
        print(f"    Prediction: {p} (GT: {q.label})  [{correct}]")
        print(f"    Probabilities: {' '.join(f'{h}={probs[h]:.2f}' for h in HYPOTHESES)}")

    if len(set(labels)) > 1:
        result = evaluate(final_predictions, labels, bootstrap_n=0)
        print(f"\nMicro metrics (n={len(queries)}, don't read too much into smoke test numbers):")
        print(f"  {result}")
    else:
        print("\n(All queries have same label — cannot compute F1 on smoke set)")

    # Sanity check: are we getting non-degenerate probability distributions?
    non_degenerate = 0
    for pred in final_predictions:
        probs_arr = np.array(list(pred["probabilities"].values()))
        kl_to_uniform = np.sum(probs_arr * np.log(probs_arr * 4 + 1e-9))
        if kl_to_uniform > 0.1:  # not uniform distribution
            non_degenerate += 1

    print(f"\nNon-degenerate predictions: {non_degenerate}/{len(queries)}")
    if non_degenerate >= len(queries) * 0.6:
        print("OVERALL: PASS — pipeline produces informative predictions")
    else:
        print("OVERALL: FAIL — most predictions are near-uniform (check LLM outputs)")


if __name__ == "__main__":
    main()
