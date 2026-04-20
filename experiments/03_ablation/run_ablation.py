"""
Experiment group 3: Ablation study (Table 3 in paper).

Removes one component at a time from the full system.
Reuses cached batch outputs where possible; only runs new batches when retrieval changes.

Ablations:
  full            — reference (must run 02_emrach first)
  no_multiagent   — replace multi-agent with single-agent (reuses A, I)
  no_deepanalysis — skip Step 7 (reuses A, I, judge output)
  no_diag         — d_j = 1 uniformly (CPU only)
  no_absence      — skip background row (CPU only)
  no_calib        — heuristic mapping (CPU only)
  no_contrastive  — generic indicator prompt (NEW Step 1+2 batch)
  no_mmr          — BM25-only retrieval (NEW Step 4 batch)
  no_decay        — no temporal decay in retrieval (NEW Step 4 batch)

Usage:
  python experiments/03_ablation/run_ablation.py --mode smoke
  python experiments/03_ablation/run_ablation.py
"""

import json
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    print_cost_estimate, load_predictions, timing,
)


def main():
    args = parse_args("Ablation study")
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

    # ------------------------------------------------------------------ #
    # Shared: load/compute full system intermediate results               #
    # ------------------------------------------------------------------ #
    print("\nComputing shared intermediate results (full-system retrieval)...")

    # Contrastive indicators (shared with full system)
    ind_reqs = build_indicator_requests(queries, model=model, contrastive=True, config=cfg)
    ind_results = client.run(ind_reqs, job_name=f"ablation_{mode}_indicators_contrastive")
    indicators_contrastive = parse_indicator_responses(ind_results, queries, cfg)

    inf_reqs = build_influence_requests(queries, indicators_contrastive, model=model, config=cfg)
    inf_results = client.run(inf_reqs, job_name=f"ablation_{mode}_influence")
    I_by_query = parse_influence_responses(inf_results, queries, indicators_contrastive, cfg)

    # Full retrieval (with MMR + time decay)
    retriever_full = get_retriever(cfg, retrieval_type="mock" if mode == "smoke" else None)
    articles_full = retriever_full.retrieve_batch(queries)

    pres_reqs = build_presence_requests(queries, articles_full, indicators_contrastive, model=model, config=cfg)
    pres_results = client.run(pres_reqs, job_name=f"ablation_{mode}_presence_full")
    A_by_query_full = parse_presence_responses(pres_results, queries, articles_full, indicators_contrastive, cfg)

    bg_reqs = build_background_prior_requests(queries, indicators_contrastive, model=model, config=cfg)
    bg_results = client.run(bg_reqs, job_name=f"ablation_{mode}_background")
    phi_by_query = parse_background_prior_responses(bg_results, queries, indicators_contrastive)

    # Multi-agent outputs (shared)
    n_agents = cfg.get("pipeline", "multi_agent", "n_agents", default=4)
    adv_reqs = build_advocate_requests(queries, articles_full, indicators_contrastive, n_agents=n_agents, model=model, config=cfg)
    adv_results = client.run(adv_reqs, job_name=f"ablation_{mode}_advocates")
    advocates_by_query = parse_advocate_responses(adv_results, queries, n_agents)
    judge_reqs = build_judge_requests(queries, advocates_by_query, model=model, config=cfg)
    judge_results = client.run(judge_reqs, job_name=f"ablation_{mode}_judge")
    judge_by_query = parse_judge_responses(judge_results, queries)

    # Deep analysis outputs (shared)
    da_reqs = build_deep_analysis_requests(queries, articles_full, model=model, config=cfg)
    da_results = client.run(da_reqs, job_name=f"ablation_{mode}_deepanalysis")
    da_by_query = parse_deep_analysis_responses(da_results, queries, articles_full)

    # ------------------------------------------------------------------ #
    # New batches for retrieval ablations                                 #
    # ------------------------------------------------------------------ #
    # no_mmr: BM25-only retrieval
    retriever_bm25 = get_retriever(cfg, retrieval_type="mock" if mode == "smoke" else "weaviate_bm25")
    articles_bm25 = retriever_bm25.retrieve_batch(queries) if mode != "smoke" else articles_full
    pres_reqs_bm25 = build_presence_requests(queries, articles_bm25, indicators_contrastive, model=model, config=cfg)
    pres_results_bm25 = client.run(pres_reqs_bm25, job_name=f"ablation_{mode}_presence_bm25")
    A_by_query_bm25 = parse_presence_responses(pres_results_bm25, queries, articles_bm25, indicators_contrastive, cfg)

    # no_decay: no temporal decay
    # For smoke mode, reuse full results (mock retriever doesn't implement decay)
    A_by_query_nodecay = A_by_query_full  # same in smoke mode
    if mode != "smoke":
        retriever_nodecay = get_retriever(cfg, retrieval_type="weaviate_nodecay")
        articles_nodecay = retriever_nodecay.retrieve_batch(queries)
        pres_reqs_nd = build_presence_requests(queries, articles_nodecay, indicators_contrastive, model=model, config=cfg)
        pres_results_nd = client.run(pres_reqs_nd, job_name=f"ablation_{mode}_presence_nodecay")
        A_by_query_nodecay = parse_presence_responses(pres_results_nd, queries, articles_nodecay, indicators_contrastive, cfg)

    # no_contrastive: generic (non-contrastive) indicators
    ind_reqs_gen = build_indicator_requests(queries, model=model, contrastive=False, config=cfg)
    ind_results_gen = client.run(ind_reqs_gen, job_name=f"ablation_{mode}_indicators_generic")
    indicators_generic = parse_indicator_responses(ind_results_gen, queries, cfg)
    inf_reqs_gen = build_influence_requests(queries, indicators_generic, model=model, config=cfg)
    inf_results_gen = client.run(inf_reqs_gen, job_name=f"ablation_{mode}_influence_generic")
    I_by_query_gen = parse_influence_responses(inf_results_gen, queries, indicators_generic, cfg)
    pres_reqs_gen = build_presence_requests(queries, articles_full, indicators_generic, model=model, config=cfg)
    pres_results_gen = client.run(pres_reqs_gen, job_name=f"ablation_{mode}_presence_generic")
    A_by_query_gen = parse_presence_responses(pres_results_gen, queries, articles_full, indicators_generic, cfg)

    # ------------------------------------------------------------------ #
    # Compute all ablation variants                                       #
    # ------------------------------------------------------------------ #
    ablations = {
        "full": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_full, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=True,
            use_multiagent=True, use_da=True,
        ),
        "no_multiagent": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_full, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=True,
            use_multiagent=False, use_da=True,
        ),
        "no_deepanalysis": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_full, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=True,
            use_multiagent=True, use_da=False,
        ),
        "no_diag": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_full, phi=phi_by_query,
            use_cal=True, use_diag=False, use_absence=True,
            use_multiagent=True, use_da=True,
        ),
        "no_absence": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_full, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=False,
            use_multiagent=True, use_da=True,
        ),
        "no_calib": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_full, phi=phi_by_query,
            use_cal=False, use_diag=True, use_absence=True,
            use_multiagent=True, use_da=True,
        ),
        "no_contrastive": dict(
            indicators=indicators_generic, I=I_by_query_gen, A=A_by_query_gen, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=True,
            use_multiagent=True, use_da=True,
        ),
        "no_mmr": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_bm25, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=True,
            use_multiagent=True, use_da=True,
        ),
        "no_decay": dict(
            indicators=indicators_contrastive, I=I_by_query, A=A_by_query_nodecay, phi=phi_by_query,
            use_cal=True, use_diag=True, use_absence=True,
            use_multiagent=True, use_da=True,
        ),
    }

    print("\n--- Ablation Results ---")
    all_results = {}
    for abl_name, abl_cfg in ablations.items():
        preds = _compute_ablation(
            queries, abl_cfg, judge_by_query, da_by_query, cal,
        )
        save_predictions(preds, results_dir / f"predictions_abl_{abl_name}.jsonl")
        if len(set(labels)) > 1:
            result = evaluate(preds, labels, bootstrap_n=0)
            all_results[abl_name] = result
            delta = ""
            if abl_name != "full" and "full" in all_results:
                df = result.f1 - all_results["full"].f1
                delta = f"  ΔF1={df*100:+.1f}%"
            print(f"  {abl_name:20s}: {result}{delta}")
            save_metrics(result.to_dict(), results_dir / f"metrics_abl_{abl_name}.json")

    print("\nAblation study complete.")


def _compute_ablation(queries, abl, judge_by_query, da_by_query, cal):
    preds = []
    for q in queries:
        indicators = abl["indicators"].get(q.id, [])
        m = len(indicators)
        I = abl["I"].get(q.id, np.full((m, 4), 0.25))
        A = abl["A"].get(q.id, np.zeros((5, m)))
        phi = abl["phi"].get(q.id, np.full(m, 0.5))
        use_cal = abl["use_cal"]
        use_diag = abl["use_diag"]
        use_absence = abl["use_absence"]
        use_multiagent = abl["use_multiagent"]
        use_da = abl["use_da"]

        if use_multiagent and q.id in judge_by_query:
            j = judge_by_query[q.id]
            ranking = j["ranking"]
            probs = j["probabilities"]
        else:
            if use_cal:
                A = cal(A)
                I = cal(I)
            A_tilde = build_augmented_A(A, phi, cal if use_cal else None) if use_absence else A
            agg = aggregate(A_tilde, I, use_diagnostic_weighting=use_diag)
            ranking = agg["ranking"]
            probs = {h: float(agg["probs"][i]) for i, h in enumerate(HYPOTHESES)}

        if use_da and q.id in da_by_query:
            ranking = apply_deep_analysis(ranking, da_by_query[q.id])

        preds.append({
            "query_id": q.id,
            "prediction": ranking[0],
            "probabilities": probs,
            "ranking": ranking,
        })
    return preds


if __name__ == "__main__":
    main()
