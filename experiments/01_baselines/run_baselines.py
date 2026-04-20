"""
Experiment group 1: Baselines B1 (direct), B2 (CoT), B3 (RAG-only).

Usage:
  python experiments/01_baselines/run_baselines.py --mode smoke  # 5 queries, direct API
  python experiments/01_baselines/run_baselines.py               # full, batch API
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.data.mirai import MiraiDataset, make_mock_queries
from src.eval.metrics import evaluate
from src.baselines.direct_prompting import build_direct_requests, parse_direct_responses
from src.baselines.cot import build_cot_requests, parse_cot_responses
from src.baselines.rag_only import build_rag_only_requests, parse_rag_only_responses
from src.pipeline.retrieval import get_retriever
from experiments.runner import (
    parse_args, get_client, get_n_queries, save_predictions, save_metrics,
    print_cost_estimate, timing,
)


def run_b1_direct(queries, client, cfg, dry_run=False):
    with timing("B1 direct prompting"):
        reqs = build_direct_requests(queries, model=cfg.experiment_model, config=cfg)
        print_cost_estimate(client, reqs)
        if dry_run:
            return None
        results = client.run(reqs, job_name="b1_direct")
        preds = parse_direct_responses(results, queries)
    return preds


def run_b2_cot(queries, client, cfg, dry_run=False):
    with timing("B2 CoT"):
        reqs = build_cot_requests(queries, model=cfg.experiment_model, config=cfg)
        print_cost_estimate(client, reqs)
        if dry_run:
            return None
        results = client.run(reqs, job_name="b2_cot")
        preds = parse_cot_responses(results, queries)
    return preds


def run_b3_rag_only(queries, client, cfg, dry_run=False, mode="full"):
    retriever = get_retriever(cfg, retrieval_type="mock" if mode == "smoke" else None)
    with timing("Retrieval"):
        articles_by_query = retriever.retrieve_batch(queries)
    with timing("B3 RAG-only"):
        reqs = build_rag_only_requests(queries, articles_by_query, model=cfg.experiment_model, config=cfg)
        print_cost_estimate(client, reqs)
        if dry_run:
            return None
        results = client.run(reqs, job_name="b3_ragonly")
        preds = parse_rag_only_responses(results, queries)
    return preds


def main():
    args = parse_args("Baseline experiments (B1-B3)")
    cfg = get_config(args.config)
    client = get_client(args.mode, cfg)
    n = get_n_queries(args.mode, args)

    # Load data
    try:
        ds = MiraiDataset(cfg)
        queries = ds.queries(n)
    except FileNotFoundError:
        print("[WARN] Real data not found, using mock queries")
        queries = make_mock_queries(n or 5)

    labels = [q.label for q in queries]
    results_dir = cfg.results_dir / "processed"
    print(f"\nMode={args.mode}  Queries={len(queries)}  Model={cfg.experiment_model}")
    print("=" * 60)

    # Run each baseline
    for name, run_fn in [("b1_direct", run_b1_direct), ("b2_cot", run_b2_cot), ("b3_ragonly", run_b3_rag_only)]:
        print(f"\n--- {name} ---")
        kwargs = {"dry_run": args.dry_run}
        if name == "b3_ragonly":
            kwargs["mode"] = args.mode
        preds = run_fn(queries, client, cfg, **kwargs)
        if preds is None:
            continue

        save_predictions(preds, results_dir / f"predictions_{name}.jsonl")

        if len(set(labels)) > 1:
            result = evaluate(preds, labels, bootstrap_n=(0 if args.mode == "smoke" else 1000))
            print(f"  Results: {result}")
            save_metrics(result.to_dict(), results_dir / f"metrics_{name}.json")
        else:
            print("  (Cannot compute F1: all labels identical in smoke set)")

    print("\nBaseline experiments complete.")


if __name__ == "__main__":
    main()
