"""
Smoke test: Step 4 — Indicator presence scoring.

Uses mock retrieval (no Weaviate needed).
Run: python experiments/00_smoke/smoke_presence.py
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import make_mock_queries
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses
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

    try:
        from src.data.mirai import MiraiDataset
        ds = MiraiDataset(cfg)
        queries = ds.queries(N)
    except FileNotFoundError:
        queries = make_mock_queries(N)
        print("Using mock queries and mock articles.")

    # Step 1: indicators
    print("Generating indicators...")
    ind_reqs = build_indicator_requests(queries, model=cfg.smoke_model, config=cfg)
    ind_results = client.run(ind_reqs, job_name="smoke_presence_indicators")
    indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)

    # Step 3: retrieve (mock)
    print("Retrieving articles (mock)...")
    articles_by_query = retriever.retrieve_batch(queries, n=cfg.get("pipeline", "n", default=10))

    # Step 4: presence scoring
    print("Scoring presence...")
    pres_reqs = build_presence_requests(queries, articles_by_query, indicators_by_query, model=cfg.smoke_model, config=cfg)
    print(f"  {len(pres_reqs)} requests. {client.estimate_cost(pres_reqs)}")
    pres_results = client.run(pres_reqs, job_name="smoke_presence_matrix")
    A_by_query = parse_presence_responses(pres_results, queries, articles_by_query, indicators_by_query, cfg)

    # Step 4b: background prior
    print("Scoring background priors...")
    bg_reqs = build_background_prior_requests(queries, indicators_by_query, model=cfg.smoke_model, config=cfg)
    bg_results = client.run(bg_reqs, job_name="smoke_presence_background")
    phi_by_query = parse_background_prior_responses(bg_results, queries, indicators_by_query)

    n_pass = 0
    for q in queries:
        A = A_by_query.get(q.id)
        phi = phi_by_query.get(q.id)
        print(f"\n{'='*60}")
        print(f"Query: {q.query_text}  |  GT: {q.label}")

        if A is None:
            print("  FAIL: No A matrix")
            continue

        n_arts, m = A.shape
        print(f"  A matrix: {n_arts} articles × {m} indicators")
        print(f"  A values: min={A.min():.3f} max={A.max():.3f} mean={A.mean():.3f}")
        print(f"  Non-zero cells: {(A > 0.1).sum()}/{A.size} ({100*(A>0.1).mean():.0f}%)")
        print(f"  phi (background): min={phi.min():.3f} max={phi.max():.3f}")

        # Build augmented matrix
        A_tilde = build_augmented_A(A, phi)
        print(f"  A_tilde shape: {A_tilde.shape} (added absence row)")

        issues = []
        if not (0 <= A.min() and A.max() <= 1.0):
            issues.append("A values outside [0,1]")
        if A.mean() > 0.8:
            issues.append("A mean too high — presence scores are inflated")
        if (A > 0.1).sum() == 0:
            issues.append("All A values are near zero — presence prompt may not be working")

        if issues:
            print(f"  ISSUES: {issues}")
        else:
            print("  PASS")
            n_pass += 1

    print(f"\nSUMMARY: {n_pass}/{len(queries)} passed")


if __name__ == "__main__":
    main()
