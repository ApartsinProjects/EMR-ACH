"""
Smoke test: Step 2 — Influence matrix scoring.

Prerequisites: smoke_indicators.py must pass first.
Run: python experiments/00_smoke/smoke_influence.py
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import make_mock_queries
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses
from src.pipeline.influence import (
    build_influence_requests,
    parse_influence_responses,
    compute_diagnosticity_weights,
)

N = 5
HYPOTHESES = ["VC", "MC", "VK", "MK"]


def main():
    cfg = get_config()
    client = BatchClient(mode="direct", config=cfg)

    try:
        from src.data.mirai import MiraiDataset
        ds = MiraiDataset(cfg)
        queries = ds.queries(N)
    except FileNotFoundError:
        queries = make_mock_queries(N)
        print("Using mock queries.")

    # Step 1: get indicators
    print("Step 1: generating indicators...")
    ind_requests = build_indicator_requests(queries, model=cfg.smoke_model, config=cfg)
    ind_results = client.run(ind_requests, job_name="smoke_influence_indicators")
    indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)

    # Step 2: score influence
    print("\nStep 2: scoring influence...")
    inf_requests = build_influence_requests(queries, indicators_by_query, model=cfg.smoke_model, config=cfg)
    print(f"Built {len(inf_requests)} requests. {client.estimate_cost(inf_requests)}")
    inf_results = client.run(inf_requests, job_name="smoke_influence_matrix")
    influence_matrices = parse_influence_responses(inf_results, queries, indicators_by_query, cfg)

    n_pass = 0
    for q in queries:
        I = influence_matrices.get(q.id)
        indicators = indicators_by_query.get(q.id, [])
        print(f"\n{'='*60}")
        print(f"Query: {q.query_text}  |  GT: {q.label}")

        if I is None:
            print("  FAIL: No influence matrix")
            continue

        m, h = I.shape
        print(f"  I matrix shape: {m} x {h}  (values: min={I.min():.3f} max={I.max():.3f} mean={I.mean():.3f})")

        d = compute_diagnosticity_weights(I)
        print(f"  Diagnosticity weights d_j: min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f}")

        # Top 3 most diagnostic indicators
        top_idx = np.argsort(d)[::-1][:3]
        print("  Top 3 most diagnostic indicators:")
        for idx in top_idx:
            text = indicators[idx]["text"] if idx < len(indicators) else "?"
            row = I[idx]
            print(f"    [{idx+1}] d={d[idx]:.4f}  [{HYPOTHESES[0]}={row[0]:.2f} {HYPOTHESES[1]}={row[1]:.2f} {HYPOTHESES[2]}={row[2]:.2f} {HYPOTHESES[3]}={row[3]:.2f}]  {text[:60]}")

        # Validation
        issues = []
        if not (0 <= I.min() and I.max() <= 1.0):
            issues.append(f"Values outside [0,1]: min={I.min():.3f} max={I.max():.3f}")
        if d.max() < 0.001:
            issues.append("All indicators have near-zero diagnosticity (all scores equal)")
        uniform_rows = sum(1 for j in range(m) if np.std(I[j]) < 0.05)
        if uniform_rows > m * 0.5:
            issues.append(f"{uniform_rows}/{m} indicators have near-uniform scores (not discriminative)")

        if issues:
            print(f"  ISSUES: {issues}")
            print("  -> Edit prompts/influence.yaml to emphasize score diversity.")
        else:
            print("  PASS")
            n_pass += 1

    print(f"\nSUMMARY: {n_pass}/{len(queries)} passed")


if __name__ == "__main__":
    main()
