"""
Smoke test: Step 1 — Indicator generation.

Runs 5 queries, prints indicators, validates JSON and coverage.
Iterate on prompts/indicators.yaml until this passes cleanly.

Run: python experiments/00_smoke/smoke_indicators.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import MiraiDataset, make_mock_queries
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses

N = 5
HYPOTHESES = ["VC", "MC", "VK", "MK"]


def main():
    cfg = get_config()
    client = BatchClient(mode="direct", config=cfg)

    # Load queries
    try:
        ds = MiraiDataset(cfg)
        queries = ds.queries(N)
        print(f"Using real MIRAI data ({len(queries)} queries)")
    except FileNotFoundError:
        queries = make_mock_queries(N)
        print(f"Real data not found. Using {len(queries)} mock queries.")

    # Build requests
    requests = build_indicator_requests(
        queries,
        model=cfg.smoke_model,
        contrastive=True,
        config=cfg,
    )
    print(f"\nBuilt {len(requests)} requests.")
    print(f"Estimated cost: {client.estimate_cost(requests)}\n")
    print("=" * 60)

    # Run
    results = client.run(requests, job_name="smoke_indicators")

    # Parse and validate
    indicators_by_query = parse_indicator_responses(results, queries, cfg)

    n_pass = 0
    for q in queries:
        indicators = indicators_by_query.get(q.id, [])
        print(f"\n{'='*60}")
        print(f"Query: {q.query_text}  |  Ground truth: {q.label}")
        print(f"Indicators generated: {len(indicators)}")

        coverage = {h: 0 for h in HYPOTHESES}
        for ind in indicators:
            ps = ind.get("primarily_supports", "?")
            coverage[ps] = coverage.get(ps, 0) + 1

        print(f"Coverage: {coverage}")
        print("Top 5 indicators:")
        for ind in indicators[:5]:
            print(f"  {ind['id']:2d}. [{ind['primarily_supports']}] {ind['text']}")

        # Validation
        issues = []
        if len(indicators) < 20:
            issues.append(f"Too few indicators: {len(indicators)} (expected 24)")
        for h in HYPOTHESES:
            if coverage.get(h, 0) == 0:
                issues.append(f"Zero coverage for hypothesis {h}")
        if any(not ind.get("text") for ind in indicators):
            issues.append("Some indicators have empty text")

        if issues:
            print(f"  ISSUES: {issues}")
        else:
            print(f"  PASS")
            n_pass += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_pass}/{len(queries)} queries passed validation")
    if n_pass < len(queries):
        print("  -> Edit prompts/indicators.yaml and re-run until all pass.")
    else:
        print("  -> Indicators prompt is ready for full experiment.")


if __name__ == "__main__":
    main()
