"""
Smoke test: Step 7 — Deep Analysis (V/M disambiguation).

Run: python experiments/00_smoke/smoke_deep_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import make_mock_queries
from src.pipeline.deep_analysis import (
    build_deep_analysis_requests,
    parse_deep_analysis_responses,
    apply_deep_analysis,
    SCORING,
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
        print("Using mock queries and articles.")

    articles_by_query = retriever.retrieve_batch(queries, n=5)

    print("Running Deep Analysis...")
    reqs = build_deep_analysis_requests(queries, articles_by_query, model=cfg.smoke_model, config=cfg)
    print(f"  {len(reqs)} requests. {client.estimate_cost(reqs)}")
    results = client.run(reqs, job_name="smoke_deep_analysis")
    da_by_query = parse_deep_analysis_responses(results, queries, articles_by_query)

    valid_labels = set(SCORING.keys())
    n_pass = 0
    for q in queries:
        da = da_by_query.get(q.id, {})
        print(f"\n{'='*60}")
        print(f"Query: {q.query_text}  |  GT: {q.label}")
        print(f"  Verbal score: {da.get('verbal_score', 0):.1f}")
        print(f"  Material score: {da.get('material_score', 0):.1f}")
        print(f"  V/M winner: {da.get('vm_winner', '?')}")

        classifications = da.get("article_classifications", [])
        issues = []
        invalid = [c for c in classifications if c["classification"] not in valid_labels]
        if invalid:
            issues.append(f"{len(invalid)} invalid classifications: {[c['classification'] for c in invalid]}")
        if not classifications:
            issues.append("No article classifications returned")

        print("  Article classifications:")
        for c in classifications[:5]:
            print(f"    [{c['article_id']}] {c['classification']}  — {c.get('reasoning', '')[:80]}")

        # Test apply_deep_analysis
        initial_ranking = ["VK", "VC", "MK", "MC"]  # hypothetical
        refined = apply_deep_analysis(initial_ranking, da)
        print(f"  Refinement: {initial_ranking[:2]} -> {refined[:2]}")

        if issues:
            print(f"  ISSUES: {issues}")
            print("  -> Edit prompts/deep_analysis.yaml")
        else:
            print("  PASS")
            n_pass += 1

    print(f"\nSUMMARY: {n_pass}/{len(queries)} passed")


if __name__ == "__main__":
    main()
