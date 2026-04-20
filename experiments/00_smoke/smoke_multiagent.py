"""
Smoke test: Multi-agent debate (advocate + judge).

Run: python experiments/00_smoke/smoke_multiagent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient
from src.config import get_config
from src.data.mirai import make_mock_queries, HYPOTHESES
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses
from src.pipeline.multi_agent import (
    build_advocate_requests,
    parse_advocate_responses,
    build_judge_requests,
    parse_judge_responses,
)
from src.pipeline.retrieval import MockRetriever

N = 3  # fewer queries — multi-agent is expensive even for smoke tests
N_AGENTS = 4


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
        print("Using mock queries.")

    # Get indicators
    ind_reqs = build_indicator_requests(queries, model=cfg.smoke_model, config=cfg)
    ind_results = client.run(ind_reqs, job_name="smoke_ma_indicators")
    indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)

    # Get articles
    articles_by_query = retriever.retrieve_batch(queries, n=5)

    # Stage A: advocates
    print(f"\nRunning {N_AGENTS} advocate agents per query...")
    adv_reqs = build_advocate_requests(
        queries, articles_by_query, indicators_by_query,
        n_agents=N_AGENTS, model=cfg.smoke_model, config=cfg,
    )
    print(f"  {len(adv_reqs)} advocate requests. {client.estimate_cost(adv_reqs)}")
    adv_results = client.run(adv_reqs, job_name="smoke_ma_advocates")
    advocates_by_query = parse_advocate_responses(adv_results, queries, N_AGENTS)

    # Stage B: judge
    print("Running judge agent...")
    judge_reqs = build_judge_requests(queries, advocates_by_query, model=cfg.smoke_model, config=cfg)
    judge_results = client.run(judge_reqs, job_name="smoke_ma_judge")
    judge_by_query = parse_judge_responses(judge_results, queries)

    n_pass = 0
    for q in queries:
        advocates = advocates_by_query.get(q.id, [])
        judge = judge_by_query.get(q.id, {})
        print(f"\n{'='*60}")
        print(f"Query: {q.query_text}  |  GT: {q.label}")

        print("  Advocate summaries:")
        for adv in advocates:
            print(f"    [{adv['hypothesis']}] conf={adv['confidence']:.2f}  {adv['argument_summary'][:80]}")

        probs = judge.get("probabilities", {})
        ranking = judge.get("ranking", [])
        print(f"  Judge probabilities: {probs}")
        print(f"  Judge ranking: {ranking}")
        print(f"  Reasoning: {judge.get('reasoning', '')[:100]}")

        issues = []
        if not probs:
            issues.append("No probabilities from judge")
        else:
            total = sum(probs.values())
            if abs(total - 1.0) > 0.05:
                issues.append(f"Probabilities don't sum to 1.0: {total:.3f}")
        if len(advocates) < N_AGENTS:
            issues.append(f"Only {len(advocates)}/{N_AGENTS} advocates returned output")

        if issues:
            print(f"  ISSUES: {issues}")
        else:
            print("  PASS")
            n_pass += 1

    print(f"\nSUMMARY: {n_pass}/{len(queries)} passed")


if __name__ == "__main__":
    main()
