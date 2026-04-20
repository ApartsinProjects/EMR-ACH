"""
Debug script: run 5 queries through all pipeline flows, print raw responses.

Usage:
  python scripts/debug_flows.py
  python scripts/debug_flows.py --flow b1  (just one flow)
  python scripts/debug_flows.py --n 3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.mirai import MiraiDataset, make_mock_queries
from src.batch_client import BatchClient, BatchRequest, parse_json_response
from src.pipeline.prompts import build_messages
from src.pipeline.retrieval import get_retriever
from src.baselines.direct_prompting import build_direct_requests, parse_direct_responses
from src.baselines.cot import build_cot_requests, parse_cot_responses
from src.baselines.rag_only import build_rag_only_requests, parse_rag_only_responses
from src.pipeline.indicators import build_indicator_requests, parse_indicator_responses
from src.pipeline.influence import build_influence_requests, parse_influence_responses
from src.pipeline.presence import (
    build_presence_requests, parse_presence_responses,
    build_background_prior_requests, parse_background_prior_responses,
    build_augmented_A,
)
from src.pipeline.aggregation import aggregate


SEP = "=" * 70


def header(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


def show_query(q):
    print(f"  ID: {q.id}")
    print(f"  Subject: {q.subject}")
    print(f"  Object:  {q.object}")
    print(f"  Date:    {q.timestamp}")
    print(f"  Label:   {q.label}")


def show_raw(custom_id: str, results: dict):
    res = results.get(custom_id)
    if res is None:
        print(f"  [MISSING] No result for key {custom_id!r}")
        return
    if not res.ok:
        print(f"  [ERROR] {res.error}")
        return
    print(f"  Raw response ({len(res.content)} chars):")
    print("  " + res.content[:800].replace("\n", "\n  "))


def show_prediction(pred: dict):
    print(f"  Prediction: {pred.get('prediction', '?')}")
    probs = pred.get("probabilities", {})
    for h, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"    {h}: {p:.3f}")
    if pred.get("reasoning"):
        print(f"  Reasoning: {str(pred['reasoning'])[:200]}")


def run_b1(queries, client, cfg):
    header("B1: Direct Prompting")
    reqs = build_direct_requests(queries, model=cfg.smoke_model, config=cfg)
    results = client.run(reqs, job_name="debug_b1")
    preds = parse_direct_responses(results, queries)

    for q, pred in zip(queries, preds):
        print(f"\nQuery: {q.subject} -> {q.object} ({q.timestamp}) [GT={q.label}]")
        show_raw(f"{q.id}__direct", results)
        print("  Parsed:")
        show_prediction(pred)
    return preds


def run_b2(queries, client, cfg):
    header("B2: Chain-of-Thought")
    reqs = build_cot_requests(queries, model=cfg.smoke_model, config=cfg)
    results = client.run(reqs, job_name="debug_b2")
    preds = parse_cot_responses(results, queries)

    for q, pred in zip(queries, preds):
        print(f"\nQuery: {q.subject} -> {q.object} ({q.timestamp}) [GT={q.label}]")
        show_raw(f"{q.id}__cot", results)
        print("  Parsed:")
        show_prediction(pred)
        if pred.get("step2_bilateral_events"):
            print(f"  Step2 bilateral: {str(pred['step2_bilateral_events'])[:200]}")
    return preds


def run_b3(queries, client, cfg):
    header("B3: RAG-only")
    retriever = get_retriever(cfg, retrieval_type="manual")
    articles_by_query = retriever.retrieve_batch(queries)
    reqs = build_rag_only_requests(queries, articles_by_query, model=cfg.smoke_model, config=cfg)
    results = client.run(reqs, job_name="debug_b3")
    preds = parse_rag_only_responses(results, queries)

    for q, pred in zip(queries, preds):
        arts = articles_by_query.get(q.id, [])
        print(f"\nQuery: {q.subject} -> {q.object} ({q.timestamp}) [GT={q.label}]")
        print(f"  Retrieved {len(arts)} articles")
        for a in arts[:2]:
            print(f"    [{a.date}] {a.title[:80]}")
        show_raw(f"{q.id}__ragonly", results)
        print("  Parsed:")
        show_prediction(pred)
    return preds


def run_emrach_core(queries, client, cfg):
    header("EMR-ACH Core (Indicators + Influence + Presence + Aggregation)")
    retriever = get_retriever(cfg, retrieval_type="manual")
    model = cfg.smoke_model

    # Stage 1: Indicators
    print("\n-- Stage 1: Indicators --")
    ind_reqs = build_indicator_requests(queries, model=model, contrastive=True, config=cfg)
    print(f"  {len(ind_reqs)} requests")
    # Show one prompt
    if ind_reqs:
        print("  Sample prompt (first query):")
        for msg in ind_reqs[0].messages:
            print(f"    [{msg['role']}] {str(msg['content'])[:300]}")
    ind_results = client.run(ind_reqs, job_name="debug_ind")
    indicators_by_query = parse_indicator_responses(ind_results, queries, cfg)

    for q in queries[:2]:
        inds = indicators_by_query.get(q.id, [])
        print(f"\n  Query: {q.subject} -> {q.object} ({q.timestamp}) [GT={q.label}]")
        show_raw(f"{q.id}__indicators", ind_results)
        print(f"  Parsed indicators: {len(inds)}")
        for ind in inds[:3]:
            h = ind.get("primarily_supports", "?") if isinstance(ind, dict) else getattr(ind, "hypothesis", "?")
            txt = ind.get("text", str(ind))[:80] if isinstance(ind, dict) else str(ind)[:80]
            print(f"    [{h}] {txt}")

    # Stage 2: Influence
    print("\n-- Stage 2: Influence --")
    inf_reqs = build_influence_requests(queries, indicators_by_query, model=model, config=cfg)
    print(f"  {len(inf_reqs)} requests")
    inf_results = client.run(inf_reqs, job_name="debug_inf")
    I_by_query = parse_influence_responses(inf_results, queries, indicators_by_query, cfg)

    for q in queries[:2]:
        print(f"\n  Query: {q.subject} -> {q.object} ({q.timestamp})")
        show_raw(f"{q.id}__influence", inf_results)
        I = I_by_query.get(q.id)
        if I is not None:
            print(f"  Influence matrix shape: {getattr(I, 'shape', 'N/A')}")

    # Stage 3: Presence
    print("\n-- Stage 3: Presence --")
    articles_by_query = retriever.retrieve_batch(queries)
    pres_reqs = build_presence_requests(
        queries, articles_by_query, indicators_by_query, model=model, config=cfg
    )
    print(f"  {len(pres_reqs)} requests")
    pres_results = client.run(pres_reqs, job_name="debug_pres")
    presence_by_query = parse_presence_responses(pres_results, queries, articles_by_query, indicators_by_query, cfg)

    # Background priors
    prior_reqs = build_background_prior_requests(queries, indicators_by_query, model=model, config=cfg)
    prior_results = client.run(prior_reqs, job_name="debug_prior")
    prior_by_query = parse_background_prior_responses(prior_results, queries, indicators_by_query)

    # Show first presence result key available
    for q in queries[:2]:
        print(f"\n  Query: {q.subject} -> {q.object}")
        arts = articles_by_query.get(q.id, [])
        first_key = f"{q.id}__presence__{arts[0].id}" if arts else f"{q.id}__presence__unknown"
        show_raw(first_key, pres_results)

    # Aggregation
    print("\n-- Aggregation --")
    preds = []
    for q in queries:
        indicators = indicators_by_query.get(q.id, [])
        I = I_by_query.get(q.id)
        presence = presence_by_query.get(q.id, {})
        prior = prior_by_query.get(q.id, {})
        if I is None or not indicators:
            preds.append({"query_id": q.id, "prediction": "VC",
                          "probabilities": {"VC": 0.25, "MC": 0.25, "VK": 0.25, "MK": 0.25},
                          "ranking": ["VC", "MC", "VK", "MK"]})
            continue
        try:
            A = presence_by_query.get(q.id)
            phi = prior_by_query.get(q.id)
            if A is None or phi is None or I is None:
                raise ValueError(f"Missing data: A={A is not None}, phi={phi is not None}, I={I is not None}")
            A_tilde = build_augmented_A(A, phi)
            agg = aggregate(A_tilde, I)
            probs = {h: float(agg["probs"][i]) for i, h in enumerate(["VC", "MC", "VK", "MK"])}
            prediction = agg["ranking"][0]
            preds.append({
                "query_id": q.id,
                "prediction": prediction,
                "probabilities": probs,
                "ranking": agg["ranking"],
            })
        except Exception as e:
            print(f"  [ERROR] Aggregation failed for {q.id}: {e}")
            preds.append({"query_id": q.id, "prediction": "VC",
                          "probabilities": {"VC": 0.25, "MC": 0.25, "VK": 0.25, "MK": 0.25},
                          "ranking": ["VC", "MC", "VK", "MK"]})

    for q, pred in zip(queries, preds):
        print(f"\n  Query: {q.subject} -> {q.object} ({q.timestamp}) [GT={q.label}]")
        show_prediction(pred)

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=["b1", "b2", "b3", "emrach", "all"], default="all")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--diverse", action="store_true", help="Pick 1 query per label class")
    args = parser.parse_args()

    cfg = get_config()
    client = BatchClient(mode="direct", config=cfg)

    try:
        ds = MiraiDataset(cfg)
        all_queries = ds.queries(None)
        if args.diverse:
            # Pick 1 from each class for better coverage
            seen = set()
            queries = []
            for q in all_queries:
                if q.label not in seen:
                    queries.append(q)
                    seen.add(q.label)
                if len(seen) == 4:
                    break
            # fill remaining
            remaining = [q for q in all_queries if q not in queries][:max(0, args.n - len(queries))]
            queries = queries + remaining
        else:
            queries = all_queries[:args.n]
        print(f"Loaded {len(all_queries)} queries, using first {len(queries)}")
    except FileNotFoundError:
        print("[WARN] Real data not found, using mock queries")
        queries = make_mock_queries(args.n)

    print(f"\nQueries selected ({len(queries)}):")
    for q in queries:
        show_query(q)

    flows = [args.flow] if args.flow != "all" else ["b1", "b2", "b3", "emrach"]

    all_preds = {}
    if "b1" in flows:
        all_preds["B1"] = run_b1(queries, client, cfg)
    if "b2" in flows:
        all_preds["B2"] = run_b2(queries, client, cfg)
    if "b3" in flows:
        all_preds["B3"] = run_b3(queries, client, cfg)
    if "emrach" in flows:
        all_preds["EMR-ACH"] = run_emrach_core(queries, client, cfg)

    # Summary comparison
    if len(all_preds) > 1:
        header("Summary Comparison")
        labels = [q.label for q in queries]
        for flow, preds in all_preds.items():
            correct = sum(1 for p, l in zip(preds, labels) if p.get("prediction") == l)
            print(f"  {flow}: {correct}/{len(labels)} correct")
            pred_dist = {}
            for p in preds:
                h = p.get("prediction", "?")
                pred_dist[h] = pred_dist.get(h, 0) + 1
            print(f"    Predictions: {pred_dist}")


if __name__ == "__main__":
    main()
