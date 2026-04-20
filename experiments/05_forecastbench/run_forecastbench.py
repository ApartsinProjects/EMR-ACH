"""
Experiment group 5: Cross-domain generalization on ForecastBench.

Adapts the EMR-ACH pipeline to binary Yes/No forecasting:
  - Hypotheses: ["Yes", "No"]
  - No Weaviate (ForecastBench questions are standalone)
  - Evaluation: Brier score + ECE vs. crowd forecasters

Usage:
  python experiments/05_forecastbench/run_forecastbench.py --mode smoke
  python experiments/05_forecastbench/run_forecastbench.py
"""

import json
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_client import BatchClient, BatchRequest, parse_json_response
from src.config import get_config
from src.data.forecastbench import ForecastBenchDataset, ForecastBenchQuery, make_mock_fb_queries
from src.eval.metrics import brier_score
from experiments.runner import (
    parse_args, get_client, get_n_queries, save_predictions, save_metrics,
    print_cost_estimate, timing,
)

FB_HYPOTHESES = ["Yes", "No"]

# Adapted prompts (inline for ForecastBench — uses binary hypotheses)
_FB_SYSTEM = (
    "You are an expert geopolitical forecaster. "
    "Always respond with valid JSON only. Do not wrap in code blocks."
)

_FB_DIRECT_USER = """Forecasting question: {question}
Resolution date: {resolution_date}

Base rate guidance: approximately 17% of binary geopolitical questions resolve Yes.
Assign a calibrated probability to Yes and No that reflects the specific evidence for THIS
question. Do not default to generic values. Probabilities must sum to 1.0.

Return JSON - no other text:
{{
  "probabilities": {{"Yes": <your_estimate>, "No": <your_estimate>}},
  "prediction": "<Yes_or_No>",
  "reasoning": "<one sentence specific to this question>"
}}"""

_FB_EMR_USER = """Forecasting question: {question}
Resolution date: {resolution_date}

Analyze this as an Analysis of Competing Hypotheses problem:
- Hypothesis Yes: the described event resolves YES by {resolution_date}
- Hypothesis No: the described event does NOT resolve Yes

Base rate: approximately 17% of such geopolitical questions resolve Yes.
Only assign high Yes probability (>0.5) if there is specific strong evidence.

Step 1: What observable indicators would most distinguish Yes from No for THIS question?
Step 2: What is the base rate and how does this specific question differ from average?
Step 3: What recent geopolitical context is relevant to this specific question?
Step 4: Weigh the bilateral/specific evidence and assign calibrated probability.

Calibration guidance:
- Extreme values (>0.85 or <0.10) require overwhelming evidence
- When uncertain, stay near the 0.20 base rate
- Use the full probability range; do not anchor to any fixed value

Return JSON - no other text:
{{
  "indicators_yes": ["<specific indicator1>", "<specific indicator2>"],
  "indicators_no": ["<specific indicator1>", "<specific indicator2>"],
  "base_rate_adjustment": "<why this question differs from 20% base rate>",
  "context": "<one sentence on specific relevant recent events>",
  "probabilities": {{"Yes": <your_estimate>, "No": <your_estimate>}},
  "prediction": "<Yes_or_No>",
  "reasoning": "<one sentence citing specific evidence>"
}}"""


def build_fb_direct_requests(
    queries: list[ForecastBenchQuery],
    model: str,
) -> list[BatchRequest]:
    return [
        BatchRequest(
            custom_id=f"{q.id}__fb_direct",
            messages=[
                {"role": "system", "content": _FB_SYSTEM},
                {"role": "user", "content": _FB_DIRECT_USER.format(
                    question=q.question,
                    resolution_date=q.resolution_date,
                )},
            ],
            model=model,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        for q in queries
    ]


def build_fb_emrach_requests(
    queries: list[ForecastBenchQuery],
    model: str,
) -> list[BatchRequest]:
    return [
        BatchRequest(
            custom_id=f"{q.id}__fb_emrach",
            messages=[
                {"role": "system", "content": _FB_SYSTEM},
                {"role": "user", "content": _FB_EMR_USER.format(
                    question=q.question,
                    resolution_date=q.resolution_date,
                )},
            ],
            model=model,
            max_tokens=768,
            response_format={"type": "json_object"},
        )
        for q in queries
    ]


def parse_fb_responses(results, queries, suffix: str) -> list[dict]:
    preds = []
    for q in queries:
        key = f"{q.id}__{suffix}"
        result = results.get(key)
        fallback = {"query_id": q.id, "prob_yes": 0.5, "prediction": "No", "gt": q.ground_truth}
        if result is None or not result.ok:
            preds.append(fallback)
            continue
        data = parse_json_response(result.content)
        if not isinstance(data, dict):
            preds.append(fallback)
            continue
        probs = data.get("probabilities", {})
        prob_yes = float(probs.get("Yes", 0.5))
        prob_yes = float(np.clip(prob_yes, 0.0, 1.0))
        preds.append({
            "query_id": q.id,
            "prob_yes": prob_yes,
            "prediction": "Yes" if prob_yes >= 0.5 else "No",
            "gt": q.ground_truth,
            "reasoning": data.get("reasoning", ""),
        })
    return preds


def evaluate_fb(preds: list[dict], queries: list[ForecastBenchQuery]) -> dict:
    probs = [p["prob_yes"] for p in preds]
    labels = [q.ground_truth for q in queries]
    bs = brier_score(probs, labels)
    accuracy = float(np.mean([p["prediction"] == q.label for p, q in zip(preds, queries)]))

    # ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(probs)
    probs_arr = np.array(probs)
    labels_arr = np.array(labels, dtype=float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_arr >= lo) & (probs_arr < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels_arr[mask].mean() - probs_arr[mask].mean())

    return {"brier_score": round(bs, 4), "ece": round(ece, 4), "accuracy": round(accuracy * 100, 1), "n": n}


def main():
    args = parse_args("ForecastBench cross-domain")
    cfg = get_config(args.config)
    mode = args.mode
    n = get_n_queries(mode, args)
    client = get_client(mode, cfg)
    model = cfg.smoke_model if mode == "smoke" else cfg.experiment_model
    results_dir = cfg.results_dir / "processed"

    try:
        ds = ForecastBenchDataset(cfg)
        queries = ds.queries(n, post_cutoff_only=(mode == "full"))
        crowd_brier = sum(
            (q.crowd_probability - q.ground_truth) ** 2 for q in queries
        ) / len(queries)
        print(f"[fb] Crowd probability baseline Brier: {crowd_brier:.4f}")
        print(f"Crowd Brier score (baseline): {ds.crowd_brier_score():.4f}")
    except FileNotFoundError:
        queries = make_mock_fb_queries(n or 5)
        print("[WARN] Using mock ForecastBench queries")

    print(f"\nMode={mode}  N={len(queries)}  Model={model}")
    print("=" * 60)

    # Direct prompting baseline
    print("\n--- FB-Direct ---")
    direct_reqs = build_fb_direct_requests(queries, model)
    print_cost_estimate(client, direct_reqs)
    if not args.dry_run:
        direct_results = client.run(direct_reqs, job_name=f"fb_{mode}_direct")
        direct_preds = parse_fb_responses(direct_results, queries, "fb_direct")
        save_predictions(direct_preds, results_dir / "predictions_fb_direct.jsonl")
        fb_direct_metrics = evaluate_fb(direct_preds, queries)
        print(f"  Brier={fb_direct_metrics['brier_score']}  ECE={fb_direct_metrics['ece']}  Acc={fb_direct_metrics['accuracy']}%")
        save_metrics(fb_direct_metrics, results_dir / "metrics_fb_direct.json")

    # EMR-ACH adapted
    print("\n--- FB-EMR-ACH ---")
    emrach_reqs = build_fb_emrach_requests(queries, model)
    print_cost_estimate(client, emrach_reqs)
    if not args.dry_run:
        emrach_results = client.run(emrach_reqs, job_name=f"fb_{mode}_emrach")
        emrach_preds = parse_fb_responses(emrach_results, queries, "fb_emrach")
        save_predictions(emrach_preds, results_dir / "predictions_fb_emrach.jsonl")
        fb_emrach_metrics = evaluate_fb(emrach_preds, queries)
        print(f"  Brier={fb_emrach_metrics['brier_score']}  ECE={fb_emrach_metrics['ece']}  Acc={fb_emrach_metrics['accuracy']}%")
        save_metrics(fb_emrach_metrics, results_dir / "metrics_fb_emrach.json")

    print("\nForecastBench experiment complete.")


if __name__ == "__main__":
    main()
