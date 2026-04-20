"""
Baseline B2: Chain-of-Thought prompting.
"""

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery, HYPOTHESES
from src.pipeline.prompts import build_messages
from src.baselines.direct_prompting import _fallback_prediction


def build_cot_requests(
    queries: list[MiraiQuery],
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    cfg = config or get_config()
    model = model or cfg.experiment_model
    requests = []
    for q in queries:
        variables = {
            "subject": q.subject,
            "object": q.object,
            "timestamp": q.timestamp,
        }
        msgs = build_messages("cot", variables, cfg)
        requests.append(BatchRequest(
            custom_id=f"{q.id}__cot",
            messages=msgs,
            model=model,
            max_tokens=1024,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_cot_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
) -> list[dict]:
    predictions = []
    for q in queries:
        key = f"{q.id}__cot"
        result = results.get(key)
        pred = _fallback_prediction(q.id)
        if result and result.ok:
            data = parse_json_response(result.content)
            if isinstance(data, dict):
                probs_raw = data.get("probabilities", {})
                probs = {h: float(probs_raw.get(h, 0.25)) for h in HYPOTHESES}
                total = sum(probs.values())
                if total > 0:
                    probs = {h: v / total for h, v in probs.items()}
                prediction = data.get("prediction") or max(probs, key=probs.get)
                if prediction not in HYPOTHESES:
                    prediction = max(probs, key=probs.get)
                pred = {
                    "query_id": q.id,
                    "prediction": prediction,
                    "probabilities": probs,
                    "ranking": sorted(probs, key=probs.get, reverse=True),
                    "step1_relationship": data.get("step1_relationship", data.get("step1_history", "")),
                    "step2_bilateral_events": data.get("step2_bilateral_events", data.get("step2_context", "")),
                    "step3_evidence": data.get("step3_evidence", {}),
                }
        predictions.append(pred)
    return predictions
