"""
Baseline B1: Direct prompting (no retrieval, no reasoning structure).
"""

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery, HYPOTHESES
from src.pipeline.prompts import build_messages


def build_direct_requests(
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
        msgs = build_messages("direct", variables, cfg)
        requests.append(BatchRequest(
            custom_id=f"{q.id}__direct",
            messages=msgs,
            model=model,
            max_tokens=512,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_direct_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
) -> list[dict]:
    """Returns list of prediction dicts aligned with queries."""
    predictions = []
    for q in queries:
        key = f"{q.id}__direct"
        result = results.get(key)
        pred = _fallback_prediction(q.id)
        if result and result.ok:
            data = parse_json_response(result.content)
            if isinstance(data, dict):
                pred = _normalize_prediction(data, q.id)
        predictions.append(pred)
    return predictions


def _normalize_prediction(data: dict, query_id: str) -> dict:
    probs_raw = data.get("probabilities", {})
    probs = {}
    for h in HYPOTHESES:
        probs[h] = float(probs_raw.get(h, 1.0 / len(HYPOTHESES)))
    total = sum(probs.values())
    if total > 0:
        probs = {h: v / total for h, v in probs.items()}
    prediction = data.get("prediction") or max(probs, key=probs.get)
    if prediction not in HYPOTHESES:
        prediction = max(probs, key=probs.get)
    ranking = sorted(probs, key=probs.get, reverse=True)
    return {
        "query_id": query_id,
        "prediction": prediction,
        "probabilities": probs,
        "ranking": ranking,
        "reasoning": data.get("reasoning", ""),
    }


def _fallback_prediction(query_id: str) -> dict:
    probs = {h: 0.25 for h in HYPOTHESES}
    return {
        "query_id": query_id,
        "prediction": HYPOTHESES[0],
        "probabilities": probs,
        "ranking": HYPOTHESES[:],
        "reasoning": "fallback",
    }
