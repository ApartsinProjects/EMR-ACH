"""
Step 1: Contrastive indicator generation.

Builds BatchRequests; parses and validates responses.
"""

import json
from typing import Any

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery
from src.pipeline.prompts import build_messages, format_hypotheses_block


def build_indicator_requests(
    queries: list[MiraiQuery],
    model: str | None = None,
    contrastive: bool = True,
    config=None,
) -> list[BatchRequest]:
    cfg = config or get_config()
    m = cfg.get("pipeline", "m", default=24)
    min_per = cfg.get("pipeline", "pairs_per_hypothesis", default=2)
    hypotheses_block = format_hypotheses_block(cfg)
    model = model or cfg.experiment_model

    requests = []
    for q in queries:
        variables = {
            "subject": q.subject,
            "object": q.object,
            "timestamp": q.timestamp,
            "m": m,
            "min_per_hypothesis": min_per,
            "hypotheses_block": hypotheses_block,
        }
        msgs = build_messages("indicators", variables, cfg)
        requests.append(BatchRequest(
            custom_id=f"{q.id}__indicators",
            messages=msgs,
            model=model,
            max_tokens=2048,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_indicator_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
    config=None,
) -> dict[str, list[dict]]:
    """Returns {query_id: [indicator_dict, ...]}."""
    cfg = config or get_config()
    m = cfg.get("pipeline", "m", default=24)
    valid_supports = {"VC", "MC", "VK", "MK"}

    parsed: dict[str, list[dict]] = {}
    errors = 0

    for q in queries:
        key = f"{q.id}__indicators"
        result = results.get(key)
        if result is None or not result.ok:
            print(f"  [WARN] No result for {q.id}: {getattr(result, 'error', 'missing')}")
            errors += 1
            parsed[q.id] = _fallback_indicators(m)
            continue

        data = parse_json_response(result.content)
        if data is None:
            print(f"  [WARN] JSON parse failed for {q.id}. Raw:\n{result.content[:300]}")
            errors += 1
            parsed[q.id] = _fallback_indicators(m)
            continue

        # Handle both {"indicators": [...]} and plain [...]
        indicators = data if isinstance(data, list) else data.get("indicators", data)
        if not isinstance(indicators, list):
            print(f"  [WARN] Unexpected format for {q.id}: {type(data)}")
            errors += 1
            parsed[q.id] = _fallback_indicators(m)
            continue

        # Validate and normalize
        cleaned = []
        for ind in indicators:
            if not isinstance(ind, dict) or "text" not in ind:
                continue
            cleaned.append({
                "id": len(cleaned) + 1,
                "text": str(ind["text"]).strip(),
                "primarily_supports": ind.get("primarily_supports", "VC"),
                "distinguishes": ind.get("distinguishes", ""),
            })

        if len(cleaned) < m // 2:
            print(f"  [WARN] Only {len(cleaned)} indicators parsed for {q.id} (expected {m})")

        # Pad if fewer than expected
        while len(cleaned) < m:
            cleaned.append({
                "id": len(cleaned) + 1,
                "text": f"Generic indicator {len(cleaned)+1}",
                "primarily_supports": "VC",
                "distinguishes": "",
            })

        parsed[q.id] = cleaned[:m]

    if errors > 0:
        print(f"  [indicators] {errors}/{len(queries)} queries had parse errors")
    return parsed


def _fallback_indicators(m: int) -> list[dict]:
    """Minimal fallback when LLM call fails — enables pipeline to continue."""
    hypotheses = ["VC", "MC", "VK", "MK"]
    return [
        {
            "id": i + 1,
            "text": f"Indicator {i+1} (fallback)",
            "primarily_supports": hypotheses[i % 4],
            "distinguishes": "",
        }
        for i in range(m)
    ]
