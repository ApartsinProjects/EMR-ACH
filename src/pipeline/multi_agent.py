"""
Multi-agent adversarial ACH.

Protocol:
  1. One advocate agent per hypothesis argues for its assigned hypothesis.
  2. A single judge agent reads all advocate outputs and produces a probability distribution.

This is implemented as TWO sequential batch stages:
  - Stage "advocate": n_agents × N_queries requests
  - Stage "judge":    N_queries requests (depends on advocate outputs)
"""

import numpy as np

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery, HYPOTHESES
from src.pipeline.prompts import (
    build_messages,
    format_articles_block,
    format_advocates_block,
    format_indicators_list,
)
from src.pipeline.retrieval import RetrievedArticle

HYPOTHESIS_NAMES = {
    "VC": "Verbal Cooperation",
    "MC": "Material Cooperation",
    "VK": "Verbal Conflict",
    "MK": "Material Conflict",
}
HYPOTHESIS_DESCRIPTIONS = {
    "VC": (
        "Diplomatic statements of support, joint communiques, praise, verbal agreements, "
        "expressions of solidarity — words without direct material commitment."
    ),
    "MC": (
        "Physical or material assistance: aid delivery, treaty implementation, "
        "joint military exercises, infrastructure projects, economic agreements enacted."
    ),
    "VK": (
        "Official accusations, threats, condemnations, diplomatic protests, "
        "warnings, expulsion of ambassadors through statements."
    ),
    "MK": (
        "Military operations, airstrikes, troop movements, economic sanctions imposed, "
        "physical attacks, blockades, forced expulsions."
    ),
}


def build_advocate_requests(
    queries: list[MiraiQuery],
    articles_by_query: dict[str, list[RetrievedArticle]],
    indicators_by_query: dict[str, list[dict]],
    n_agents: int = 4,
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    """One request per (query, hypothesis) pair. n_agents must be <= 4."""
    cfg = config or get_config()
    model = model or cfg.experiment_model
    hypotheses = HYPOTHESES[:n_agents]

    requests = []
    for q in queries:
        articles = articles_by_query.get(q.id, [])
        indicators = indicators_by_query.get(q.id, [])
        articles_block = format_articles_block([a.to_dict() for a in articles])
        indicators_block = format_indicators_list(indicators)

        for h in hypotheses:
            variables = {
                "subject": q.subject,
                "object": q.object,
                "timestamp": q.timestamp,
                "hypothesis": h,
                "hypothesis_name": HYPOTHESIS_NAMES[h],
                "hypothesis_description": HYPOTHESIS_DESCRIPTIONS[h],
                "articles_block": articles_block,
                "indicators_block": indicators_block,
            }
            msgs = build_messages("multiagent_advocate", variables, cfg)
            requests.append(BatchRequest(
                custom_id=f"{q.id}__advocate__{h}",
                messages=msgs,
                model=model,
                max_tokens=512,
                response_format={"type": "json_object"},
            ))
    return requests


def parse_advocate_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
    n_agents: int = 4,
) -> dict[str, list[dict]]:
    """Returns {query_id: [advocate_dict_per_hypothesis]}."""
    hypotheses = HYPOTHESES[:n_agents]
    advocates_by_query: dict[str, list[dict]] = {}

    for q in queries:
        advocate_outputs = []
        for h in hypotheses:
            key = f"{q.id}__advocate__{h}"
            result = results.get(key)
            fallback = {
                "hypothesis": h,
                "key_evidence": [],
                "confidence": 0.25,
                "argument_summary": f"No argument available for {h}.",
            }
            if result is None or not result.ok:
                advocate_outputs.append(fallback)
                continue
            data = parse_json_response(result.content)
            if not isinstance(data, dict):
                advocate_outputs.append(fallback)
                continue
            advocate_outputs.append({
                "hypothesis": data.get("hypothesis", h),
                "key_evidence": data.get("key_evidence", []),
                "confidence": float(np.clip(data.get("confidence", 0.25), 0.0, 1.0)),
                "argument_summary": data.get("argument_summary", ""),
                "supporting_articles": data.get("supporting_articles", []),
                "counter_to_alternatives": data.get("counter_to_alternatives", ""),
            })
        advocates_by_query[q.id] = advocate_outputs

    return advocates_by_query


def build_judge_requests(
    queries: list[MiraiQuery],
    advocates_by_query: dict[str, list[dict]],
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    """One request per query — judge reads all advocate outputs."""
    cfg = config or get_config()
    model = model or cfg.experiment_model

    requests = []
    for q in queries:
        advocates = advocates_by_query.get(q.id, [])
        if not advocates:
            continue
        variables = {
            "subject": q.subject,
            "object": q.object,
            "timestamp": q.timestamp,
            "advocates_block": format_advocates_block(advocates),
        }
        msgs = build_messages("multiagent_judge", variables, cfg)
        requests.append(BatchRequest(
            custom_id=f"{q.id}__judge",
            messages=msgs,
            model=model,
            max_tokens=512,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_judge_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
) -> dict[str, dict]:
    """Returns {query_id: {probabilities, ranking, reasoning}}."""
    output = {}

    for q in queries:
        key = f"{q.id}__judge"
        fallback_probs = {h: 0.25 for h in HYPOTHESES}
        fallback = {"probabilities": fallback_probs, "ranking": HYPOTHESES[:], "reasoning": ""}

        result = results.get(key)
        if result is None or not result.ok:
            output[q.id] = fallback
            continue

        data = parse_json_response(result.content)
        if not isinstance(data, dict):
            output[q.id] = fallback
            continue

        probs = data.get("probabilities", {})
        # Normalize
        total = sum(probs.get(h, 0.0) for h in HYPOTHESES)
        if total < 1e-9:
            probs = {h: 0.25 for h in HYPOTHESES}
        else:
            probs = {h: probs.get(h, 0.0) / total for h in HYPOTHESES}

        ranking = sorted(probs, key=probs.get, reverse=True)
        output[q.id] = {
            "probabilities": probs,
            "ranking": ranking,
            "reasoning": data.get("reasoning", ""),
            "strongest_advocate": data.get("strongest_advocate", ""),
            "key_deciding_factors": data.get("key_deciding_factors", []),
        }

    return output
