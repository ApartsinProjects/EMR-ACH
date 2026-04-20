"""
Step 7: Deep Analysis — Verbal vs Material disambiguation.

Second-stage LLM pass that classifies each retrieved article as:
  Verbal/High, Verbal/Low, Material/High, Material/Low, Uncertain

The category (Verbal/Material) with the higher cumulative score wins.
This refines the top-1 prediction between VC/VK (both Verbal) or MC/MK (both Material).
"""

import numpy as np

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery, HYPOTHESES
from src.pipeline.prompts import build_messages, load_prompt
from src.pipeline.retrieval import RetrievedArticle

# Classification score weights
SCORING = {
    "Verbal/High": 1.0,
    "Verbal/Low": 0.5,
    "Material/High": 1.0,
    "Material/Low": 0.5,
    "Uncertain": 0.0,
}

# Which hypotheses are Verbal vs Material
VERBAL_HYPOTHESES = {"VC", "VK"}
MATERIAL_HYPOTHESES = {"MC", "MK"}


def build_deep_analysis_requests(
    queries: list[MiraiQuery],
    articles_by_query: dict[str, list[RetrievedArticle]],
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    cfg = config or get_config()
    model = model or cfg.experiment_model

    requests = []
    for q in queries:
        articles = articles_by_query.get(q.id, [])
        for art in articles:
            variables = {
                "subject": q.subject,
                "object": q.object,
                "timestamp": q.timestamp,
                "article_title": art.title,
                "article_text": art.content[:1500],
            }
            msgs = build_messages("deep_analysis", variables, cfg)
            requests.append(BatchRequest(
                custom_id=f"{q.id}__deepanalysis__{art.id}",
                messages=msgs,
                model=model,
                max_tokens=256,
                response_format={"type": "json_object"},
            ))
    return requests


def parse_deep_analysis_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
    articles_by_query: dict[str, list[RetrievedArticle]],
) -> dict[str, dict]:
    """
    Returns {query_id: {"verbal_score": float, "material_score": float, "vm_winner": str}}.
    """
    valid = set(SCORING.keys())
    output = {}

    for q in queries:
        articles = articles_by_query.get(q.id, [])
        verbal_total = 0.0
        material_total = 0.0
        article_classifications = []

        for art in articles:
            key = f"{q.id}__deepanalysis__{art.id}"
            result = results.get(key)
            classification = "Uncertain"
            reasoning = ""

            if result and result.ok:
                data = parse_json_response(result.content)
                if data and isinstance(data, dict):
                    classification = data.get("classification", "Uncertain")
                    reasoning = data.get("reasoning", "")
                    if classification not in valid:
                        classification = "Uncertain"

            score = SCORING[classification]
            if classification.startswith("Verbal"):
                verbal_total += score
            elif classification.startswith("Material"):
                material_total += score

            article_classifications.append({
                "article_id": art.id,
                "classification": classification,
                "reasoning": reasoning,
                "score": score,
            })

        vm_winner = "Verbal" if verbal_total >= material_total else "Material"

        output[q.id] = {
            "verbal_score": verbal_total,
            "material_score": material_total,
            "vm_winner": vm_winner,
            "article_classifications": article_classifications,
        }

    return output


def apply_deep_analysis(
    initial_ranking: list[str],
    da_result: dict,
) -> list[str]:
    """
    Refine the initial ranking using the V/M winner from Deep Analysis.

    Logic:
    - If top-1 and top-2 differ in V/M category, no refinement needed.
    - If top-1 and top-2 are in the same V/M group (both Verbal or both Material),
      use the DA winner to pick between them.
    - In all other cases, return the original ranking unchanged.
    """
    if not initial_ranking or da_result is None:
        return initial_ranking

    top1 = initial_ranking[0]
    top2 = initial_ranking[1] if len(initial_ranking) > 1 else None
    vm_winner = da_result.get("vm_winner", "")

    if top2 is None:
        return initial_ranking

    top1_is_verbal = top1 in VERBAL_HYPOTHESES
    top2_is_verbal = top2 in VERBAL_HYPOTHESES

    # Top-1 and top-2 are in the same V/M group — use DA to decide
    if top1_is_verbal == top2_is_verbal:
        preferred_group = VERBAL_HYPOTHESES if vm_winner == "Verbal" else MATERIAL_HYPOTHESES
        if top1 not in preferred_group and top2 in preferred_group:
            # Swap top-1 and top-2
            new_ranking = [top2, top1] + initial_ranking[2:]
            return new_ranking

    return initial_ranking
