"""
Step 2: Influence matrix scoring.

For each query, sends ONE batch request that scores all m indicators against all 4 hypotheses.
Returns I matrix: shape (m, 4), values in [0, 1].
Also computes diagnosticity weights d_j = Var_k[I_{jk}].
"""

import numpy as np

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery, HYPOTHESES
from src.pipeline.prompts import (
    build_messages,
    format_hypotheses_block,
    format_indicators_list,
)


def build_influence_requests(
    queries: list[MiraiQuery],
    indicators_by_query: dict[str, list[dict]],
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    cfg = config or get_config()
    model = model or cfg.experiment_model
    hypotheses_block = format_hypotheses_block(cfg)

    requests = []
    for q in queries:
        indicators = indicators_by_query.get(q.id, [])
        if not indicators:
            continue
        variables = {
            "subject": q.subject,
            "object": q.object,
            "timestamp": q.timestamp,
            "hypotheses_block": hypotheses_block,
            "indicators_list": format_indicators_list(indicators),
        }
        msgs = build_messages("influence", variables, cfg)
        requests.append(BatchRequest(
            custom_id=f"{q.id}__influence",
            messages=msgs,
            model=model,
            max_tokens=2048,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_influence_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
    indicators_by_query: dict[str, list[dict]],
    config=None,
) -> dict[str, np.ndarray]:
    """Returns {query_id: I_matrix} where I has shape (m, 4)."""
    cfg = config or get_config()
    m = cfg.get("pipeline", "m", default=24)

    influence_matrices: dict[str, np.ndarray] = {}

    for q in queries:
        key = f"{q.id}__influence"
        n_indicators = len(indicators_by_query.get(q.id, []))
        fallback = np.full((n_indicators, 4), 0.5)

        result = results.get(key)
        if result is None or not result.ok:
            print(f"  [WARN] No influence result for {q.id}")
            influence_matrices[q.id] = fallback
            continue

        data = parse_json_response(result.content)
        if data is None:
            print(f"  [WARN] Influence JSON parse failed for {q.id}")
            influence_matrices[q.id] = fallback
            continue

        scores_list = data.get("scores", [])
        if not isinstance(scores_list, list) or len(scores_list) == 0:
            influence_matrices[q.id] = fallback
            continue

        I = np.full((n_indicators, len(HYPOTHESES)), 0.5)
        for entry in scores_list:
            idx = int(entry.get("id", 0)) - 1
            if not (0 <= idx < n_indicators):
                continue
            for j, h in enumerate(HYPOTHESES):
                val = float(entry.get(h, 0.5))
                I[idx, j] = float(np.clip(val, 0.0, 1.0))

        influence_matrices[q.id] = I

    return influence_matrices


def compute_diagnosticity_weights(I: np.ndarray) -> np.ndarray:
    """
    d_j = Var_k[I_{jk}] — variance of row j across hypotheses.

    High d_j: indicator strongly discriminates between hypotheses (diagnostic).
    Low d_j: indicator gives similar scores to all hypotheses (uninformative).
    Returns shape (m,).
    """
    return np.var(I, axis=1)


def normalize_diagnosticity(d: np.ndarray) -> np.ndarray:
    """Normalize so weights sum to 1: d_tilde_j = d_j / ||d||_1."""
    total = d.sum()
    if total < 1e-10:
        return np.ones_like(d) / len(d)
    return d / total
