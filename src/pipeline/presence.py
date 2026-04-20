"""
Step 4 + 4b: Analysis matrix A and background prior (absence-of-evidence row).

Step 4: For each (query, article) pair, score all m indicators in one batch call.
        Returns A matrix: shape (n, m), values in [0, 1].

Step 4b: For each query, score background priors for all m indicators.
         Returns phi vector: shape (m,), values in [0, 1].
         Absence row: A_tilde[n+1, j] = 1 - phi[j]
"""

import numpy as np

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery
from src.pipeline.prompts import build_messages, format_indicators_list
from src.pipeline.retrieval import RetrievedArticle


def build_presence_requests(
    queries: list[MiraiQuery],
    articles_by_query: dict[str, list[RetrievedArticle]],
    indicators_by_query: dict[str, list[dict]],
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    """One request per (query, article) pair."""
    cfg = config or get_config()
    model = model or cfg.experiment_model

    requests = []
    for q in queries:
        articles = articles_by_query.get(q.id, [])
        indicators = indicators_by_query.get(q.id, [])
        ind_list = format_indicators_list(indicators)

        for art in articles:
            variables = {
                "subject": q.subject,
                "object": q.object,
                "timestamp": q.timestamp,
                "article_title": art.title,
                "article_text": art.content[:1500],  # truncate to keep tokens manageable
                "article_id": art.id,
                "indicators_list": ind_list,
            }
            msgs = build_messages("presence", variables, cfg)
            requests.append(BatchRequest(
                custom_id=f"{q.id}__presence__{art.id}",
                messages=msgs,
                model=model,
                max_tokens=1024,
                response_format={"type": "json_object"},
            ))
    return requests


def parse_presence_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
    articles_by_query: dict[str, list[RetrievedArticle]],
    indicators_by_query: dict[str, list[dict]],
    config=None,
) -> dict[str, np.ndarray]:
    """Returns {query_id: A_matrix} where A has shape (n_articles, m)."""
    matrices: dict[str, np.ndarray] = {}

    for q in queries:
        articles = articles_by_query.get(q.id, [])
        m = len(indicators_by_query.get(q.id, []))
        n = len(articles)
        A = np.zeros((n, m))

        for i, art in enumerate(articles):
            key = f"{q.id}__presence__{art.id}"
            result = results.get(key)
            if result is None or not result.ok:
                continue

            data = parse_json_response(result.content)
            if data is None:
                continue

            scores_list = data.get("scores", [])
            for entry in scores_list:
                idx = int(entry.get("id", 0)) - 1
                if 0 <= idx < m:
                    A[i, idx] = float(np.clip(float(entry.get("presence", 0.0)), 0.0, 1.0))

        matrices[q.id] = A

    return matrices


# ----- Step 4b: Background prior -----

def build_background_prior_requests(
    queries: list[MiraiQuery],
    indicators_by_query: dict[str, list[dict]],
    model: str | None = None,
    config=None,
) -> list[BatchRequest]:
    """One request per query — scores background priors for all indicators."""
    cfg = config or get_config()
    model = model or cfg.experiment_model

    requests = []
    for q in queries:
        indicators = indicators_by_query.get(q.id, [])
        if not indicators:
            continue
        variables = {
            "subject": q.subject,
            "object": q.object,
            "timestamp": q.timestamp,
            "indicators_list": format_indicators_list(indicators),
        }
        msgs = build_messages("background_prior", variables, cfg)
        requests.append(BatchRequest(
            custom_id=f"{q.id}__background",
            messages=msgs,
            model=model,
            max_tokens=1024,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_background_prior_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
    indicators_by_query: dict[str, list[dict]],
) -> dict[str, np.ndarray]:
    """Returns {query_id: phi_vector} shape (m,)."""
    priors: dict[str, np.ndarray] = {}

    for q in queries:
        m = len(indicators_by_query.get(q.id, []))
        phi = np.full(m, 0.5)  # neutral fallback

        key = f"{q.id}__background"
        result = results.get(key)
        if result is None or not result.ok:
            priors[q.id] = phi
            continue

        data = parse_json_response(result.content)
        if data is None:
            priors[q.id] = phi
            continue

        priors_list = data.get("priors", [])
        for entry in priors_list:
            idx = int(entry.get("id", 0)) - 1
            if 0 <= idx < m:
                phi[idx] = float(np.clip(float(entry.get("prior", 0.5)), 0.0, 1.0))

        priors[q.id] = phi

    return priors


def build_augmented_A(
    A: np.ndarray,
    phi: np.ndarray,
    calibration_fn=None,
) -> np.ndarray:
    """
    Append the background-absence row to A.

    A_tilde[n+1, j] = 1 - phi_cal(phi[j])

    If phi[j] is high (indicator commonly observable as background),
    not seeing it is not very informative. If phi[j] is low,
    not seeing the indicator is strong disconfirming evidence.
    """
    if calibration_fn is not None:
        phi_cal = calibration_fn(phi)
    else:
        phi_cal = phi
    absence_row = 1.0 - phi_cal  # shape (m,)
    return np.vstack([A, absence_row.reshape(1, -1)])
