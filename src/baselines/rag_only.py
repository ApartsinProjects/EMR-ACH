"""
Baseline B3: RAG-only (retrieve articles, concatenate, single LLM call — no ACH structure).
"""

from src.batch_client import BatchRequest, BatchResult, parse_json_response
from src.config import get_config
from src.data.mirai import MiraiQuery, HYPOTHESES
from src.baselines.direct_prompting import _fallback_prediction
from src.pipeline.retrieval import RetrievedArticle

_RAG_SYSTEM = (
    "You are an expert geopolitical forecaster. Analyze the provided news articles "
    "to predict the most likely type of interaction between the two countries. "
    "Always respond with valid JSON only. Do not wrap in code blocks."
)

_RAG_USER = """Forecasting query: {subject} - {object} ({timestamp})

News articles retrieved for the period around this query date:
{articles_block}

Focus on articles that specifically describe interactions between {subject} and {object}.
If no directly relevant articles are found, rely on your background knowledge of the bilateral relationship.

Predict the outcome category for this specific bilateral relationship:
- VC (Verbal Cooperation): Diplomatic statements, agreements, expressions of support (~50% base rate)
- MC (Material Cooperation): Physical aid, implemented treaties, joint operations (~15% base rate)
- VK (Verbal Conflict): Accusations, threats, condemnations, diplomatic protests (~25% base rate)
- MK (Material Conflict): Military action, sanctions, physical attacks (~10% base rate)

Return JSON - no other text:
{{
  "probabilities": {{"VC": <your_estimate>, "MC": <your_estimate>, "VK": <your_estimate>, "MK": <your_estimate>}},
  "prediction": "<most_likely_category>",
  "reasoning": "<one sentence citing specific evidence from the articles>"
}}"""


def build_rag_only_requests(
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
        articles_block = _format_articles(articles)
        user_text = _RAG_USER.format(
            subject=q.subject,
            object=q.object,
            timestamp=q.timestamp,
            articles_block=articles_block,
        )
        msgs = [
            {"role": "system", "content": _RAG_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        requests.append(BatchRequest(
            custom_id=f"{q.id}__ragonly",
            messages=msgs,
            model=model,
            max_tokens=512,
            response_format={"type": "json_object"},
        ))
    return requests


def parse_rag_only_responses(
    results: dict[str, BatchResult],
    queries: list[MiraiQuery],
) -> list[dict]:
    predictions = []
    for q in queries:
        key = f"{q.id}__ragonly"
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
                    "reasoning": data.get("reasoning", ""),
                }
        predictions.append(pred)
    return predictions


def _format_articles(articles: list[RetrievedArticle], max_chars: int = 300) -> str:
    lines = []
    for art in articles:
        content = art.content[:max_chars]
        lines.append(f"[{art.id}] ({art.date}) {art.title}\n  {content}")
    return "\n\n".join(lines) if lines else "(no articles retrieved)"
