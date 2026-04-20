"""
Step 3: Article retrieval.

Pluggable interface:
  - WeaviateRetriever: production retrieval using Weaviate vector DB
  - ManualRetriever: use MIRAI-provided doc IDs (oracle retrieval)
  - MockRetriever: returns dummy articles for smoke tests (no Weaviate needed)

Usage:
    retriever = get_retriever(config)
    articles = retriever.retrieve(query, n=10)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.config import get_config
from src.data.mirai import MiraiQuery, MiraiArticle, MiraiDataset, make_mock_articles


@dataclass
class RetrievedArticle:
    id: str
    title: str
    abstract: str
    date: str
    score: float = 1.0  # retrieval relevance score

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "date": self.date,
            "score": self.score,
        }

    @property
    def content(self) -> str:
        return self.abstract or self.title


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: MiraiQuery, n: int = 10) -> list[RetrievedArticle]:
        pass

    def retrieve_batch(self, queries: list[MiraiQuery], n: int = 10) -> dict[str, list[RetrievedArticle]]:
        return {q.id: self.retrieve(q, n) for q in queries}


class ManualRetriever(BaseRetriever):
    """Use pre-defined doc_ids from the MIRAI dataset (oracle setting)."""

    def __init__(self, dataset: MiraiDataset):
        self.dataset = dataset

    def retrieve(self, query: MiraiQuery, n: int = 10) -> list[RetrievedArticle]:
        arts = self.dataset.get_articles_for_query(query)
        results = []
        for art in arts[:n]:
            results.append(RetrievedArticle(
                id=art.id,
                title=art.title,
                abstract=art.abstract,
                date=art.date,
                score=1.0,
            ))
        return results


class MockRetriever(BaseRetriever):
    """Returns synthetic articles — for smoke tests only."""

    def retrieve(self, query: MiraiQuery, n: int = 10) -> list[RetrievedArticle]:
        mocks = make_mock_articles(n)
        return [
            RetrievedArticle(
                id=art.id,
                title=art.title.format(s=query.subject, o=query.object),
                abstract=art.abstract.format(s=query.subject, o=query.object),
                date=query.timestamp,
            )
            for art in mocks
        ]


class WeaviateRetriever(BaseRetriever):
    """
    Production retriever using Weaviate with:
      - Hybrid search (BM25 + dense vectors via RRF)
      - MMR re-ranking for diversity
      - Exponential time-decay weighting
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        self._client = None  # lazy init
        self._use_mock = False  # set True on first connection failure

    def _get_client(self):
        if self._client is None:
            import weaviate
            url = self.cfg.get("retrieval", "weaviate_url", default="http://localhost:8080")
            api_key = self.cfg.get("retrieval", "weaviate_api_key")
            if api_key:
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key),
                )
            else:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                self._client = weaviate.connect_to_local(host=host, port=port)
        return self._client

    def retrieve(self, query: MiraiQuery, n: int = 10) -> list[RetrievedArticle]:
        if self._use_mock:
            return MockRetriever().retrieve(query, n)
        try:
            client = self._get_client()
        except BaseException as e:
            print(f"[WARN] Weaviate unavailable, falling back to mock retriever.")
            self._use_mock = True
            return MockRetriever().retrieve(query, n)

        collection = client.collections.get("Article")

        alpha = self.cfg.get("retrieval", "time_decay_alpha", default=0.015)
        use_mmr = self.cfg.get("retrieval", "mmr", default=True)
        mmr_lambda = self.cfg.get("retrieval", "mmr_lambda", default=0.5)
        top_k_initial = n * 3  # retrieve more, then re-rank

        # Build query string
        query_text = f"{query.subject} {query.object} {query.timestamp}"

        # Hybrid search with metadata filter
        from weaviate.classes.query import Filter, MetadataQuery, HybridFusion
        try:
            response = collection.query.hybrid(
                query=query_text,
                alpha=0.5,  # 50/50 BM25 and dense
                fusion_type=HybridFusion.RELATIVE_SCORE,
                filters=(
                    Filter.by_property("country_mentions").contains_any(
                        [query.subject, query.object]
                    )
                ),
                limit=top_k_initial,
                return_metadata=MetadataQuery(score=True),
            )
            candidates = [
                RetrievedArticle(
                    id=obj.properties.get("article_id", str(obj.uuid)),
                    title=obj.properties.get("title", ""),
                    abstract=obj.properties.get("abstract", ""),
                    date=obj.properties.get("date", ""),
                    score=float(obj.metadata.score or 0.0),
                )
                for obj in response.objects
            ]
        except Exception as exc:
            print(f"  [WARN] Weaviate query failed: {exc}. Using empty results.")
            return []

        # Apply time decay
        candidates = self._apply_time_decay(candidates, query.timestamp, alpha)

        # MMR re-ranking
        if use_mmr and len(candidates) > n:
            candidates = self._mmr(candidates, n, mmr_lambda)
        else:
            candidates = candidates[:n]

        return candidates

    def _apply_time_decay(
        self,
        articles: list[RetrievedArticle],
        query_date: str,
        alpha: float,
    ) -> list[RetrievedArticle]:
        from datetime import datetime
        try:
            qdate = datetime.strptime(query_date, "%Y-%m-%d")
        except ValueError:
            return articles

        for art in articles:
            try:
                adate = datetime.strptime(art.date, "%Y-%m-%d")
                delta_days = abs((qdate - adate).days)
                decay = math.exp(-alpha * delta_days)
                art.score *= decay
            except (ValueError, AttributeError):
                pass
        return sorted(articles, key=lambda a: a.score, reverse=True)

    def _mmr(
        self,
        candidates: list[RetrievedArticle],
        n: int,
        lam: float,
    ) -> list[RetrievedArticle]:
        """Maximal Marginal Relevance: balance relevance vs. diversity."""
        if not candidates:
            return []
        selected = [candidates[0]]
        remaining = list(candidates[1:])

        while len(selected) < n and remaining:
            best_score = -1e9
            best = None
            for cand in remaining:
                relevance = cand.score
                # Approximate diversity as negative max similarity to selected
                # Use title overlap as a simple similarity proxy
                max_sim = max(
                    _title_overlap(cand.title, sel.title) for sel in selected
                )
                mmr_score = lam * relevance - (1 - lam) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = cand
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)

        return selected


def _title_overlap(t1: str, t2: str) -> float:
    """Jaccard similarity on word sets."""
    w1 = set(t1.lower().split())
    w2 = set(t2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def get_retriever(config=None, retrieval_type: str | None = None) -> BaseRetriever:
    cfg = config or get_config()
    rtype = retrieval_type or cfg.get("retrieval", "type", default="weaviate")
    if rtype == "mock":
        return MockRetriever()
    if rtype == "manual":
        return ManualRetriever(MiraiDataset(cfg))
    if rtype == "weaviate":
        return WeaviateRetriever(cfg)
    raise ValueError(f"Unknown retrieval type: {rtype!r}")
