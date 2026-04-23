"""src.common.retrieval_router: per-benchmark retrieval dispatch (v2.2 [A13]).

Dispatches the retrieval call to the correct algorithm based on the
benchmark:

* forecastbench   -> SBERT cosine over the full pool (current v2.1 behavior)
* gdelt_cameo     -> actor-pair prefilter over the editorial-only pool, then SBERT cosine
* earnings        -> ticker x date relational join (shipped in commit ac0b031;
                     scripts/link_earnings_articles.py ~1 sec for 535 FDs,
                     zero embedding work)

This module is the declarative frontend; the actual retrieval primitives
live in ``scripts/compute_relevance.py`` (SBERT) and
``scripts/link_earnings_articles.py`` (relational join). We dispatch via
``RetrievalMode`` from :mod:`src.retrieval.contract` so the benchmark-
to-mode mapping has exactly one owner.

Callers inject the primitive callables (``sbert_fn``, ``join_fn``) at
construction time so this module has no import-time dependency on the
SBERT or GDELT stacks. Unit-testable with stub callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from src.retrieval.contract import (
    Benchmark,
    RetrievalMode,
    contract_for,
    retrieval_mode_for,
)

__all__ = [
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalRouter",
]


@dataclass(frozen=True)
class RetrievalRequest:
    """One retrieval call: FD metadata + the benchmark context."""

    fd_id: str
    benchmark: str
    question: str = ""
    background: str = ""
    # Optional fields the non-SBERT paths consume.
    ticker: str | None = None
    forecast_point: str | None = None
    actor_pair: tuple[str, str] | None = None
    # Carry-through arbitrary extras; each primitive function picks what it needs.
    extras: dict | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized output contract shared by every retrieval mode."""

    fd_id: str
    article_ids: list[str]
    mode: RetrievalMode
    notes: str = ""


# Primitive callable signatures. Kept permissive so real implementations
# can add kwargs without invalidating the router.
SbertFn = Callable[[RetrievalRequest], list[str]]
JoinFn = Callable[[RetrievalRequest], list[str]]
ActorPrefilterFn = Callable[[RetrievalRequest], list[str]]


class RetrievalRouter:
    """Dispatcher for the per-benchmark retrieval table."""

    def __init__(
        self,
        *,
        sbert_fn: SbertFn,
        join_fn: JoinFn,
        actor_prefilter_fn: ActorPrefilterFn | None = None,
    ) -> None:
        self._sbert = sbert_fn
        self._join = join_fn
        self._actor_prefilter = actor_prefilter_fn

    def route(self, req: RetrievalRequest) -> RetrievalResult:
        """Dispatch a single request. Raises ``ValueError`` if the benchmark
        is unknown (surfaces typos early rather than silently falling
        through to SBERT).
        """
        mode = retrieval_mode_for(req.benchmark)
        if mode == RetrievalMode.SBERT_COSINE:
            ids = list(self._sbert(req))
            return RetrievalResult(
                fd_id=req.fd_id, article_ids=ids, mode=mode,
                notes="full-pool SBERT cosine",
            )
        if mode == RetrievalMode.TICKER_DATE_JOIN:
            ids = list(self._join(req))
            return RetrievalResult(
                fd_id=req.fd_id, article_ids=ids, mode=mode,
                notes="ticker x date relational join (earnings)",
            )
        if mode == RetrievalMode.ACTOR_PAIR_PREFILTER_THEN_SBERT:
            prefiltered = (
                list(self._actor_prefilter(req))
                if self._actor_prefilter is not None
                else []
            )
            # If prefilter yields nothing, fall back to straight SBERT so
            # we do not starve the benchmark.
            if not prefiltered:
                ids = list(self._sbert(req))
                notes = "actor-pair prefilter empty; SBERT fallback"
            else:
                # Hand the prefiltered pool to SBERT via extras so the
                # primitive can scope its search.
                scoped = RetrievalRequest(
                    fd_id=req.fd_id,
                    benchmark=req.benchmark,
                    question=req.question,
                    background=req.background,
                    ticker=req.ticker,
                    forecast_point=req.forecast_point,
                    actor_pair=req.actor_pair,
                    extras={**(req.extras or {}), "candidate_ids": prefiltered},
                )
                ids = list(self._sbert(scoped))
                notes = f"actor-pair prefilter -> {len(prefiltered)} candidates; then SBERT"
            return RetrievalResult(
                fd_id=req.fd_id, article_ids=ids, mode=mode, notes=notes,
            )
        raise ValueError(f"unhandled retrieval mode: {mode!r}")

    def route_many(self, reqs: Iterable[RetrievalRequest]) -> list[RetrievalResult]:
        return [self.route(r) for r in reqs]

    @staticmethod
    def mode_for(benchmark: str) -> RetrievalMode:
        """Re-export for callers that want to branch on the mode without
        constructing a router (e.g. orchestrator diagnostics)."""
        return retrieval_mode_for(benchmark)
