"""Smoke tests for src.common.retrieval_router (v2.2 [A13])."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.retrieval_router import (
    RetrievalRequest,
    RetrievalRouter,
)
from src.retrieval.contract import RetrievalMode


def _sbert_stub(req):
    # Echo something derived from candidate_ids if provided by prefilter.
    cand = (req.extras or {}).get("candidate_ids")
    if cand:
        return [f"sbert:{c}" for c in cand[:2]]
    return ["sbert:a1", "sbert:a2"]


def _join_stub(req):
    return [f"join:{req.ticker}:{req.forecast_point}"]


def _actor_prefilter_stub(req):
    a, b = req.actor_pair or ("", "")
    return [f"cand:{a}:{b}:1", f"cand:{a}:{b}:2", f"cand:{a}:{b}:3"]


def test_forecastbench_routes_to_sbert():
    router = RetrievalRouter(sbert_fn=_sbert_stub, join_fn=_join_stub)
    req = RetrievalRequest(fd_id="fd_fb", benchmark="forecastbench")
    res = router.route(req)
    assert res.mode == RetrievalMode.SBERT_COSINE
    assert res.article_ids == ["sbert:a1", "sbert:a2"]


def test_earnings_routes_to_ticker_date_join():
    router = RetrievalRouter(sbert_fn=_sbert_stub, join_fn=_join_stub)
    req = RetrievalRequest(
        fd_id="fd_e", benchmark="earnings",
        ticker="AAPL", forecast_point="2026-03-01",
    )
    res = router.route(req)
    assert res.mode == RetrievalMode.TICKER_DATE_JOIN
    assert res.article_ids == ["join:AAPL:2026-03-01"]


def test_gdelt_cameo_uses_actor_prefilter_then_sbert():
    router = RetrievalRouter(
        sbert_fn=_sbert_stub,
        join_fn=_join_stub,
        actor_prefilter_fn=_actor_prefilter_stub,
    )
    req = RetrievalRequest(
        fd_id="fd_g", benchmark="gdelt_cameo",
        actor_pair=("PAK", "AFG"),
    )
    res = router.route(req)
    assert res.mode == RetrievalMode.ACTOR_PAIR_PREFILTER_THEN_SBERT
    # SBERT stub consumed candidate_ids from the prefilter and returned 2.
    assert res.article_ids == ["sbert:cand:PAK:AFG:1", "sbert:cand:PAK:AFG:2"]


def test_gdelt_cameo_falls_back_when_prefilter_empty():
    router = RetrievalRouter(
        sbert_fn=_sbert_stub,
        join_fn=_join_stub,
        actor_prefilter_fn=lambda req: [],
    )
    req = RetrievalRequest(fd_id="fd_g2", benchmark="gdelt_cameo")
    res = router.route(req)
    assert res.article_ids == ["sbert:a1", "sbert:a2"]
    assert "fallback" in res.notes


def test_unknown_benchmark_raises():
    router = RetrievalRouter(sbert_fn=_sbert_stub, join_fn=_join_stub)
    req = RetrievalRequest(fd_id="fd_x", benchmark="not-a-bench")
    with pytest.raises(ValueError):
        router.route(req)


def test_route_many_preserves_order():
    router = RetrievalRouter(sbert_fn=_sbert_stub, join_fn=_join_stub)
    reqs = [
        RetrievalRequest(fd_id="fd_1", benchmark="forecastbench"),
        RetrievalRequest(fd_id="fd_2", benchmark="earnings",
                         ticker="T", forecast_point="2026-03"),
    ]
    out = router.route_many(reqs)
    assert [r.fd_id for r in out] == ["fd_1", "fd_2"]
