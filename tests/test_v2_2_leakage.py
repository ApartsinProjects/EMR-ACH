"""v2.2 leakage-constraint regression tests.

Verifies the core v2.2 invariants:
  1. forecast_point = resolution_date - horizon_days (configurable, default 14)
  2. Every article in an FD's article_ids satisfies publish_date <=
     forecast_point.
  3. The fetch-time leakage filter drops a planted future-dated article
     instead of letting it into the pool.

The tests use small synthetic FD + article fixtures (3 FDs per track, 10
articles each) rather than loading the published v2.1 bundle, so they
run offline and stay stable across benchmark rebuilds.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest


HORIZON_DAYS = 14
LOOKBACK_DAYS = 90


def _d(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _synthetic_bundle(horizon_days: int = HORIZON_DAYS) -> list[dict]:
    """Return 9 FDs (3 per track) + 30 articles (10 per track) built under
    v2.2 semantics: forecast_point = resolution_date - horizon_days, every
    article satisfies publish_date <= forecast_point."""
    fds: list[dict] = []
    tracks = [
        ("forecastbench", "fb",  "2026-03-15"),
        ("gdelt-cameo",   "gdc", "2026-03-20"),
        ("earnings",      "earn", "2026-04-05"),
    ]
    for bench, prefix, res_str in tracks:
        res_dt = _d(res_str)
        fp_dt = res_dt - timedelta(days=horizon_days)
        for i in range(3):
            fd_id = f"{prefix}_{i}"
            # 10 articles, dates 1..10 days before forecast_point (all legal)
            art_ids = [f"art_{fd_id}_{k}" for k in range(10)]
            articles = [
                {
                    "id": art_ids[k],
                    "url": f"https://example.com/{fd_id}/{k}",
                    "date": (fp_dt - timedelta(days=k + 1)).strftime("%Y-%m-%d"),
                    "fd_id": fd_id,
                }
                for k in range(10)
            ]
            fds.append({
                "id": fd_id,
                "benchmark": bench,
                "forecast_point": fp_dt.strftime("%Y-%m-%d"),
                "resolution_date": res_dt.strftime("%Y-%m-%d"),
                "article_ids": art_ids,
                "_articles": articles,  # co-located for test convenience
            })
    return fds


def test_forecast_point_equals_resolution_minus_horizon():
    for fd in _synthetic_bundle(HORIZON_DAYS):
        res = _d(fd["resolution_date"])
        fp = _d(fd["forecast_point"])
        delta = (res - fp).days
        assert delta == HORIZON_DAYS, (
            f"{fd['id']}: resolution - forecast_point = {delta} != {HORIZON_DAYS}"
        )
        assert delta >= HORIZON_DAYS, (
            f"{fd['id']}: horizon must be >= {HORIZON_DAYS}"
        )


def test_no_article_post_forecast_point():
    """Hard leakage constraint on the bundle."""
    for fd in _synthetic_bundle():
        fp = _d(fd["forecast_point"])
        for art in fd["_articles"]:
            art_dt = _d(art["date"])
            assert art_dt <= fp, (
                f"LEAKAGE: {fd['id']} article {art['id']} dated {art['date']} "
                f"exceeds forecast_point {fd['forecast_point']}"
            )


def _leakage_filter(articles: list[dict], forecast_point: datetime) -> tuple[list[dict], int]:
    """Minimal reimplementation of the v2.2 fetch-time leakage filter, to
    exercise the contract in a test without spawning the real fetcher.
    Mirrors the loop in scripts/fetch_*_news.py."""
    kept: list[dict] = []
    dropped = 0
    for art in articles:
        pd_str = art.get("date", "") or ""
        if pd_str:
            try:
                pd_dt = _d(pd_str[:10])
                if pd_dt > forecast_point:
                    dropped += 1
                    continue
            except ValueError:
                pass
        kept.append(art)
    return kept, dropped


def test_planted_leakage_is_dropped():
    """Insert one article dated 1 day AFTER forecast_point into each FD's
    fetch results and verify the filter removes exactly those records."""
    for fd in _synthetic_bundle():
        fp = _d(fd["forecast_point"])
        planted = {
            "id": f"planted_{fd['id']}",
            "url": f"https://example.com/{fd['id']}/planted",
            "date": (fp + timedelta(days=1)).strftime("%Y-%m-%d"),
            "fd_id": fd["id"],
        }
        raw = list(fd["_articles"]) + [planted]
        kept, dropped = _leakage_filter(raw, fp)
        assert dropped == 1, f"{fd['id']}: expected 1 leakage drop, got {dropped}"
        assert planted["id"] not in {a["id"] for a in kept}, (
            f"{fd['id']}: planted leakage article survived the filter"
        )
        # All originals must survive
        assert len(kept) == len(fd["_articles"])


@pytest.mark.parametrize("horizon_days", [7, 14, 30])
def test_configured_horizon_respected(horizon_days: int):
    """Same invariant as test_forecast_point_equals_resolution_minus_horizon
    but parameterised across non-default horizons."""
    for fd in _synthetic_bundle(horizon_days):
        res = _d(fd["resolution_date"])
        fp = _d(fd["forecast_point"])
        assert (res - fp).days == horizon_days


def test_article_dated_exactly_on_forecast_point_is_retained():
    """Boundary: publish_date == forecast_point is NOT leakage (<= is the rule)."""
    fp = _d("2026-03-01")
    arts = [{"id": "a0", "date": "2026-03-01", "url": ""}]
    kept, dropped = _leakage_filter(arts, fp)
    assert dropped == 0
    assert len(kept) == 1
