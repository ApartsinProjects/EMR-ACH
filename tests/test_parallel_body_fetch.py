"""Tests for src.common.parallel_body_fetch (v2.2 [A5])."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.parallel_body_fetch import (
    BodyFetchResult,
    fetch_bodies_parallel,
)


def test_empty_input_returns_empty():
    assert fetch_bodies_parallel([], fetch_fn=lambda u: u) == []


def test_fetch_all_ok_preserves_order():
    urls = [f"u{i}" for i in range(10)]
    out = fetch_bodies_parallel(urls, fetch_fn=lambda u: u.upper(), max_workers=4)
    assert [r.url for r in out] == urls
    assert all(r.ok for r in out)
    assert [r.body for r in out] == [u.upper() for u in urls]


def test_fetch_captures_per_url_exceptions():
    def flaky(u: str) -> str | None:
        if u.endswith("bad"):
            raise RuntimeError("boom")
        return u.upper()

    urls = ["good1", "bad", "good2"]
    out = fetch_bodies_parallel(urls, fetch_fn=flaky, max_workers=3)
    errs = {r.url: r.error for r in out}
    assert errs["good1"] is None
    assert errs["good2"] is None
    assert "RuntimeError" in (errs["bad"] or "")


def test_fetch_none_body_is_not_ok():
    def always_none(u: str):
        return None

    out = fetch_bodies_parallel(["u1", "u2"], fetch_fn=always_none)
    assert all((not r.ok) for r in out)
    assert all(r.error is None for r in out)


def test_result_dataclass_fields():
    r = BodyFetchResult(url="u", body="b")
    assert r.url == "u"
    assert r.body == "b"
    assert r.ok is True
    assert BodyFetchResult(url="u", body=None, error="x").ok is False
