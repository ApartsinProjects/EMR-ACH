"""Per-validator tests for src.etd.date_validators (v2.2 [B7] + [E4]).

The ``> vs >=`` regression in commit 7237553 (Phase A; ~8,647 wrongly-
rejected facts) is pinned by ``test_is_post_publish_same_day_is_safe``.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etd.date_validators import (
    ParseError,
    is_calendar_valid,
    is_iso_format,
    is_not_future,
    is_post_publish,
    is_within_window,
    parse_iso_date,
)


def test_parse_iso_date_happy():
    assert parse_iso_date("2026-04-23") == _dt.date(2026, 4, 23)


def test_parse_iso_date_with_time_suffix():
    assert parse_iso_date("2026-04-23T12:34:56Z") == _dt.date(2026, 4, 23)


def test_parse_iso_date_rejects_garbage():
    for bad in ("", "not-a-date", "2026/04/23", "20260423", None, 12345):
        with pytest.raises(ParseError):
            parse_iso_date(bad)  # type: ignore[arg-type]


def test_is_iso_format_true_false_cases():
    assert is_iso_format("2026-04-23")
    assert is_iso_format("2026-04-23T00:00:00")
    assert not is_iso_format("2026/04/23")
    assert not is_iso_format("23-04-2026")
    assert not is_iso_format("")
    assert not is_iso_format(None)


def test_is_calendar_valid_rejects_impossible_dates():
    assert is_calendar_valid("2026-04-23")
    assert not is_calendar_valid("2026-02-30")
    assert not is_calendar_valid("2026-13-01")
    assert not is_calendar_valid("2026-04-31")


def test_is_post_publish_same_day_is_safe():
    """Regression for commit 7237553: a fact dated on the article's
    publish day is NOT a leak (must return False)."""
    assert is_post_publish("2026-04-23", "2026-04-23") is False


def test_is_post_publish_strictly_after_is_leak():
    assert is_post_publish("2026-04-24", "2026-04-23") is True


def test_is_post_publish_before_is_safe():
    assert is_post_publish("2026-04-22", "2026-04-23") is False


def test_is_post_publish_accepts_date_objects():
    assert (
        is_post_publish(_dt.date(2026, 5, 1), _dt.date(2026, 4, 23)) is True
    )


def test_is_within_window_inclusive():
    assert is_within_window("2026-04-23", "2026-04-01", "2026-04-30")
    assert is_within_window("2026-04-01", "2026-04-01", "2026-04-30")
    assert is_within_window("2026-04-30", "2026-04-01", "2026-04-30")
    assert not is_within_window("2026-05-01", "2026-04-01", "2026-04-30")


def test_is_within_window_half_open():
    assert is_within_window(
        "2026-04-29", "2026-04-01", "2026-04-30", inclusive=False
    )
    assert not is_within_window(
        "2026-04-30", "2026-04-01", "2026-04-30", inclusive=False
    )


def test_is_not_future():
    today = _dt.date(2026, 4, 23)
    assert is_not_future("2026-04-23", today=today)
    assert is_not_future("2025-12-31", today=today)
    assert not is_not_future("2026-04-24", today=today)
