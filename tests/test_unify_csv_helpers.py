"""Tests for src.unify.csv_helpers (v2.2 [B3+E2])."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.unify.csv_helpers import (
    raise_csv_field_limit,
    with_raised_csv_field_limit,
)


def test_raise_csv_field_limit_returns_positive_int():
    new = raise_csv_field_limit()
    assert isinstance(new, int)
    assert new > 0
    # Verify the global limit was actually set.
    assert csv.field_size_limit() == new


def test_decorator_runs_and_passes_args_through():
    @with_raised_csv_field_limit
    def add(a, b, *, c=0):
        return a + b + c

    assert add(1, 2, c=4) == 7


def test_decorator_idempotent_when_stacked():
    @with_raised_csv_field_limit
    @with_raised_csv_field_limit
    def f():
        return csv.field_size_limit()

    assert f() > 0
