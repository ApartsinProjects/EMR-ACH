"""src.unify.csv_helpers: shared CSV plumbing for the unifier (v2.2 [B3+E2]).

Lifts the ``_with_raised_csv_field_limit`` decorator out of
``scripts/unify_articles.py`` so other consumers (e.g. the GDELT KG
loaders) can reuse it without duplicating the pattern.
"""

from __future__ import annotations

import csv
import functools
import sys
from typing import Callable, TypeVar

__all__ = ["with_raised_csv_field_limit", "raise_csv_field_limit"]


F = TypeVar("F", bound=Callable[..., object])


def raise_csv_field_limit(target: int | None = None) -> int:
    """Idempotently raise ``csv.field_size_limit`` toward ``sys.maxsize``.
    Returns the new limit. ``target`` may be supplied to set an explicit
    cap; defaults to the largest value Python's underlying C field
    accepts. The standard pattern (halving on OverflowError) is the
    same one used inline in unify_articles today.
    """
    cap = target if target is not None else sys.maxsize
    while True:
        try:
            csv.field_size_limit(cap)
            return cap
        except OverflowError:
            cap = cap // 2


def with_raised_csv_field_limit(func: F) -> F:
    """Decorator: raise the CSV field-size limit before invoking ``func``.
    Idempotent; safe to stack.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        raise_csv_field_limit()
        return func(*args, **kwargs)

    return _wrapper  # type: ignore[return-value]
