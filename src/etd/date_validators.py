"""src.etd.date_validators: pure-function date validators for ETD facts
(v2.2 [B7], absorbs [E4]).

Lifted out of ``scripts/articles_to_facts.py`` so each predicate is
unit-testable in isolation. The ``> vs >=`` bug recovered in commit
7237553 (Phase A; ~8,647 wrongly-rejected facts) would have been
caught by a per-validator test on ``is_post_publish``; that scenario is
now in :mod:`tests.test_etd_date_validators`.

All predicates take ISO-format date strings ("YYYY-MM-DD"); the parser
is tolerant of trailing time components ("YYYY-MM-DDTHH:MM:SSZ") so it
can ingest both fact dates and article publish dates without the caller
having to pre-trim.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final, Optional

__all__ = [
    "parse_iso_date",
    "is_iso_format",
    "is_calendar_valid",
    "is_post_publish",
    "is_within_window",
    "is_not_future",
    "ParseError",
]


_ISO_DATE_LEN: Final[int] = 10


class ParseError(ValueError):
    """Raised when a date string cannot be coerced to a calendar date."""


def parse_iso_date(s: str) -> _dt.date:
    """Parse ``YYYY-MM-DD`` (with optional ``THH:MM:SSZ`` trailing
    time) into a :class:`datetime.date`. Raises :class:`ParseError` on
    any other shape; callers that want a soft check should use
    :func:`is_iso_format` + :func:`is_calendar_valid` first.
    """
    if not isinstance(s, str) or len(s) < _ISO_DATE_LEN:
        raise ParseError(f"not an ISO date: {s!r}")
    head = s[:_ISO_DATE_LEN]
    try:
        return _dt.date.fromisoformat(head)
    except ValueError as exc:
        raise ParseError(f"not an ISO date: {s!r} ({exc})") from exc


def is_iso_format(s: object) -> bool:
    """``True`` if ``s`` is a string whose first 10 characters look like
    ``YYYY-MM-DD`` (digits + hyphens in the right slots). Does NOT
    check that the date is real; use :func:`is_calendar_valid` for that.
    """
    if not isinstance(s, str) or len(s) < _ISO_DATE_LEN:
        return False
    head = s[:_ISO_DATE_LEN]
    return (
        head[4] == "-"
        and head[7] == "-"
        and head[0:4].isdigit()
        and head[5:7].isdigit()
        and head[8:10].isdigit()
    )


def is_calendar_valid(s: str) -> bool:
    """``True`` if the string parses to a real calendar date (no
    Feb-30, no month-13)."""
    if not is_iso_format(s):
        return False
    try:
        parse_iso_date(s)
    except ParseError:
        return False
    return True


def is_post_publish(
    fact_date: str | _dt.date, article_publish_date: str | _dt.date
) -> bool:
    """``True`` iff the fact's date is strictly AFTER the article's
    publish date. The strict inequality is the v2.1 leakage rule: a
    fact dated on or before the article's publish day is allowed
    because the article could legitimately report a same-day or
    earlier event.

    NOTE: this is the inverse polarity used elsewhere; see commit
    7237553 ("relax leakage validator to >") which changed the gate
    from ``>=`` (over-rejecting same-day facts) to ``>`` (the correct
    rule). The exhaustive test suite below pins both endpoints.

    Returns ``False`` (i.e. "this fact is NOT a leak") for the safe
    case so the caller can read it as ``if is_post_publish(...): drop``.
    """
    fd = fact_date if isinstance(fact_date, _dt.date) else parse_iso_date(fact_date)
    ad = (
        article_publish_date
        if isinstance(article_publish_date, _dt.date)
        else parse_iso_date(article_publish_date)
    )
    return fd > ad


def is_within_window(
    fact_date: str | _dt.date,
    window_start: str | _dt.date,
    window_end: str | _dt.date,
    *,
    inclusive: bool = True,
) -> bool:
    """Whether ``fact_date`` falls in ``[window_start, window_end]``
    (inclusive by default; pass ``inclusive=False`` for the half-open
    ``[start, end)`` variant used by some leakage probes).
    """
    fd = fact_date if isinstance(fact_date, _dt.date) else parse_iso_date(fact_date)
    ws = (
        window_start
        if isinstance(window_start, _dt.date)
        else parse_iso_date(window_start)
    )
    we = (
        window_end
        if isinstance(window_end, _dt.date)
        else parse_iso_date(window_end)
    )
    if inclusive:
        return ws <= fd <= we
    return ws <= fd < we


def is_not_future(
    fact_date: str | _dt.date, *, today: Optional[_dt.date] = None
) -> bool:
    """``True`` iff the fact's date is on or before ``today``. Used by
    the production filter recipe (``--no-future``).
    """
    fd = fact_date if isinstance(fact_date, _dt.date) else parse_iso_date(fact_date)
    today = today or _dt.date.today()
    return fd <= today
