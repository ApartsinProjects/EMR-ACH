"""ETD-fact -> EMR-ACH indicator-row projection.

Standalone helpers (no dependency on EMR-ACH's existing pipeline) that
project ETD atomic facts (see ``docs/etd.schema.json``) into the
indicator-row dict shape consumed by EMR-ACH's analysis matrix A
(see paper section 4.3). When the F3 wiring lands, EMR-ACH will import
this module to source per-FD evidence rows from facts.jsonl alongside
articles.jsonl.

Schema notes
------------
ETD fact text lives under the canonical key ``fact`` (per
``docs/etd.schema.json``). Some pre-v2.2 callers spelled it
``fact_text``. We accept both: ``fact_text`` wins if present, else
``fact``.

Confidence in the ETD schema is the categorical
``extraction_confidence`` in {"high", "medium", "low", None}; we map it
to a numeric prior weight in [0.0, 1.0] for downstream use.
"""

from __future__ import annotations

from typing import Any, Iterable

# Map ETD's categorical extraction_confidence to EMR-ACH's evidence-strength
# prior. None / unknown collapses to a neutral midpoint so downstream priors
# don't silently drop the row.
_CONFIDENCE_TO_WEIGHT: dict[str | None, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
    None: 0.5,
}


def _fact_text(fact: dict) -> str:
    """Return the fact's natural-language descriptor.

    Prefer ``fact_text`` for legacy callers; fall back to the canonical
    ``fact`` field per docs/etd.schema.json.
    """
    text = fact.get("fact_text")
    if text:
        return str(text)
    return str(fact.get("fact") or "")


def _confidence_weight(fact: dict) -> float:
    raw = fact.get("extraction_confidence")
    if isinstance(raw, (int, float)):
        # Already numeric (some callers carry a probability). Clamp to [0,1].
        return max(0.0, min(1.0, float(raw)))
    return _CONFIDENCE_TO_WEIGHT.get(raw, 0.5)


def fact_to_indicator_row(fact: dict) -> dict:
    """Project an ETD atomic fact into an indicator-row dict.

    The returned dict is compatible with EMR-ACH's analysis matrix A
    (see paper section 4.3). Uses ``fact.time`` + ``fact.entities`` +
    fact-text as the indicator descriptor; ``fact.extraction_confidence``
    maps to the row's evidence-strength prior.

    Parameters
    ----------
    fact:
        A single ETD fact dict (one row from facts.jsonl).

    Returns
    -------
    dict with keys:
        ``fact_id``, ``time``, ``entities``, ``descriptor``,
        ``article_ids``, ``primary_article_id``, ``source``,
        ``evidence_strength_prior``, ``extraction_confidence``.
    """
    entities = fact.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    return {
        "fact_id": fact.get("id"),
        "time": fact.get("time"),
        "entities": [
            {
                "name": e.get("name"),
                "type": e.get("type"),
                "role": e.get("role"),
            }
            for e in entities
            if isinstance(e, dict)
        ],
        "descriptor": _fact_text(fact),
        "article_ids": list(fact.get("article_ids") or []),
        "primary_article_id": fact.get("primary_article_id"),
        "source": fact.get("source"),
        "evidence_strength_prior": _confidence_weight(fact),
        "extraction_confidence": fact.get("extraction_confidence"),
    }


def _sort_key(fact: dict) -> tuple[str, float]:
    """Sort key: time desc (lexicographic on ISO-prefixed strings), then
    confidence desc.

    'unknown' or missing time sorts last by being mapped to the empty
    string (which is < any real ISO date). We invert order by negating
    the confidence and by sorting reverse=True on the date.
    """
    time = fact.get("time") or ""
    if time == "unknown":
        time = ""
    return (time, _confidence_weight(fact))


def fd_facts_to_rows(
    fd: dict,
    facts: Iterable[dict],
    top_k: int = 20,
) -> list[dict]:
    """Return at most ``top_k`` indicator rows for a given FD.

    Sorted by ``fact.time`` desc, then ``extraction_confidence`` desc.
    The ``fd`` argument is accepted for API symmetry with future
    FD-aware filtering (e.g. entity overlap); the current implementation
    treats ``facts`` as already-filtered to the FD's relevant facts.

    Parameters
    ----------
    fd:
        The forecast dossier dict (used only for shape validation today).
    facts:
        Iterable of ETD fact dicts pre-filtered to the FD.
    top_k:
        Maximum number of rows to return. Must be > 0.

    Returns
    -------
    list of indicator-row dicts (length <= top_k).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    # fd is accepted for future filtering hooks; reference it lightly
    # so callers that pass garbage get a clear error.
    if not isinstance(fd, dict):
        raise TypeError(f"fd must be a dict, got {type(fd).__name__}")

    materialized = [f for f in facts if isinstance(f, dict)]
    materialized.sort(key=_sort_key, reverse=True)
    return [fact_to_indicator_row(f) for f in materialized[:top_k]]


__all__: list[str] = ["fact_to_indicator_row", "fd_facts_to_rows"]
