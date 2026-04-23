"""Smoke tests for src/emrach/facts_rows.py.

The module is a standalone helper (no EMR-ACH pipeline dependency); these
tests cover the projection contract and the sort ordering used by F3.
"""

from __future__ import annotations

import pytest

from src.emrach.facts_rows import fact_to_indicator_row, fd_facts_to_rows


def _make_fact(**overrides):
    base = {
        "id": "f_abc12345",
        "schema_version": "1.0",
        "time": "2025-12-15",
        "fact": "Country A imposed sanctions on Country B.",
        "article_ids": ["art_1", "art_2"],
        "primary_article_id": "art_1",
        "article_date": "2025-12-16",
        "extractor": "test",
        "extract_run": "test",
        "extracted_at": "2025-12-16T00:00:00Z",
        "extraction_confidence": "high",
        "entities": [
            {"name": "Country A", "type": "country", "role": "actor"},
            {"name": "Country B", "type": "country", "role": "target"},
        ],
        "source": "test-source",
    }
    base.update(overrides)
    return base


def test_fact_to_indicator_row_canonical_field():
    fact = _make_fact()
    row = fact_to_indicator_row(fact)
    assert row["fact_id"] == "f_abc12345"
    assert row["time"] == "2025-12-15"
    assert row["descriptor"] == "Country A imposed sanctions on Country B."
    assert row["article_ids"] == ["art_1", "art_2"]
    assert row["primary_article_id"] == "art_1"
    assert row["evidence_strength_prior"] == pytest.approx(0.9)
    assert row["extraction_confidence"] == "high"
    assert len(row["entities"]) == 2
    assert row["entities"][0]["name"] == "Country A"


def test_fact_to_indicator_row_legacy_fact_text():
    fact = _make_fact(fact=None, fact_text="Legacy descriptor wins.")
    row = fact_to_indicator_row(fact)
    assert row["descriptor"] == "Legacy descriptor wins."


def test_confidence_mapping():
    assert fact_to_indicator_row(_make_fact(extraction_confidence="medium"))[
        "evidence_strength_prior"
    ] == pytest.approx(0.6)
    assert fact_to_indicator_row(_make_fact(extraction_confidence="low"))[
        "evidence_strength_prior"
    ] == pytest.approx(0.3)
    assert fact_to_indicator_row(_make_fact(extraction_confidence=None))[
        "evidence_strength_prior"
    ] == pytest.approx(0.5)


def test_confidence_numeric_passthrough_clamped():
    assert fact_to_indicator_row(_make_fact(extraction_confidence=0.42))[
        "evidence_strength_prior"
    ] == pytest.approx(0.42)
    assert fact_to_indicator_row(_make_fact(extraction_confidence=1.5))[
        "evidence_strength_prior"
    ] == pytest.approx(1.0)
    assert fact_to_indicator_row(_make_fact(extraction_confidence=-0.5))[
        "evidence_strength_prior"
    ] == pytest.approx(0.0)


def test_fd_facts_to_rows_sort_and_limit():
    fd = {"id": "fd_test"}
    facts = [
        _make_fact(id="f_old", time="2024-01-01", extraction_confidence="high"),
        _make_fact(id="f_new_low", time="2025-12-15", extraction_confidence="low"),
        _make_fact(id="f_new_high", time="2025-12-15", extraction_confidence="high"),
        _make_fact(id="f_unknown", time="unknown", extraction_confidence="high"),
    ]
    rows = fd_facts_to_rows(fd, facts, top_k=3)
    assert len(rows) == 3
    # newest date wins over older; within-date confidence breaks the tie.
    assert rows[0]["fact_id"] == "f_new_high"
    assert rows[1]["fact_id"] == "f_new_low"
    assert rows[2]["fact_id"] == "f_old"
    # 'unknown' was excluded because top_k=3.


def test_fd_facts_to_rows_top_k_zero_raises():
    with pytest.raises(ValueError):
        fd_facts_to_rows({"id": "x"}, [], top_k=0)


def test_fd_facts_to_rows_bad_fd_raises():
    with pytest.raises(TypeError):
        fd_facts_to_rows("not-a-dict", [], top_k=5)  # type: ignore[arg-type]


def test_fd_facts_to_rows_filters_non_dicts():
    rows = fd_facts_to_rows({"id": "x"}, [_make_fact(), None, "junk"], top_k=10)
    assert len(rows) == 1
