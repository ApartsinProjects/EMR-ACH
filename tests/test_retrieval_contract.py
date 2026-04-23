"""Smoke tests for src.retrieval.contract (v2.2 [B1])."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.contract import (
    CONTRACT,
    Benchmark,
    RetrievalMode,
    Source,
    all_sources_for,
    contract_for,
    primary_source_for,
    requires_editorial_filter,
    retrieval_mode_for,
)


def test_all_three_benchmarks_have_contracts():
    for b in Benchmark:
        assert b in CONTRACT


def test_forecastbench_cascade():
    c = contract_for("forecastbench")
    assert c.primary == Source.GOOGLE_NEWS
    assert Source.GDELT_DOC in c.secondaries
    assert c.editorial_filter is False
    assert c.retrieval_mode == RetrievalMode.SBERT_COSINE


def test_gdelt_cameo_uses_editorial_filter():
    assert requires_editorial_filter("gdelt_cameo") is True
    assert requires_editorial_filter("forecastbench") is False
    assert requires_editorial_filter("earnings") is False


def test_earnings_uses_relational_join_not_sbert():
    assert retrieval_mode_for("earnings") == RetrievalMode.TICKER_DATE_JOIN
    assert retrieval_mode_for("forecastbench") == RetrievalMode.SBERT_COSINE


def test_primary_source_for_each_benchmark():
    assert primary_source_for("forecastbench") == Source.GOOGLE_NEWS
    assert primary_source_for("gdelt_cameo") == Source.GDELT_DOC
    assert primary_source_for("earnings") == Source.SEC_EDGAR


def test_all_sources_cascade_order_primary_first():
    sources = all_sources_for("gdelt_cameo")
    assert sources[0] == Source.GDELT_DOC
    assert Source.GOOGLE_NEWS in sources[1:]


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError):
        contract_for("not-a-benchmark")


def test_contract_is_frozen_dataclass():
    c = contract_for("earnings")
    with pytest.raises(Exception):
        c.primary = Source.GOOGLE_NEWS  # type: ignore[misc]
