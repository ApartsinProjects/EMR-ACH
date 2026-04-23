"""Smoke tests for src.common.layout (v2.2 [B9])."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.layout import CutoffLayout, layout_for


def test_layout_paths_under_cutoff_root(tmp_path):
    layout = layout_for("2026-01-01", benchmark_root=tmp_path)
    assert layout.cutoff == "2026-01-01"
    assert layout.root == tmp_path / "2026-01-01"
    assert layout.forecasts == layout.root / "forecasts.jsonl"
    assert layout.forecasts_change == layout.root / "forecasts_change.jsonl"
    assert layout.forecasts_stability == layout.root / "forecasts_stability.jsonl"
    assert layout.build_manifest == layout.root / "build_manifest.json"
    assert layout.benchmark_yaml == layout.root / "benchmark.yaml"


def test_per_benchmark_articles(tmp_path):
    layout = layout_for("c", benchmark_root=tmp_path)
    assert layout.per_benchmark_articles("earnings") == layout.root / "earnings_articles.jsonl"
    assert (
        layout.per_benchmark_articles_checksum("earnings")
        == layout.root / "earnings_articles.checksums.json"
    )


def test_etd_production_path_uses_cutoff(tmp_path):
    layout = layout_for("2026-04-01", benchmark_root=tmp_path)
    assert layout.etd_facts_production.name == "facts.v1_production_2026-04-01.jsonl"


def test_all_deliverable_files_includes_core_set(tmp_path):
    layout = layout_for("c", benchmark_root=tmp_path)
    names = {p.name for p in layout.all_deliverable_files()}
    for expected in (
        "forecasts.jsonl",
        "forecasts_change.jsonl",
        "forecasts_stability.jsonl",
        "articles.jsonl",
        "benchmark.yaml",
        "build_manifest.json",
    ):
        assert expected in names


def test_layout_is_frozen(tmp_path):
    layout = layout_for("c", benchmark_root=tmp_path)
    with pytest.raises(Exception):
        layout.cutoff = "other"  # type: ignore[misc]
