"""Smoke tests for src.common.paths (v2.2 [B4] + [B4a])."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Local bootstrap so the test runs whether pytest discovers it from the
# repo root or from the tests/ directory.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.common.paths import (
    BENCHMARK_DIR,
    DATA_DIR,
    REPO_ROOT,
    UNIFIED_DIR,
    bootstrap_sys_path,
    benchmark_data_dir,
    ensure_dir,
    known_benchmarks,
    per_benchmark_articles_path,
    per_benchmark_dir,
)


def test_repo_root_contains_expected_dirs():
    assert (REPO_ROOT / "src").is_dir()
    assert (REPO_ROOT / "scripts").is_dir()


def test_data_dir_layout():
    assert DATA_DIR == REPO_ROOT / "data"
    assert UNIFIED_DIR == DATA_DIR / "unified"
    assert BENCHMARK_DIR == REPO_ROOT / "benchmark" / "data"


def test_bootstrap_idempotent():
    root_str = str(REPO_ROOT)
    # Strip every existing occurrence so we can verify the helper inserts
    # exactly one copy and refuses to add a second.
    sys.path[:] = [p for p in sys.path if p != root_str]
    bootstrap_sys_path()
    first_count = sys.path.count(root_str)
    bootstrap_sys_path()
    second_count = sys.path.count(root_str)
    assert sys.path[0] == root_str
    assert first_count == 1
    assert second_count == 1


def test_per_benchmark_paths():
    for bench in known_benchmarks():
        assert per_benchmark_dir(bench) == DATA_DIR / bench
        assert per_benchmark_articles_path(bench) == (
            DATA_DIR / bench / f"{bench}_articles.jsonl"
        )


def test_benchmark_data_dir_with_and_without_cutoff():
    assert benchmark_data_dir() == BENCHMARK_DIR
    assert benchmark_data_dir("2026-01-01") == BENCHMARK_DIR / "2026-01-01"


def test_ensure_dir_creates_and_returns(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    out = ensure_dir(target)
    assert out == target
    assert target.is_dir()


def test_known_benchmarks_stable_order():
    assert tuple(known_benchmarks()) == ("forecastbench", "gdelt_cameo", "earnings")
