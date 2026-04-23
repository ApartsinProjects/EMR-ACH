"""Tests for query_gdelt_doc_index shard-prune logic (v2.2 [A3] + [A8])."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import from the script directly. The script uses a local sys.path
# insert so `from scripts.query_gdelt_doc_index` works after that runs;
# we import via spec below to avoid side effects on argv.
import importlib.util as _iu

_SPEC = _iu.spec_from_file_location(
    "_qgdi",
    Path(__file__).resolve().parent.parent / "scripts" / "query_gdelt_doc_index.py",
)
_mod = _iu.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]

shard_intersects_window = _mod.shard_intersects_window
compute_prune_plan = _mod.compute_prune_plan
filter_candidates_by_domain = _mod.filter_candidates_by_domain


def test_shard_inside_window_kept():
    assert shard_intersects_window("2026-03", date(2026, 1, 1), date(2026, 4, 1))


def test_shard_before_window_pruned():
    assert not shard_intersects_window(
        "2025-12", date(2026, 2, 1), date(2026, 4, 1)
    )


def test_shard_after_window_pruned():
    assert not shard_intersects_window(
        "2026-05", date(2026, 1, 1), date(2026, 4, 1)
    )


def test_shard_overlapping_boundary_kept():
    # Shard's end-of-month equals window start -> intersects.
    assert shard_intersects_window(
        "2026-01", date(2026, 1, 15), date(2026, 3, 1)
    )


def test_compute_prune_plan_partitions():
    plan = compute_prune_plan(
        ["2025-11", "2025-12", "2026-01", "2026-02", "2026-05"],
        window_start=date(2026, 1, 1),
        window_end=date(2026, 3, 1),
    )
    assert plan["kept"] == ["2026-01", "2026-02"]
    assert plan["pruned"] == ["2025-11", "2025-12", "2026-05"]


def test_filter_candidates_by_domain_drops_aggregators():
    cands = [
        {"url": "https://news.google.com/x"},
        {"url": "https://reuters.com/y"},
        {"url": "https://archive.org/z"},
    ]
    kept = filter_candidates_by_domain(cands)
    assert [c["url"] for c in kept] == ["https://reuters.com/y"]
