"""Tests for src.common.stage_cache (v2.2 [G1] + [B6] + [C4])."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.stage_cache import StageCache, StageMeta


@pytest.fixture
def cfg():
    return {
        "cutoff": "2026-01-01",
        "relevance": {
            "encoder": {
                "model": "mpnet", "batch_size": 256, "fp16": True,
                "backend": "sbert",
            },
            "top_k_per_fd": 50,
            "lookback_days": 90,
        },
    }


def test_is_valid_false_when_no_meta(tmp_path, cfg):
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    assert cache.is_valid(cfg) is False


def test_record_then_is_valid(tmp_path, cfg):
    out_file = tmp_path / "out.jsonl"
    out_file.write_text("{}\n")
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[out_file], n_rows=1)
    assert cache.is_valid(cfg) is True


def test_is_valid_false_after_config_change(tmp_path, cfg):
    out_file = tmp_path / "out.jsonl"
    out_file.write_text("{}\n")
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[out_file], n_rows=1)
    cfg["relevance"]["encoder"]["batch_size"] = 128
    assert cache.is_valid(cfg) is False


def test_unrelated_config_change_keeps_cache_valid(tmp_path, cfg):
    out_file = tmp_path / "out.jsonl"
    out_file.write_text("{}\n")
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[out_file], n_rows=1)
    cfg["unrelated"] = "whatever"
    assert cache.is_valid(cfg) is True


def test_is_valid_false_when_output_missing(tmp_path, cfg):
    out_file = tmp_path / "out.jsonl"
    out_file.write_text("{}\n")
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[out_file], n_rows=1)
    out_file.unlink()
    assert cache.is_valid(cfg) is False


def test_invalidate_removes_meta(tmp_path, cfg):
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[], n_rows=0)
    assert cache.meta_path.exists()
    cache.invalidate()
    assert not cache.meta_path.exists()


def test_record_is_atomic_no_partial_files(tmp_path, cfg):
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[], n_rows=0)
    # Only the final file; no .tmp leftovers.
    dirlist = list(cache.meta_path.parent.iterdir())
    assert [p.name for p in dirlist] == ["compute_relevance.json"]


def test_load_returns_stage_meta(tmp_path, cfg):
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.record(cfg, outputs=[], n_rows=42)
    meta = cache.load()
    assert isinstance(meta, StageMeta)
    assert meta.stage == "compute_relevance"
    assert meta.n_rows == 42


def test_malformed_meta_does_not_crash(tmp_path, cfg):
    cache = StageCache("compute_relevance", "2026-01-01", meta_root=tmp_path)
    cache.meta_path.parent.mkdir(parents=True, exist_ok=True)
    cache.meta_path.write_text("not-json")
    assert cache.load() is None
    assert cache.is_valid(cfg) is False
