"""Smoke tests for src.common.config_slices (v2.2 [B5] + [C5])."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config_slices import (
    STAGE_SLICES,
    hash_slice,
    known_stages,
    slice_config,
    stage_cache_key,
)


@pytest.fixture
def sample_config():
    return {
        "cutoff": "2026-01-01",
        "fetch": {
            "lookback_days": 90,
            "spam_blocklist_revision": "r3",
            "forecastbench": {"eligible_sources": ["nytimes.com", "reuters.com"]},
            "gdelt_cameo": {"aggregator_blocklist_revision": "r1"},
            "earnings": {"tickers_revision": "sp500.v2"},
        },
        "relevance": {
            "encoder": {
                "model": "all-MiniLM-L6-v2",
                "batch_size": 256,
                "fp16": True,
                "backend": "sbert",
            },
            "top_k_per_fd": 50,
            "lookback_days": 90,
        },
        "etd": {
            "prompt_version": "v3",
            "model": "gpt-5-nano",
            "strict_dates": True,
            "strict_quotes": True,
        },
        # Intentionally unrelated key; mutating this must NOT change the
        # compute_relevance cache key.
        "unrelated_key": {"value": 42},
    }


def test_known_stages_nonempty():
    stages = list(known_stages())
    assert "compute_relevance" in stages
    assert "step_publish" in stages
    assert len(stages) == len(STAGE_SLICES)


def test_slice_config_returns_only_declared_keys(sample_config):
    sl = slice_config(sample_config, "compute_relevance")
    assert "relevance.encoder.model" in sl
    assert "unrelated_key" not in sl
    # Every key listed in the slice declaration must appear (with value
    # or None) in the output.
    for k in STAGE_SLICES["compute_relevance"]:
        assert k in sl


def test_slice_config_unknown_stage_raises(sample_config):
    with pytest.raises(KeyError):
        slice_config(sample_config, "no_such_stage")


def test_hash_slice_stable_under_dict_reorder():
    a = {"cutoff": "2026-01-01", "x": 1}
    b = {"x": 1, "cutoff": "2026-01-01"}
    assert hash_slice(a) == hash_slice(b)


def test_unrelated_mutation_does_not_change_cache_key(sample_config):
    """Regression for C5: editing a key outside a stage's slice must not
    invalidate the stage's cache."""
    key1 = stage_cache_key(sample_config, "compute_relevance")
    sample_config["unrelated_key"]["value"] = 9999
    key2 = stage_cache_key(sample_config, "compute_relevance")
    assert key1 == key2


def test_related_mutation_changes_cache_key(sample_config):
    key1 = stage_cache_key(sample_config, "compute_relevance")
    sample_config["relevance"]["encoder"]["batch_size"] = 128
    key2 = stage_cache_key(sample_config, "compute_relevance")
    assert key1 != key2


def test_missing_keys_resolve_to_none(sample_config):
    del sample_config["relevance"]
    sl = slice_config(sample_config, "compute_relevance")
    assert sl["relevance.encoder.model"] is None
    # And the hash is still computable (no exception).
    assert isinstance(hash_slice(sl), str)
