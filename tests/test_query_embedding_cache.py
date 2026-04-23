"""Tests for src.common.query_embedding_cache (v2.2 [A9])."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.query_embedding_cache import (
    QueryEmbeddingCache,
    query_cache_key,
)


def test_query_cache_key_stable():
    a = query_cache_key("q", "b", backend="sbert", model="m")
    b = query_cache_key("q", "b", backend="sbert", model="m")
    assert a == b
    assert len(a) == 32


def test_query_cache_key_differs_on_backend():
    a = query_cache_key("q", "b", backend="sbert", model="m")
    c = query_cache_key("q", "b", backend="openai", model="m")
    assert a != c


def test_query_cache_key_differs_on_model():
    a = query_cache_key("q", "b", backend="sbert", model="m1")
    c = query_cache_key("q", "b", backend="sbert", model="m2")
    assert a != c


def test_put_then_get_roundtrip(tmp_path):
    cache = QueryEmbeddingCache(backend="sbert", model="m", root=tmp_path)
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.put("q", "b", v)
    got = cache.get("q", "b")
    assert got is not None
    assert np.allclose(got, v)


def test_miss_returns_none(tmp_path):
    cache = QueryEmbeddingCache(backend="sbert", model="m", root=tmp_path)
    assert cache.get("q", "b") is None


def test_get_or_encode_calls_encoder_once(tmp_path):
    cache = QueryEmbeddingCache(backend="sbert", model="m", root=tmp_path)
    calls = []

    def encoder(text: str):
        calls.append(text)
        return np.ones(4, dtype=np.float32)

    v1 = cache.get_or_encode("q", "b", encoder)
    v2 = cache.get_or_encode("q", "b", encoder)
    assert np.allclose(v1, v2)
    assert len(calls) == 1


def test_cross_backend_isolation(tmp_path):
    c1 = QueryEmbeddingCache(backend="sbert", model="m", root=tmp_path)
    c2 = QueryEmbeddingCache(backend="openai", model="m", root=tmp_path)
    v = np.array([1.0], dtype=np.float32)
    c1.put("q", "b", v)
    assert c2.get("q", "b") is None
