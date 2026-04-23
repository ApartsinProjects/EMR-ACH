"""Smoke tests for src.common.embeddings_backend (v2.2 [A6])."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.embeddings_backend import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SBERT_MODEL,
    backend_identity,
    encode,
)


def test_module_imports_without_torch_or_openai():
    # Importing the module must not require heavy ML deps.
    import src.common.embeddings_backend as mod

    assert mod.DEFAULT_SBERT_MODEL
    assert mod.DEFAULT_OPENAI_MODEL


def test_backend_identity_sbert_default():
    backend, model, _rev = backend_identity("sbert")
    assert backend == "sbert"
    assert model == DEFAULT_SBERT_MODEL


def test_backend_identity_openai_default():
    backend, model, rev = backend_identity("openai")
    assert backend == "openai"
    assert model == DEFAULT_OPENAI_MODEL
    # Revision falls back to model id for OpenAI.
    assert rev == model


def test_backend_identity_openai_batch_default():
    backend, model, rev = backend_identity("openai_batch")
    assert backend == "openai_batch"
    assert model == DEFAULT_OPENAI_MODEL
    assert rev == model


def test_backend_identity_unknown_raises():
    with pytest.raises(ValueError):
        backend_identity("not-a-backend")  # type: ignore[arg-type]


def test_encode_unknown_backend_raises():
    with pytest.raises(ValueError):
        encode(["hello"], backend="not-a-backend")  # type: ignore[arg-type]
