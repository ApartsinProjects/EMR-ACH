"""src.common.embeddings_backend: unified ``encode(...)`` API across SBERT
and OpenAI embedding backends (v2.2 [A6], absorbs residual [A7]).

Per the v2.2 review (REC-03), the OpenAI Batch backend has already
shipped under :mod:`src.common.openai_embeddings` (commits 9a27816 and
a373e89). What was still missing is the umbrella API that lets every
caller (compute_relevance, build_gdelt_doc_index, baselines) ask for
``encode(texts, backend=...)`` without knowing whether SBERT or OpenAI
will run it.

Design notes:

* The SBERT backend is a thin wrapper over ``sentence_transformers``
  using the same ``batch_size=256 + fp16`` defaults used inside
  ``compute_relevance.py:embed`` (the production path). The GPU is
  optional; if torch+CUDA is unavailable the backend falls back to CPU
  with a warning so the wrapper is safe to import in environments
  without a GPU. Heavy ML imports are lazy.

* The OpenAI backend dispatches to :mod:`src.common.openai_embeddings`,
  preserving its native 1536-dim output (we do NOT project to 768; per
  REC-03 the parallel-cache approach keeps SBERT cache validity intact
  on backend swap).

* The function returns L2-normalized float32 ``numpy.ndarray`` of shape
  ``(N, D)``. Cosine similarity downstream stays unchanged.

* :func:`backend_identity` returns the metadata triple
  ``(backend, model, model_revision)`` that callers should persist into
  ``build_manifest.json`` (per backlog C3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

__all__ = [
    "BackendName",
    "EmbeddingResult",
    "encode",
    "backend_identity",
    "DEFAULT_SBERT_MODEL",
    "DEFAULT_OPENAI_MODEL",
]

BackendName = Literal["sbert", "openai", "openai_batch"]

DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding tensor plus the identity triple that produced it."""

    vectors: "object"  # numpy.ndarray, untyped to keep numpy import lazy.
    backend: str
    model: str
    model_revision: str
    dim: int


def _import_numpy():
    import numpy as np  # local import so this module is cheap to import
    return np


def _encode_sbert(texts: Sequence[str], model: str, batch_size: int, fp16: bool):
    """SBERT path: lazy-imports torch + sentence_transformers."""
    np = _import_numpy()
    from sentence_transformers import SentenceTransformer  # type: ignore

    enc = SentenceTransformer(model)
    # Best-effort FP16 on CUDA; falls back transparently otherwise.
    try:
        import torch  # type: ignore

        if fp16 and torch.cuda.is_available():
            enc = enc.half()
    except Exception:
        pass

    arr = enc.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return arr.astype(np.float32, copy=False)


def _encode_openai(texts: Sequence[str], model: str, mode: str):
    """OpenAI path: dispatches to src.common.openai_embeddings."""
    from src.common import openai_embeddings as _oe  # lazy

    if mode == "batch":
        return _oe.encode_batch(list(texts), model=model)
    return _oe.encode_sync(list(texts), model=model)


def encode(
    texts: Iterable[str],
    *,
    backend: BackendName = "sbert",
    model: str | None = None,
    batch_size: int = 256,
    fp16: bool = True,
) -> EmbeddingResult:
    """Encode ``texts`` with the requested backend and return an
    :class:`EmbeddingResult`.

    ``backend="openai"`` runs the synchronous OpenAI path (fast for small
    inputs); ``backend="openai_batch"`` runs the 50%-discount Batch API
    path (recommended for production-scale runs). ``backend="sbert"`` is
    the local SBERT GPU/CPU path.

    All backends emit L2-normalized float32 vectors so cosine similarity
    is the dot product downstream.
    """
    text_list = list(texts)
    if backend == "sbert":
        m = model or DEFAULT_SBERT_MODEL
        vectors = _encode_sbert(text_list, m, batch_size=batch_size, fp16=fp16)
        rev = _sbert_revision(m)
        return EmbeddingResult(
            vectors=vectors, backend="sbert", model=m, model_revision=rev,
            dim=int(vectors.shape[1]) if vectors.size else 0,
        )

    if backend in ("openai", "openai_batch"):
        m = model or DEFAULT_OPENAI_MODEL
        mode = "batch" if backend == "openai_batch" else "sync"
        vectors = _encode_openai(text_list, m, mode)
        return EmbeddingResult(
            vectors=vectors, backend=backend, model=m,
            # OpenAI does not expose a model commit-SHA; we record the
            # date-stamped model name as the "revision" so manifests are
            # at least monotone with respect to known OpenAI rollouts.
            model_revision=m,
            dim=int(vectors.shape[1]) if vectors.size else 0,
        )

    raise ValueError(f"unknown backend {backend!r}; expected sbert / openai / openai_batch")


def _sbert_revision(model: str) -> str:
    """Best-effort: return the HuggingFace commit SHA for the SBERT
    model. Falls back to the bare model string if the HF cache is not
    populated. Pinning the revision into manifests is backlog item B17
    (proposed); this helper is the substrate.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        info = HfApi().model_info(model)
        sha = getattr(info, "sha", None)
        if sha:
            return str(sha)
    except Exception:
        pass
    return model


def backend_identity(
    backend: BackendName,
    model: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(backend, model, model_revision)`` without doing any
    encoding. Use this from ``step_publish`` (C3) when writing the
    manifest before the actual encode runs.
    """
    if backend == "sbert":
        m = model or DEFAULT_SBERT_MODEL
        return ("sbert", m, _sbert_revision(m))
    if backend in ("openai", "openai_batch"):
        m = model or DEFAULT_OPENAI_MODEL
        return (str(backend), m, m)
    raise ValueError(f"unknown backend {backend!r}")
