"""src.common.query_embedding_cache: per-FD query-embedding memoization
(v2.2 [A9]).

Most builds re-query the same FDs (full-pool rebuild on a new cutoff;
leakage-probe reruns; baseline ablations). Encoding the same
``question + background`` string repeatedly is wasted SBERT / OpenAI
cost. This module is a content-addressed cache keyed by
``MD5(question + background + model + backend)`` so the cache is
correctly invalidated when the embedding backend or model changes.

Storage: one ``.npy`` per (key, backend, model) tuple under
``data/cache/query_embeddings/{backend}/{model}/{key}.npy``. The keys
are deterministic so the cache is portable across machines.

No SBERT / OpenAI import at module load; the encoder is injected.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from src.common.paths import DATA_DIR

__all__ = [
    "query_cache_key",
    "QueryEmbeddingCache",
    "default_cache_root",
]


def default_cache_root() -> Path:
    return DATA_DIR / "cache" / "query_embeddings"


def query_cache_key(
    question: str, background: str, *, backend: str, model: str,
) -> str:
    """Stable MD5 of the query text plus backend identity."""
    payload = "\n".join([backend, model, question or "", background or ""])
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


class QueryEmbeddingCache:
    """Content-addressed .npy cache for per-FD query embeddings."""

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        root: Path | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.root = (root or default_cache_root()) / backend / self._safe(model)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}.npy"

    def get(self, question: str, background: str):
        """Return the cached embedding or ``None``."""
        import numpy as np  # local import

        key = query_cache_key(
            question, background, backend=self.backend, model=self.model
        )
        p = self._path_for(key)
        if not p.exists():
            return None
        try:
            return np.load(p, allow_pickle=False)
        except Exception:
            return None

    def put(self, question: str, background: str, vector) -> Path:
        import numpy as np  # local import

        key = query_cache_key(
            question, background, backend=self.backend, model=self.model
        )
        p = self._path_for(key)
        # np.save appends ".npy" unless the filename already ends with it,
        # which breaks simple .tmp suffix patterns. Use allow_pickle=False
        # and pass a file handle so the exact target path is respected.
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, vector, allow_pickle=False)
        tmp.replace(p)
        return p

    def get_or_encode(
        self,
        question: str,
        background: str,
        encoder: Callable[[str], "object"],
    ):
        """Return the cached vector if present; otherwise call
        ``encoder(text)`` and store the result.
        """
        hit = self.get(question, background)
        if hit is not None:
            return hit
        text = "\n".join([question or "", background or ""]).strip()
        vector = encoder(text)
        self.put(question, background, vector)
        return vector
