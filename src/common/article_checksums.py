"""src.common.article_checksums: per-benchmark article-file checksums and
integrity assertions (v2.2 [G2] + [G5]).

Closes the failure mode from 2026-04-22 where
``data/earnings/earnings_articles.jsonl`` was deleted between fetcher
completion and ``step_publish``; the unifier silently loaded an empty
pool and the build shipped without any signal that anything was wrong.

The functions here record SHA-256, line count, and ``fd_id`` coverage
for each per-benchmark articles JSONL at copy time, write a sidecar
``data/{bench}/{bench}_articles.checksums.json``, and provide an
``assert_articles_present`` helper that ``step_publish`` calls before
the write step (G5 integrity check). Stand-alone CLI invocation lives
in :mod:`scripts.benchmark_article_checksums` (G2 sidecar, NEW script).

Pure library: no orchestrator edits required. The build pipeline picks
this up by importing it from a new sidecar script (no edit to the
in-flight ``build_benchmark.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

__all__ = [
    "ArticleChecksum",
    "compute_checksum",
    "write_checksum_sidecar",
    "read_checksum_sidecar",
    "assert_articles_present",
    "checksum_sidecar_path",
]


@dataclass(frozen=True)
class ArticleChecksum:
    """Provenance triple recorded per per-benchmark articles file."""

    benchmark: str
    path: str
    sha256: str
    n_lines: int
    n_unique_fd_ids: int
    bytes: int
    mtime_iso: str

    def to_json(self) -> dict:
        return asdict(self)


def _sha256_and_count(path: Path) -> tuple[str, int, int, int]:
    """Stream the file, returning ``(sha256, n_lines, n_unique_fd_ids,
    bytes)``. Counts unique ``fd_id`` references per article so a
    coverage-degradation downstream is detectable.
    """
    h = hashlib.sha256()
    n_lines = 0
    n_bytes = 0
    fd_ids: set[str] = set()
    with path.open("rb") as raw:
        for raw_line in raw:
            n_bytes += len(raw_line)
            h.update(raw_line)
            stripped = raw_line.strip()
            if not stripped:
                continue
            n_lines += 1
            try:
                obj = json.loads(stripped)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            # Multiple shapes in v2.1: linked_fd_ids list or single fd_id.
            for k in ("linked_fd_ids", "fd_ids"):
                v = obj.get(k)
                if isinstance(v, list):
                    fd_ids.update(str(x) for x in v if x)
            v = obj.get("fd_id")
            if v:
                fd_ids.add(str(v))
    return h.hexdigest(), n_lines, len(fd_ids), n_bytes


def compute_checksum(benchmark: str, path: Path) -> ArticleChecksum:
    """Stream-hash the per-benchmark articles file and return the
    metadata record. Raises FileNotFoundError if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"articles file missing: {path}")
    sha, n_lines, n_unique_fd_ids, n_bytes = _sha256_and_count(path)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return ArticleChecksum(
        benchmark=benchmark,
        path=str(path),
        sha256=sha,
        n_lines=n_lines,
        n_unique_fd_ids=n_unique_fd_ids,
        bytes=n_bytes,
        mtime_iso=mtime,
    )


def checksum_sidecar_path(articles_path: Path) -> Path:
    """Convention: ``foo_articles.jsonl`` -> ``foo_articles.checksums.json``."""
    return articles_path.with_suffix(".checksums.json")


def write_checksum_sidecar(checksum: ArticleChecksum, sidecar_path: Path) -> Path:
    """Atomically write the sidecar JSON next to the articles file."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(json.dumps(checksum.to_json(), indent=2, sort_keys=True))
    tmp.replace(sidecar_path)
    return sidecar_path


def read_checksum_sidecar(sidecar_path: Path) -> ArticleChecksum:
    """Inverse of :func:`write_checksum_sidecar`."""
    obj = json.loads(sidecar_path.read_text())
    return ArticleChecksum(**obj)


def assert_articles_present(
    article_paths: Iterable[tuple[str, Path]],
    *,
    min_lines: int = 1,
) -> list[ArticleChecksum]:
    """Pre-publish gate: every (benchmark, path) tuple must reference a
    file that exists, is non-empty, and has at least ``min_lines`` JSON
    lines. Returns the list of computed checksums on success; raises
    :class:`RuntimeError` listing every offender on failure.

    Call site (G2 / G5): the new ``scripts/preflight_publish.py``
    invokes this before the in-flight ``step_publish`` runs; the
    orchestrator does not need to be edited.
    """
    failures: list[str] = []
    out: list[ArticleChecksum] = []
    for bench, path in article_paths:
        if not path.exists():
            failures.append(f"{bench}: missing file {path}")
            continue
        try:
            cks = compute_checksum(bench, path)
        except Exception as exc:
            failures.append(f"{bench}: failed to checksum {path} ({exc})")
            continue
        if cks.n_lines < min_lines:
            failures.append(
                f"{bench}: {path} has {cks.n_lines} lines (< {min_lines})"
            )
            continue
        out.append(cks)
    if failures:
        msg = "; ".join(failures)
        raise RuntimeError(f"assert_articles_present: {msg}")
    return out
