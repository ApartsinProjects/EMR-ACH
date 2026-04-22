"""Fast JSONL I/O helpers — uses orjson when available, falls back to stdlib.

orjson is ~3x faster on both encode and decode for typical JSONL records.
The fallback path (`json`) keeps scripts functional on machines without orjson.

Usage:
    from _fast_jsonl import load_jsonl, write_jsonl, dumps, loads

    rows = load_jsonl("articles.jsonl")
    write_jsonl("out.jsonl", rows)

    # Or line-at-a-time:
    for row in iter_jsonl("articles.jsonl"):
        ...
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import orjson as _oj
    _HAVE_ORJSON = True
except ImportError:
    import json as _oj
    _HAVE_ORJSON = False


def loads(s: bytes | str) -> Any:
    if _HAVE_ORJSON:
        if isinstance(s, str):
            s = s.encode("utf-8")
        return _oj.loads(s)
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return _oj.loads(s)


def dumps(obj: Any) -> bytes:
    """Returns bytes — write directly to a binary-mode file."""
    if _HAVE_ORJSON:
        return _oj.dumps(obj)
    return _oj.dumps(obj, ensure_ascii=False).encode("utf-8")


def dumps_str(obj: Any) -> str:
    """Returns str — for human-readable logging."""
    return dumps(obj).decode("utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "wb") as f:
        for r in rows:
            f.write(dumps(r))
            f.write(b"\n")
            n += 1
    return n


def available() -> bool:
    return _HAVE_ORJSON
