"""src.common.fast_jsonl: orjson-accelerated JSONL helpers (v2.2 [E7]).

Promoted from the script-local ``scripts/_fast_jsonl.py`` so callers
no longer need to ``_sys.path.insert`` to reach it. The original
script-local module remains in place as a backwards-compat shim until
all callers migrate; both expose the identical API:

    loads(s) -> Any
    dumps(obj) -> bytes
    dumps_str(obj) -> str
    iter_jsonl(path) -> Iterator[dict]
    load_jsonl(path) -> list[dict]
    write_jsonl(path, rows) -> int
    available() -> bool

orjson is ~3x faster on both encode and decode for typical JSONL
records. The fallback path (``json``) keeps the helper functional on
machines without orjson.

Adds a small atomic-write convenience (:func:`write_jsonl_atomic`)
that the v2.2 mutators (B15) can adopt incrementally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import orjson as _oj  # type: ignore
    _HAVE_ORJSON = True
except ImportError:
    import json as _oj  # type: ignore
    _HAVE_ORJSON = False


__all__ = [
    "loads",
    "dumps",
    "dumps_str",
    "iter_jsonl",
    "load_jsonl",
    "write_jsonl",
    "write_jsonl_atomic",
    "available",
]


def loads(s: bytes | str) -> Any:
    if _HAVE_ORJSON:
        if isinstance(s, str):
            s = s.encode("utf-8")
        return _oj.loads(s)
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return _oj.loads(s)


def dumps(obj: Any) -> bytes:
    """Returns bytes; write directly to a binary-mode file."""
    if _HAVE_ORJSON:
        return _oj.dumps(obj)
    return _oj.dumps(obj, ensure_ascii=False).encode("utf-8")


def dumps_str(obj: Any) -> str:
    """Returns str; for human-readable logging."""
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
    """Non-atomic write (truncate-and-write). Use
    :func:`write_jsonl_atomic` for the safer write-then-rename pattern.
    """
    n = 0
    with open(path, "wb") as f:
        for r in rows:
            f.write(dumps(r))
            f.write(b"\n")
            n += 1
    return n


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> int:
    """Atomic write (B15 helper): write to a tmp sibling and rename.
    Defends against Ctrl-C-mid-write leaving a half-truncated file
    that the next build silently resumes against.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    n = 0
    with open(tmp, "wb") as f:
        for r in rows:
            f.write(dumps(r))
            f.write(b"\n")
            n += 1
    os.replace(tmp, p)
    return n


def available() -> bool:
    return _HAVE_ORJSON
