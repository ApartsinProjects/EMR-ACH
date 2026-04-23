"""Tests for src.common.fast_jsonl (v2.2 [E7])."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.fast_jsonl import (
    available,
    dumps,
    dumps_str,
    iter_jsonl,
    load_jsonl,
    loads,
    write_jsonl,
    write_jsonl_atomic,
)


def test_dumps_loads_round_trip():
    obj = {"a": 1, "b": "two", "c": [3, 4]}
    s = dumps(obj)
    assert isinstance(s, bytes)
    assert loads(s) == obj
    assert json.loads(dumps_str(obj)) == obj


def test_write_then_load_jsonl(tmp_path):
    rows = [{"i": i} for i in range(5)]
    p = tmp_path / "out.jsonl"
    n = write_jsonl(p, rows)
    assert n == 5
    assert load_jsonl(p) == rows


def test_iter_jsonl_skips_blanks(tmp_path):
    p = tmp_path / "spaces.jsonl"
    p.write_bytes(b'{"x":1}\n\n{"x":2}\n')
    out = list(iter_jsonl(p))
    assert out == [{"x": 1}, {"x": 2}]


def test_write_jsonl_atomic_no_tmp_left(tmp_path):
    p = tmp_path / "atomic.jsonl"
    write_jsonl_atomic(p, [{"k": 1}])
    assert [c.name for c in tmp_path.iterdir()] == ["atomic.jsonl"]
    assert load_jsonl(p) == [{"k": 1}]


def test_available_returns_bool():
    assert isinstance(available(), bool)
