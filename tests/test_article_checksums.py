"""Tests for src.common.article_checksums (v2.2 [G2] + [G5])."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.article_checksums import (
    assert_articles_present,
    checksum_sidecar_path,
    compute_checksum,
    read_checksum_sidecar,
    write_checksum_sidecar,
)


def _write_articles(path: Path, n: int, fd_ids_per: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            obj = {
                "article_id": f"art_{i}",
                "url": f"https://example.com/{i}",
                "linked_fd_ids": [f"fd_{i}_{k}" for k in range(fd_ids_per)],
            }
            f.write(json.dumps(obj) + "\n")


def test_compute_checksum_happy(tmp_path):
    p = tmp_path / "earnings_articles.jsonl"
    _write_articles(p, n=5, fd_ids_per=2)
    cks = compute_checksum("earnings", p)
    assert cks.benchmark == "earnings"
    assert cks.n_lines == 5
    assert cks.n_unique_fd_ids == 10
    assert len(cks.sha256) == 64
    assert cks.bytes > 0


def test_compute_checksum_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_checksum("earnings", tmp_path / "nope.jsonl")


def test_sidecar_round_trip(tmp_path):
    p = tmp_path / "fb_articles.jsonl"
    _write_articles(p, n=3)
    cks = compute_checksum("forecastbench", p)
    sidecar = checksum_sidecar_path(p)
    write_checksum_sidecar(cks, sidecar)
    assert sidecar.exists()
    loaded = read_checksum_sidecar(sidecar)
    assert loaded == cks


def test_assert_articles_present_passes_when_all_files_nonempty(tmp_path):
    paths = []
    for bench in ("forecastbench", "gdelt_cameo", "earnings"):
        p = tmp_path / bench / f"{bench}_articles.jsonl"
        _write_articles(p, n=4)
        paths.append((bench, p))
    out = assert_articles_present(paths)
    assert len(out) == 3
    assert {c.benchmark for c in out} == {
        "forecastbench",
        "gdelt_cameo",
        "earnings",
    }


def test_assert_articles_present_fails_on_missing_file(tmp_path):
    p1 = tmp_path / "fb_articles.jsonl"
    _write_articles(p1, n=4)
    missing = tmp_path / "earn_articles.jsonl"
    with pytest.raises(RuntimeError) as ei:
        assert_articles_present([("fb", p1), ("earnings", missing)])
    assert "earnings" in str(ei.value)
    assert "missing" in str(ei.value)


def test_assert_articles_present_fails_on_empty_file(tmp_path):
    p = tmp_path / "earnings_articles.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    with pytest.raises(RuntimeError) as ei:
        assert_articles_present([("earnings", p)])
    assert "earnings" in str(ei.value)
