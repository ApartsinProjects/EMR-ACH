"""Smoke tests for src.common.news_fetcher (v2.2 [B2])."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.news_fetcher import (
    DEFAULT_HEADERS,
    FetchedArticle,
    NewsFetcher,
    art_id_for,
    domain_of,
)


def test_art_id_deterministic_and_prefixed():
    a = art_id_for("forecastbench", "https://example.com/x", "2026-01-01")
    b = art_id_for("forecastbench", "https://example.com/x", "2026-01-01")
    assert a == b
    assert a.startswith("fbn_")
    assert art_id_for("gdelt_cameo", "u", "d").startswith("gdc_")
    assert art_id_for("earnings", "u", "d").startswith("earn_")
    assert art_id_for("unknown_bench", "u", "d").startswith("art_")


def test_domain_of_variants():
    assert domain_of("https://WWW.Example.COM/foo") == "example.com"
    assert domain_of("") == ""
    assert domain_of("not-a-url") == ""


class _StubFetcher(NewsFetcher):
    benchmark = "forecastbench"

    def __init__(self, out_path, queue, **kw):
        super().__init__(out_path, **kw)
        self._queue = queue

    def build_queries(self, fd):
        return ["q"]

    def eligible_sources(self):
        return ["google-news"]

    def fetch_for_query(self, query, fd):
        for item in self._queue:
            yield item


def test_process_fd_deduplicates(tmp_path):
    out = tmp_path / "fbn.jsonl"
    queue = [
        {"url": "https://nytimes.com/1", "title": "T1", "publish_date": "2026-03-01"},
        {"url": "https://nytimes.com/1", "title": "T1 dup", "publish_date": "2026-03-01"},
        {"url": "https://reuters.com/2", "title": "T2", "publish_date": "2026-03-02"},
    ]
    f = _StubFetcher(out, queue)
    arts = f.process_fd({"id": "fd_1"})
    assert len(arts) == 2
    assert {a.url for a in arts} == {
        "https://nytimes.com/1", "https://reuters.com/2",
    }


def test_process_fd_applies_spam_filter(tmp_path):
    queue = [
        {"url": "https://spam.com/1", "title": "bad"},
        {"url": "https://nytimes.com/2", "title": "good"},
    ]
    out = tmp_path / "fbn.jsonl"
    f = _StubFetcher(out, queue, spam_blocklist={"spam.com"})
    arts = f.process_fd({"id": "fd_1"})
    assert [a.url for a in arts] == ["https://nytimes.com/2"]


def test_write_all_atomic(tmp_path):
    out = tmp_path / "fbn.jsonl"
    f = _StubFetcher(out, [])
    arts = [
        FetchedArticle(
            article_id="fbn_a", url="https://u/1", title="T", text="b",
            publish_date="2026-03-01", source_domain="u", provenance="google-news",
            linked_fd_ids=["fd_1"],
        ),
    ]
    n = f.write_all(arts)
    assert n == 1
    # No .tmp leftover.
    assert [p.name for p in tmp_path.iterdir()] == ["fbn.jsonl"]
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert loaded[0]["article_id"] == "fbn_a"


def test_append_then_dedup_across_session(tmp_path):
    out = tmp_path / "fbn.jsonl"
    # Seed the file with one existing article.
    out.write_text(
        json.dumps({"article_id": art_id_for("forecastbench",
                                             "https://u/1", "2026-03-01")})
        + "\n"
    )
    queue = [
        {"url": "https://u/1", "title": "dup", "publish_date": "2026-03-01"},
        {"url": "https://u/2", "title": "new", "publish_date": "2026-03-02"},
    ]
    f = _StubFetcher(out, queue)
    arts = f.process_fd({"id": "fd_1"})
    assert len(arts) == 1
    assert arts[0].url == "https://u/2"
