"""Round-trip tests for the GDELT editorial filter (v2.2 [B8] + [C6])."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.gdelt_aggregator_domains import (
    BLOCKLIST_REVISION,
    domain_of,
    filter_articles,
    is_aggregator_domain,
)


def test_blocklist_revision_set():
    assert BLOCKLIST_REVISION
    assert isinstance(BLOCKLIST_REVISION, str)


def test_known_aggregator_blocked():
    assert is_aggregator_domain("https://news.google.com/articles/abc")
    assert is_aggregator_domain("archive.org")
    assert is_aggregator_domain("https://web.archive.org/web/foo")


def test_known_state_syndication_blocked():
    assert is_aggregator_domain("https://news.fjsen.com/2026/x")
    assert is_aggregator_domain("world.people.com.cn")


def test_legitimate_editorial_outlet_allowed():
    # Reuters, NYT, BBC, Guardian, AP wire are not aggregator surfaces.
    for u in (
        "https://www.reuters.com/world/asia-pacific/foo",
        "https://www.nytimes.com/2026/04/22/world/asia/foo.html",
        "https://www.bbc.com/news/world-asia-1234",
        "https://www.theguardian.com/world/2026/apr/22/foo",
    ):
        assert not is_aggregator_domain(u), u


def test_subdomain_suffix_match_blocks_correctly():
    # Subdomain of a blocked registered domain is blocked.
    assert is_aggregator_domain("data.gdeltproject.org")
    # But a substring-only match must NOT block (no false positives).
    assert not is_aggregator_domain("gdeltproject-impostor.com")


def test_domain_of_strips_www_and_lowercases():
    assert domain_of("https://WWW.Example.COM/foo") == "example.com"
    assert domain_of("https://www.reuters.com/x") == "reuters.com"
    assert domain_of("") == ""
    assert domain_of("not-a-url") == ""


def test_filter_articles_round_trip():
    arts = [
        {"url": "https://www.reuters.com/x", "domain": "reuters.com"},
        {"url": "https://news.google.com/foo", "domain": "news.google.com"},
        {"url": "https://www.nytimes.com/y", "domain": "nytimes.com"},
        {"url": "https://archive.org/q", "domain": "archive.org"},
        # Ambiguous case: missing domain field falls through to URL parse.
        {"url": "https://prnewswire.com/release/123"},
    ]
    out = filter_articles(arts)
    out_urls = {a["url"] for a in out}
    assert "https://www.reuters.com/x" in out_urls
    assert "https://www.nytimes.com/y" in out_urls
    assert "https://news.google.com/foo" not in out_urls
    assert "https://archive.org/q" not in out_urls
    assert "https://prnewswire.com/release/123" not in out_urls


def test_empty_input_safe():
    assert is_aggregator_domain("") is False
    assert filter_articles([]) == []


def test_malformed_url_does_not_crash():
    arts = [{"url": "://broken"}, {"url": None}, {"url": ""}]
    # None / empty / malformed must be passed through (fail-open at the
    # filter; spam check is a separate concern).
    out = filter_articles(arts)
    assert len(out) == 3
