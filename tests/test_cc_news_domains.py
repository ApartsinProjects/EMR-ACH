"""Smoke tests for the CC-News editorial domain whitelist."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.cc_news_domains import host_in_whitelist, load_whitelist


def test_whitelist_size_reasonable():
    wl = load_whitelist()
    assert 30 <= len(wl) <= 200, f"unexpected whitelist size {len(wl)}"


def test_whitelist_has_core_outlets():
    wl = load_whitelist()
    for expected in ("nytimes.com", "bbc.com", "reuters.com", "bloomberg.com",
                     "theguardian.com", "wsj.com", "ft.com"):
        assert expected in wl, f"missing core outlet {expected}"


def test_host_exact_match():
    wl = load_whitelist()
    assert host_in_whitelist("reuters.com", wl)
    assert host_in_whitelist("www.reuters.com", wl)


def test_host_subdomain_match():
    wl = load_whitelist()
    assert host_in_whitelist("edition.cnn.com", {"cnn.com"})
    assert host_in_whitelist("feeds.reuters.com", wl)


def test_host_reject_unknown():
    wl = load_whitelist()
    assert not host_in_whitelist("randomspam.example", wl)
    assert not host_in_whitelist("", wl)


def test_whitelist_extra_file(tmp_path):
    extras = tmp_path / "extras.txt"
    extras.write_text("# comment\nexample.org\n   another.test   \n\n", encoding="utf-8")
    wl = load_whitelist(extras)
    assert "example.org" in wl
    assert "another.test" in wl


def test_whitelist_normalizes_www():
    # extras should have www stripped
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("www.extra-outlet.com\n")
        path = Path(fh.name)
    try:
        wl = load_whitelist(path)
        assert "extra-outlet.com" in wl
    finally:
        path.unlink()
