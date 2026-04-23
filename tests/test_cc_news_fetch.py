"""Smoke tests for scripts/fetch_cc_news_archive.py.

Mocks the HTTP layer so no network call is made; uses a tiny in-memory
WARC built with warcio to verify the streaming filter logic.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import fetch_cc_news_archive as fcn  # type: ignore


def test_month_range_inclusive():
    assert fcn._month_range("2026-01", "2026-03") == [(2026, 1), (2026, 2), (2026, 3)]
    assert fcn._month_range("2025-12", "2026-02") == [(2025, 12), (2026, 1), (2026, 2)]
    assert fcn._month_range("2026-04", "2026-04") == [(2026, 4)]


def test_extract_host():
    assert fcn._extract_host("https://www.reuters.com/world/foo") == "www.reuters.com"
    assert fcn._extract_host("not-a-url") == ""


def test_parse_date_header_handles_rfc1123():
    dt = fcn._parse_date_header("Tue, 23 Apr 2026 12:34:56 GMT")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 4 and dt.day == 23
    assert dt.tzinfo is not None


def test_parse_date_header_returns_none_on_garbage():
    assert fcn._parse_date_header(None) is None
    assert fcn._parse_date_header("not a date") is None


def _make_mini_warc() -> bytes:
    """Construct a 2-record gzipped WARC in memory using warcio."""
    pytest.importorskip("warcio")
    from warcio.warcwriter import WARCWriter  # type: ignore
    from warcio.statusandheaders import StatusAndHeaders  # type: ignore

    buf = io.BytesIO()
    writer = WARCWriter(buf, gzip=True)

    body1 = (b"<html><head><title>Hello World From Reuters</title></head>"
             b"<body><p>" + b"This is a long enough article body to clear trafilatura's "
             b"length floor. " * 30 + b"</p></body></html>")
    headers1 = StatusAndHeaders("200 OK",
                                [("Content-Type", "text/html"),
                                 ("Date", "Tue, 23 Apr 2026 12:00:00 GMT")],
                                protocol="HTTP/1.0")
    rec1 = writer.create_warc_record("https://www.reuters.com/world/foo", "response",
                                     payload=io.BytesIO(body1),
                                     http_headers=headers1)
    writer.write_record(rec1)

    body2 = b"<html><body>spam from a non-whitelisted domain</body></html>"
    headers2 = StatusAndHeaders("200 OK",
                                [("Content-Type", "text/html"),
                                 ("Date", "Tue, 23 Apr 2026 12:30:00 GMT")],
                                protocol="HTTP/1.0")
    rec2 = writer.create_warc_record("https://random-spam.example/x", "response",
                                     payload=io.BytesIO(body2),
                                     http_headers=headers2)
    writer.write_record(rec2)

    return buf.getvalue()


def test_iter_warc_records_yields_responses_only():
    data = _make_mini_warc()
    records = list(fcn.iter_warc_records(io.BytesIO(data)))
    assert len(records) == 2
    assert records[0]["url"].startswith("https://www.reuters.com")
    assert "html" in records[0]["content_type"].lower()
    assert records[0]["http_date"] is not None


def test_iter_warc_records_filters_by_host_and_date():
    """Filter logic outside of process_shard's network path."""
    data = _make_mini_warc()
    wl = frozenset({"reuters.com"})
    lo = datetime(2026, 4, 1, tzinfo=timezone.utc)
    hi = datetime(2026, 4, 30, tzinfo=timezone.utc)

    kept = []
    for record in fcn.iter_warc_records(io.BytesIO(data)):
        host = fcn._extract_host(record["url"])
        if not fcn.host_in_whitelist(host, wl):
            continue
        dt = fcn._parse_date_header(record["http_date"])
        if dt is None or dt < lo or dt > hi:
            continue
        kept.append(record)
    assert len(kept) == 1
    assert "reuters.com" in kept[0]["url"]


def test_list_shards_for_month_uses_manifest(monkeypatch):
    """Verify the manifest URL pattern and gzip parsing."""
    payload = gzip.compress(b"crawl-data/CC-NEWS/2026/01/CC-NEWS-20260101000000-00000.warc.gz\n"
                            b"crawl-data/CC-NEWS/2026/01/CC-NEWS-20260101001500-00001.warc.gz\n")

    class FakeResp:
        content = payload

        def raise_for_status(self):
            return None

    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return FakeResp()

    out = fcn.list_shards_for_month(2026, 1, http_get=fake_get)
    assert "warc.paths.gz" in captured["url"]
    assert "/CC-NEWS/2026/01/" in captured["url"]
    assert len(out) == 2
    assert out[0].endswith(".warc.gz")
