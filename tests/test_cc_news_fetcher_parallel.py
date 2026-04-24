"""Parity, retry, and cancel tests for the parallel CC-News fetcher.

These tests stub out the HTTPS download path (``_download_shard``) so
no real request hits commoncrawl.org. Each shard's "download" is
satisfied by writing a pre-built in-memory WARC to the worker's raw
staging path.

Tests cover H8-1 (parallel parity, .done resume, max-shards bound,
KeyboardInterrupt cleanup) and H8-2 (3-attempt retry on HTTP 503).
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import fetch_cc_news_archive as fcn  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mini_warc() -> bytes:
    """Two records: one whitelisted reuters URL, one off-list spam URL.

    The body of the kept record is long enough to clear trafilatura's
    200-char floor.
    """
    pytest.importorskip("warcio")
    from warcio.warcwriter import WARCWriter
    from warcio.statusandheaders import StatusAndHeaders

    buf = io.BytesIO()
    writer = WARCWriter(buf, gzip=True)

    body1 = (b"<html><head><title>Hello From Reuters</title></head><body><p>"
             + b"This is a long enough article body for trafilatura. " * 30
             + b"</p></body></html>")
    h1 = StatusAndHeaders("200 OK",
                          [("Content-Type", "text/html"),
                           ("Date", "Tue, 23 Apr 2026 12:00:00 GMT")],
                          protocol="HTTP/1.0")
    writer.write_record(writer.create_warc_record(
        "https://www.reuters.com/world/foo",
        "response",
        payload=io.BytesIO(body1),
        http_headers=h1,
    ))

    body2 = b"<html><body>spam, off-list domain</body></html>"
    h2 = StatusAndHeaders("200 OK",
                          [("Content-Type", "text/html"),
                           ("Date", "Tue, 23 Apr 2026 12:30:00 GMT")],
                          protocol="HTTP/1.0")
    writer.write_record(writer.create_warc_record(
        "https://random-spam.example/x",
        "response",
        payload=io.BytesIO(body2),
        http_headers=h2,
    ))
    return buf.getvalue()


_FAKE_MANIFEST = [f"crawl-data/CC-NEWS/2026/04/CC-NEWS-mock-{i:05d}.warc.gz"
                  for i in range(10)]


def _stub_list_shards(monkeypatch, manifest: Iterable[str] = _FAKE_MANIFEST):
    monkeypatch.setattr(fcn, "list_shards_for_month",
                        lambda y, m, http_get=None: list(manifest))


def _install_download_stub(monkeypatch, warc_bytes: bytes,
                           fail_count: int = 0,
                           fail_status: int = 503):
    """Patch _download_shard so it writes ``warc_bytes`` to raw_path.

    If ``fail_count`` > 0, the first N calls raise ``_TransientHTTPError``
    with ``fail_status`` before a success on attempt N+1.
    """
    state = {"calls": 0}

    def fake_dl(url, raw_path, max_bytes, session=None):
        state["calls"] += 1
        if state["calls"] <= fail_count:
            # Use the same exception type the real code uses for 429/503
            # so the retry path picks the longer (5-15s) sleep window.
            raise fcn._TransientHTTPError(fail_status, "mock failure")
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "wb") as fh:
            fh.write(warc_bytes)
        return len(warc_bytes)

    monkeypatch.setattr(fcn, "_download_shard", fake_dl)
    return state


def _patch_no_throttle(monkeypatch):
    """Make retry sleeps near-instant so tests stay fast."""
    monkeypatch.setattr(fcn.time, "sleep", lambda s: None)
    # Random sleep range collapses to ~0.
    monkeypatch.setattr(fcn.random, "uniform", lambda a, b: 0.0)


# ---------------------------------------------------------------------------
# Test 1: workers=1 vs workers=4 produce identical outputs.
# ---------------------------------------------------------------------------

def _run_main(tmp_path, workers, manifest_size=4, max_shards=0):
    out = tmp_path / f"out_w{workers}"
    args = [
        "--start", "2026-04",
        "--end", "2026-04",
        "--out", str(out),
        "--workers", str(workers),
        "--date-lo", "2026-04-01",
        "--date-hi", "2026-04-30",
    ]
    if max_shards:
        args += ["--max-shards", str(max_shards)]
    rc = fcn.main(args)
    return rc, out


def test_workers_1_equals_workers_4_output(tmp_path, monkeypatch):
    """Parity: serial vs parallel dispatch yields identical output files.

    monkeypatch does not cross the ProcessPoolExecutor boundary (worker
    children re-import the module fresh and lose the stub). So this test
    exercises parallelism via a thread pool to validate the contract
    that the production process pool relies on: the per-shard worker
    is deterministic by shard idx, so concurrent dispatch produces
    byte-identical .jsonl.zst files.
    """
    import concurrent.futures as cf
    warc = _build_mini_warc()
    _patch_no_throttle(monkeypatch)
    _install_download_stub(monkeypatch, warc)

    out1 = tmp_path / "out_serial"
    out4 = tmp_path / "out_threaded"
    month1 = out1 / "2026-04"; month1.mkdir(parents=True)
    month4 = out4 / "2026-04"; month4.mkdir(parents=True)

    wl_list = sorted(fcn.load_whitelist())
    common = dict(date_lo_iso="2026-04-01T00:00:00+00:00",
                  date_hi_iso="2026-04-30T00:00:00+00:00",
                  whitelist_list=wl_list,
                  max_bytes=fcn.DEFAULT_MAX_SHARD_BYTES)

    for idx, shard in enumerate(_FAKE_MANIFEST[:4]):
        res = fcn._process_shard(shard, idx, str(month1), **common)
        assert res["status"] == "ok"

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(fcn._process_shard, shard, idx, str(month4), **common)
                for idx, shard in enumerate(_FAKE_MANIFEST[:4])]
        for f in cf.as_completed(futs):
            assert f.result()["status"] == "ok"

    files1 = sorted(p.name for p in month1.glob("shard_*.jsonl.zst"))
    files4 = sorted(p.name for p in month4.glob("shard_*.jsonl.zst"))
    assert files1 == files4
    assert len(files1) == 4

    import zstandard as zstd
    for name in files1:
        b1 = (month1 / name).read_bytes()
        b4 = (month4 / name).read_bytes()
        d1 = zstd.ZstdDecompressor().decompress(b1, max_output_size=10 * 1024 * 1024)
        d4 = zstd.ZstdDecompressor().decompress(b4, max_output_size=10 * 1024 * 1024)
        assert d1 == d4, f"contents differ for {name}"
        rows = [json.loads(line) for line in d1.decode("utf-8").splitlines() if line]
        assert all(r["host"].endswith("reuters.com") for r in rows)


# ---------------------------------------------------------------------------
# Test 2: pre-existing .done skips the shard.
# ---------------------------------------------------------------------------

def test_done_marker_skips_shard(tmp_path, monkeypatch):
    warc = _build_mini_warc()
    _stub_list_shards(monkeypatch, _FAKE_MANIFEST[:3])
    state = _install_download_stub(monkeypatch, warc)
    _patch_no_throttle(monkeypatch)

    out = tmp_path / "out"
    month = out / "2026-04"
    month.mkdir(parents=True)
    # Pre-create .done for shard_0001 (so only 0000 + 0002 should run).
    placeholder = month / "shard_0001.jsonl.zst"
    placeholder.write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00\x00")  # zstd magic stub
    (month / "shard_0001.jsonl.zst.done").write_text(json.dumps({"shard_name": "x"}))

    rc = fcn.main(["--start", "2026-04", "--end", "2026-04",
                   "--out", str(out), "--workers", "1",
                   "--date-lo", "2026-04-01", "--date-hi", "2026-04-30"])
    assert rc == 0
    # 3 shards in manifest, 1 pre-done -> 2 actual downloads.
    assert state["calls"] == 2
    assert (month / "shard_0001.jsonl.zst").read_bytes().startswith(b"\x28\xb5\x2f\xfd")


# ---------------------------------------------------------------------------
# Test 3: 3-attempt retry on 503; permanent failure on 4 strikes.
# ---------------------------------------------------------------------------

def test_failing_shard_retries_three_times(tmp_path, monkeypatch):
    warc = _build_mini_warc()
    _stub_list_shards(monkeypatch, [_FAKE_MANIFEST[0]])
    # Fail twice, then succeed on attempt 3.
    state = _install_download_stub(monkeypatch, warc, fail_count=2, fail_status=503)
    _patch_no_throttle(monkeypatch)

    out = tmp_path / "out"
    rc = fcn.main(["--start", "2026-04", "--end", "2026-04",
                   "--out", str(out), "--workers", "1",
                   "--date-lo", "2026-04-01", "--date-hi", "2026-04-30"])
    assert rc == 0
    assert state["calls"] == 3
    done = out / "2026-04" / "shard_0000.jsonl.zst.done"
    assert done.exists()
    meta = json.loads(done.read_text())
    assert meta["download_attempts"] == 3


def test_failing_shard_permanent_after_four_strikes(tmp_path, monkeypatch):
    warc = _build_mini_warc()
    _stub_list_shards(monkeypatch, [_FAKE_MANIFEST[0]])
    # 4 failures > RETRY_ATTEMPTS=3 -> permanent.
    state = _install_download_stub(monkeypatch, warc, fail_count=4, fail_status=503)
    _patch_no_throttle(monkeypatch)

    out = tmp_path / "out"
    rc = fcn.main(["--start", "2026-04", "--end", "2026-04",
                   "--out", str(out), "--workers", "1",
                   "--date-lo", "2026-04-01", "--date-hi", "2026-04-30"])
    assert rc != 0  # n_fatal > 0
    assert state["calls"] == fcn.RETRY_ATTEMPTS
    done = out / "2026-04" / "shard_0000.jsonl.zst.done"
    assert not done.exists(), "no .done should be written for failed shards"
    out_file = out / "2026-04" / "shard_0000.jsonl.zst"
    assert not out_file.exists(), "no .jsonl.zst should remain after retry-exhausted failure"


# ---------------------------------------------------------------------------
# Test 4: --max-shards bounds dispatch.
# ---------------------------------------------------------------------------

def test_max_shards_bounds_dispatch(tmp_path, monkeypatch):
    warc = _build_mini_warc()
    _stub_list_shards(monkeypatch, _FAKE_MANIFEST)  # 10 entries
    state = _install_download_stub(monkeypatch, warc)
    _patch_no_throttle(monkeypatch)

    out = tmp_path / "out"
    rc = fcn.main(["--start", "2026-04", "--end", "2026-04",
                   "--out", str(out), "--workers", "1",
                   "--max-shards", "2",
                   "--date-lo", "2026-04-01", "--date-hi", "2026-04-30"])
    assert rc == 0
    assert state["calls"] == 2
    files = sorted((out / "2026-04").glob("shard_*.jsonl.zst"))
    assert len(files) == 2


# ---------------------------------------------------------------------------
# Test 5: simulated worker crash leaves no .jsonl.zst without .done.
# Skipped on Windows because real Ctrl-C plumbing through ProcessPool is
# brittle; instead we simulate by raising in the worker.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Real SIGINT plumbing through ProcessPoolExecutor is brittle on Windows; "
           "the equivalent invariant (no orphan .jsonl.zst) is exercised by the "
           "permanent-failure test above which uses the same atomic-rename guard.",
)
def test_keyboard_interrupt_no_partial_jsonl(tmp_path, monkeypatch):
    warc = _build_mini_warc()
    _stub_list_shards(monkeypatch, _FAKE_MANIFEST[:3])

    def crash_dl(url, raw_path, max_bytes, session=None):
        raise KeyboardInterrupt("simulated SIGINT mid-fetch")

    monkeypatch.setattr(fcn, "_download_shard", crash_dl)
    _patch_no_throttle(monkeypatch)

    out = tmp_path / "out"
    try:
        fcn.main(["--start", "2026-04", "--end", "2026-04",
                  "--out", str(out), "--workers", "1",
                  "--date-lo", "2026-04-01", "--date-hi", "2026-04-30"])
    except KeyboardInterrupt:
        pass

    # Invariant: every .jsonl.zst file must have a matching .done sibling.
    month = out / "2026-04"
    if month.exists():
        for p in month.glob("shard_*.jsonl.zst"):
            assert p.with_suffix(p.suffix + ".done").exists(), \
                f"orphan .jsonl.zst without .done: {p}"
