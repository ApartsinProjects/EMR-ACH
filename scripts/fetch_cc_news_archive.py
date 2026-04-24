"""Stream CC-News WARC shards from Common Crawl and filter-while-download.

CC-News is the public news-only Common Crawl archive. Shards live at
``s3://commoncrawl/crawl-data/CC-NEWS/{YYYY}/{MM}/CC-NEWS-{timestamp}.warc.gz``
and are also mirrored on HTTPS at
``https://data.commoncrawl.org/crawl-data/CC-NEWS/{YYYY}/{MM}/``.

This fetcher:

1. Discovers shard URLs for the requested month range using the public
   ``warc.paths.gz`` manifest that Common Crawl publishes.
2. Streams each shard over HTTPS (no AWS auth needed). Multiple shards
   are processed in parallel via ``ProcessPoolExecutor`` (see
   ``--workers``); each worker runs the WARC + trafilatura loop in its
   own process so the GIL is not a bottleneck.
3. Iterates WARC records using ``warcio``; for each response record,
   extracts the host from the target URI and the publish date from the
   ``Date`` header.
4. Keeps the record only if the host is in the editorial whitelist AND
   the publish date is inside the window.
5. Extracts body text with ``trafilatura`` and writes one JSON object per
   kept record to ``data/cc_news/{YYYY-MM}/shard_{NNN}.jsonl.zst``.
6. Writes a per-shard ``.done`` marker with stats on success; next run
   skips shards that already have a done marker. Atomic ``.tmp`` then
   ``os.replace`` ensures Ctrl-C never leaves a half-written
   ``.jsonl.zst`` in place without its sibling ``.done``.

Sibling pipeline: ``docs/V2_2_ARCHITECTURE.md`` Section 3 (GDELT DOC).
CC-News complements GDELT DOC by providing raw editorial body text
rather than DOC's metadata-plus-snippet records.

Usage
-----
    python scripts/fetch_cc_news_archive.py \\
        --start 2026-01 --end 2026-03 \\
        --domain-whitelist src/common/cc_news_domains.py \\
        --workers 8 \\
        --out data/cc_news/

    # sample: download ONE shard only, just to prove plumbing works
    python scripts/fetch_cc_news_archive.py \\
        --start 2026-01 --end 2026-01 \\
        --out data/cc_news_sample/ --max-shards 1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.common.cc_news_domains import load_whitelist, host_in_whitelist  # noqa: E402

CC_BASE = "https://data.commoncrawl.org"
# Manifest of all shards for a given month (one shard path per line).
# e.g. crawl-data/CC-NEWS/2026/01/warc.paths.gz
MANIFEST_TPL = "crawl-data/CC-NEWS/{yyyy}/{mm}/warc.paths.gz"

# Max shard bytes; abort if a single shard exceeds this.
DEFAULT_MAX_SHARD_BYTES = 5 * 1024 ** 3  # 5 GB

# Soft floor: warn if shard is smaller than this (likely truncated).
MIN_SHARD_BYTES = 10 * 1024 ** 2  # 10 MB

# Retry tuning for H8-2: 3 attempts with exponential backoff.
RETRY_ATTEMPTS = 3
RETRY_BASE_SLEEP = 1.0  # 1s, 2s, 4s
THROTTLE_SLEEP_RANGE = (5.0, 15.0)  # seconds, used for HTTP 429 / 503


@dataclass
class ShardStats:
    shard_name: str = ""
    bytes_downloaded: int = 0
    records_seen: int = 0
    records_kept: int = 0
    records_malformed: int = 0
    download_attempts: int = 1
    domains_kept: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "shard_name": self.shard_name,
            "bytes_downloaded": self.bytes_downloaded,
            "records_seen": self.records_seen,
            "records_kept": self.records_kept,
            "records_malformed": self.records_malformed,
            "download_attempts": self.download_attempts,
            "domains_kept": self.domains_kept,
            "duration_s": round(self.duration_s, 2),
        }


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Return a list of (YYYY, MM) tuples inclusive of both endpoints."""
    sy, sm = [int(x) for x in start.split("-")]
    ey, em = [int(x) for x in end.split("-")]
    out: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def list_shards_for_month(year: int, month: int, http_get=None) -> list[str]:
    """Download and parse the ``warc.paths.gz`` manifest for one month.

    Returns a list of shard paths (relative to CC_BASE), for example
    ``crawl-data/CC-NEWS/2026/01/CC-NEWS-20260101001500-00000.warc.gz``.
    """
    import requests

    http_get = http_get or requests.get
    manifest_url = f"{CC_BASE}/{MANIFEST_TPL.format(yyyy=year, mm=f'{month:02d}')}"
    r = http_get(manifest_url, timeout=60)
    r.raise_for_status()
    body = gzip.decompress(r.content)
    return [line.strip() for line in body.decode("utf-8").splitlines() if line.strip()]


def _parse_date_header(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        dt = parsedate_to_datetime(val)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_host(uri: str) -> str:
    try:
        return urlparse(uri).netloc.lower()
    except Exception:
        return ""


def iter_warc_records(stream: io.IOBase) -> Iterator[dict]:
    """Yield minimally-parsed WARC ``response`` records from ``stream``.

    Each yield is a dict ``{url, http_date, content_type, payload_bytes}``.
    Callers decide whether to extract body text. Malformed individual
    records are skipped (warcio raises inside ``ArchiveIterator``); the
    iterator does not abort the whole shard for one bad entry.
    """
    from warcio.archiveiterator import ArchiveIterator  # type: ignore

    for record in ArchiveIterator(stream):
        try:
            if record.rec_type != "response":
                continue
            uri = record.rec_headers.get_header("WARC-Target-URI") or ""
            http_headers = record.http_headers
            if http_headers is None:
                continue
            ctype = http_headers.get_header("Content-Type") or ""
            if "html" not in ctype.lower():
                continue
            http_date = http_headers.get_header("Date") or record.rec_headers.get_header("WARC-Date")
            payload = record.content_stream().read()
        except Exception:
            # Surface as a sentinel so caller can count it without aborting.
            yield {"_malformed": True}
            continue
        yield {
            "url": uri,
            "http_date": http_date,
            "content_type": ctype,
            "payload_bytes": payload,
        }


def _extract_article(payload: bytes, url: str) -> dict | None:
    """Run trafilatura on a raw HTML payload. Returns None if extraction fails."""
    try:
        import trafilatura  # type: ignore
    except ImportError:
        return None
    try:
        html = payload.decode("utf-8", errors="replace")
    except Exception:
        return None
    text = trafilatura.extract(html, include_comments=False, include_tables=False,
                               favor_recall=False, url=url)
    if not text or len(text) < 200:
        return None
    meta = trafilatura.extract_metadata(html) if hasattr(trafilatura, "extract_metadata") else None
    title = getattr(meta, "title", None) if meta else None
    meta_date = getattr(meta, "date", None) if meta else None
    return {"text": text, "title": title, "meta_date": meta_date}


# --- HTTP helpers ----------------------------------------------------------

class _TransientHTTPError(Exception):
    """Raised on HTTP 429 or 5xx; signals the retry loop to back off harder."""

    def __init__(self, status: int, msg: str = ""):
        super().__init__(f"HTTP {status}: {msg}")
        self.status = status


def _download_shard(url: str, raw_path: Path, max_bytes: int,
                    session=None) -> int:
    """Stream a single shard URL to ``raw_path``. Returns bytes written.

    Raises ``_TransientHTTPError`` on 429 / 5xx so the caller can sleep
    longer between retries. Other errors propagate as-is.
    """
    import requests

    session = session or requests.Session()
    bytes_dl = 0
    with session.get(url, stream=True, timeout=120) as r:
        if r.status_code == 429 or r.status_code >= 500:
            raise _TransientHTTPError(r.status_code, r.reason or "")
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            raise RuntimeError(
                f"shard size {int(cl):,} exceeds cap {max_bytes:,}"
            )
        with open(raw_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                bytes_dl += len(chunk)
                if bytes_dl > max_bytes:
                    fh.close()
                    try:
                        raw_path.unlink()
                    except FileNotFoundError:
                        pass
                    raise RuntimeError(
                        f"shard exceeded cap {max_bytes:,} mid-stream"
                    )
                fh.write(chunk)
    return bytes_dl


def process_shard(shard_path: str, out_path: Path, whitelist: frozenset[str],
                  date_lo: datetime, date_hi: datetime,
                  max_bytes: int = DEFAULT_MAX_SHARD_BYTES,
                  session=None) -> ShardStats:
    """Stream a WARC shard over HTTPS and write kept records to zstd-JSONL.

    Returns a ``ShardStats`` summary. Writes atomically (``.tmp`` then
    rename). Wraps the network + WARC loop in a 3-attempt retry with
    exponential backoff (H8-2). Raises ``RuntimeError`` if the shard
    exceeds ``max_bytes`` or all retries are exhausted.
    """
    import zstandard as zstd  # type: ignore

    url = f"{CC_BASE}/{shard_path}"
    stats = ShardStats(shard_name=Path(shard_path).name)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".warc.gz.dl")

    t0 = time.time()
    last_err: Exception | None = None
    downloaded = False
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            stats.bytes_downloaded = _download_shard(url, raw_path, max_bytes,
                                                     session=session)
            stats.download_attempts = attempt
            downloaded = True
            break
        except _TransientHTTPError as exc:
            last_err = exc
            if attempt >= RETRY_ATTEMPTS:
                break
            sleep_s = random.uniform(*THROTTLE_SLEEP_RANGE)
            print(f"[cc-news]   {shard_path}: {exc}, sleeping {sleep_s:.1f}s "
                  f"(attempt {attempt}/{RETRY_ATTEMPTS})")
            time.sleep(sleep_s)
        except Exception as exc:
            last_err = exc
            if attempt >= RETRY_ATTEMPTS:
                break
            sleep_s = RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            print(f"[cc-news]   {shard_path}: {exc!r}, sleeping {sleep_s:.1f}s "
                  f"(attempt {attempt}/{RETRY_ATTEMPTS})")
            time.sleep(sleep_s)
    if not downloaded:
        raise RuntimeError(
            f"shard {shard_path} failed after {RETRY_ATTEMPTS} retries: {last_err}"
        ) from last_err

    # Now stream through warcio, filter, and write kept records to zstd.
    cctx = zstd.ZstdCompressor(level=10)
    with open(raw_path, "rb") as fin, open(tmp, "wb") as fout, cctx.stream_writer(fout) as zwriter:
        for record in iter_warc_records(fin):
            if record.get("_malformed"):
                # H8-2: bad WARC entry; skip without aborting the shard.
                stats.records_malformed += 1
                continue
            stats.records_seen += 1
            try:
                host = _extract_host(record["url"])
                if not host_in_whitelist(host, whitelist):
                    continue
                dt = _parse_date_header(record["http_date"])
                if dt is None or dt < date_lo or dt > date_hi:
                    continue
                article = _extract_article(record["payload_bytes"], record["url"])
                if article is None:
                    continue
                out_row = {
                    "url": record["url"],
                    "host": host,
                    "publish_date": dt.isoformat(),
                    "title": article.get("title"),
                    "meta_date": article.get("meta_date"),
                    "text": article["text"],
                    "shard": Path(shard_path).name,
                }
                zwriter.write((json.dumps(out_row, ensure_ascii=False) + "\n").encode("utf-8"))
                stats.records_kept += 1
                dom = host.replace("www.", "")
                stats.domains_kept[dom] = stats.domains_kept.get(dom, 0) + 1
            except Exception:
                # H8-2: a single malformed record (e.g. truncated payload)
                # must not kill the whole shard. Count and continue.
                stats.records_malformed += 1
                continue

    try:
        raw_path.unlink()
    except FileNotFoundError:
        pass
    os.replace(tmp, out_path)
    stats.duration_s = time.time() - t0
    return stats


# --- Top-level worker for ProcessPoolExecutor ------------------------------

def _process_shard(shard_url: str, shard_idx: int, month_dir: str,
                   date_lo_iso: str, date_hi_iso: str,
                   whitelist_list: list[str], max_bytes: int) -> dict:
    """Module-level worker so multiprocessing can pickle it on Windows.

    Returns a stats dict; does NOT raise. Writes the .done marker on
    success only. On failure the .jsonl.zst is removed (atomic rename
    is gated on success), so the resume scan only sees fully-finished
    shards.
    """
    out_path = Path(month_dir) / f"shard_{shard_idx:04d}.jsonl.zst"
    done_path = out_path.with_suffix(out_path.suffix + ".done")
    date_lo = _parse_iso(date_lo_iso)
    date_hi = _parse_iso(date_hi_iso)
    whitelist = frozenset(whitelist_list)

    t0 = time.time()
    try:
        stats = process_shard(shard_url, out_path, whitelist, date_lo, date_hi,
                              max_bytes=max_bytes)
    except Exception as exc:
        # Clean up any partial outputs (the atomic rename inside
        # process_shard means out_path only exists on success, but
        # belt + suspenders here).
        for p in (out_path.with_suffix(out_path.suffix + ".tmp"),
                  out_path.with_suffix(".warc.gz.dl")):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        return {
            "idx": shard_idx,
            "shard_name": Path(shard_url).name,
            "n_in": 0,
            "n_out": 0,
            "bytes_in": 0,
            "bytes_out": 0,
            "wall_sec": round(time.time() - t0, 2),
            "status": "failed",
            "error": str(exc),
        }

    done_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    out_size = out_path.stat().st_size if out_path.exists() else 0
    return {
        "idx": shard_idx,
        "shard_name": stats.shard_name,
        "n_in": stats.records_seen,
        "n_out": stats.records_kept,
        "bytes_in": stats.bytes_downloaded,
        "bytes_out": out_size,
        "wall_sec": round(stats.duration_s, 2),
        "status": "ok",
        "n_malformed": stats.records_malformed,
        "download_attempts": stats.download_attempts,
    }


def _default_workers() -> int:
    cpu = os.cpu_count() or 2
    return min(8, max(2, cpu - 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start month YYYY-MM")
    p.add_argument("--end", required=True, help="End month YYYY-MM (inclusive)")
    p.add_argument("--out", default="data/cc_news/", help="Output root directory")
    p.add_argument("--domain-whitelist", default=None,
                   help="Optional extra whitelist file (one domain per line)")
    p.add_argument("--date-lo", default=None,
                   help="Keep articles with publish_date >= this ISO date (default: --start month start)")
    p.add_argument("--date-hi", default=None,
                   help="Keep articles with publish_date <= this ISO date (default: --end month end)")
    p.add_argument("--max-shards", type=int, default=0,
                   help="Max shards to process across all months (0 = all)")
    p.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES,
                   help=f"Abort any shard exceeding this size (default: {DEFAULT_MAX_SHARD_BYTES:,})")
    p.add_argument("--workers", type=int, default=_default_workers(),
                   help="Parallel worker processes (default: min(8, cpu-1)). Use 1 for legacy single-process path.")
    p.add_argument("--force", action="store_true", help="Re-process shards even if .done exists")
    p.add_argument("--dry-run", action="store_true", help="List shards without downloading")
    return p


def _enumerate_jobs(months: list[tuple[int, int]], out_root: Path,
                    max_shards: int, force: bool, dry_run: bool) -> list[tuple[int, str, Path]]:
    """Walk each month manifest and return (idx, shard_url, month_dir).

    ``idx`` is per-month so that ``shard_{NNN}`` filenames stay stable
    across runs (matches the legacy single-thread enumeration).
    """
    jobs: list[tuple[int, str, Path]] = []
    total = 0
    for (y, m) in months:
        shards = list_shards_for_month(y, m)
        print(f"[cc-news] {y}-{m:02d}: {len(shards)} shard(s) in manifest")
        month_dir = out_root / f"{y:04d}-{m:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        for idx, shard in enumerate(shards):
            out_file = month_dir / f"shard_{idx:04d}.jsonl.zst"
            done_file = out_file.with_suffix(out_file.suffix + ".done")
            if done_file.exists() and not force:
                if dry_run:
                    print(f"[cc-news]   skip (done): {out_file.name}")
                continue
            if dry_run:
                print(f"[cc-news]   would download: {shard}")
            jobs.append((idx, shard, month_dir))
            total += 1
            if max_shards and total >= max_shards:
                return jobs
    return jobs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wl = load_whitelist(Path(args.domain_whitelist) if args.domain_whitelist else None)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    months = _month_range(args.start, args.end)
    date_lo = _parse_iso(args.date_lo) if args.date_lo else datetime(months[0][0], months[0][1], 1, tzinfo=timezone.utc)
    ey, em = months[-1]
    date_hi = _parse_iso(args.date_hi) if args.date_hi else _month_end(ey, em)

    print(f"[cc-news] window={args.start}..{args.end}, "
          f"date_lo={date_lo.isoformat()}, date_hi={date_hi.isoformat()}, "
          f"whitelist_size={len(wl)}, workers={args.workers}")

    jobs = _enumerate_jobs(months, out_root, args.max_shards, args.force, args.dry_run)
    if args.dry_run:
        print(f"[cc-news] dry-run: {len(jobs)} shard(s) would be fetched")
        return 0
    if not jobs:
        print("[cc-news] nothing to do (all shards .done or 0 in manifest)")
        return 0

    wl_list = sorted(wl)
    date_lo_iso = date_lo.isoformat()
    date_hi_iso = date_hi.isoformat()

    n_succeeded = n_retried = n_fatal = 0
    total_kept = 0
    completed = 0

    if args.workers <= 1:
        # Legacy single-process path. Keeps stack traces simple for debug.
        for (idx, shard, month_dir) in jobs:
            res = _process_shard(shard, idx, str(month_dir), date_lo_iso,
                                 date_hi_iso, wl_list, args.max_shard_bytes)
            completed += 1
            if res["status"] == "ok":
                n_succeeded += 1
                total_kept += res["n_out"]
                if res.get("download_attempts", 1) > 1:
                    n_retried += 1
            else:
                n_fatal += 1
                print(f"[cc-news] shard {idx} FAILED after {RETRY_ATTEMPTS} retries: "
                      f"{res.get('error')}")
            if completed % 10 == 0 or completed == len(jobs):
                print(f"[cc-news] progress: {completed}/{len(jobs)} "
                      f"(ok={n_succeeded} fail={n_fatal} kept={total_kept:,})")
        print(f"[cc-news] complete: {n_succeeded} ok, {n_fatal} failed, "
              f"{total_kept:,} records kept")
        return 0 if n_fatal == 0 else 1

    # Parallel path. Use ProcessPoolExecutor; on Ctrl-C, cancel pending
    # futures and let in-flight workers finish (their tmp/raw files get
    # cleaned up either by process_shard itself or by _process_shard's
    # except branch).
    futures: dict[cf.Future, tuple[int, str]] = {}
    pool = cf.ProcessPoolExecutor(max_workers=args.workers)
    interrupted = False
    try:
        for (idx, shard, month_dir) in jobs:
            fut = pool.submit(_process_shard, shard, idx, str(month_dir),
                              date_lo_iso, date_hi_iso, wl_list,
                              args.max_shard_bytes)
            futures[fut] = (idx, shard)

        for fut in cf.as_completed(futures):
            idx, shard = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                n_fatal += 1
                print(f"[cc-news] shard {idx} crashed in worker: {exc!r}")
                completed += 1
                continue
            completed += 1
            if res["status"] == "ok":
                n_succeeded += 1
                total_kept += res["n_out"]
                if res.get("download_attempts", 1) > 1:
                    n_retried += 1
            else:
                n_fatal += 1
                print(f"[cc-news] shard {idx} FAILED after {RETRY_ATTEMPTS} retries: "
                      f"{res.get('error')}")
            if completed % 10 == 0 or completed == len(jobs):
                print(f"[cc-news] progress: {completed}/{len(jobs)} "
                      f"(ok={n_succeeded} fail={n_fatal} kept={total_kept:,})")
    except KeyboardInterrupt:
        interrupted = True
        print("[cc-news] KeyboardInterrupt, cancelling pending futures...")
        for fut in futures:
            fut.cancel()
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=True)

    print(f"[cc-news] complete: {n_succeeded} ok, {n_fatal} failed, "
          f"{total_kept:,} records kept "
          f"({n_retried} retried, interrupted={interrupted})")
    return 0 if (n_fatal == 0 and not interrupted) else 1


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _month_end(year: int, month: int) -> datetime:
    if month == 12:
        ny, nm = year + 1, 1
    else:
        ny, nm = year, month + 1
    from datetime import timedelta
    return datetime(ny, nm, 1, tzinfo=timezone.utc) - timedelta(seconds=1)


if __name__ == "__main__":
    raise SystemExit(main())
