"""Stream CC-News WARC shards from Common Crawl and filter-while-download.

CC-News is the public news-only Common Crawl archive. Shards live at
``s3://commoncrawl/crawl-data/CC-NEWS/{YYYY}/{MM}/CC-NEWS-{timestamp}.warc.gz``
and are also mirrored on HTTPS at
``https://data.commoncrawl.org/crawl-data/CC-NEWS/{YYYY}/{MM}/``.

This fetcher:

1. Discovers shard URLs for the requested month range using the public
   ``warc.paths.gz`` manifest that Common Crawl publishes.
2. Streams each shard over HTTPS (no AWS auth needed).
3. Iterates WARC records using ``warcio``; for each response record,
   extracts the host from the target URI and the publish date from the
   ``Date`` header.
4. Keeps the record only if the host is in the editorial whitelist AND
   the publish date is inside the window.
5. Extracts body text with ``trafilatura`` and writes one JSON object per
   kept record to ``data/cc_news/{YYYY-MM}/shard_{NNN}.jsonl.zst``.
6. Writes a per-shard ``.done`` marker with stats on success; next run
   skips shards that already have a done marker.

Sibling pipeline: ``docs/V2_2_ARCHITECTURE.md`` Section 3 (GDELT DOC).
CC-News complements GDELT DOC by providing raw editorial body text
rather than DOC's metadata-plus-snippet records.

Usage
-----
    python scripts/fetch_cc_news_archive.py \\
        --start 2026-01 --end 2026-03 \\
        --domain-whitelist src/common/cc_news_domains.py \\
        --out data/cc_news/

    # sample: download ONE shard only, just to prove plumbing works
    python scripts/fetch_cc_news_archive.py \\
        --start 2026-01 --end 2026-01 \\
        --out data/cc_news_sample/ --max-shards 1
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Iterator
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


@dataclass
class ShardStats:
    shard_name: str = ""
    bytes_downloaded: int = 0
    records_seen: int = 0
    records_kept: int = 0
    domains_kept: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "shard_name": self.shard_name,
            "bytes_downloaded": self.bytes_downloaded,
            "records_seen": self.records_seen,
            "records_kept": self.records_kept,
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
    Callers decide whether to extract body text.
    """
    from warcio.archiveiterator import ArchiveIterator  # type: ignore

    for record in ArchiveIterator(stream):
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


def process_shard(shard_path: str, out_path: Path, whitelist: frozenset[str],
                  date_lo: datetime, date_hi: datetime,
                  max_bytes: int = DEFAULT_MAX_SHARD_BYTES,
                  session=None) -> ShardStats:
    """Stream a WARC shard over HTTPS and write kept records to zstd-JSONL.

    Returns a ``ShardStats`` summary. Writes atomically (``.tmp`` then
    rename). Raises ``RuntimeError`` if the shard exceeds ``max_bytes``.
    """
    import requests
    import zstandard as zstd  # type: ignore

    session = session or requests.Session()
    url = f"{CC_BASE}/{shard_path}"
    stats = ShardStats(shard_name=Path(shard_path).name)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        # Enforce the 5 GB cap if the server advertises Content-Length.
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            raise RuntimeError(
                f"shard {shard_path} size {int(cl):,} exceeds cap {max_bytes:,}"
            )

        # Buffer the download to a tempfile so warcio can seek if needed.
        # We still cap the byte count.
        raw_path = out_path.with_suffix(".warc.gz.dl")
        with open(raw_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                stats.bytes_downloaded += len(chunk)
                if stats.bytes_downloaded > max_bytes:
                    fh.close()
                    try:
                        raw_path.unlink()
                    except FileNotFoundError:
                        pass
                    raise RuntimeError(
                        f"shard {shard_path} exceeded cap {max_bytes:,} mid-stream"
                    )
                fh.write(chunk)

    # Now stream through warcio, filter, and write kept records to zstd.
    cctx = zstd.ZstdCompressor(level=10)
    with open(raw_path, "rb") as fin, open(tmp, "wb") as fout, cctx.stream_writer(fout) as zwriter:
        for record in iter_warc_records(fin):
            stats.records_seen += 1
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

    try:
        raw_path.unlink()
    except FileNotFoundError:
        pass
    os.replace(tmp, out_path)
    stats.duration_s = time.time() - t0
    return stats


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
    p.add_argument("--force", action="store_true", help="Re-process shards even if .done exists")
    p.add_argument("--dry-run", action="store_true", help="List shards without downloading")
    return p


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
          f"whitelist_size={len(wl)}")

    processed = 0
    total_kept = 0
    for (y, m) in months:
        shards = list_shards_for_month(y, m)
        print(f"[cc-news] {y}-{m:02d}: {len(shards)} shard(s) in manifest")
        month_dir = out_root / f"{y:04d}-{m:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        for idx, shard in enumerate(shards):
            if args.max_shards and processed >= args.max_shards:
                print(f"[cc-news] --max-shards={args.max_shards} reached, stopping")
                return 0
            out_file = month_dir / f"shard_{idx:04d}.jsonl.zst"
            done_file = out_file.with_suffix(out_file.suffix + ".done")
            if done_file.exists() and not args.force:
                print(f"[cc-news]   skip (done): {out_file.name}")
                continue
            if args.dry_run:
                print(f"[cc-news]   would download: {shard}")
                processed += 1
                continue
            print(f"[cc-news]   downloading shard {idx+1}/{len(shards)}: {shard}")
            try:
                stats = process_shard(shard, out_file, wl, date_lo, date_hi,
                                      max_bytes=args.max_shard_bytes)
            except Exception as exc:
                print(f"[cc-news]     FAILED: {exc}")
                continue
            done_file.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
            total_kept += stats.records_kept
            print(f"[cc-news]     done: {stats.bytes_downloaded:,} B, "
                  f"{stats.records_seen:,} recs seen, {stats.records_kept:,} kept, "
                  f"{len(stats.domains_kept)} domains, {stats.duration_s:.1f}s")
            processed += 1

    print(f"[cc-news] complete: {processed} shard(s) processed, {total_kept:,} records kept")
    return 0


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
