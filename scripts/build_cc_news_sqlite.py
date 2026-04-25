"""Build a SQLite index over CC-News shards for v2.2 strategy 6.1B.

Reads zstd-compressed JSONL shards produced by
``scripts/fetch_cc_news_archive.py`` and writes a single SQLite database
with two tables:

    articles (
        url            TEXT PRIMARY KEY,
        host           TEXT,
        publish_date   TEXT,    -- ISO YYYY-MM-DD (date only)
        publish_ts     TEXT,    -- full ISO timestamp from CC-News
        title          TEXT,
        text           TEXT,
        shard          TEXT,
        meta_date      TEXT
    )

    articles_fts USING fts5(title, text, content='articles', content_rowid='rowid')

Plus indices on (publish_date), (host).

This is a metadata + full-text index, complementing the FAISS+parquet
semantic index built by ``scripts/build_cc_news_index.py``. Used by
``scripts/unify_articles_ccnews.py`` to retrieve CC-News articles per FD
via keyword (FB) or ticker/host (earnings) selectors with date prefilter.

Usage
-----
    /c/Python314/python scripts/build_cc_news_sqlite.py \
        --in data/cc_news/2025-12/ \
        --out data/cc_news/index_2025-12.sqlite

The script is idempotent for a given output path: if --out exists it is
overwritten. A build log is written next to --out as ``index_build.log``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))


def _iter_zst_jsonl(path: Path) -> Iterator[dict]:
    import zstandard as zstd  # type: ignore

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        yield json.loads(line.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
        if buf.strip():
            try:
                yield json.loads(buf.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                pass


def _iso_date(ts: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp; empty on failure."""
    if not ts:
        return ""
    s = str(ts)
    # Fast path: already a date or starts with one.
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return ""


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;

        DROP TABLE IF EXISTS articles_fts;
        DROP TABLE IF EXISTS articles;

        CREATE TABLE articles (
            rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
            url            TEXT UNIQUE,
            host           TEXT,
            publish_date   TEXT,
            publish_ts     TEXT,
            title          TEXT,
            text           TEXT,
            shard          TEXT,
            meta_date      TEXT
        );

        CREATE VIRTUAL TABLE articles_fts USING fts5(
            title, text,
            content='articles',
            content_rowid='rowid',
            tokenize='porter unicode61'
        );
        """
    )


def _create_indices(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_articles_publish_date ON articles(publish_date);
        CREATE INDEX IF NOT EXISTS idx_articles_host ON articles(host);
        """
    )


def build(in_dir: Path, out_path: Path, max_text_chars: int = 8000) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(str(out_path))
    try:
        _create_schema(con)

        shards = sorted(in_dir.glob("shard_*.jsonl.zst"))
        print(f"[cc-news-sqlite] {len(shards)} shard(s) under {in_dir}")
        n_rows = 0
        n_skipped_shards = 0
        n_dup_url = 0
        t0 = time.time()
        cur = con.cursor()
        cur.execute("BEGIN")
        BATCH = 5000
        batch_rows: list[tuple] = []

        def flush() -> int:
            if not batch_rows:
                return 0
            cur.executemany(
                "INSERT OR IGNORE INTO articles "
                "(url, host, publish_date, publish_ts, title, text, shard, meta_date) "
                "VALUES (?,?,?,?,?,?,?,?)",
                batch_rows,
            )
            n = cur.rowcount
            batch_rows.clear()
            return n

        for si, shard in enumerate(shards):
            done = shard.with_suffix(shard.suffix + ".done")
            if not done.exists():
                n_skipped_shards += 1
                continue
            for r in _iter_zst_jsonl(shard):
                url = (r.get("url") or "").strip()
                if not url:
                    continue
                publish_ts = r.get("publish_date") or ""
                publish_date = _iso_date(publish_ts) or _iso_date(r.get("meta_date") or "")
                title = (r.get("title") or "").strip()
                text = (r.get("text") or "").strip()
                if max_text_chars and len(text) > max_text_chars:
                    text = text[:max_text_chars]
                batch_rows.append(
                    (
                        url,
                        (r.get("host") or "").lower(),
                        publish_date,
                        publish_ts,
                        title,
                        text,
                        r.get("shard") or shard.name,
                        r.get("meta_date") or "",
                    )
                )
                if len(batch_rows) >= BATCH:
                    inserted = flush()
                    n_rows += inserted
                    n_dup_url += BATCH - inserted
            if (si + 1) % 50 == 0:
                inserted = flush()
                n_rows += inserted
                con.commit()
                cur.execute("BEGIN")
                dt = time.time() - t0
                print(f"[cc-news-sqlite]   shard {si+1}/{len(shards)} rows={n_rows:,} dt={dt:.1f}s")

        inserted = flush()
        n_rows += inserted
        con.commit()

        # Populate FTS5 from the articles table.
        print("[cc-news-sqlite] building FTS5 index...")
        t1 = time.time()
        con.execute(
            "INSERT INTO articles_fts(rowid, title, text) "
            "SELECT rowid, title, text FROM articles"
        )
        con.commit()
        print(f"[cc-news-sqlite]   FTS5 built in {time.time()-t1:.1f}s")

        print("[cc-news-sqlite] creating B-tree indices...")
        _create_indices(con)
        con.commit()

        # Sanity counts.
        n_articles = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        n_with_date = con.execute(
            "SELECT COUNT(*) FROM articles WHERE publish_date != ''"
        ).fetchone()[0]
        n_distinct_hosts = con.execute(
            "SELECT COUNT(DISTINCT host) FROM articles"
        ).fetchone()[0]

        stats = {
            "build_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "in_dir": str(in_dir),
            "out_path": str(out_path),
            "shards_total": len(shards),
            "shards_skipped_no_done": n_skipped_shards,
            "rows_inserted": n_articles,
            "rows_with_publish_date": n_with_date,
            "distinct_hosts": n_distinct_hosts,
            "duplicate_urls_ignored": n_dup_url,
            "duration_s": round(time.time() - t0, 2),
        }
        return stats
    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="in_dir", required=True, help="CC-News shard month directory")
    p.add_argument("--out", required=True, help="Output SQLite path")
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=8000,
        help="Truncate stored text to this many chars (FTS5 indexes the truncated text). 0 = keep full text.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    if not in_dir.exists():
        print(f"[cc-news-sqlite] ERROR: input dir not found: {in_dir}", file=sys.stderr)
        return 2

    stats = build(in_dir, out_path, max_text_chars=args.max_text_chars)
    print("=== CC-News SQLite index build ===")
    print(json.dumps(stats, indent=2))

    log_path = out_path.parent / "index_build.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stats) + "\n")
    print(f"[cc-news-sqlite] log -> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
