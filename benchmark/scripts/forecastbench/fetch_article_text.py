"""
Fetch full article text for ForecastBench questions.

Reads data/gdelt_articles.jsonl (URLs from GDELT DOC API),
downloads each article page, extracts main text with trafilatura,
and writes data/fb_articles_full.jsonl with a populated 'text' field.

Usage:
  python benchmark/scripts/forecastbench/fetch_article_text.py --n 100       # first 100 questions
  python benchmark/scripts/forecastbench/fetch_article_text.py --all          # all questions
  python benchmark/scripts/forecastbench/fetch_article_text.py --workers 8    # parallel workers
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import trafilatura
import requests

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
IN_FILE  = DATA / "gdelt_articles.jsonl"
OUT_FILE = DATA / "fb_articles_full.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 12


def fetch_text(url: str) -> str:
    """Download URL and extract main article text. Returns '' on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return ""
        text = trafilatura.extract(
            resp.content,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text or ""
    except Exception:
        return ""


def process_article(record: dict) -> dict:
    url = record.get("url", "")
    text = fetch_text(url) if url else ""
    return {**record, "text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Max questions to process")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not IN_FILE.exists():
        print(f"[ERROR] {IN_FILE} not found. Run download_gdelt_news.py first.")
        sys.exit(1)

    records = [json.loads(l) for l in open(IN_FILE, encoding="utf-8")]
    if not args.all and args.n:
        # limit by question count
        seen_qids: set[str] = set()
        kept = []
        for r in records:
            seen_qids.add(r["question_id"])
            if len(seen_qids) > args.n:
                break
            kept.append(r)
        records = kept

    # skip already-fetched
    existing_ids: set[str] = set()
    if OUT_FILE.exists():
        for line in open(OUT_FILE, encoding="utf-8"):
            d = json.loads(line)
            if d.get("text"):
                existing_ids.add(d["id"])
        print(f"Skipping {len(existing_ids)} already-fetched articles")

    pending = [r for r in records if r["id"] not in existing_ids]
    print(f"Fetching text for {len(pending)} articles ({args.workers} workers)")

    fetched = skipped = 0
    with open(OUT_FILE, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_article, r): r for r in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                f_out.write(json.dumps(result) + "\n")
                if result.get("text"):
                    fetched += 1
                else:
                    skipped += 1
                if i % 50 == 0:
                    print(f"  [{i}/{len(pending)}] fetched={fetched} empty={skipped}")

    total = sum(1 for _ in open(OUT_FILE))
    non_empty = sum(1 for l in open(OUT_FILE) if json.loads(l).get("text"))
    print(f"\nDone. Total={total}, with_text={non_empty} ({non_empty/total*100:.0f}%)")


if __name__ == "__main__":
    main()
