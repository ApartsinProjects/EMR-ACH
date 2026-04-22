"""
Retry article text fetch for failed URLs from fb_articles_full.jsonl.

Addresses the four main failure modes we diagnosed:
  - HTTP 403 (bot detection): rotate between Chrome/Firefox UA + browser-like Accept headers
  - Non-Latin charsets (RO/IT/RU/CZ sites): use chardet + response.apparent_encoding
  - Extraction miss on 200 OK: trafilatura → readability-lxml → BeautifulSoup fallback
  - Timeout / 429 / 503: 30s timeout, one retry after 5s backoff

Writes updated fb_articles_full.jsonl in place, only modifying rows with empty text.

Usage:
  python scripts/retry_article_text.py --workers 6
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chardet
import requests
import trafilatura
from bs4 import BeautifulSoup
from readability import Document

ROOT = Path(__file__).parent.parent
FILE = ROOT / "data" / "fb_articles_full.jsonl"

UA_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
             "AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/122.0.0.0 Safari/537.36")
UA_FIREFOX = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
              "Gecko/20100101 Firefox/123.0")

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

TIMEOUT = 30


def decode_body(resp: requests.Response) -> str:
    """Decode response body, handling non-standard encodings."""
    raw = resp.content
    # trust HTTP header first
    enc = (resp.encoding or "").lower() if resp.encoding else ""
    # requests defaults to ISO-8859-1 when no charset in header - unreliable
    if not enc or enc == "iso-8859-1":
        # detect via chardet
        detected = chardet.detect(raw[:50000])
        enc = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def extract_with_fallbacks(html: str, url: str) -> str:
    """Try trafilatura first; fall back to readability; then BeautifulSoup."""
    # 1. trafilatura (best for news sites)
    text = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False) or ""
    if len(text) >= 200:
        return text
    # 2. readability-lxml (used by Firefox reader mode)
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text2 = soup.get_text("\n", strip=True)
        if len(text2) >= 200:
            return text2
    except Exception:
        pass
    # 3. naive BeautifulSoup — strip nav/footer/script, keep article/main
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        target = soup.find("article") or soup.find("main") or soup.body or soup
        text3 = target.get_text("\n", strip=True)
        # only return if it looks like an article, not just a menu
        if len(text3) >= 300:
            return text3
    except Exception:
        pass
    return text  # return whatever trafilatura gave (may be empty or <200)


def fetch_via_wayback(url: str) -> str:
    """Try the closest Wayback Machine snapshot (archive.org)."""
    try:
        api = f"http://archive.org/wayback/available?url={url}"
        meta = requests.get(api, timeout=15).json()
        snap = (meta.get("archived_snapshots", {}) or {}).get("closest", {})
        if not snap.get("available"):
            return ""
        archived_url = snap["url"]
        # request plain-content variant (id_ = no wayback toolbar wrap)
        if "/web/" in archived_url and "id_" not in archived_url:
            archived_url = archived_url.replace("/web/", "/web/", 1)
            parts = archived_url.split("/web/", 1)
            ts_and_rest = parts[1]
            ts, real = ts_and_rest.split("/", 1)
            archived_url = f"{parts[0]}/web/{ts}id_/{real}"
        resp = requests.get(archived_url, headers={"User-Agent": UA_CHROME},
                            timeout=TIMEOUT, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        html = decode_body(resp)
        return extract_with_fallbacks(html, url)
    except Exception:
        return ""


def fetch_one(url: str) -> str:
    """5-layer cascade: plain HTTP + UA rotation → Playwright → Wayback
    → archive.today → Common Crawl. All layers wrapped in try/except.
    Implemented by scripts/fetch_text_multi.py."""
    try:
        from fetch_text_multi import fetch_text, FetchOptions   # local import for speed
    except ImportError:
        # fetch_text_multi.py lives next to this file; ensure it's importable
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from fetch_text_multi import fetch_text, FetchOptions
    return fetch_text(url, FetchOptions(
        timeout=TIMEOUT,
        min_chars=200,
        enable_playwright=True,
        enable_wayback=True,
        enable_archive_today=True,
        enable_common_crawl=True,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Only retry first N failures (debug)")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(FILE, encoding="utf-8")]
    failed_idx = [i for i, r in enumerate(records) if not r.get("text")]
    if args.limit:
        failed_idx = failed_idx[: args.limit]
    print(f"Total records: {len(records)} | Empty: {len(failed_idx)} | Retrying: {len(failed_idx)}")

    recovered = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(fetch_one, records[i].get("url", "")): i for i in failed_idx
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                text = fut.result()
            except Exception:
                text = ""
            if text and len(text) >= 200:
                records[idx]["text"] = text
                recovered += 1
            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(failed_idx)}] recovered={recovered}")

    # write back
    tmp = FILE.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(FILE)

    with_text = sum(1 for r in records if r.get("text"))
    print(f"\nDone. Recovered {recovered} of {len(failed_idx)}.")
    print(f"File now has {with_text}/{len(records)} ({with_text/len(records)*100:.1f}%) articles with text.")


if __name__ == "__main__":
    main()
