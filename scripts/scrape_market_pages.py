"""
Step C of coverage recovery: scrape prediction-market question pages for
outbound links to news articles, then fetch those articles.

Strategy: the question page itself often contains a "Background" or "Related
Articles" section with direct URLs to news stories. These are human-curated
and often higher quality than GDELT keyword matches.

Scope: Metaculus (IDs like fb_7664 -> metaculus.com/questions/7664/) is the
cleanest case and where we can reliably construct URLs. Polymarket/Manifold
use opaque condition IDs that require additional API calls to resolve.

Outputs:
  data/gdelt_articles_supplement.jsonl (appended, same file as Step B)

Usage:
  python scripts/scrape_market_pages.py
  python scripts/scrape_market_pages.py --limit 5 --dry-run
"""
import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

ROOT = Path(__file__).parent.parent
FC_FILE = ROOT / "data" / "unified" / "forecasts.jsonl"
ART_FILE = ROOT / "data" / "unified" / "articles.jsonl"
SUPP = ROOT / "data" / "gdelt_articles_supplement.jsonl"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}

META_URL_TPL = "https://www.metaculus.com/api2/questions/{n}/"

NEWS_DOMAIN_HINTS = (
    "nytimes.com", "washingtonpost.com", "bbc.", "reuters.com", "apnews.com",
    "bloomberg.com", "wsj.com", "ft.com", "cnn.com", "nbcnews.com", "cbsnews.com",
    "theguardian.com", "politico.com", "axios.com", "semafor.com", "npr.org",
    "aljazeera.com", "economist.com", "foreignpolicy.com", "foreignaffairs.com",
    "defensenews.com", "military.com", "csmonitor.com", "newsweek.com",
    "techcrunch.com", "arstechnica.com", "theverge.com", "wired.com",
)
INTERNAL_SKIP = ("metaculus.com", "polymarket.com", "manifold.markets",
                 "infer-pub.com", "twitter.com", "x.com", "reddit.com",
                 "youtube.com", "youtu.be", "facebook.com", "instagram.com",
                 "wikipedia.org", "wikimedia.org")


def looks_like_news(url: str) -> bool:
    """Heuristic: URL is external, not social, has a long path."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    dom = p.netloc.lower().replace("www.", "")
    if not dom:
        return False
    if any(dom.endswith(s) for s in INTERNAL_SKIP):
        return False
    # extract path depth
    path_parts = [x for x in p.path.split("/") if x]
    if len(path_parts) < 2:
        return False
    # hint domain OR looks article-ish (has slug with 3+ words)
    if any(h in dom for h in NEWS_DOMAIN_HINTS):
        return True
    slug = path_parts[-1]
    return len(slug) > 20 and slug.count("-") >= 2


def fetch_metaculus_links(meta_id: str) -> list[str]:
    """Return external URLs found in a Metaculus question's description+criteria."""
    url = META_URL_TPL.format(n=meta_id)
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code != 200:
            return []
        d = r.json()
    except Exception:
        return []
    q = d.get("question") or {}
    text_fields = [q.get("description", ""), q.get("resolution_criteria", ""),
                   q.get("fine_print", ""), d.get("description", "")]
    blob = "\n".join(t or "" for t in text_fields)
    urls = re.findall(r"https?://[^\s)\]\"<>]+", blob)
    return [u.rstrip(".,;:)\"'") for u in urls if looks_like_news(u)]


def fetch_article(url: str) -> tuple[str, str]:
    """Return (title, text). Empty strings on failure."""
    try:
        r = requests.get(url, headers=UA, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return "", ""
        text = trafilatura.extract(r.content, include_comments=False,
                                   include_tables=False, no_fallback=False) or ""
        meta = trafilatura.extract_metadata(r.content)
        title = (meta.title if meta else "") or ""
        return title, text
    except Exception:
        return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    forecasts = [json.loads(l) for l in open(FC_FILE, encoding="utf-8")]
    orphans = [f for f in forecasts if not f.get("article_ids")]
    # scope to Metaculus orphans (only source where we can reliably construct URL)
    meta_orphans = []
    for f in orphans:
        if f.get("source") != "metaculus":
            continue
        # strip "fb_" prefix
        raw = f["id"].replace("fb_", "", 1)
        if raw.isdigit():
            meta_orphans.append((f, raw))
    if args.limit:
        meta_orphans = meta_orphans[: args.limit]
    print(f"Metaculus orphans with resolvable IDs: {len(meta_orphans)} "
          f"(of {sum(1 for f in orphans if f.get('source')=='metaculus')} Metaculus orphans, "
          f"{len(orphans)} total orphans)")

    existing = set()
    if ART_FILE.exists():
        for l in open(ART_FILE, encoding="utf-8"):
            existing.add(json.loads(l).get("url", ""))
    if SUPP.exists():
        for l in open(SUPP, encoding="utf-8"):
            existing.add(json.loads(l).get("url", ""))
    print(f"Known article URLs: {len(existing)}")

    new_count = 0
    recovered = 0
    with open(SUPP, "a", encoding="utf-8") as fout:
        for fc, meta_id in meta_orphans:
            print(f"\n  [{fc['id']}] Metaculus {meta_id}")
            if args.dry_run:
                links = ["(dry-run)"]
            else:
                links = fetch_metaculus_links(meta_id)
            print(f"    found {len(links)} external news-looking links")
            if args.dry_run:
                continue
            added_here = 0
            for url in links[:10]:
                if url in existing:
                    continue
                title, text = fetch_article(url)
                if not text or len(text) < 200:
                    existing.add(url)
                    continue
                existing.add(url)
                fout.write(json.dumps({
                    "id": f"gdeltc_{abs(hash(url)) % 10**10}",
                    "question_id": fc["id"],
                    "title": title,
                    "abstract": "",
                    "text": text,
                    "date": fc.get("resolution_date", ""),  # best-effort; article dates unknown
                    "url": url,
                    "source": urlparse(url).netloc.lower().replace("www.", ""),
                    "tone": 0.0,
                    "country_mentions": [],
                    "retry_query": "metaculus-page-scrape",
                }, ensure_ascii=False) + "\n")
                added_here += 1
                new_count += 1
            if added_here:
                recovered += 1
                print(f"    +{added_here} articles saved")
            time.sleep(0.5)

    print(f"\nDone. New articles: {new_count}")
    print(f"Metaculus orphans recovered: {recovered} / {len(meta_orphans)}")
    print(f"Supplement updated: {SUPP}")
    print("Next: re-run unify_articles.py then compute_relevance.py")


if __name__ == "__main__":
    main()
