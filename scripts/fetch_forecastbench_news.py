"""Fetch news for each ForecastBench FD across multiple sources.

Mirrors scripts/fetch_earnings_news.py structure but uses question-text keyword
extraction instead of ticker metadata, and uses three sources:

  1. GDELT DOC API (no key, global coverage, time-sliced for day-spread)
  2. Google News RSS (no key, editorial breadth via feedparser)
  3. Guardian Open Platform (requires GUARDIAN_API_KEY, high-quality archive)

All fetches are strictly pre-forecast_point (leakage guard).

Reads:
  data/forecastbench_geopolitics.jsonl  - FB FDs (source list from existing build)
  OR benchmark/data/{cutoff}/forecasts.jsonl (filter to benchmark=forecastbench)

Writes:
  data/forecastbench/forecastbench_articles.jsonl   - unified article records

Usage:
  python scripts/fetch_forecastbench_news.py                       # all FB FDs
  python scripts/fetch_forecastbench_news.py --source guardian     # Guardian only
  python scripts/fetch_forecastbench_news.py --limit 10            # debug
  python scripts/fetch_forecastbench_news.py --fd-ids fb_xx,fb_yy  # specific FDs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, quote_plus

import requests

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from src.common.optional_imports import optional  # noqa: E402

feedparser = optional("feedparser")   # only needed for Google News RSS source

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
# FB FDs are in the unified forecasts file from the pipeline, or can be pulled
# from the published benchmark. We default to the latter if available.
OUT_DIR = DATA / "forecastbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "forecastbench_articles.jsonl"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GUARDIAN_API = "https://content.guardianapis.com/search"
NYT_API = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 30


# Shared spam blocklist — see src/common/spam_domains.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from src.common.spam_domains import is_spam_url as _is_spam_url_shared  # noqa: E402


def art_id(url: str) -> str:
    return "fbn_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_spam_url(url: str) -> bool:
    """Delegate to the shared blocklist in src/common/spam_domains.py."""
    return _is_spam_url_shared(url)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOPS = {
    "will", "the", "a", "an", "be", "at", "in", "of", "to", "and", "or", "is",
    "are", "was", "were", "has", "have", "had", "by", "between", "from", "for",
    "with", "on", "that", "this", "than", "more", "least", "most", "before",
    "after", "during", "whether", "would", "could", "when", "which", "who",
    "what", "how", "many", "much", "any", "some", "each", "every", "all",
    "about", "over", "under", "up", "down", "out", "into", "through",
    # forecasting-template words
    "question", "predict", "forecast", "probability", "resolve", "resolves",
    "resolved", "market", "date", "time", "year", "month", "day", "week",
}


def extract_keywords(question: str, background: str = "", max_terms: int = 5) -> str:
    """Pick 3-5 search terms from the question. Prefer proper nouns (likely
    entities like places/people/orgs), fall back to content words."""
    text = (question or "") + " " + (background or "")
    # Tokenize keeping capitalization
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-]{2,}\b', text)
    filtered = [w for w in words if w.lower() not in _STOPS]
    proper = []
    common_seen = set()
    common = []
    for w in filtered:
        if w[0].isupper():
            if w not in proper:
                proper.append(w)
        else:
            lw = w.lower()
            if lw not in common_seen:
                common_seen.add(lw)
                common.append(lw)
    picks = proper[:4] + common[:max_terms - min(len(proper), 4)]
    return " ".join(picks[:max_terms])


# ---------------------------------------------------------------------------
# Source 0: NYT Article Search (free w/ key, 1000/day, 1851+ archive)
# ---------------------------------------------------------------------------

def fetch_nyt(query: str, forecast_point: datetime, lookback_days: int,
              max_pages: int = 2) -> list[dict]:
    key = os.environ.get("NYT_API_KEY", "")
    if not key:
        return []
    start = forecast_point - timedelta(days=lookback_days)
    end = forecast_point - timedelta(days=1)
    out = []
    for page in range(max_pages):
        params = {
            "q":          query,
            "begin_date": start.strftime("%Y%m%d"),
            "end_date":   end.strftime("%Y%m%d"),
            "api-key":    key,
            "page":       page,
            "sort":       "relevance",
        }
        try:
            r = requests.get(NYT_API, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(6)
                continue
            if r.status_code != 200:
                break
            docs = r.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"  [WARN] NYT: {e}")
            break
        if not docs:
            break
        for d in docs:
            url = d.get("web_url", "")
            if not url:
                continue
            headline = d.get("headline", {}).get("main", "") or ""
            abstract = d.get("abstract", "") or d.get("lead_paragraph", "") or ""
            pub = (d.get("pub_date", "") or "")[:10]
            out.append({
                "url":        url,
                "title":      headline,
                "summary":    abstract,
                "date":       pub,
                "publisher":  "The New York Times",
                "provenance": "nyt",
            })
        # NYT rate-limits aggressively (~6s/req safe). Sleep between pages.
        time.sleep(6)
    return out


# ---------------------------------------------------------------------------
# Source 1: Guardian
# ---------------------------------------------------------------------------

def fetch_guardian(query: str, forecast_point: datetime, lookback_days: int,
                   page_size: int = 15) -> list[dict]:
    key = os.environ.get("GUARDIAN_API_KEY", "")
    if not key:
        return []
    start = forecast_point - timedelta(days=lookback_days)
    end = forecast_point - timedelta(days=1)
    params = {
        "q":            query,
        "from-date":    start.strftime("%Y-%m-%d"),
        "to-date":      end.strftime("%Y-%m-%d"),
        "show-fields":  "trailText,byline",
        "page-size":    page_size,
        "order-by":     "relevance",
        "api-key":      key,
    }
    try:
        r = requests.get(GUARDIAN_API, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        results = r.json().get("response", {}).get("results", [])
    except Exception as e:
        print(f"  [WARN] Guardian: {e}")
        return []
    out = []
    for a in results:
        url = a.get("webUrl", "")
        if not url:
            continue
        pub = a.get("webPublicationDate", "")[:10]
        fields = a.get("fields", {}) or {}
        out.append({
            "url":        url,
            "title":      a.get("webTitle", "") or "",
            "summary":    fields.get("trailText", "") or "",
            "date":       pub,
            "publisher":  "The Guardian",
            "provenance": "guardian",
        })
    return out


# ---------------------------------------------------------------------------
# Source 2: Google News RSS
# ---------------------------------------------------------------------------

def fetch_google_news(query: str, forecast_point: datetime, lookback_days: int,
                      max_records: int = 25) -> list[dict]:
    if not feedparser:            # feedparser not installed; soft-skip Google News
        return []
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    start = forecast_point - timedelta(days=lookback_days)
    end = forecast_point
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"  [WARN] Google News: {e}")
        return []
    out = []
    for entry in getattr(feed, "entries", [])[: max_records * 3]:
        link = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        if not link:
            continue
        pub = getattr(entry, "published_parsed", None)
        pub_dt = None
        if pub:
            try:
                pub_dt = datetime(*pub[:6])
            except Exception:
                pass
        if pub_dt is None or not (start <= pub_dt < end):
            continue
        out.append({
            "url":        link,
            "title":      title,
            "summary":    getattr(entry, "summary", "") or "",
            "date":       pub_dt.strftime("%Y-%m-%d"),
            "provenance": "google-news",
        })
        if len(out) >= max_records:
            break
    return out


# ---------------------------------------------------------------------------
# Source 3: GDELT DOC (time-sliced)
# ---------------------------------------------------------------------------

def _gdelt_query(query: str, sd: str, ed: str, max_records: int, fd_id: str) -> list[dict]:
    params = {
        "query":         query,
        "mode":          "ArtList",
        "maxrecords":    max_records,
        "startdatetime": sd,
        "enddatetime":   ed,
        "format":        "json",
        "sort":          "DateDesc",
    }
    for attempt in range(3):
        try:
            r = requests.get(GDELT_DOC_API, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json().get("articles", [])
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] GDELT {fd_id} {sd}..{ed}: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def fetch_gdelt(query: str, forecast_point: datetime, lookback_days: int,
                max_records: int, n_slices: int = 3, fd_id: str = "") -> list[dict]:
    total_end = forecast_point - timedelta(days=1)
    slice_days = max(1, lookback_days // n_slices)
    per_slice = max(3, max_records // n_slices)
    out = []
    seen: set[str] = set()
    for i in range(n_slices):
        slice_end = total_end - timedelta(days=i * slice_days)
        slice_start = slice_end - timedelta(days=slice_days - 1)
        if slice_start < forecast_point - timedelta(days=lookback_days):
            slice_start = forecast_point - timedelta(days=lookback_days)
        sd = slice_start.strftime("%Y%m%d000000")
        ed = slice_end.strftime("%Y%m%d235959")
        articles = _gdelt_query(query, sd, ed, per_slice, fd_id)
        time.sleep(0.7)
        for a in articles:
            url = a.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            seendate = a.get("seendate", "")
            date_str = ""
            if seendate:
                try:
                    date_str = datetime.strptime(seendate[:8], "%Y%m%d").strftime("%Y-%m-%d")
                except ValueError:
                    pass
            out.append({
                "url":        url,
                "title":      a.get("title", "") or "",
                "summary":    "",
                "date":       date_str,
                "provenance": "gdelt-fb",
            })
    return out


# ---------------------------------------------------------------------------
# FD loading
# ---------------------------------------------------------------------------

def load_fb_fds(cutoff: str | None = None) -> list[dict]:
    """Load FB FDs from the unified forecasts file. We want all 530 original
    FDs (including those that may currently have zero article_ids) so we can
    enrich them."""
    # Prefer unified forecasts, fall back to published benchmark
    unified = DATA / "unified" / "forecasts.jsonl"
    if unified.exists():
        src = unified
    elif cutoff:
        src = ROOT / "benchmark" / "data" / cutoff / "forecasts.jsonl"
    else:
        # Try the staged snapshots as last resort
        staged = sorted((DATA / "staged").glob("*/01_after_first_unify/forecasts.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if not staged:
            raise FileNotFoundError("No FD source found")
        src = staged[0]
    print(f"[load] reading FDs from {src}")
    out = []
    for line in src.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("benchmark") == "forecastbench":
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["all", "gdelt", "google", "guardian", "nyt"], default="all")
    ap.add_argument("--lookback", type=int, default=90,
                    help="Days before forecast_point to search (default 90 — unified analysis window)")
    ap.add_argument("--max-gdelt", type=int, default=18)
    ap.add_argument("--max-guardian", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N FDs (debug)")
    ap.add_argument("--fd-ids", default=None,
                    help="Comma-separated FD id whitelist.")
    ap.add_argument("--skip-completed", action="store_true",
                    help="Skip FDs that already have any article in the output file.")
    ap.add_argument("--cutoff", default=None,
                    help="Benchmark cutoff (e.g. 2026-01-01) for FD-source lookup.")
    args = ap.parse_args()

    fds = load_fb_fds(args.cutoff)
    if args.fd_ids:
        wl = {x.strip() for x in args.fd_ids.split(",") if x.strip()}
        fds = [fd for fd in fds if fd["id"] in wl]
    if args.skip_completed and OUT_FILE.exists():
        done: set[str] = set()
        for line in OUT_FILE.open(encoding="utf-8"):
            try: done.add(json.loads(line).get("fd_id", ""))
            except: pass
        before = len(fds)
        fds = [fd for fd in fds if fd["id"] not in done]
        print(f"[filter] --skip-completed dropped {before - len(fds)} FDs")
    if args.limit:
        fds = fds[:args.limit]

    print(f"Processing {len(fds)} FB FDs  lookback={args.lookback}d  sources={args.source}")

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()  # per-FD title dedup across sources
    if OUT_FILE.exists():
        for line in OUT_FILE.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                seen_urls.add(rec.get("url", ""))
                t = (rec.get("title") or "").strip().lower()[:80]
                if t:
                    seen_titles.add(f"{rec.get('fd_id','')}::{t}")
            except: pass
        print(f"existing: {len(seen_urls)} unique URLs, {len(seen_titles)} url+title pairs")

    def _url_key(url: str) -> str:
        """Normalize URL for cross-source dedup: strip query/fragment, lowercase host."""
        from urllib.parse import urlparse, urlunparse
        try:
            p = urlparse(url)
            return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
        except Exception:
            return url

    written = 0
    per_prov = Counter()
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for i, fd in enumerate(fds):
            fd_id = fd["id"]
            fp_str = fd.get("forecast_point")
            try:
                fp_dt = datetime.strptime(fp_str, "%Y-%m-%d")
            except Exception:
                continue
            query = extract_keywords(fd.get("question", ""), fd.get("background", ""))
            if not query:
                continue
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(fds)}] {fd_id}  q={query!r}  written={written}  prov={dict(per_prov)}")

            collected: list[dict] = []
            if args.source in ("all", "nyt"):
                collected.extend(fetch_nyt(query, fp_dt, args.lookback))
            if args.source in ("all", "guardian"):
                collected.extend(fetch_guardian(query, fp_dt, args.lookback, args.max_guardian))
            if args.source in ("all", "google"):
                collected.extend(fetch_google_news(query, fp_dt, args.lookback))
            if args.source in ("all", "gdelt"):
                collected.extend(fetch_gdelt(query, fp_dt, args.lookback, args.max_gdelt, fd_id=fd_id))

            for art in collected:
                url = art["url"]
                if not url:
                    continue
                # v2.2 leakage guard: publish_date must not exceed the FD's
                # forecast_point. The GDELT DOC API, Google News proxy URLs,
                # and NYT occasionally surface future-dated items; re-assert
                # here rather than trusting the source-side filter.
                _pd = art.get("date", "") or ""
                if _pd:
                    try:
                        _pd_dt = datetime.strptime(_pd[:10], "%Y-%m-%d")
                        if _pd_dt > fp_dt:
                            per_prov["__dropped_leakage__"] += 1
                            continue
                    except ValueError:
                        pass
                if is_spam_url(url):
                    per_prov["__dropped_spam__"] += 1
                    continue
                norm_url = _url_key(url)
                if url in seen_urls or norm_url in seen_urls:
                    per_prov["__dedup_url__"] += 1
                    continue
                title_key = (art.get("title") or "").strip().lower()[:80]
                fd_title_key = f"{fd_id}::{title_key}" if title_key else ""
                if fd_title_key and fd_title_key in seen_titles:
                    per_prov["__dedup_title__"] += 1
                    continue
                seen_urls.add(url); seen_urls.add(norm_url)
                if fd_title_key:
                    seen_titles.add(fd_title_key)
                source_disp = art.get("publisher") or domain_of(url)
                rec = {
                    "id":          art_id(url),
                    "fd_id":       fd_id,
                    "query":       query,
                    "url":         url,
                    "title":       art.get("title", ""),
                    "text":        art.get("summary", "") or "",
                    "date":        art.get("date", "") or "",
                    "source":      source_disp,
                    "provenance":  art["provenance"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_prov[art["provenance"]] += 1

    print(f"\nDone. Wrote {written} new articles to {OUT_FILE.relative_to(ROOT)}")
    for p, n in sorted(per_prov.items(), key=lambda x: -x[1]):
        print(f"  {p:20s}: {n}")
    _leak = per_prov.get("__dropped_leakage__", 0)
    _total_seen = sum(per_prov.values())
    print(f"[forecastbench] leakage-filtered: {_leak} of {_total_seen} articles")


if __name__ == "__main__":
    main()
