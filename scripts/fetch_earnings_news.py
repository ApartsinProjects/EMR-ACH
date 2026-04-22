"""
Fetch earnings-specific news for each FD in data/earnings/earnings_forecasts.jsonl.

Two sources per FD:
  1. Yahoo Finance (yfinance.Ticker.news) — ticker-tagged editorial news
  2. GDELT DOC API — broader wire coverage via keyword query

Filters articles to publish_date in [report_date - lookback_days, report_date),
which matches the FD's forecast window (no leakage past forecast_point).

Reads:
  data/earnings/earnings_forecasts.jsonl  - each record has _earnings_meta with
                                            ticker, company, report_date

Writes:
  data/earnings/earnings_articles.jsonl   - standard article schema matching
                                            data/gdelt_articles.jsonl

Output schema (per line):
  {
    "id":       "earn_<sha1(url)[:12]>",
    "question_id": "<FD id>",
    "ticker":   "AAPL",
    "url":      "...",
    "title":    "...",
    "text":     "...",         # full text via trafilatura (may be empty)
    "date":     "2026-01-15",
    "source":   "domain.tld",
    "provenance": "yfinance" | "gdelt-earnings",
  }

Usage:
  python scripts/fetch_earnings_news.py                 # all FDs, both sources
  python scripts/fetch_earnings_news.py --source yfinance
  python scripts/fetch_earnings_news.py --lookback 30 --max-gdelt 15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.common.optional_imports import optional  # noqa: E402

yf = optional("yfinance")           # only needed for the yfinance source
trafilatura = optional("trafilatura")  # only needed when --fetch-text is on

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FD_FILE = DATA / "earnings" / "earnings_forecasts.jsonl"
OUT_FILE = DATA / "earnings" / "earnings_articles.jsonl"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 12


def art_id(url: str) -> str:
    return "earn_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


# Spam blocklist + is_spam_url are shared across all three news fetchers.
# See src/common/spam_domains.py for the authoritative list.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from src.common.spam_domains import is_spam_url  # noqa: E402


def url_key(url: str) -> str:
    """Normalize URL for cross-source dedup: drop query/fragment, lowercase host."""
    try:
        from urllib.parse import urlparse as _up, urlunparse as _uu
        p = _up(url)
        return _uu((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def fetch_text(url: str) -> str:
    if not trafilatura:           # module not installed; soft-skip full-text fetch
        return ""
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


# ---------------------------------------------------------------------------
# Source 1: Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_yfinance_news(ticker: str, forecast_point: datetime, lookback_days: int) -> list[dict]:
    """Pull ticker.news, filter to the pre-forecast_point window.

    Note: yfinance.Ticker.news returns only the most recent ~10 items, with no
    historical lookup API. For FDs whose forecast_point is weeks/months in the
    past, yfinance typically returns 0 matches. Use only as supplement.
    """
    if not yf:                    # yfinance not installed; soft-skip
        return []
    start = forecast_point - timedelta(days=lookback_days)
    end = forecast_point  # strict < forecast_point (leakage guard)
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"  [WARN] yfinance news failed for {ticker}: {e}")
        return []
    out = []
    for it in items:
        # yfinance returns either {content: {...}} (newer) or flat dict (older)
        content = it.get("content") if isinstance(it.get("content"), dict) else it
        url = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else None
        url = url or content.get("clickThroughUrl", {}).get("url") if isinstance(content.get("clickThroughUrl"), dict) else url
        url = url or content.get("link") or it.get("link", "")
        if not url:
            continue
        title = content.get("title") or it.get("title", "") or ""
        summary = content.get("summary") or content.get("description") or it.get("summary", "") or ""
        # publish time: unix seconds (older) or ISO string (newer)
        pub = content.get("pubDate") or content.get("providerPublishTime") or it.get("providerPublishTime")
        pub_dt = None
        if isinstance(pub, (int, float)):
            pub_dt = datetime.utcfromtimestamp(int(pub))
        elif isinstance(pub, str):
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
                try:
                    pub_dt = datetime.strptime(pub[:len(fmt.replace("%Y", "2026"))], fmt)
                    break
                except ValueError:
                    continue
        if pub_dt is None or not (start <= pub_dt < end):
            continue
        out.append({
            "url":      url,
            "title":    title,
            "summary":  summary,
            "date":     pub_dt.strftime("%Y-%m-%d"),
            "provenance": "yfinance",
        })
    return out


# ---------------------------------------------------------------------------
# Source 1a: Finnhub /company-news (finance-first, historical, free API key)
# ---------------------------------------------------------------------------

FINNHUB_API = "https://finnhub.io/api/v1/company-news"


def fetch_finnhub(ticker: str, forecast_point: datetime, lookback_days: int) -> list[dict]:
    """Finnhub company-news endpoint. Requires FINNHUB_API_KEY env var.
    Returns finance-focused news items with date-range filtering at the API level.
    Free tier: 60 calls/min, unlimited/day. Stricly < forecast_point.
    """
    import os
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        return []
    end = forecast_point - timedelta(days=1)
    start = forecast_point - timedelta(days=lookback_days)
    params = {
        "symbol": ticker,
        "from":   start.strftime("%Y-%m-%d"),
        "to":     end.strftime("%Y-%m-%d"),
        "token":  key,
    }
    try:
        r = requests.get(FINNHUB_API, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  [WARN] Finnhub {ticker} status={r.status_code}")
            return []
        items = r.json() or []
    except Exception as e:
        print(f"  [WARN] Finnhub {ticker}: {e}")
        return []
    out = []
    for it in items:
        url = it.get("url") or ""
        if not url:
            continue
        # Finnhub's own relevance signal: `related` is a comma-sep ticker list.
        # Keep only items where the queried ticker is in `related`.
        related = (it.get("related") or "").upper()
        if ticker.upper() not in related.split(","):
            continue
        # Category filter: 'company' is earnings-relevant; skip 'general'/'top news'.
        cat = (it.get("category") or "").lower()
        if cat not in ("company", "earnings"):
            continue
        ts = it.get("datetime")
        date_str = ""
        if ts:
            try:
                date_str = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
            except Exception:
                pass
        # Use Finnhub's `source` (actual publisher: Yahoo, Benzinga, SeekingAlpha),
        # not the proxy URL's finnhub.io domain.
        publisher = (it.get("source") or "").strip() or domain_of(url)
        out.append({
            "url":           url,
            "title":         it.get("headline", "") or "",
            "summary":       it.get("summary", "") or "",
            "date":          date_str,
            "publisher":     publisher,   # carried through to record.source below
            "provenance":    "finnhub",
        })
    return out


# ---------------------------------------------------------------------------
# Source 1b: Google News RSS (historical, no API key, unlimited)
# ---------------------------------------------------------------------------

def fetch_google_news(ticker: str, company: str, forecast_point: datetime,
                       lookback_days: int, max_records: int = 30) -> list[dict]:
    """Query Google News RSS for (company OR ticker) news in the pre-earnings
    window. No API key required. Google doesn't expose date-range in the RSS
    URL cleanly, so we filter client-side by pubDate.

    Note: Google News RSS returns up to ~100 items and often includes articles
    older than the requested window; we filter locally.
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        return []
    from urllib.parse import quote_plus
    clean_company = company.replace('"', '').split(',')[0].strip()
    query = quote_plus(f'"{clean_company}" OR {ticker} earnings')
    url = (
        f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )
    start = forecast_point - timedelta(days=lookback_days)
    end = forecast_point
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"  [WARN] Google News failed for {ticker}: {e}")
        return []
    out = []
    for entry in getattr(feed, "entries", [])[: max_records * 3]:
        link = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        if not link:
            continue
        # Parse pubDate: "Mon, 12 Jan 2026 14:30:00 GMT"
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
# Source 2: GDELT DOC
# ---------------------------------------------------------------------------

def _gdelt_query(query: str, sd: str, ed: str, max_records: int, ticker: str) -> list[dict]:
    """Single GDELT DOC call with 3-attempt retry."""
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
            r = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("articles", [])
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] GDELT failed for {ticker} {sd}..{ed}: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def _run_slices(query: str, forecast_point: datetime, lookback_days: int,
                max_records: int, n_slices: int, ticker: str) -> list[dict]:
    """Run the given GDELT query across n_slices time windows within
    [forecast_point - lookback_days, forecast_point). Returns deduplicated
    article records (url, title, date, provenance='gdelt-earnings')."""
    total_end = forecast_point - timedelta(days=1)
    slice_days = max(1, lookback_days // n_slices)
    per_slice = max(3, max_records // n_slices)
    out = []
    seen_urls: set[str] = set()
    for i in range(n_slices):
        slice_end = total_end - timedelta(days=i * slice_days)
        slice_start = slice_end - timedelta(days=slice_days - 1)
        if slice_start < forecast_point - timedelta(days=lookback_days):
            slice_start = forecast_point - timedelta(days=lookback_days)
        sd = slice_start.strftime("%Y%m%d000000")
        ed = slice_end.strftime("%Y%m%d235959")
        articles = _gdelt_query(query, sd, ed, per_slice, ticker)
        time.sleep(0.7)
        for a in articles:
            url = a.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            seen = a.get("seendate", "")
            date_str = ""
            if seen:
                try:
                    date_str = datetime.strptime(seen[:8], "%Y%m%d").strftime("%Y-%m-%d")
                except ValueError:
                    pass
            out.append({
                "url":        url,
                "title":      a.get("title", "") or "",
                "summary":    "",
                "date":       date_str,
                "provenance": "gdelt-earnings",
            })
    return out


def fetch_gdelt(ticker: str, company: str, forecast_point: datetime,
                lookback_days: int, max_records: int, n_slices: int = 3,
                fallback_threshold: int = 5) -> list[dict]:
    """Fetch pre-earnings news. Tries three queries in escalating breadth,
    stopping as soon as we have enough articles:

      1. Primary:  ("<company>" OR <ticker>) earnings       (tight, finance-focused)
      2. Fallback: ("<company>" OR <ticker>)                (drops "earnings" keyword)
      3. Fallback: "<company>"                              (company name alone)

    Each fallback is only triggered if the previous query returned fewer than
    `fallback_threshold` articles. Strictly no articles on/after forecast_point.
    """
    clean_company = company.replace('"', '').split(',')[0].strip()
    # Primary: tightest query
    q1 = f'("{clean_company}" OR {ticker}) earnings'
    arts = _run_slices(q1, forecast_point, lookback_days, max_records, n_slices, ticker)
    if len(arts) >= fallback_threshold:
        return arts

    # Fallback 1: drop "earnings" keyword — broader company news
    q2 = f'("{clean_company}" OR {ticker})'
    print(f"    [fallback-1] {ticker}: primary returned {len(arts)} arts, trying broader query")
    extra = _run_slices(q2, forecast_point, lookback_days, max_records, n_slices, ticker)
    seen = {a["url"] for a in arts}
    arts.extend(a for a in extra if a["url"] not in seen)
    if len(arts) >= fallback_threshold:
        return arts

    # Fallback 2: company name alone (no ticker, no filter)
    q3 = f'"{clean_company}"'
    print(f"    [fallback-2] {ticker}: still {len(arts)} arts, trying company-name-only")
    extra = _run_slices(q3, forecast_point, lookback_days, max_records, n_slices, ticker)
    seen = {a["url"] for a in arts}
    arts.extend(a for a in extra if a["url"] not in seen)
    return arts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_fds() -> list[dict]:
    if not FD_FILE.exists():
        print(f"[ERROR] {FD_FILE} not found. Run build_earnings_benchmark.py first.")
        sys.exit(1)
    out = []
    for line in FD_FILE.open(encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("_earnings_meta"):
                out.append(r)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["all", "yfinance", "gdelt", "google", "finnhub"], default="all")
    ap.add_argument("--lookback", type=int, default=90,
                    help="Days before report_date to search (default 90 — unified 'forecast from prior news' window)")
    ap.add_argument("--max-gdelt", type=int, default=15,
                    help="Max GDELT articles per FD (default 15)")
    ap.add_argument("--fetch-text", action="store_true",
                    help="Fetch full article text via trafilatura (slow, ~1s/article)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N FDs (debug)")
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated ticker whitelist (e.g. 'WFC,GS,MS'). "
                         "Only FDs whose _earnings_meta.ticker is in the list will be processed. "
                         "Use to split work across parallel workers.")
    ap.add_argument("--skip-completed", action="store_true",
                    help="Skip FDs that already have any article in the output file.")
    args = ap.parse_args()

    fds = load_fds()
    if args.tickers:
        wl = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        fds = [fd for fd in fds if fd.get("_earnings_meta", {}).get("ticker", "").upper() in wl]
        print(f"[filter] --tickers kept {len(fds)} FDs matching {sorted(wl)}")
    if args.skip_completed and OUT_FILE.exists():
        done_fds: set[str] = set()
        for line in OUT_FILE.open(encoding="utf-8"):
            try:
                done_fds.add(json.loads(line).get("question_id", ""))
            except Exception:
                pass
        before = len(fds)
        fds = [fd for fd in fds if fd["id"] not in done_fds]
        print(f"[filter] --skip-completed dropped {before - len(fds)} already-fetched FDs")
    if args.limit:
        fds = fds[:args.limit]
    print(f"Processing {len(fds)} earnings FDs")
    print(f"Sources: {args.source}  lookback: {args.lookback}d  fetch_text: {args.fetch_text}")

    # Load existing URLs (raw + normalized) and per-FD title prefixes for cross-source dedup
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                u = rec.get("url", "")
                seen_urls.add(u); seen_urls.add(url_key(u))
                t = (rec.get("title") or "").strip().lower()[:80]
                if t:
                    seen_titles.add(f"{rec.get('question_id','')}::{t}")
            except Exception:
                pass
        print(f"Already have {len(seen_urls)} URL keys, {len(seen_titles)} title keys")

    written = 0
    per_prov_count = Counter()
    # (imports)
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for i, fd in enumerate(fds):
            meta = fd["_earnings_meta"]
            ticker = meta["ticker"]
            company = meta.get("company", ticker)
            # forecast_point is the leakage-guard boundary; fetch must end before it.
            fp_str = fd.get("forecast_point") or meta.get("report_date")
            try:
                forecast_point_dt = datetime.strptime(fp_str, "%Y-%m-%d")
            except Exception:
                continue

            fd_id = fd["id"]
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(fds)}] {fd_id}  ticker={ticker}  "
                      f"written so far: {written}  per_prov={dict(per_prov_count)}")

            collected: list[dict] = []
            if args.source in ("all", "finnhub"):
                collected.extend(fetch_finnhub(ticker, forecast_point_dt, args.lookback))
            if args.source in ("all", "yfinance"):
                collected.extend(fetch_yfinance_news(ticker, forecast_point_dt, args.lookback))
            if args.source in ("all", "google"):
                collected.extend(fetch_google_news(ticker, company, forecast_point_dt, args.lookback))
            if args.source in ("all", "gdelt"):
                collected.extend(fetch_gdelt(ticker, company, forecast_point_dt, args.lookback, args.max_gdelt))

            for art in collected:
                url = art["url"]
                if not url:
                    continue
                if is_spam_url(url):
                    per_prov_count["__dropped_spam__"] += 1
                    continue
                norm = url_key(url)
                if url in seen_urls or norm in seen_urls:
                    per_prov_count["__dedup_url__"] += 1
                    continue
                title_key = (art.get("title") or "").strip().lower()[:80]
                fd_title_key = f"{fd_id}::{title_key}" if title_key else ""
                if fd_title_key and fd_title_key in seen_titles:
                    per_prov_count["__dedup_title__"] += 1
                    continue
                seen_urls.add(url); seen_urls.add(norm)
                if fd_title_key:
                    seen_titles.add(fd_title_key)

                text = art.get("summary", "") or ""
                if args.fetch_text and url:
                    full = fetch_text(url)
                    if full and len(full) > len(text):
                        text = full

                # Prefer the explicit publisher (Finnhub provides one); otherwise
                # parse the URL. For finnhub.io proxy URLs the publisher field
                # holds the real outlet ("Yahoo", "SeekingAlpha", etc.).
                source = art.get("publisher") or domain_of(url)

                rec = {
                    "id":          art_id(url),
                    "question_id": fd_id,
                    "ticker":      ticker,
                    "url":         url,
                    "title":       art.get("title", "") or "",
                    "text":        text,
                    "date":        art.get("date", "") or "",
                    "source":      source,
                    "provenance":  art["provenance"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_prov_count[art["provenance"]] += 1

    print(f"\nDone. Wrote {written} new articles to {OUT_FILE.relative_to(ROOT)}")
    for prov, n in sorted(per_prov_count.items(), key=lambda x: -x[1]):
        print(f"  {prov:20s}: {n}")
    total = sum(1 for _ in OUT_FILE.open(encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"Total in file: {total} articles")


if __name__ == "__main__":
    main()
