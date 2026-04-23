"""
Source #2: NYT Article Search + Guardian Open Platform editorial fetch.

For each unique S&P 500 ticker x report_date in earnings_forecasts.jsonl,
queries NYT and Guardian for company-name + earnings-context query in the
window [report_date - lookback, report_date). Joins back to the FD by ticker
and writes to data/earnings/earnings_articles.jsonl in append mode with
URL-dedup.

NYT rate limits: ~10 req/min (sleep 6s between calls).
Guardian: 12 calls/sec, 5000/day; we sleep 0.3s for safety.

Usage:
  python scripts/enrich_earnings_from_editorial.py
  python scripts/enrich_earnings_from_editorial.py --source nyt --limit 5
  python scripts/enrich_earnings_from_editorial.py --source guardian
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.common.spam_domains import is_spam_url  # noqa: E402

DATA = ROOT / "data"
FD_FILE = DATA / "earnings" / "earnings_forecasts.jsonl"
OUT_FILE = DATA / "earnings" / "earnings_articles.jsonl"

NYT_API = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
GUARDIAN_API = "https://content.guardianapis.com/search"
TIMEOUT = 30


def _load_dotenv():
    """Best-effort load of E:/Projects/ACH/.env into os.environ."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


def art_id(url: str) -> str:
    return "earn_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def url_key(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10] if "-" in s else s[:8], fmt)
        except ValueError:
            continue
    return None


def load_fds() -> list[dict]:
    out = []
    for line in FD_FILE.open(encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("_earnings_meta", {}).get("ticker"):
                out.append(r)
        except Exception:
            pass
    return out


def load_existing_urls() -> set[str]:
    seen: set[str] = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.open(encoding="utf-8"):
            try:
                u = json.loads(line).get("url", "")
                if u:
                    seen.add(u); seen.add(url_key(u))
            except Exception:
                pass
    return seen


def clean_company(name: str) -> str:
    """Strip corporate suffixes for query construction."""
    n = name
    for suf in [
        ", Inc.", ", Inc", " Corporation", " Incorporated", " Inc.",
        " Inc", " Corp.", " Corp", " Company", " Co.", " plc",
        " Holdings", " Group",
    ]:
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip().rstrip(",.").strip()


# ---------------------------------------------------------------------------
# NYT
# ---------------------------------------------------------------------------

def fetch_nyt(company: str, ticker: str, fp: datetime, lookback: int,
              session: requests.Session, max_pages: int = 1) -> list[dict]:
    key = os.environ.get("NYT_API_KEY", "")
    if not key:
        return []
    start = fp - timedelta(days=lookback)
    end = fp - timedelta(days=1)
    q = f'"{company}" AND (earnings OR EPS OR revenue OR guidance OR quarterly)'
    out = []
    for page in range(max_pages):
        params = {
            "q":          q,
            "begin_date": start.strftime("%Y%m%d"),
            "end_date":   end.strftime("%Y%m%d"),
            "api-key":    key,
            "page":       page,
            "sort":       "relevance",
        }
        try:
            r = session.get(NYT_API, params=params, timeout=TIMEOUT)
        except Exception as e:
            print(f"  [WARN] NYT {ticker}: {e}")
            return out
        if r.status_code == 429:
            time.sleep(10)
            continue
        if r.status_code != 200:
            print(f"  [WARN] NYT {ticker} status {r.status_code}")
            break
        try:
            docs = r.json().get("response", {}).get("docs", [])
        except Exception:
            break
        if not docs:
            break
        for d in docs:
            url = d.get("web_url", "")
            if not url:
                continue
            head = (d.get("headline", {}) or {}).get("main", "") or ""
            abstract = d.get("abstract", "") or d.get("lead_paragraph", "") or ""
            pub = (d.get("pub_date", "") or "")[:10]
            out.append({
                "url":        url,
                "title":      head,
                "summary":    abstract,
                "date":       pub,
                "publisher":  "The New York Times",
                "provenance": "nyt",
            })
        time.sleep(6)  # NYT rate-limit
    return out


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------

def fetch_guardian(company: str, ticker: str, fp: datetime, lookback: int,
                   session: requests.Session, page_size: int = 15) -> list[dict]:
    key = os.environ.get("GUARDIAN_API_KEY", "")
    if not key:
        return []
    start = fp - timedelta(days=lookback)
    end = fp - timedelta(days=1)
    q = f'"{company}" AND (earnings OR revenue OR profit OR quarterly)'
    params = {
        "q":           q,
        "from-date":   start.strftime("%Y-%m-%d"),
        "to-date":     end.strftime("%Y-%m-%d"),
        "show-fields": "trailText,byline",
        "page-size":   page_size,
        "order-by":    "relevance",
        "api-key":     key,
    }
    try:
        r = session.get(GUARDIAN_API, params=params, timeout=TIMEOUT)
    except Exception as e:
        print(f"  [WARN] Guardian {ticker}: {e}")
        return []
    if r.status_code != 200:
        return []
    try:
        results = r.json().get("response", {}).get("results", [])
    except Exception:
        return []
    out = []
    for a in results:
        url = a.get("webUrl", "")
        if not url:
            continue
        pub = (a.get("webPublicationDate", "") or "")[:10]
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
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["all", "nyt", "guardian"], default="all")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    fds = load_fds()
    if args.tickers:
        wl = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        fds = [fd for fd in fds if fd["_earnings_meta"]["ticker"].upper() in wl]
    if args.limit:
        fds = fds[: args.limit]
    print(f"[fds] processing {len(fds)} FDs (source={args.source}, "
          f"lookback={args.lookback}d)")

    seen_urls = load_existing_urls()
    print(f"[dedup] {len(seen_urls)} pre-existing URL keys")

    nyt_key = os.environ.get("NYT_API_KEY", "")
    guard_key = os.environ.get("GUARDIAN_API_KEY", "")
    print(f"[keys] NYT={'yes' if nyt_key else 'NO'} Guardian={'yes' if guard_key else 'NO'}")

    session = requests.Session()
    tmp = OUT_FILE.with_suffix(".editorial.tmp")
    written = 0
    dedup_url = 0
    fd_hits: set[str] = set()
    per_prov: Counter = Counter()

    with tmp.open("w", encoding="utf-8") as fout:
        for i, fd in enumerate(fds):
            m = fd["_earnings_meta"]
            ticker = m["ticker"]
            company = clean_company(m.get("company") or ticker)
            fp = parse_date(fd.get("forecast_point") or m.get("report_date") or "")
            if fp is None:
                continue

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(fds)}] {ticker} written={written} "
                      f"fds_hit={len(fd_hits)} per_prov={dict(per_prov)}")

            collected: list[dict] = []
            if args.source in ("all", "guardian") and guard_key:
                collected.extend(fetch_guardian(company, ticker, fp, args.lookback, session))
                time.sleep(0.3)
            if args.source in ("all", "nyt") and nyt_key:
                collected.extend(fetch_nyt(company, ticker, fp, args.lookback, session))

            for art in collected:
                url = art["url"]
                if not url or is_spam_url(url):
                    continue
                pub_dt = parse_date(art.get("date") or "")
                if pub_dt is None:
                    continue
                # Strict pre-forecast_point window guard.
                if not (pub_dt < fp):
                    continue
                if url in seen_urls or url_key(url) in seen_urls:
                    composite = f"{fd['id']}::{url_key(url)}"
                    if composite in seen_urls:
                        dedup_url += 1
                        continue
                    seen_urls.add(composite)
                seen_urls.add(url); seen_urls.add(url_key(url))

                rec = {
                    "id":          art_id(url + "::" + fd["id"]),
                    "question_id": fd["id"],
                    "ticker":      ticker,
                    "url":         url,
                    "title":       art["title"],
                    "text":        art.get("summary", ""),
                    "date":        pub_dt.strftime("%Y-%m-%d"),
                    "source":      art.get("publisher") or domain_of(url),
                    "provenance":  art["provenance"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_prov[art["provenance"]] += 1
                fd_hits.add(fd["id"])

    if written > 0:
        with OUT_FILE.open("a", encoding="utf-8") as afn, tmp.open("r", encoding="utf-8") as rfn:
            for line in rfn:
                afn.write(line)
    tmp.unlink(missing_ok=True)

    print()
    print("=== Source #2 (NYT + Guardian editorial) summary ===")
    print(f"  FDs processed       : {len(fds)}")
    print(f"  records written     : {written}")
    print(f"  dedup (url)         : {dedup_url}")
    print(f"  FDs covered (delta) : {len(fd_hits)}")
    print(f"  per-provenance      : {dict(per_prov)}")
    total = sum(1 for _ in OUT_FILE.open(encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"  total in pool now   : {total}")


if __name__ == "__main__":
    main()
