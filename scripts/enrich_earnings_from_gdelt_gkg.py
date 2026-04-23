"""
Source #3: GDELT 2.0 news-corpus URL/slug ticker match.

The task originally specified scanning `data_kg.csv` V2Organizations. However
the version of `data_kg.csv` produced by the project build is a pre-aggregated
country-pair CAMEO event table (DateStr, Actor1CountryCode, ..., Docids) and
does NOT contain a V2Organizations column, and `data_news.csv` has empty
Title/Abstract/Text for every row (only URL + Date). As a pragmatic fallback
we scan the URL slug for ticker word-boundary or company-name substring
matches (news URLs frequently include slugs like
"apple-q4-earnings-beat-estimates") and join to FDs by ticker and date window
[report_date - lookback, report_date).

Writes to data/earnings/earnings_articles.jsonl in append mode with URL-dedup.
Atomic: write to .tmp then concat then rename.

Usage:
  python scripts/enrich_earnings_from_gdelt_gkg.py
  python scripts/enrich_earnings_from_gdelt_gkg.py --lookback 60
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse, unquote

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.common.spam_domains import is_spam_url  # noqa: E402

# Reuse the stoplists from Source #1 for a consistent ticker/name gate.
from scripts.enrich_earnings_from_unified import (  # noqa: E402
    HIGH_COLLISION_TICKERS,
    COMPANY_NAME_STOPLIST,
    FINANCE_CONTEXT_WORDS,
    normalize_company,
    parse_date,
)

DATA = ROOT / "data"
FD_FILE = DATA / "earnings" / "earnings_forecasts.jsonl"
OUT_FILE = DATA / "earnings" / "earnings_articles.jsonl"
NEWS_CSV = DATA / "gdelt_cameo" / "data_news.csv"

csv.field_size_limit(2**28)


def art_id(url: str) -> str:
    return "earn_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def url_key(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def slug_of(url: str) -> str:
    """Extract a searchable slug from a URL: path + query with hyphens -> spaces."""
    try:
        p = urlparse(url)
        path = unquote(p.path + " " + p.query)
        path = re.sub(r"[-_/]+", " ", path)
        return path.lower()
    except Exception:
        return url.lower()


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


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


def build_index(fds: list[dict], lookback: int):
    ticker_to_fds: dict[str, list] = defaultdict(list)
    name_to_ticker: dict[str, str] = {}
    tickers: set[str] = set()

    for fd in fds:
        m = fd["_earnings_meta"]
        t = m["ticker"].upper()
        rd = parse_date(m.get("report_date") or "")
        if rd is None:
            continue
        start = rd - timedelta(days=lookback)
        end = rd
        ticker_to_fds[t].append((fd["id"], t, start, end))
        tickers.add(t)
        cname = normalize_company(m.get("company") or "")
        if not cname or cname in COMPANY_NAME_STOPLIST:
            continue
        is_multi = " " in cname
        if (is_multi and len(cname) >= 6) or (not is_multi and len(cname) >= 7):
            name_to_ticker.setdefault(cname, t)

    safe = [t for t in tickers if t not in HIGH_COLLISION_TICKERS]
    # Lowercase word-boundary regex for slug matching (slug is already lowercase).
    ticker_re = re.compile(r"\b(" + "|".join(sorted(
        [t.lower() for t in safe], key=len, reverse=True)) + r")\b")
    ticker_case = {t.lower(): t for t in safe}
    return ticker_to_fds, ticker_re, ticker_case, name_to_ticker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not NEWS_CSV.exists():
        print(f"[ERROR] {NEWS_CSV} not found.")
        sys.exit(1)

    fds = load_fds()
    print(f"[fds] {len(fds)} earnings FDs")
    ticker_to_fds, ticker_re, ticker_case, name_to_ticker = build_index(fds, args.lookback)
    print(f"[index] {len(ticker_to_fds)} tickers, {len(name_to_ticker)} name keys")

    seen_urls = load_existing_urls()
    print(f"[dedup] {len(seen_urls)} pre-existing URL keys")

    tmp = OUT_FILE.with_suffix(".gdelt_gkg.tmp")
    scanned = 0
    written = 0
    matched = 0
    dedup_url = 0
    spam = 0
    fd_hits: set[str] = set()
    per_ticker: Counter = Counter()

    with tmp.open("w", encoding="utf-8") as fout, \
         NEWS_CSV.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            if args.limit is not None and scanned >= args.limit:
                break
            scanned += 1
            if scanned % 20000 == 0:
                print(f"  [scan {scanned}] matched={matched} written={written} "
                      f"fds_hit={len(fd_hits)}")
            url = (row.get("URL") or "").strip()
            date = (row.get("Date") or "").strip()
            if not url or not date:
                continue
            if is_spam_url(url):
                spam += 1
                continue
            slug = slug_of(url)
            if not slug:
                continue

            hits: set[str] = set()
            # Name hits
            for cname, tk in name_to_ticker.items():
                if cname in slug:
                    hits.add(tk)
            # Ticker hits in slug; require finance context word too
            tk_matches = set(ticker_re.findall(slug))
            if tk_matches:
                has_ctx = any(w in slug for w in FINANCE_CONTEXT_WORDS)
                if has_ctx:
                    for lc in tk_matches:
                        hits.add(ticker_case[lc])
            if not hits:
                continue
            matched += 1

            pub_dt = parse_date(date)
            if pub_dt is None:
                continue

            for tk in hits:
                for fd_id, _, start, end in ticker_to_fds.get(tk, []):
                    if not (start <= pub_dt < end):
                        continue
                    if url in seen_urls or url_key(url) in seen_urls:
                        composite = f"{fd_id}::{url_key(url)}"
                        if composite in seen_urls:
                            dedup_url += 1
                            continue
                        seen_urls.add(composite)
                    seen_urls.add(url); seen_urls.add(url_key(url))

                    rec = {
                        "id":          art_id(url + "::" + fd_id),
                        "question_id": fd_id,
                        "ticker":      tk,
                        "url":         url,
                        "title":       "",
                        "text":        "",
                        "date":        pub_dt.strftime("%Y-%m-%d"),
                        "source":      domain_of(url),
                        "provenance":  "gdelt-news-slug",
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    per_ticker[tk] += 1
                    fd_hits.add(fd_id)

    if written > 0:
        with OUT_FILE.open("a", encoding="utf-8") as afn, tmp.open("r", encoding="utf-8") as rfn:
            for line in rfn:
                afn.write(line)
    tmp.unlink(missing_ok=True)

    print()
    print("=== Source #3 (GDELT news-slug match) summary ===")
    print(f"  scanned rows        : {scanned}")
    print(f"  spam dropped        : {spam}")
    print(f"  articles matched    : {matched}")
    print(f"  records written     : {written}")
    print(f"  dedup (url)         : {dedup_url}")
    print(f"  FDs covered (delta) : {len(fd_hits)}")
    print(f"  top tickers         : {per_ticker.most_common(10)}")
    total = sum(1 for _ in OUT_FILE.open(encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"  total in pool now   : {total}")


if __name__ == "__main__":
    main()
