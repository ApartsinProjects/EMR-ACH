"""
Source #4: SEC EDGAR Full-Text Search (10-Q, 10-K, DEF 14A).

Queries the SEC EDGAR full-text search API (efts.sec.gov) per ticker for
filings of form 10-Q, 10-K, DEF 14A within [forecast_point - lookback, forecast_point).
Writes matching filings as article records into data/earnings/earnings_articles.jsonl
in append mode with URL-dedup.

Leakage note: the filing that IS the current-quarter earnings release (10-Q or
10-K for the current reporting period) would leak ground truth if its filing
date happens to be < forecast_point. We keep the strict half-open window
[start, forecast_point) to exclude any same-day-or-later filing. For 10-K and
DEF 14A the information is stale relative to the current quarter so leakage
risk is low. For 10-Q, within the window these are PRIOR-quarter 10-Qs; the
current-quarter 10-Q is by definition filed ON or AFTER the report_date.

API docs: https://efts.sec.gov/LATEST/search-index

Usage:
  python scripts/enrich_earnings_from_edgar_fts.py
  python scripts/enrich_earnings_from_edgar_fts.py --limit 10
  python scripts/enrich_earnings_from_edgar_fts.py --forms 10-Q,DEF+14A
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
from urllib.parse import urlparse, urlunparse

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.common.spam_domains import is_spam_url  # noqa: E402

DATA = ROOT / "data"
FD_FILE = DATA / "earnings" / "earnings_forecasts.jsonl"
OUT_FILE = DATA / "earnings" / "earnings_articles.jsonl"

SEC_FTS_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_UA = "EMR-ACH research (research@emr-ach.example)"

_TICKER_CIK: dict[str, int] | None = None


def load_ticker_to_cik() -> dict[str, int]:
    global _TICKER_CIK
    if _TICKER_CIK is not None:
        return _TICKER_CIK
    try:
        r = requests.get(SEC_TICKERS_URL, headers={"User-Agent": SEC_UA}, timeout=TIMEOUT)
        data = r.json()
        _TICKER_CIK = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    except Exception as e:
        print(f"[WARN] ticker map fetch failed: {e}")
        _TICKER_CIK = {}
    return _TICKER_CIK
DEFAULT_FORMS = "10-Q,10-K,DEF 14A"
TIMEOUT = 20


def art_id(url: str) -> str:
    return "earn_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def url_key(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[: len(fmt) - len(fmt.replace("%", ""))] if False else s[:10], "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(s[:8], "%Y%m%d")
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
                    seen.add(u)
                    seen.add(url_key(u))
            except Exception:
                pass
    return seen


def fts_search(ticker: str, forms: str, start: datetime, end: datetime,
               session: requests.Session, target_cik: int | None) -> list[dict]:
    """Query SEC EDGAR FTS for filings of the given forms by ticker in window.

    If target_cik is provided, the `ciks` param restricts results to filings
    submitted by that CIK (the ticker's own filings only), which is what we
    want for earnings-relevant filings.

    Returns dicts with url, title, date, accession.
    """
    params = {
        "q": f'"{ticker}"',
        "forms": forms,
        "dateRange": "custom",
        "startdt": start.strftime("%Y-%m-%d"),
        "enddt": (end - timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    if target_cik is not None:
        params["ciks"] = f"{target_cik:010d}"
    try:
        r = session.get(SEC_FTS_URL, params=params,
                        headers={"User-Agent": SEC_UA}, timeout=TIMEOUT)
    except Exception as e:
        print(f"  [WARN] FTS request failed for {ticker}: {e}")
        return []
    if r.status_code != 200:
        print(f"  [WARN] FTS {ticker} status {r.status_code}")
        return []
    try:
        data = r.json()
    except Exception:
        return []
    hits = data.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {})
        fdate = src.get("file_date") or ""
        form = src.get("form") or src.get("root_forms", [""])[0]
        adsh = src.get("adsh") or ""
        display_names = src.get("display_names", [])
        ciks = src.get("ciks", [])
        # id format: "<adsh>:<primary_doc>"
        hit_id = h.get("_id", "")
        primary_doc = hit_id.split(":", 1)[1] if ":" in hit_id else ""
        cik = ciks[0] if ciks else ""
        if not (fdate and adsh and cik):
            continue
        acc_nodash = adsh.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_doc}"
        company = display_names[0] if display_names else ""
        title = f"{form} filing ({ticker}): {company}".strip()
        out.append({
            "url":   url,
            "title": title,
            "date":  fdate,
            "form":  form,
            "accession": adsh,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--forms", default=DEFAULT_FORMS,
                    help=f"Comma-separated form list (default: {DEFAULT_FORMS!r})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N FDs (smoke)")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="Sleep between API calls (default 0.3s)")
    args = ap.parse_args()

    fds = load_fds()
    if args.tickers:
        wl = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        fds = [fd for fd in fds if fd["_earnings_meta"]["ticker"].upper() in wl]
    if args.limit:
        fds = fds[: args.limit]
    print(f"[fds] processing {len(fds)} FDs, forms={args.forms!r}, "
          f"lookback={args.lookback}d")

    seen_urls = load_existing_urls()
    print(f"[dedup] {len(seen_urls)} pre-existing URL keys")

    session = requests.Session()
    tk_cik = load_ticker_to_cik()
    print(f"[cik-map] {len(tk_cik)} tickers mapped")
    tmp_path = OUT_FILE.with_suffix(".edgar_fts.tmp")
    written = 0
    dedup_url = 0
    errors = 0
    fd_hits: set[str] = set()
    per_form: Counter = Counter()

    # Convert the task's "DEF 14A" to the URL-safe "DEF+14A" for the forms arg
    forms_param = args.forms.replace(" ", "+")

    with tmp_path.open("w", encoding="utf-8") as fout:
        for i, fd in enumerate(fds):
            m = fd["_earnings_meta"]
            ticker = m["ticker"]
            fp = fd.get("forecast_point") or m.get("report_date")
            fp_dt = parse_date(fp)
            if fp_dt is None:
                continue
            start = fp_dt - timedelta(days=args.lookback)
            end = fp_dt  # exclusive

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(fds)}] ticker={ticker} written={written} "
                      f"fds_hit={len(fd_hits)} errors={errors}")

            cik = tk_cik.get(ticker.upper())
            try:
                results = fts_search(ticker, forms_param, start, end, session, cik)
            except Exception as e:
                errors += 1
                print(f"  [ERR] {ticker}: {e}")
                time.sleep(1.0)
                continue
            time.sleep(args.sleep)

            for r in results:
                fd_dt = parse_date(r["date"])
                if fd_dt is None:
                    continue
                # Strict half-open window; exclude filings on or after forecast_point
                if not (start <= fd_dt < end):
                    continue
                url = r["url"]
                if not url or is_spam_url(url):
                    continue
                if url in seen_urls or url_key(url) in seen_urls:
                    dedup_url += 1
                    continue
                seen_urls.add(url); seen_urls.add(url_key(url))

                rec = {
                    "id":          art_id(url + "::" + fd["id"]),
                    "question_id": fd["id"],
                    "ticker":      ticker,
                    "url":         url,
                    "title":       r["title"],
                    "text":        f"SEC Form {r['form']} filed by {ticker} on "
                                  f"{r['date']}. Accession {r['accession']}.",
                    "date":        fd_dt.strftime("%Y-%m-%d"),
                    "source":      "SEC EDGAR",
                    "provenance":  "sec-edgar-fts",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_form[r["form"]] += 1
                fd_hits.add(fd["id"])

    if written > 0:
        with OUT_FILE.open("a", encoding="utf-8") as appendf, \
             tmp_path.open("r", encoding="utf-8") as readf:
            for line in readf:
                appendf.write(line)
    tmp_path.unlink(missing_ok=True)

    print()
    print(f"=== Source #4 (SEC EDGAR FTS) summary ===")
    print(f"  FDs processed        : {len(fds)}")
    print(f"  records written      : {written}")
    print(f"  dedup (url)          : {dedup_url}")
    print(f"  errors               : {errors}")
    print(f"  FDs covered (delta)  : {len(fd_hits)}")
    print(f"  per-form             : {dict(per_form)}")
    final_total = sum(1 for _ in OUT_FILE.open(encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"  total in pool now    : {final_total}")


if __name__ == "__main__":
    main()
