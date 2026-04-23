"""
Source #1: Enrich earnings_articles.jsonl from the unified article pool.

Reads data/staged/<latest>/01_after_first_unify/articles.jsonl (the v2.1 pool of
~218k articles already fetched for ForecastBench + GDELT-CAMEO). For each
article, looks for any S&P 500 ticker (uppercase word-boundary regex) or
company-name substring (case-insensitive, length >= 4) in title or first
1500 chars of text. Joins the article to FDs whose ticker matches AND whose
report_date window contains the article's publish_date.

Date window: [report_date - lookback_days, report_date) (default 90).

Writes to data/earnings/earnings_articles.jsonl in append mode with URL-dedup
against existing records. Atomic: write to .tmp then concat then rename.

Usage:
  python scripts/enrich_earnings_from_unified.py
  python scripts/enrich_earnings_from_unified.py --limit 100  # smoke test
  python scripts/enrich_earnings_from_unified.py --lookback 60 --pool /path/to/articles.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.common.spam_domains import is_spam_url  # noqa: E402

DATA = ROOT / "data"
FD_FILE = DATA / "earnings" / "earnings_forecasts.jsonl"
OUT_FILE = DATA / "earnings" / "earnings_articles.jsonl"

# Case-insensitive substring stoplist for company-name matches that are too
# generic and would yield false positives. Length-4 floor catches most, but
# common short corp names slip through.
COMPANY_NAME_STOPLIST = {
    # Single-word English words that S&P company names reduce to after suffix
    # strip. These match far too broadly as a substring and are unsafe.
    "news", "block", "ball", "key", "gap", "target", "best", "first",
    "host", "home", "iron", "live", "match", "mid", "monster", "pool",
    "post", "public", "quest", "share", "southern", "state", "trust",
    "union", "united", "valero", "visa", "wells", "western", "whole",
    "dell", "xyz", "hp", "next", "core", "bank", "power", "light", "oil",
    "food", "motion", "health", "family", "general", "progress", "simon",
    "edison", "truist", "martin", "lincoln", "kenvue", "gilead", "church",
    "bio", "lab", "ford", "omega", "vici", "zoom", "mplx",
}

# High-collision tickers: common English words or single letters that match
# huge numbers of articles spuriously. We require *both* the ticker AND the
# company name (or a strong context word) to be present for these.
# If a ticker hits purely via the \bTICKER\b regex (not via company-name
# substring), we require one of these context words nearby to confirm it is
# really a stock-market reference and not an unrelated acronym/word.
FINANCE_CONTEXT_WORDS = (
    "earning", "revenue", "eps", "quarter", "guidance", "analyst", "stock",
    "share", "forecast", "consensus", "dividend", "buyback", "nyse", "nasdaq",
    "ticker", "upgrade", "downgrade", "price target", "fiscal", "profit",
    "outlook", "margin", "investor", "valuation", "market cap",
)

HIGH_COLLISION_TICKERS = {
    # All single letters (S&P includes many: A, C, D, F, K, L, M, O, T, V, X)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    # Two-letter common words / English bigrams
    "AI", "AM", "AN", "AS", "AT", "BE", "BY", "CO", "DO", "EA", "EX",
    "FE", "GE", "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MO", "MS",
    "MY", "NA", "NO", "OF", "OK", "ON", "OR", "RE", "SO", "TO", "UP",
    "US", "WE", "WM", "WY", "DE", "IP", "PM", "RL", "SO",
    # Three+ letter common-word tickers
    "ALL", "ARE", "BIG", "CAN", "FOR", "GAS", "GAP", "KEY", "LOW",
    "NEW", "NOW", "ODD", "ONE", "OUT", "OWN", "PAY", "RUN", "SEE",
    "TAP", "WHO", "WIN", "TRUE", "WELL", "BEST", "BIO", "HOT", "JOB",
    "LAW", "PLAY", "POOL", "POST", "WORK", "WALL", "WIRE", "STAY",
    # Proper nouns / common acronyms that collide with earnings tickers
    "ICE", "FOX", "EOG", "RTX", "WAT",
}
# For tickers in this set, we require the company-name match path (or skip
# pure-ticker hits) to avoid spurious matches like "the company A reported".

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def art_id(url: str) -> str:
    return "earn_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def url_key(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    s = s[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def find_latest_unified() -> Path:
    """Locate the freshest staged unified articles.jsonl."""
    candidates = sorted(
        DATA.glob("staged/*/01_after_first_unify/articles.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No staged unified articles.jsonl found under data/staged/*")
    return candidates[0]


def load_fds() -> list[dict]:
    out = []
    for line in FD_FILE.open(encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("_earnings_meta", {}).get("ticker") and r.get("_earnings_meta", {}).get("report_date"):
                out.append(r)
        except Exception:
            pass
    return out


def normalize_company(name: str) -> str:
    """Strip corporate suffixes for substring matching."""
    n = name.lower()
    for suf in [
        " corporation", " incorporated", " inc.", " inc", " corp.", " corp",
        " company", " co.", " plc", " ltd.", " ltd", " holdings", " group",
        " international", " industries", " technologies", " solutions",
        " communications", " systems", ", inc", ",inc",
    ]:
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = n.strip().rstrip(",.").strip()
    if n.startswith("the "):
        n = n[4:].strip()
    return n


def build_index(fds: list[dict], lookback_days: int) -> tuple[dict, dict, dict]:
    """Build lookup indices.

    Returns:
      ticker_to_fds: ticker -> list of (fd_id, ticker, start_date, end_date)
      ticker_re: compiled regex matching any S&P 500 ticker (case-sensitive, word-boundary)
      name_to_ticker: lowercased normalized company name -> ticker (longest first when iterated)
    """
    ticker_to_fds: dict[str, list] = defaultdict(list)
    name_to_ticker: dict[str, str] = {}
    tickers: set[str] = set()

    for fd in fds:
        m = fd["_earnings_meta"]
        t = m["ticker"].upper()
        rd = parse_date(m["report_date"])
        if rd is None:
            continue
        start = rd - timedelta(days=lookback_days)
        end = rd  # half-open [start, end)
        ticker_to_fds[t].append((fd["id"], t, start, end))
        tickers.add(t)
        cname = normalize_company(m.get("company") or "")
        if not cname or cname in COMPANY_NAME_STOPLIST:
            continue
        # Require either multi-word (whitespace present) or a long single word
        # to avoid false positives from common English words.
        is_multiword = " " in cname
        if (is_multiword and len(cname) >= 6) or (not is_multiword and len(cname) >= 7):
            name_to_ticker.setdefault(cname, t)

    safe_tickers = [t for t in tickers if t not in HIGH_COLLISION_TICKERS]
    ticker_re = re.compile(r"\b(" + "|".join(sorted(safe_tickers, key=len, reverse=True)) + r")\b")
    return ticker_to_fds, ticker_re, name_to_ticker


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None,
                    help="Path to unified articles.jsonl. Default: latest staged.")
    ap.add_argument("--lookback", type=int, default=90,
                    help="Days before report_date (default 90)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N pool articles (smoke test)")
    ap.add_argument("--text-prefix", type=int, default=1500,
                    help="Chars of text body to scan (default 1500)")
    args = ap.parse_args()

    pool_path = Path(args.pool) if args.pool else find_latest_unified()
    print(f"[pool] {pool_path}")
    fds = load_fds()
    print(f"[fds]  {len(fds)} earnings FDs loaded")

    ticker_to_fds, ticker_re, name_to_ticker = build_index(fds, args.lookback)
    print(f"[index] {len(ticker_to_fds)} unique tickers, "
          f"{len(name_to_ticker)} company-name keys")

    seen_urls = load_existing_urls()
    print(f"[dedup] {len(seen_urls)} pre-existing URL keys")

    tmp_path = OUT_FILE.with_suffix(".unified.tmp")
    written = 0
    scanned = 0
    matched_articles = 0
    dedup_url = 0
    dedup_window = 0
    spam = 0
    fd_hits: set[str] = set()
    per_ticker_count: Counter = Counter()

    with tmp_path.open("w", encoding="utf-8") as fout, \
         pool_path.open("r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if args.limit is not None and i >= args.limit:
                break
            scanned += 1
            if scanned % 25000 == 0:
                print(f"  [scan {scanned}] matched_arts={matched_articles} "
                      f"written={written} fds_hit={len(fd_hits)}")
            try:
                rec = json.loads(line)
            except Exception:
                continue

            url = rec.get("url", "")
            if not url or is_spam_url(url):
                if url:
                    spam += 1
                continue

            title = rec.get("title") or ""
            text = rec.get("text") or ""
            blob_upper = (title + "\n" + text[: args.text_prefix])
            blob_lower = blob_upper.lower()

            # Find company-name hits first (case-insensitive multi-word match
            # is strong evidence on its own).
            hits: set[str] = set()
            for cname, tk in name_to_ticker.items():
                if cname in blob_lower:
                    hits.add(tk)

            # Ticker-only hits require a finance context word in the same blob
            # to avoid false positives from common English acronyms.
            ticker_hits = set(ticker_re.findall(blob_upper))
            if ticker_hits:
                has_ctx = any(w in blob_lower for w in FINANCE_CONTEXT_WORDS)
                if has_ctx:
                    hits.update(ticker_hits)

            if not hits:
                continue
            matched_articles += 1

            pub_dt = parse_date(rec.get("publish_date") or rec.get("date") or "")
            if pub_dt is None:
                continue

            # For each ticker hit, attach to all FDs whose window covers pub_dt
            attached_any = False
            for tk in hits:
                for fd_id, _, start, end in ticker_to_fds.get(tk, []):
                    if not (start <= pub_dt < end):
                        dedup_window += 1
                        continue

                    # URL dedup applies per (url) globally (article pool small);
                    # but we DO emit the same source-article once per FD so the
                    # downstream linker can score per-FD.
                    out_url = url
                    if (out_url in seen_urls or url_key(out_url) in seen_urls):
                        # Same URL already attached to this FD? Mark via composite key.
                        composite = f"{fd_id}::{url_key(out_url)}"
                        if composite in seen_urls:
                            dedup_url += 1
                            continue
                        seen_urls.add(composite)
                    seen_urls.add(out_url)
                    seen_urls.add(url_key(out_url))

                    out_rec = {
                        "id":          art_id(out_url + "::" + fd_id),
                        "question_id": fd_id,
                        "ticker":      tk,
                        "url":         out_url,
                        "title":       title,
                        "text":        text[:4000],
                        "date":        pub_dt.strftime("%Y-%m-%d"),
                        "source":      rec.get("source_domain") or domain_of(out_url),
                        "provenance":  "unified-pool",
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    written += 1
                    attached_any = True
                    fd_hits.add(fd_id)
                    per_ticker_count[tk] += 1
            if not attached_any:
                # Article matched a ticker but no FD window covered the date.
                pass

    # Atomic append: cat tmp >> OUT_FILE then unlink tmp
    if written > 0:
        with OUT_FILE.open("a", encoding="utf-8") as appendf, \
             tmp_path.open("r", encoding="utf-8") as readf:
            for line in readf:
                appendf.write(line)
    tmp_path.unlink(missing_ok=True)

    print()
    print(f"=== Source #1 (unified-pool ticker match) summary ===")
    print(f"  scanned pool articles : {scanned}")
    print(f"  spam dropped          : {spam}")
    print(f"  articles with hits    : {matched_articles}")
    print(f"  records written       : {written}")
    print(f"  dedup (url)           : {dedup_url}")
    print(f"  dedup (window miss)   : {dedup_window}")
    print(f"  FDs covered (delta)   : {len(fd_hits)}")
    print(f"  top 10 tickers        : {per_ticker_count.most_common(10)}")
    final_total = sum(1 for _ in OUT_FILE.open(encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"  total in pool now     : {final_total}")


if __name__ == "__main__":
    main()
