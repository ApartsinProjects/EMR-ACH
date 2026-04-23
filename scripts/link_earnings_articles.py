"""Ticker-match retrieval for earnings FDs (v2.1.1).

Replaces SBERT cosine relevance for the earnings benchmark with a pure
relational join: an earnings FD's `article_ids` = the first top_k earnings
articles whose `ticker` equals the FD's ticker AND whose `date` falls
within [report_date - lookback_days, report_date). No embeddings, no
LLM, no GPU. O(n) pass over the earnings article pool.

Motivation (v2.1 audit): earnings FD questions are formulaic, so SBERT
returns semantic-near-neighbors (sector news, macro buzz) instead of
ticker-specific coverage. The fetchers (Finnhub, yfinance, Google News,
GDELT DOC, SEC EDGAR) already ticker-tag every article; discard SBERT
entirely for this benchmark and join on the key.

Usage (typical):
  python scripts/link_earnings_articles.py                # update data/unified/forecasts.jsonl in place
  python scripts/link_earnings_articles.py --top-k 15
  python scripts/link_earnings_articles.py --lookback-days 30
  python scripts/link_earnings_articles.py \
      --articles benchmark/data/2026-01-01/articles.jsonl \
      --forecasts benchmark/data/2026-01-01/forecasts.jsonl \
      --in-place                                         # patch a published bundle

Reads (default paths):
  data/earnings/earnings_articles.jsonl   raw, ticker-tagged (from fetch_earnings_news.py)
  data/unified/forecasts.jsonl            target FD pool; earnings FDs are patched

Writes (atomic):
  data/unified/forecasts.jsonl            article_ids filled for earnings only
  data/unified/earnings_link_meta.json    audit: per-FD match count + coverage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_ARTS = DATA / "earnings" / "earnings_articles.jsonl"
DEFAULT_FDS = DATA / "unified" / "forecasts.jsonl"
DEFAULT_META = DATA / "unified" / "earnings_link_meta.json"


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _atomic_write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _derive_unified_id(article: dict) -> str:
    """The fetcher emits `id: earn_<hash>`, but unify_articles rewrites
    every article under a canonical `art_<urlhash>` id. To cite articles
    in FD.article_ids we need the unified id. If the fetcher record
    already carries `unified_id`, use it; else fall back to `id`.

    This function is the seam: if `article_ids` should point at unified
    art_... ids and this script is invoked BEFORE unify_articles, we
    need the URL hash to be computed. We delegate by importing
    unify_articles.art_id() when available.
    """
    if "unified_id" in article:
        return article["unified_id"]
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from unify_articles import art_id  # type: ignore
        return art_id(article.get("url", ""))
    except Exception:
        # Fall back to the fetcher-local id; the caller is responsible for
        # ensuring the target FD references match whatever id scheme they use.
        return article.get("id", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default=str(DEFAULT_ARTS),
                    help="Path to earnings_articles.jsonl (ticker-tagged raw).")
    ap.add_argument("--forecasts", default=str(DEFAULT_FDS),
                    help="Target FD file to patch in place.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lookback-days", type=int, default=90)
    ap.add_argument("--meta-out", default=str(DEFAULT_META))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arts_path = Path(args.articles)
    fds_path = Path(args.forecasts)
    if not arts_path.exists():
        print(f"[ERROR] {arts_path} not found. Re-run scripts/fetch_earnings_news.py first.")
        return 1
    if not fds_path.exists():
        print(f"[ERROR] {fds_path} not found.")
        return 1

    # 1. Build ticker -> [article records] index, keyed on ticker + parseable date
    print(f"[link_earnings] indexing {arts_path}")
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    n_arts = 0
    n_no_date = 0
    n_no_ticker = 0
    for line in open(arts_path, encoding="utf-8"):
        try:
            a = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_arts += 1
        ticker = (a.get("ticker") or "").upper().strip()
        if not ticker:
            n_no_ticker += 1
            continue
        d = _parse_date(a.get("date"))
        if d is None:
            n_no_date += 1
            continue
        a["_parsed_date"] = d
        a["_unified_id"] = _derive_unified_id(a)
        by_ticker[ticker].append(a)
    print(f"[link_earnings] articles: {n_arts}  (no-ticker={n_no_ticker}  no-date={n_no_date})")
    print(f"[link_earnings] unique tickers: {len(by_ticker)}")

    # Pre-sort each ticker's list by date desc so top_k is a simple slice.
    for tk, lst in by_ticker.items():
        lst.sort(key=lambda a: a["_parsed_date"], reverse=True)

    # 2. Walk FDs; for each earnings FD, join on ticker + date window.
    print(f"[link_earnings] patching {fds_path}")
    n_total = 0
    n_earnings = 0
    n_filled = 0
    n_zero = 0
    match_hist = Counter()
    missing_ticker_fds = 0

    out_records: list[dict] = []
    for line in open(fds_path, encoding="utf-8"):
        try:
            fd = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_total += 1
        if fd.get("benchmark") != "earnings":
            out_records.append(fd)
            continue

        n_earnings += 1
        meta = fd.get("_earnings_meta") or {}
        ticker = (meta.get("ticker") or "").upper().strip()
        report_date = _parse_date(meta.get("report_date"))
        # v2.2 leakage-guard anchor: forecast_point, not report_date. For v2.1
        # FDs forecast_point == resolution_date so the boundary is identical;
        # for v2.2 forecast_point = report_date - horizon_days (default 14),
        # so the retrieval upper bound moves 2 weeks earlier.
        forecast_point = _parse_date(fd.get("forecast_point")) or report_date
        if not ticker or report_date is None:
            missing_ticker_fds += 1
            out_records.append(fd)
            continue

        start = forecast_point - timedelta(days=args.lookback_days)
        candidates = by_ticker.get(ticker, [])
        # Hard leakage constraint: article publish date must be <= forecast_point.
        pre_leak = 0
        matches = []
        for a in candidates:
            if a["_parsed_date"] is None:
                continue
            if a["_parsed_date"] > forecast_point:
                pre_leak += 1
                continue
            if start <= a["_parsed_date"] <= forecast_point:
                matches.append(a)
        if pre_leak:
            # Rare but possible: an earnings article in the pool dated AFTER
            # the FD's forecast_point. Log so we can track source-side drift.
            print(f"[link_earnings] {fd['id']} leakage-filtered {pre_leak} "
                  f"articles with publish_date > {forecast_point.date()}")
        # Already sorted desc; take top_k
        picks = matches[: args.top_k]
        fd["article_ids"] = [a["_unified_id"] for a in picks]
        fd["earnings_retrieval_method"] = "ticker-date-join"
        fd["earnings_retrieval_meta"] = {
            "ticker": ticker,
            "report_date": meta.get("report_date"),
            "lookback_days": args.lookback_days,
            "top_k": args.top_k,
            "candidate_count": len(matches),
        }
        if fd["article_ids"]:
            n_filled += 1
        else:
            n_zero += 1
        match_hist[min(len(picks), 15)] += 1
        out_records.append(fd)

    print(f"[link_earnings] FDs total: {n_total}")
    print(f"[link_earnings] earnings FDs: {n_earnings}")
    print(f"[link_earnings] filled:  {n_filled}  ({100*n_filled/max(1,n_earnings):.1f}%)")
    print(f"[link_earnings] zero:    {n_zero}")
    print(f"[link_earnings] missing ticker/report_date: {missing_ticker_fds}")
    print(f"[link_earnings] per-FD article count histogram (capped at 15):")
    for k in sorted(match_hist):
        print(f"    {k}: {match_hist[k]}")

    if args.dry_run:
        print("[link_earnings] DRY-RUN, not writing.")
        return 0

    # 3. Atomic write back
    _atomic_write_jsonl(fds_path, out_records)
    print(f"[link_earnings] wrote -> {fds_path}")

    meta = {
        "articles_input": str(arts_path),
        "forecasts_input": str(fds_path),
        "top_k": args.top_k,
        "lookback_days": args.lookback_days,
        "n_earnings_fds": n_earnings,
        "n_filled": n_filled,
        "n_zero": n_zero,
        "n_missing_ticker_fds": missing_ticker_fds,
        "unique_tickers_in_articles": len(by_ticker),
        "article_count_histogram": dict(match_hist),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _atomic_write_json(Path(args.meta_out), meta)
    print(f"[link_earnings] meta -> {args.meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
