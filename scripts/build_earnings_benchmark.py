"""
Build an earnings-surprise forecasting benchmark (finance domain).

For each company in the ticker list, fetch quarterly earnings release metadata
from Yahoo Finance (yfinance), compute the surprise class, and emit a
Forecast Dossier record.

Hypothesis class (3-class): Beat / Meet / Miss
  Beat:  actual EPS >= consensus + |consensus| * threshold  (default 2%)
  Miss:  actual EPS <= consensus - |consensus| * threshold
  Meet:  otherwise

Reads:
  - builtin ticker list (S&P 500 slice) OR --tickers CSV
Writes:
  data/earnings/earnings_forecasts.jsonl   (FD records)
  data/earnings/earnings_meta.json         (stats)

Usage:
  python scripts/build_earnings_benchmark.py --start 2024-07-01 --end 2025-02-28
  python scripts/build_earnings_benchmark.py --tickers AAPL,MSFT,GOOGL --start 2024-07-01
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "earnings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FC = OUT_DIR / "earnings_forecasts.jsonl"
OUT_META = OUT_DIR / "earnings_meta.json"

HYP = ["Beat", "Meet", "Miss"]

# Default ticker list: full S&P 500 (503 tickers incl. multi-class).
# Sourced from data/sp500_tickers.txt (refresh via scripts/fetch_sp500_tickers.py).
# Falls back to the original 50-ticker mega-cap list if the file is missing.

_SP500_FILE = Path(__file__).parent.parent / "data" / "sp500_tickers.txt"
_FALLBACK_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "INTC", "CSCO", "QCOM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA",
    # Healthcare / pharma
    "JNJ", "PFE", "UNH", "LLY", "MRK", "ABBV", "TMO", "ABT",
    # Consumer
    "WMT", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT", "HD", "LOW",
    # Industrial / energy / other
    "XOM", "CVX", "BA", "CAT", "GE", "DIS", "NFLX",
]
if _SP500_FILE.exists():
    DEFAULT_TICKERS = [t.strip() for t in _SP500_FILE.open(encoding="utf-8") if t.strip()]
else:
    DEFAULT_TICKERS = _FALLBACK_TICKERS


def compute_hypothesis(actual: float, estimate: float, threshold: float) -> str:
    if estimate is None or actual is None:
        return None
    if estimate == 0:
        # fall back to absolute-EPS compare for zero-estimate cases
        diff = actual - estimate
        if diff > 0.02: return "Beat"
        if diff < -0.02: return "Miss"
        return "Meet"
    rel = (actual - estimate) / abs(estimate)
    if rel >= threshold:
        return "Beat"
    if rel <= -threshold:
        return "Miss"
    return "Meet"


def fetch_ticker_earnings(ticker: str, start: datetime, end: datetime,
                          threshold: float) -> list[dict]:
    """Return list of FD records for the ticker's earnings in [start, end]."""
    out = []
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass
        company_name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "") or ""
        industry = info.get("industry", "") or ""

        # yfinance quarterly earnings history
        eh = None
        try:
            eh = t.get_earnings_dates(limit=24)  # past ~6 yrs
        except Exception as e:
            print(f"    [{ticker}] earnings_dates not available: {e}")
            return out
        if eh is None or eh.empty:
            return out

        # eh index is DatetimeIndex of report dates; columns: EPS Estimate, Reported EPS, Surprise(%)
        for dt, row in eh.iterrows():
            try:
                report_dt = dt.to_pydatetime().replace(tzinfo=None)
            except Exception:
                continue
            if report_dt < start or report_dt > end:
                continue
            est = row.get("EPS Estimate")
            act = row.get("Reported EPS")
            if est is None or act is None:
                continue
            try:
                est_f = float(est); act_f = float(act)
            except Exception:
                continue
            hyp = compute_hypothesis(act_f, est_f, threshold)
            if hyp is None:
                continue
            surprise_pct = row.get("Surprise(%)") or row.get("Surprise %")
            # forecast_point = one day before report (market close T-1)
            fp = (report_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            rd = report_dt.strftime("%Y-%m-%d")
            quarter = f"Q{((report_dt.month - 1) // 3) + 1}-{report_dt.year}"
            q_id = f"earn_{ticker}_{rd}"
            question = (f"Will {company_name} ({ticker}) beat, meet, or miss analyst "
                        f"EPS consensus for earnings reported on {rd}?")
            background = (f"Sector: {sector}. Industry: {industry}. "
                          f"Ticker: {ticker}. Reporting quarter: {quarter}. "
                          f"Analyst consensus EPS estimate: {est_f:.2f}.")
            out.append({
                "id": q_id,
                "benchmark": "earnings",
                "source": "yfinance",
                "hypothesis_set": HYP,
                "question": question,
                "background": background,
                "forecast_point": fp,
                "resolution_date": rd,
                "ground_truth": hyp,
                "ground_truth_idx": HYP.index(hyp),
                "crowd_probability": None,  # analyst consensus is in background, not a market probability
                "lookback_days": 14,        # 2 weeks of pre-release news
                "article_ids": [],
                "_earnings_meta": {
                    "ticker": ticker, "company": company_name,
                    "sector": sector, "industry": industry,
                    "eps_estimate": est_f, "eps_actual": act_f,
                    "surprise_pct": float(surprise_pct) if surprise_pct is not None else None,
                    "report_date": rd,
                },
            })
    except Exception as e:
        print(f"  [{ticker}] failed: {type(e).__name__}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated ticker list. If omitted, uses ~50 default large-caps.")
    ap.add_argument("--start", default="2024-07-01", help="Report date lower bound (post GPT-4o cutoff)")
    ap.add_argument("--end", default="2025-03-31", help="Report date upper bound")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="Beat/Miss threshold as fraction of consensus EPS (default 0.02 = 2%%)")
    ap.add_argument("--sleep", type=float, default=0.4, help="Per-ticker throttle")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("[ERROR] --start/--end must be YYYY-MM-DD")
        sys.exit(1)

    print(f"Fetching earnings for {len(tickers)} tickers from {args.start} to {args.end}")
    print(f"Beat/Miss threshold: {args.threshold*100:.1f}% of consensus EPS")

    all_fds = []
    failures = []
    for i, tk in enumerate(tickers, 1):
        recs = fetch_ticker_earnings(tk, start_dt, end_dt, args.threshold)
        if not recs:
            failures.append(tk)
        all_fds.extend(recs)
        if i % 10 == 0:
            print(f"  [{i}/{len(tickers)}] total FDs so far: {len(all_fds)}")
        time.sleep(args.sleep)

    # write (scrub NaN/Inf to None so the output is strict JSON; orjson and other
    # downstream parsers reject the JS-style NaN literal that stdlib json emits)
    import math as _math

    def _scrub(v):
        if isinstance(v, float):
            return None if (_math.isnan(v) or _math.isinf(v)) else v
        if isinstance(v, dict):
            return {k: _scrub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_scrub(x) for x in v]
        return v

    with open(OUT_FC, "w", encoding="utf-8") as f:
        for r in all_fds:
            f.write(json.dumps(_scrub(r), ensure_ascii=False, allow_nan=False) + "\n")

    # stats
    from collections import Counter
    gt = Counter(r["ground_truth"] for r in all_fds)
    by_sector = Counter(r["_earnings_meta"]["sector"] for r in all_fds)
    meta = {
        "total_fds": len(all_fds),
        "unique_tickers_kept": len({r["_earnings_meta"]["ticker"] for r in all_fds}),
        "tickers_attempted": len(tickers),
        "tickers_with_no_data": failures,
        "date_range": f"{args.start} to {args.end}",
        "threshold_pct": args.threshold * 100,
        "ground_truth_distribution": dict(gt),
        "ground_truth_base_rates": {k: round(v / len(all_fds), 3) for k, v in gt.items()} if all_fds else {},
        "by_sector": dict(by_sector.most_common()),
        "output": str(OUT_FC),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone. Wrote {len(all_fds)} earnings FDs")
    print(f"Ground-truth: {dict(gt)}")
    if all_fds:
        print(f"Base rates: {meta['ground_truth_base_rates']}")
    if failures:
        print(f"Tickers with no data: {failures}")


if __name__ == "__main__":
    main()
