#!/usr/bin/env python
"""scripts/query_gdelt_doc_index.py: per-FD lookup against the GDELT DOC
index (v2.2 [A3] + [A8] + [A12]).

For each FD, query the FAISS flat-IP index built by A2 with:

* date-window prefilter (``[forecast_point - lookback_days,
  forecast_point)``),
* language filter (default: English),
* domain blocklist from :mod:`src.common.gdelt_aggregator_domains`,
* FAISS shard-prune by date intersection (A8; skip whole shards whose
  [start_month, end_month] does not intersect the FD's lookback window),
* optional ``--fetch-bodies`` post-pass for survivors only (A12; the
  "filter first, fetch survivors" inversion).

This pass ships the orchestrator + CLI surface and the shard-prune
logic; the actual FAISS query is behind ``--dry-run`` because the
index itself is a downstream artefact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import DATA_DIR, bootstrap_sys_path  # noqa: E402

bootstrap_sys_path()

from src.common.gdelt_aggregator_domains import (  # noqa: E402
    is_aggregator_domain,
)

DEFAULT_INDEX_ROOT = DATA_DIR / "gdelt_doc" / "index"


def _parse_month(m: str) -> date:
    y, mo = (int(x) for x in m.split("-"))
    return date(y, mo, 1)


def _month_end(m: str) -> date:
    d = _parse_month(m)
    ny, nm = (d.year, d.month + 1) if d.month < 12 else (d.year + 1, 1)
    return date(ny, nm, 1) - timedelta(days=1)


def shard_intersects_window(
    shard_month: str, window_start: date, window_end: date,
) -> bool:
    """Return True iff the shard's month overlaps the half-open window
    ``[window_start, window_end)``. Callers use this to skip whole FAISS
    shards (A8).
    """
    s = _parse_month(shard_month)
    e = _month_end(shard_month)
    return (s <= window_end) and (e >= window_start)


def discover_shards(index_root: Path) -> list[str]:
    """List sorted YYYY-MM shard directory names under ``index_root``."""
    if not index_root.exists():
        return []
    return sorted(
        c.name for c in index_root.iterdir()
        if c.is_dir() and c.name != "__pycache__" and _looks_like_month(c.name)
    )


def _looks_like_month(s: str) -> bool:
    return len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:7].isdigit()


def compute_prune_plan(
    shards: list[str], window_start: date, window_end: date,
) -> dict:
    keep = [s for s in shards if shard_intersects_window(s, window_start, window_end)]
    skip = [s for s in shards if s not in keep]
    return {"kept": keep, "pruned": skip}


def filter_candidates_by_domain(candidates: list[dict]) -> list[dict]:
    """Drop candidates whose URL hits the aggregator blocklist."""
    return [
        c for c in candidates
        if not is_aggregator_domain(str(c.get("url") or c.get("source_domain") or ""))
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    p.add_argument("--fd-id", required=True)
    p.add_argument("--forecast-point", required=True,
                   help="Forecast point date, YYYY-MM-DD.")
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--language", default="en")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--fetch-bodies", action="store_true",
                   help="Post-pass HTTP body fetch for survivors only (A12).")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fp = date.fromisoformat(args.forecast_point)
    window_start = fp - timedelta(days=args.lookback_days)
    window_end = fp  # half-open: fp is excluded

    shards = discover_shards(args.index_root)
    prune = compute_prune_plan(shards, window_start, window_end)

    if args.dry_run:
        sys.stdout.write(json.dumps({
            "fd_id": args.fd_id,
            "forecast_point": args.forecast_point,
            "lookback_days": args.lookback_days,
            "language": args.language,
            "window": [window_start.isoformat(), window_end.isoformat()],
            "shards_discovered": shards,
            "shard_prune": prune,
            "top_k": args.top_k,
            "fetch_bodies": args.fetch_bodies,
            "dry_run": True,
        }, indent=2, sort_keys=True) + "\n")
        return 0

    raise NotImplementedError(
        "Real FAISS query path is not enabled in this build pass. "
        "Implement: (1) encode [question + background] with "
        "src.common.embeddings_backend.encode, (2) for each shard in "
        "prune['kept'], load faiss.flatip + meta.parquet, search top-K, "
        "(3) apply language + domain filters, (4) optionally fetch "
        "bodies for survivors via fetch_article_text."
    )


if __name__ == "__main__":
    sys.exit(main())
