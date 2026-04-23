"""Forecast Dossier filter: downstream selection without re-running quality_filter.

quality_filter.py is the FD pool's *quality gate* (drops FDs that lack enough
articles / day-spread / chars). This script is for *experiment selection*:
slice the published FD pool by benchmark, fd_type, source, hypothesis, etc.
into a smaller JSONL the baselines runner consumes directly.

Composable predicates with per-filter drop-count logging and atomic write.

Usage:
  python scripts/fd_filter.py \\
      --in benchmark/data/2026-01-01/forecasts.jsonl \\
      --out benchmark/data/2026-01-01/forecasts.change_only.jsonl \\
      --fd-type change

  python scripts/fd_filter.py \\
      --in benchmark/data/2026-01-01/forecasts.jsonl \\
      --out benchmark/data/2026-01-01/forecasts.gdelt_change.jsonl \\
      --benchmark gdelt-cameo --fd-type change \\
      --min-articles 5 --max-articles 15

  python scripts/fd_filter.py --in ... --out ... \\
      --primary-only --no-unknown-fd-type
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META_OUT = ROOT / "data" / "unified" / "audit" / "fd_filter_meta.json"


def _atomic_write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", choices=["forecastbench", "gdelt-cameo", "earnings"],
                    default=None)
    ap.add_argument("--fd-type", choices=["stability", "change", "unknown"],
                    default=None)
    ap.add_argument("--no-unknown-fd-type", action="store_true",
                    help="Drop FDs whose fd_type is null or 'unknown'.")
    ap.add_argument("--source", default=None,
                    help="Comma-separated source whitelist (e.g. 'polymarket,metaculus').")
    ap.add_argument("--ground-truth", default=None,
                    help="Restrict to FDs with this ground_truth label.")
    ap.add_argument("--primary-only", action="store_true",
                    help="Require hypothesis_set == ['Comply','Surprise'] "
                         "(filters out FDs that haven't been promoted).")
    ap.add_argument("--min-articles", type=int, default=None)
    ap.add_argument("--max-articles", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate output to first N records (post-filter).")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[ERROR] {inp} not found")
        return 1
    print(f"[fd_filter] reading {inp}")

    sources_ok = set(s.strip() for s in (args.source or "").split(",") if s.strip()) or None
    drops: Counter = Counter()
    kept: list[dict] = []
    n_in = 0
    with open(inp, encoding="utf-8") as f:
        for line in f:
            n_in += 1
            try:
                fd = json.loads(line)
            except json.JSONDecodeError:
                drops["malformed_json"] += 1
                continue

            if args.benchmark and fd.get("benchmark") != args.benchmark:
                drops["benchmark"] += 1
                continue
            if args.fd_type and fd.get("fd_type") != args.fd_type:
                drops["fd_type"] += 1
                continue
            if args.no_unknown_fd_type:
                ft = fd.get("fd_type")
                if ft is None or ft == "unknown":
                    drops["unknown_fd_type"] += 1
                    continue
            if sources_ok and fd.get("source") not in sources_ok:
                drops["source"] += 1
                continue
            if args.ground_truth and fd.get("ground_truth") != args.ground_truth:
                drops["ground_truth"] += 1
                continue
            if args.primary_only:
                hs = fd.get("hypothesis_set") or []
                if sorted(hs) != ["Comply", "Surprise"]:
                    drops["non_primary"] += 1
                    continue
            if args.min_articles is not None:
                if len(fd.get("article_ids") or []) < args.min_articles:
                    drops["min_articles"] += 1
                    continue
            if args.max_articles is not None:
                if len(fd.get("article_ids") or []) > args.max_articles:
                    drops["max_articles"] += 1
                    continue
            kept.append(fd)
            if args.limit and len(kept) >= args.limit:
                break

    out_path = Path(args.out)
    _atomic_write_jsonl(out_path, kept)

    meta = {
        "input": str(inp),
        "output": str(out_path),
        "n_in": n_in,
        "n_out": len(kept),
        "drops": dict(drops),
        "filters": {
            "benchmark": args.benchmark,
            "fd_type": args.fd_type,
            "no_unknown_fd_type": args.no_unknown_fd_type,
            "source": args.source,
            "ground_truth": args.ground_truth,
            "primary_only": args.primary_only,
            "min_articles": args.min_articles,
            "max_articles": args.max_articles,
            "limit": args.limit,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _atomic_write_json(META_OUT, meta)

    print(f"[fd_filter] in:  {n_in}")
    print(f"[fd_filter] out: {len(kept)}  ({100*len(kept)/max(1,n_in):.1f}%)")
    for k, v in drops.most_common():
        print(f"  drop {k}: {v}")
    print(f"[fd_filter] wrote {out_path}")
    print(f"[fd_filter] meta  {META_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
