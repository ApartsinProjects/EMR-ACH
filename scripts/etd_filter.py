"""ETD fact filter: composable predicates that drop low-quality facts.

Each filter targets a known failure mode surfaced by the audit / debug
tooling:

  --min-confidence   drop facts whose extraction_confidence is below threshold
  --polarity         keep only asserted facts (drop denied/hypothetical)
  --max-date-skew-days N  drop facts whose `time` is more than N days
                          BEFORE article_date (likely hallucinated dates)
  --no-future        drop facts whose `time` is AFTER article_date
  --require-entities drop facts with empty `entities` list
  --min-cluster-size N  keep only canonical facts whose cluster size >= N
                        (requires Stage-2 dedup output)
  --require-linked-fd   keep only facts attached to >=1 FD
                        (requires Stage-3 linkage output)
  --benchmark BENCH  keep only facts attached to FDs from BENCH
                     (requires Stage-3 linkage output)

Filters are applied in the order listed. Drop counts are logged per filter
so you can see which filter cuts what.

CPU-only. Atomic write. Idempotent.

Usage:
  python scripts/etd_filter.py --in data/etd/facts.v1.jsonl \\
      --out data/etd/facts.v1_filtered.jsonl \\
      --min-confidence medium --polarity asserted --no-future

  python scripts/etd_filter.py --in data/etd/facts.v1_linked.jsonl \\
      --out data/etd/facts.v1_production.jsonl \\
      --min-confidence medium --polarity asserted --no-future \\
      --require-entities --require-linked-fd --benchmark gdelt-cameo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_IN = DATA / "etd" / "facts.v1.jsonl"
DEFAULT_OUT = DATA / "etd" / "facts.v1_filtered.jsonl"
META_OUT = DATA / "etd" / "filter_meta.json"

CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}


def _parse_date(s):
    if not s: return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


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
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"],
                    default=None)
    ap.add_argument("--polarity", choices=["asserted", "denied", "hypothetical"],
                    default=None,
                    help="If set, keep only facts with this polarity.")
    ap.add_argument("--max-date-skew-days", type=int, default=None,
                    help="Drop facts whose time is >N days BEFORE article_date.")
    ap.add_argument("--no-future", action="store_true",
                    help="Drop facts whose time is AFTER article_date.")
    ap.add_argument("--require-entities", action="store_true",
                    help="Drop facts with empty entities list.")
    ap.add_argument("--min-cluster-size", type=int, default=None,
                    help="Keep only canonical facts (no canonical_id) whose "
                         "variant_ids list has >= N-1 entries (cluster size N).")
    ap.add_argument("--require-linked-fd", action="store_true",
                    help="Drop facts whose linked_fd_ids is empty/missing.")
    ap.add_argument("--benchmark", default=None,
                    help="Restrict to facts linked to FDs from this benchmark "
                         "(implies --require-linked-fd; needs FD index file).")
    ap.add_argument("--fds", default=None,
                    help="Path to forecasts.jsonl for benchmark filter "
                         "(default: benchmark/data/{cutoff}/forecasts.jsonl).")
    ap.add_argument("--cutoff", default=None,
                    help="Cutoff to resolve --fds default path.")
    ap.add_argument("--source-blocklist", default=None,
                    help="Comma-separated list of source domains to drop. "
                         "Recommended for production: known stale-republish outlets "
                         "whose publish_date metadata disagrees with body event dates "
                         "(see Phase C verifier audit). Substring match against "
                         "fact.source; case-insensitive.")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[ERROR] {inp} not found")
        return 1
    print(f"[etd_filter] reading {inp}")

    fd_bench: dict[str, str] = {}
    if args.benchmark:
        if args.fds:
            fd_path = Path(args.fds)
        elif args.cutoff:
            fd_path = ROOT / "benchmark" / "data" / args.cutoff / "forecasts.jsonl"
        else:
            print("[ERROR] --benchmark requires --fds or --cutoff")
            return 1
        if not fd_path.exists():
            print(f"[ERROR] FDs path not found: {fd_path}")
            return 1
        with open(fd_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    fd_bench[d["id"]] = d.get("benchmark", "")
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[etd_filter] indexed {len(fd_bench)} FDs from {fd_path}")

    blocklist = set()
    if args.source_blocklist:
        blocklist = {s.strip().lower() for s in args.source_blocklist.split(",") if s.strip()}
        print(f"[etd_filter] source-blocklist: {sorted(blocklist)}")

    drops: Counter = Counter()
    kept: list[dict] = []
    n_in = 0
    min_conf_rank = CONFIDENCE_RANK.get(args.min_confidence, 0)

    with open(inp, encoding="utf-8") as f:
        for line in f:
            n_in += 1
            try:
                fact = json.loads(line)
            except json.JSONDecodeError:
                drops["malformed_json"] += 1
                continue

            if blocklist:
                src = (fact.get("source") or "").lower()
                if any(b in src for b in blocklist):
                    drops["source_blocklist"] += 1
                    continue
            if min_conf_rank:
                cr = CONFIDENCE_RANK.get(fact.get("extraction_confidence") or "", 0)
                if cr < min_conf_rank:
                    drops["min_confidence"] += 1
                    continue
            if args.polarity:
                if (fact.get("polarity") or "").lower() != args.polarity:
                    drops["polarity"] += 1
                    continue
            if args.no_future or args.max_date_skew_days is not None:
                ft = _parse_date(fact.get("time"))
                ad = _parse_date(fact.get("article_date"))
                if ft is not None and ad is not None:
                    if args.no_future and ft > ad:
                        drops["future_date"] += 1
                        continue
                    if (args.max_date_skew_days is not None
                            and ad - ft > timedelta(days=args.max_date_skew_days)):
                        drops["max_date_skew"] += 1
                        continue
            if args.require_entities:
                ents = fact.get("entities") or []
                if not [e for e in ents if isinstance(e, dict) and e.get("name")]:
                    drops["no_entities"] += 1
                    continue
            if args.min_cluster_size is not None:
                # Canonical fact only; cluster_size = 1 + len(variant_ids)
                if fact.get("canonical_id") is not None:
                    drops["non_canonical"] += 1
                    continue
                cs = 1 + len(fact.get("variant_ids") or [])
                if cs < args.min_cluster_size:
                    drops["min_cluster_size"] += 1
                    continue
            linked = fact.get("linked_fd_ids") or []
            if args.require_linked_fd and not linked:
                drops["no_linked_fd"] += 1
                continue
            if args.benchmark:
                if not any(fd_bench.get(fid) == args.benchmark for fid in linked):
                    drops["wrong_benchmark"] += 1
                    continue

            kept.append(fact)

    out_path = Path(args.out)
    _atomic_write_jsonl(out_path, kept)

    meta = {
        "input": str(inp),
        "output": str(out_path),
        "n_in": n_in,
        "n_out": len(kept),
        "drops": dict(drops),
        "filters": {
            "min_confidence": args.min_confidence,
            "polarity": args.polarity,
            "no_future": args.no_future,
            "max_date_skew_days": args.max_date_skew_days,
            "require_entities": args.require_entities,
            "min_cluster_size": args.min_cluster_size,
            "require_linked_fd": args.require_linked_fd,
            "benchmark": args.benchmark,
            "source_blocklist": sorted(blocklist) if blocklist else None,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _atomic_write_json(META_OUT, meta)

    print(f"[etd_filter] in:  {n_in}")
    print(f"[etd_filter] out: {len(kept)}  ({100*len(kept)/max(1,n_in):.1f}%)")
    for k, v in drops.most_common():
        print(f"  drop {k}: {v}")
    print(f"[etd_filter] wrote {out_path}")
    print(f"[etd_filter] meta  {META_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
