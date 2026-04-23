"""ETD Stage 3: link each fact to the FDs whose evidence pool references it.

Each Stage-1 fact carries `primary_article_id` (and a wider `article_ids` list
when the same fact appears in multiple articles). Each FD carries
`article_ids: [...]` after relevance scoring. This stage builds the
inverted index `article_id -> [fd_id, ...]` and projects it onto facts to
produce `linked_fd_ids` per fact.

The linkage is what makes the ETD usable downstream: a baseline that wants
"all facts known to the system at forecast_point t* for FD X" filters
`facts_linked.jsonl` by `linked_fd_ids contains X` and then by `time < t*`.

Idempotent: re-running rebuilds the index from scratch from the current
forecasts.jsonl, so swapping in a freshly-published benchmark just works.

Reads:
  data/etd/facts.v1_canonical.jsonl                Stage-2 output (preferred)
    OR data/etd/facts.v1.jsonl                     Stage-1 output (fallback)
  benchmark/data/{cutoff}/forecasts.jsonl          v2.1 published FDs

Writes (atomic):
  data/etd/facts.v1_linked.jsonl                   facts with linked_fd_ids[]
  data/etd/link_meta.json                          coverage stats

Usage:
  python scripts/etd_link.py --cutoff 2026-01-01
  python scripts/etd_link.py --cutoff 2026-01-01 --use-stage1   # skip dedup
  python scripts/etd_link.py --cutoff 2026-01-01 --secondary-articles
       (also project facts whose `article_ids` list overlaps an FD pool, not
        just the primary_article_id; default OFF for clean primary linkage)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ETD_DIR = ROOT / "data" / "etd"
STAGE1 = ETD_DIR / "facts.v1.jsonl"
STAGE2 = ETD_DIR / "facts.v1_canonical.jsonl"
OUTPUT = ETD_DIR / "facts.v1_linked.jsonl"
META = ETD_DIR / "link_meta.json"


def _atomic_write(path: Path, render) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        render(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True,
                    help="Published benchmark cutoff (YYYY-MM-DD).")
    ap.add_argument("--use-stage1", action="store_true",
                    help="Read facts.v1.jsonl directly (skip Stage-2 dedup output).")
    ap.add_argument("--secondary-articles", action="store_true",
                    help="Also project via fact.article_ids[] (not just primary).")
    args = ap.parse_args()

    fd_path = ROOT / "benchmark" / "data" / args.cutoff / "forecasts.jsonl"
    if not fd_path.exists():
        print(f"[ERROR] FDs not found at {fd_path}. Build the benchmark first.")
        return 1

    fact_path = STAGE1 if args.use_stage1 else (STAGE2 if STAGE2.exists() else STAGE1)
    if not fact_path.exists():
        print(f"[ERROR] facts file not found at {fact_path}.")
        return 1
    print(f"[etd_link] facts:  {fact_path}")
    print(f"[etd_link] FDs:    {fd_path}")

    fds = _load_jsonl(fd_path)
    facts = _load_jsonl(fact_path)
    print(f"[etd_link] loaded {len(fds)} FDs and {len(facts)} facts")

    # Build inverted index: article_id -> list[fd_id]
    article_to_fds: dict[str, list[str]] = defaultdict(list)
    for fd in fds:
        fid = fd["id"]
        for aid in fd.get("article_ids", []):
            article_to_fds[aid].append(fid)
    print(f"[etd_link] articles referenced by any FD: {len(article_to_fds)}")

    # Project onto facts
    n_linked = 0
    link_size_hist = Counter()
    for fact in facts:
        linked: set[str] = set()
        pid = fact.get("primary_article_id")
        if pid and pid in article_to_fds:
            linked.update(article_to_fds[pid])
        if args.secondary_articles:
            for aid in fact.get("article_ids", []) or []:
                if aid in article_to_fds:
                    linked.update(article_to_fds[aid])
        fact["linked_fd_ids"] = sorted(linked)
        if linked:
            n_linked += 1
        link_size_hist[len(linked)] += 1

    print(f"[etd_link] facts linked to >=1 FD: {n_linked} / {len(facts)} "
          f"({100 * n_linked / max(1, len(facts)):.1f}%)")
    print(f"[etd_link] link-size histogram (top 10): "
          f"{dict(sorted(link_size_hist.items())[:10])}")

    def _w(f):
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    print(f"[etd_link] writing {OUTPUT}")
    _atomic_write(OUTPUT, _w)

    meta = {
        "facts_input": str(fact_path),
        "fds_input": str(fd_path),
        "output": str(OUTPUT),
        "cutoff": args.cutoff,
        "n_facts": len(facts),
        "n_fds": len(fds),
        "n_articles_in_fds": len(article_to_fds),
        "n_facts_linked": n_linked,
        "secondary_articles": args.secondary_articles,
        "link_size_histogram": dict(sorted(link_size_hist.items())),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    def _wm(f):
        json.dump(meta, f, indent=2)

    _atomic_write(META, _wm)
    print(f"[etd_link] meta -> {META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
