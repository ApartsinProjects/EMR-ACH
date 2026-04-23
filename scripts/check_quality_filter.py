#!/usr/bin/env python
"""scripts/check_quality_filter.py: assert quality_filter actually ran (v2.2 [G4]).

The 2026-04-22 audit (Section 3 of V2_2_END_TO_END_AUDIT.md) found
1,555 zero-article FDs survived the ``08_after_quality_filter`` stage
because ``forecasts_filtered.jsonl`` and ``quality_meta.json`` were
silently absent: the build orchestrator did not emit a failure even
though the spec required them.

This script is the audit-level assertion that v2.2 lifts to a
fail-fast gate WITHOUT editing the in-flight scripts/quality_filter.py
or scripts/build_benchmark.py:

    python scripts/check_quality_filter.py --cutoff 2026-01-01 \\
        --before data/staging/.../forecasts.jsonl \\
        --after  data/staging/.../forecasts_filtered.jsonl \\
        --quality-meta data/staging/.../quality_meta.json

What it checks:

1. ``forecasts_filtered.jsonl`` exists and is non-empty.
2. ``quality_meta.json`` exists.
3. ``n_after_filter < n_before_filter`` (the filter did SOMETHING).
4. For each row in ``forecasts_filtered.jsonl`` the ``article_ids``
   list (or equivalent) is non-empty (no zero-article FD survived).

Exit code is 2 on any violation; stderr lists every offender.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import bootstrap_sys_path  # noqa: E402

bootstrap_sys_path()

ZERO_ARTICLE_KEYS = ("article_ids", "linked_article_ids", "evidence_article_ids")


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                n += 1
    return n


def _row_has_articles(row: dict) -> bool:
    for k in ZERO_ARTICLE_KEYS:
        v = row.get(k)
        if isinstance(v, list) and len(v) > 0:
            return True
    return False


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--before", type=Path, required=True,
                   help="Pre-filter forecasts.jsonl (snapshot 07).")
    p.add_argument("--after", type=Path, required=True,
                   help="Post-filter forecasts_filtered.jsonl (snapshot 08).")
    p.add_argument("--quality-meta", type=Path, required=True,
                   help="Path to quality_meta.json.")
    p.add_argument("--allow-zero-article-rows", action="store_true",
                   help="Skip the per-row article-list non-empty check. "
                   "Default: refuse to proceed if any FD has zero articles.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    failures: list[str] = []

    if not args.after.exists():
        failures.append(f"forecasts_filtered.jsonl missing at {args.after}")
    if not args.quality_meta.exists():
        failures.append(f"quality_meta.json missing at {args.quality_meta}")

    if failures:
        sys.stderr.write("check_quality_filter FAILED (preconditions):\n")
        for f in failures:
            sys.stderr.write(f"  - {f}\n")
        return 2

    n_before = _count_lines(args.before) if args.before.exists() else None
    n_after = _count_lines(args.after)

    if n_after == 0:
        failures.append(f"forecasts_filtered.jsonl is empty: {args.after}")

    if n_before is not None and n_after >= n_before:
        failures.append(
            f"quality filter was a no-op: n_after={n_after} >= n_before={n_before} "
            f"(expected strict reduction)"
        )

    if not args.allow_zero_article_rows:
        bad: list[str] = []
        for i, row in enumerate(_iter_jsonl(args.after)):
            if not _row_has_articles(row):
                fd_id = row.get("fd_id") or row.get("id") or f"row{i}"
                bad.append(str(fd_id))
                if len(bad) >= 10:
                    break
        if bad:
            failures.append(
                f"{len(bad)}+ FDs in post-filter file have zero articles "
                f"(first: {', '.join(bad)}). Closes the 1,555-zero-article-FD "
                f"silent failure from 2026-04-22."
            )

    if failures:
        sys.stderr.write("check_quality_filter FAILED:\n")
        for f in failures:
            sys.stderr.write(f"  - {f}\n")
        return 2

    summary = {
        "cutoff": args.cutoff,
        "n_before": n_before,
        "n_after": n_after,
        "passed": True,
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
