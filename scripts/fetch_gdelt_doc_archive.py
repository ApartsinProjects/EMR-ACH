#!/usr/bin/env python
"""scripts/fetch_gdelt_doc_archive.py: bulk GDELT DOC monthly archive
downloader (v2.2 [A1]).

One-time bulk download of GDELT DOC monthly archives to local
zstd-JSONL shards under ``data/gdelt_doc/raw/``. Per-shard sidecar
``_progress.json`` for resume.

This pass ships the orchestrator + CLI surface; the network fetch
itself is gated behind ``--dry-run`` by default to avoid kicking off
6-8 GB of downloads from a CI / agent context. To do a real fetch,
pass ``--no-dry-run`` and supply ``--start-month`` / ``--end-month``
explicitly.

See docs/V2_2_ARCHITECTURE.md Section 3.1 for the on-disk schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import DATA_DIR, bootstrap_sys_path  # noqa: E402

bootstrap_sys_path()

DEFAULT_RAW_ROOT = DATA_DIR / "gdelt_doc" / "raw"
GDELT_DOC_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class ShardSpec:
    """One monthly shard: 'YYYY-MM' window plus output paths."""

    month: str  # "YYYY-MM"
    raw_path: Path
    progress_path: Path

    @classmethod
    def for_month(cls, month: str, raw_root: Path) -> "ShardSpec":
        d = raw_root / month
        return cls(
            month=month,
            raw_path=d / "shard.jsonl.zst",
            progress_path=d / "_progress.json",
        )


def _months_between(start: str, end: str) -> list[str]:
    """Inclusive month range: ('2024-01', '2024-04') -> 4 entries."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _read_progress(spec: ShardSpec) -> dict:
    if not spec.progress_path.exists():
        return {"month": spec.month, "completed": False, "n_records": 0}
    try:
        return json.loads(spec.progress_path.read_text())
    except json.JSONDecodeError:
        return {"month": spec.month, "completed": False, "n_records": 0}


def _write_progress(spec: ShardSpec, payload: dict) -> None:
    spec.progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.progress_path.with_suffix(spec.progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(spec.progress_path)


def fetch_shard(spec: ShardSpec, *, dry_run: bool) -> dict:
    """Fetch one shard. With ``dry_run=True`` the function only writes
    a placeholder progress file and returns the planned action; with
    ``dry_run=False`` the network fetch is invoked (intentionally not
    implemented in this pass to avoid agent-context downloads).
    """
    progress = _read_progress(spec)
    if progress.get("completed"):
        return {"shard": spec.month, "status": "skipped (already complete)"}

    if dry_run:
        _write_progress(
            spec,
            {"month": spec.month, "completed": False, "n_records": 0,
             "dry_run": True},
        )
        return {"shard": spec.month, "status": "dry-run (no network call)"}

    # Real fetch is intentionally NOT implemented here. The skeleton
    # leaves a structured TODO so the next pass can drop in the GDELT
    # DOC API client without rewriting the orchestrator.
    raise NotImplementedError(
        "Real GDELT DOC fetch is not enabled in this build pass. "
        "Implement the API call here (paginated by 'maxrecords=250'), "
        "stream JSONL to spec.raw_path with zstd compression, and "
        "checkpoint progress every batch."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    today = date.today()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                   help="Output root for shard files.")
    p.add_argument("--start-month", default=f"{today.year:04d}-01",
                   help="Inclusive lower bound, YYYY-MM.")
    p.add_argument("--end-month", default=f"{today.year:04d}-{today.month:02d}",
                   help="Inclusive upper bound, YYYY-MM.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                   help="Plan-only (default; writes placeholder progress).")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                   help="Run the real fetch (intentionally not implemented yet).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    months = _months_between(args.start_month, args.end_month)
    results = []
    for m in months:
        spec = ShardSpec.for_month(m, args.raw_root)
        try:
            results.append(fetch_shard(spec, dry_run=args.dry_run))
        except NotImplementedError as exc:
            sys.stderr.write(f"{exc}\n")
            return 3
    sys.stdout.write(json.dumps({
        "raw_root": str(args.raw_root),
        "months": months,
        "results": results,
        "dry_run": args.dry_run,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
