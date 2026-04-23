#!/usr/bin/env python
"""scripts/preflight_publish.py: pre-publish integrity gate (v2.2 [G2] + [G3] + [G5]).

Stand-alone CLI that runs BEFORE step_publish in the v2.1 build
orchestrator. Designed to be invoked manually (or via a thin wrapper)
without editing the in-flight scripts/build_benchmark.py:

    python scripts/preflight_publish.py --cutoff 2026-01-01

What it checks:

* Every per-benchmark articles file
  (data/{bench}/{bench}_articles.jsonl) exists and is non-empty.
  Closes the 2026-04-22 failure where earnings_articles.jsonl was
  silently deleted between fetch and publish (G2).

* Writes a checksum sidecar (sha256 + line count + unique fd_id
  coverage + mtime) per benchmark so step_publish can later compare
  source against destination (G5).

* Optionally compares the staged forecasts.jsonl mtime against a
  baseline mtime (--min-forecasts-mtime ISO) to guard against the
  G3 "stale forecasts.jsonl shipped" failure.

The script's exit code is non-zero on any failure, and its stderr is
the human-readable failure list. Suitable for CI / orchestrator
preflight stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Bootstrap so this script can run from anywhere.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import bootstrap_sys_path, per_benchmark_articles_path

bootstrap_sys_path()

from src.common.article_checksums import (  # noqa: E402
    assert_articles_present,
    checksum_sidecar_path,
    write_checksum_sidecar,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cutoff",
        required=True,
        help="Build cutoff (e.g. 2026-01-01). Reserved for future use; "
        "currently the per-benchmark article paths are cutoff-agnostic.",
    )
    p.add_argument(
        "--benchmarks",
        default="forecastbench,gdelt_cameo,earnings",
        help="Comma-separated list of benchmarks to gate.",
    )
    p.add_argument(
        "--min-lines",
        type=int,
        default=1,
        help="Per-benchmark minimum line count; default 1 (any non-empty file).",
    )
    p.add_argument(
        "--forecasts-path",
        type=Path,
        default=None,
        help="Optional path to staged forecasts.jsonl; --min-forecasts-mtime "
        "asserts its mtime is >= the supplied ISO timestamp.",
    )
    p.add_argument(
        "--min-forecasts-mtime",
        default=None,
        help="ISO-8601 timestamp; the staged forecasts file must be at least "
        "this fresh. Pair with --forecasts-path. Closes G3.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute checksums and report but do not write sidecars.",
    )
    return p.parse_args(argv)


def _check_forecasts_mtime(path: Path, min_iso: str) -> str | None:
    if not path.exists():
        return f"forecasts.jsonl missing at {path}"
    try:
        threshold = datetime.fromisoformat(min_iso)
    except ValueError as exc:
        return f"--min-forecasts-mtime not parseable: {exc}"
    actual = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    if threshold.tzinfo is None:
        threshold = threshold.replace(tzinfo=timezone.utc)
    if actual < threshold:
        return (
            f"forecasts.jsonl at {path} is stale: mtime {actual.isoformat()} "
            f"< required {threshold.isoformat()} (G3 stale-publish guard)"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    targets = [(b, per_benchmark_articles_path(b)) for b in benches]

    failures: list[str] = []

    # G3: optional staged-forecasts freshness gate.
    if args.min_forecasts_mtime or args.forecasts_path:
        if not (args.min_forecasts_mtime and args.forecasts_path):
            failures.append(
                "--min-forecasts-mtime and --forecasts-path must be set together"
            )
        else:
            err = _check_forecasts_mtime(args.forecasts_path, args.min_forecasts_mtime)
            if err:
                failures.append(err)

    # G2 + G5: per-benchmark articles files.
    try:
        checksums = assert_articles_present(targets, min_lines=args.min_lines)
    except RuntimeError as exc:
        failures.append(str(exc))
        checksums = []

    if failures:
        sys.stderr.write("preflight_publish FAILED:\n")
        for f in failures:
            sys.stderr.write(f"  - {f}\n")
        return 2

    # Write sidecars.
    written: list[str] = []
    for cks in checksums:
        sidecar = checksum_sidecar_path(Path(cks.path))
        if not args.dry_run:
            write_checksum_sidecar(cks, sidecar)
        written.append(str(sidecar))

    summary = {
        "cutoff": args.cutoff,
        "benchmarks": benches,
        "checksums": [c.to_json() for c in checksums],
        "sidecars_written": [] if args.dry_run else written,
        "dry_run": args.dry_run,
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
