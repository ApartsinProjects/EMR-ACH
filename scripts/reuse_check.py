#!/usr/bin/env python
"""scripts/reuse_check.py: dry-audit of stage cache reuse (v2.2 [G6]).

Reports which pipeline stages would be reused on a fresh build given
the current cache state. No side effects.

Usage:

    python scripts/reuse_check.py --cutoff 2026-01-01 \\
        --config configs/default_config.yaml

For each stage declared in src.common.config_slices.STAGE_SLICES, the
script prints:

  stage_name | slice_hash | meta_hash | matches? | next_action

``slice_hash`` is computed from the live config; ``meta_hash`` is read
from the per-stage ``stage_meta.json`` if present (default location:
``data/stage_meta/{cutoff}/{stage}.json``). When G1 lands, the reading
side migrates to ``src.common.stage_cache``; for now this script is the
read-only frontend that lets developers debug "why did this stage
re-run" questions before G1 is wired into the orchestrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import REPO_ROOT, bootstrap_sys_path  # noqa: E402

bootstrap_sys_path()

from src.common.config_slices import (  # noqa: E402
    STAGE_SLICES,
    known_stages,
    slice_config,
    stage_cache_key,
)


def _load_yaml(path: Path) -> dict:
    """Best-effort YAML load; falls back to JSON if PyYAML is unavailable."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text()) or {}
    except ImportError:
        return json.loads(path.read_text())


def _meta_hash_for(stage: str, cutoff: str, meta_root: Path) -> str | None:
    candidate = meta_root / cutoff / f"{stage}.json"
    if not candidate.exists():
        return None
    try:
        obj = json.loads(candidate.read_text())
    except json.JSONDecodeError:
        return None
    return obj.get("cache_key") or obj.get("slice_hash")


def _next_action(matches: bool, meta_present: bool) -> str:
    if not meta_present:
        return "RUN (no cached meta)"
    if matches:
        return "REUSE"
    return "RERUN (slice changed)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutoff", required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default_config.yaml",
        help="Path to the active config (yaml or json).",
    )
    p.add_argument(
        "--meta-root",
        type=Path,
        default=REPO_ROOT / "data" / "stage_meta",
        help="Root directory containing per-cutoff stage_meta files.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the table.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.config.exists():
        sys.stderr.write(f"reuse_check: config not found: {args.config}\n")
        return 2
    config = _load_yaml(args.config)

    rows: list[dict] = []
    for stage in known_stages():
        slice_hash = stage_cache_key(config, stage)
        meta_hash = _meta_hash_for(stage, args.cutoff, args.meta_root)
        matches = (meta_hash is not None) and (meta_hash == slice_hash)
        rows.append(
            {
                "stage": stage,
                "slice_keys": list(STAGE_SLICES[stage]),
                "slice_hash": slice_hash[:12],
                "meta_hash": (meta_hash[:12] if meta_hash else None),
                "matches": matches,
                "next_action": _next_action(matches, meta_hash is not None),
            }
        )

    if args.json:
        sys.stdout.write(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return 0

    # Plain-text table.
    name_w = max(len(r["stage"]) for r in rows)
    fmt = "{stage:<%d}  {slice:<14}  {meta:<14}  {match:<6}  {next}" % name_w
    sys.stdout.write(
        fmt.format(
            stage="STAGE", slice="SLICE_HASH", meta="META_HASH",
            match="MATCH", next="NEXT_ACTION",
        )
        + "\n"
    )
    sys.stdout.write("-" * (name_w + 60) + "\n")
    for r in rows:
        sys.stdout.write(
            fmt.format(
                stage=r["stage"],
                slice=r["slice_hash"],
                meta=r["meta_hash"] or "-",
                match="yes" if r["matches"] else "no",
                next=r["next_action"],
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
