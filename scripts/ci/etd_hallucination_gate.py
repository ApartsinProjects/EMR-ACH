"""ETD Stage-1 hallucination floor CI gate (v2.2 [C7]).

Wraps ``scripts/etd_verify.py`` so CI can fail the build when the
unsupported-at-high-confidence rate climbs past a configured floor.

Usage
-----
Live mode (calls etd_verify.py's main(); spends API tokens):

    python scripts/ci/etd_hallucination_gate.py --threshold 0.13 --min-sample 200

Dry-run mode (zero API calls; reads existing audit reports):

    python scripts/ci/etd_hallucination_gate.py --dry-run --threshold 0.13

Exit codes
----------
* 0 -- pass; high-confidence unsupported share is at or below threshold
       AND sample size is at or above ``--min-sample``.
* 1 -- fail; threshold breached.
* 2 -- fail; no audit data available (dry-run only).
* 3 -- fail; sample size below ``--min-sample`` (insufficient power).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_DIR = REPO_ROOT / "data" / "etd" / "audit"

# Reports we are willing to read in --dry-run priority order. Newest /
# most-trusted first.
_DRY_RUN_REPORT_GLOBS: tuple[str, ...] = (
    "verifier_report.md",
    "verifier_v3b_smoke.md",
    "verifier_v3_smoke.md",
)


def _parse_high_unsupported_share(report_text: str) -> tuple[float, int] | None:
    """Parse a verifier report's stratified-by-confidence table.

    Returns ``(unsupported_share, sample_size_at_high)`` for the
    ``high`` confidence row, or None if the row is not found.

    The table format (from scripts/etd_verify.py) is::

        | confidence | n | supported | partial | unsupported | unsupported share |
        |---|---:|---:|---:|---:|---:|
        | `high` | 200 | 169 | 0 | 31 | 15.5% |
    """
    # Match a row whose first cell is `high` (with optional backticks).
    pattern = re.compile(
        r"^\|\s*`?high`?\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([0-9.]+)\s*%\s*\|",
        re.MULTILINE,
    )
    m = pattern.search(report_text)
    if not m:
        return None
    n = int(m.group(1))
    share = float(m.group(2)) / 100.0
    return share, n


def _find_dry_run_report(audit_dir: Path) -> Path | None:
    for name in _DRY_RUN_REPORT_GLOBS:
        candidate = audit_dir / name
        if candidate.exists():
            return candidate
    return None


def run_dry_run(threshold: float, min_sample: int, audit_dir: Path) -> int:
    report = _find_dry_run_report(audit_dir)
    if report is None:
        print(
            f"[etd-gate] dry-run: no verifier report found in {audit_dir}; "
            f"tried {list(_DRY_RUN_REPORT_GLOBS)}",
            file=sys.stderr,
        )
        return 2
    text = report.read_text(encoding="utf-8")
    parsed = _parse_high_unsupported_share(text)
    if parsed is None:
        print(
            f"[etd-gate] dry-run: could not parse 'high'-confidence row from {report}",
            file=sys.stderr,
        )
        return 2
    share, n = parsed
    print(
        f"[etd-gate] dry-run: report={report.name} "
        f"high_unsupported_share={share:.4f} sample={n} "
        f"threshold={threshold:.4f} min_sample={min_sample}"
    )
    if n < min_sample:
        print(
            f"[etd-gate] FAIL: high-confidence sample {n} < min_sample {min_sample}",
            file=sys.stderr,
        )
        return 3
    if share > threshold:
        print(
            f"[etd-gate] FAIL: high-confidence unsupported share {share:.4f} "
            f"> threshold {threshold:.4f}",
            file=sys.stderr,
        )
        return 1
    print("[etd-gate] PASS")
    return 0


def run_live(threshold: float, min_sample: int, audit_dir: Path) -> int:
    """Invoke etd_verify.main() in-process, then re-enter dry-run parsing."""
    # Import-then-call (not subprocess) per the C7 contract.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import etd_verify  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime import failure
        print(f"[etd-gate] ERROR: failed to import scripts/etd_verify.py: {exc!r}", file=sys.stderr)
        return 2
    # etd_verify.main() reads sys.argv. Stage a synthetic argv that
    # asks for at least min_sample facts.
    saved_argv = sys.argv[:]
    sys.argv = ["etd_verify.py", "--n", str(min_sample)]
    try:
        rc = etd_verify.main()
    finally:
        sys.argv = saved_argv
    if rc != 0:
        print(f"[etd-gate] ERROR: etd_verify.main() returned {rc}", file=sys.stderr)
        return 2
    return run_dry_run(threshold, min_sample, audit_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="etd_hallucination_gate",
        description="CI gate on ETD Stage-1 hallucination floor.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.13,
        help="Maximum allowed unsupported-share at high confidence (default: 0.13).",
    )
    p.add_argument(
        "--min-sample",
        type=int,
        default=200,
        help="Minimum high-confidence sample size for the gate to be valid (default: 200).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Read existing audit reports instead of running the verifier (zero API tokens).",
    )
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=AUDIT_DIR,
        help=f"Override audit report directory (default: {AUDIT_DIR}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return run_dry_run(args.threshold, args.min_sample, args.audit_dir)
    return run_live(args.threshold, args.min_sample, args.audit_dir)


if __name__ == "__main__":
    sys.exit(main())
