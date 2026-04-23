"""Deprecation shim for benchmark/build.py (v2.1).

The canonical orchestrator now lives at `scripts/build_benchmark.py` at the
repo root. This file used to be a parallel entrypoint that called into a
`benchmark/scripts/` script tree. That tree drifted pre-v2.1 (no Comply/
Surprise promotion, 4-class GDELT labels, no multi-source news fetch, broken
publish path), so we deprecated this file rather than maintain two
diverging trees.

This shim simply forwards to the canonical entrypoint with the same CLI
arguments. Update your scripts and CI to call `scripts/build_benchmark.py`
directly; this file may be removed in a future release.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "scripts" / "build_benchmark.py"


def main() -> int:
    if not CANONICAL.exists():
        print(f"[ERROR] canonical orchestrator not found at {CANONICAL}",
              file=sys.stderr)
        return 1
    print(f"[benchmark/build.py] DEPRECATED: forwarding to {CANONICAL}",
          file=sys.stderr)
    print(f"[benchmark/build.py] Update callers to: "
          f"python scripts/build_benchmark.py [args]", file=sys.stderr)
    cmd = [sys.executable, str(CANONICAL), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
