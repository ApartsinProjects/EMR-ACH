"""
Shim that re-exports the small surface of `src.batch_client`, `src.config`,
and `src.eval.metrics` that the baselines need.

Why a shim (not a copy):
  `src.batch_client` (~311 LOC), `src.config` (~100 LOC), and `src.eval.metrics`
  (~268 LOC) are the canonical implementations used elsewhere in the repo. We
  want the baselines to use the same BatchClient and the same Brier/ECE code
  that the paper pipeline uses, so that results are comparable. Copying would
  fork the implementations and drift.

How it works:
  At import time we resolve the project root (4 levels up from this file:
  benchmark/evaluation/baselines/_shim.py -> <repo>) and prepend it to
  sys.path if needed. Then we re-export the symbols each baseline consumes.

If `src/` is ever removed/renamed, every import below is a single place to
fix. Callers should always import through this shim, not `src.*` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root = parents[3] from this file:
#   _shim.py -> baselines -> evaluation -> benchmark -> <repo_root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-exports. Kept narrow on purpose.
from src.batch_client import (  # noqa: E402
    BatchClient,
    BatchRequest,
    BatchResult,
    parse_json_response,
)
from src.config import get_config  # noqa: E402
from src.eval.metrics import brier_score  # noqa: E402

__all__ = [
    "BatchClient",
    "BatchRequest",
    "BatchResult",
    "parse_json_response",
    "get_config",
    "brier_score",
    "REPO_ROOT",
]

REPO_ROOT = _REPO_ROOT
