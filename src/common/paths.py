"""src.common.paths: single owner of project path conventions (v2.2 [B4] + [B4a]).

Centralizes two concerns previously scattered across the codebase:

1. Repository-root discovery and ``sys.path`` bootstrap so that scripts
   under ``scripts/`` can ``from src.common.* import ...`` without each one
   carrying its own ``_sys.path.insert(0, str(_Path(__file__).parent.parent))``
   block. The redundant pattern caused three production crashes on
   2026-04-23 (commits 060c9cf, ac0b031, a373e89). See backlog item B4a.

2. Layout helpers for the v2.1 directory split between ``data/unified/``
   and per-benchmark ``data/{benchmark}/`` trees, so callers do not have
   to reconstruct path conventions from string literals. See backlog
   item B4.

This module is import-side-effect-free except for the explicit
``bootstrap_sys_path()`` call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "UNIFIED_DIR",
    "BENCHMARK_DIR",
    "bootstrap_sys_path",
    "repo_root",
    "data_dir",
    "unified_dir",
    "benchmark_data_dir",
    "per_benchmark_dir",
    "per_benchmark_articles_path",
    "ensure_dir",
]


def _discover_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until a directory containing ``src/`` and
    either ``scripts/`` or ``pyproject.toml`` is found. Fail fast with a
    clear error if none is reachable; this matches the guarantee in B4a
    that callers know immediately when the file layout has shifted under
    them rather than discovering it at the next ``from src.* import``.
    """
    here = start.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src").is_dir() and (
            (candidate / "scripts").is_dir() or (candidate / "pyproject.toml").exists()
        ):
            return candidate
    raise RuntimeError(
        f"src.common.paths: could not locate repo root from {start!r}; "
        "expected a parent containing both src/ and scripts/."
    )


# Module-level constants computed once at import time.
REPO_ROOT: Path = _discover_repo_root(Path(__file__).parent)
DATA_DIR: Path = REPO_ROOT / "data"
UNIFIED_DIR: Path = DATA_DIR / "unified"
BENCHMARK_DIR: Path = REPO_ROOT / "benchmark" / "data"


def bootstrap_sys_path() -> Path:
    """Idempotently insert ``REPO_ROOT`` at the front of ``sys.path``.

    Intended call site: the first non-stdlib import in any script under
    ``scripts/`` that imports ``from src.* import ...``. Returns
    ``REPO_ROOT`` for the convenience of callers that want to use the
    path immediately.

    This is the single owner of the ``sys.path.insert(0, ROOT)`` idiom.
    Per backlog item B4a, all 17 scripts that previously rolled their
    own version should migrate to::

        from src.common.paths import bootstrap_sys_path
        bootstrap_sys_path()

    Note that the helper itself is in ``src/common/paths.py``, so the
    obvious chicken-and-egg question is unavoidable: the script must
    first make ``src/`` importable before it can call this helper. The
    accepted pattern, documented in backlog B4a, is one short
    bootstrap block at the top of each script::

        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from src.common.paths import bootstrap_sys_path
        bootstrap_sys_path()

    The helper then takes over and provides the canonical assertion +
    idempotency. Future work (B4b in the v2.3 backlog) may eliminate
    even this short prelude via a ``conftest``-style sitecustomize hook.
    """
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return REPO_ROOT


def repo_root() -> Path:
    """Return the project repo root."""
    return REPO_ROOT


def data_dir() -> Path:
    """Return ``<repo_root>/data``."""
    return DATA_DIR


def unified_dir() -> Path:
    """Return ``<repo_root>/data/unified`` (the cross-benchmark pool)."""
    return UNIFIED_DIR


def benchmark_data_dir(cutoff: str | None = None) -> Path:
    """Return ``<repo_root>/benchmark/data`` or, if ``cutoff`` is given,
    the per-cutoff publish directory ``<repo_root>/benchmark/data/{cutoff}``.
    """
    if cutoff is None:
        return BENCHMARK_DIR
    return BENCHMARK_DIR / cutoff


def per_benchmark_dir(benchmark: str) -> Path:
    """Return ``<repo_root>/data/{benchmark}``. Used by the per-benchmark
    fetchers (forecastbench, gdelt_cameo, earnings).
    """
    return DATA_DIR / benchmark


def per_benchmark_articles_path(benchmark: str) -> Path:
    """Convention: per-benchmark pools live at
    ``data/{benchmark}/{benchmark}_articles.jsonl``. Centralizing this
    path eliminates the literal duplication that has accumulated across
    fetchers, unifiers, and audit scripts.
    """
    return per_benchmark_dir(benchmark) / f"{benchmark}_articles.jsonl"


def ensure_dir(path: Path) -> Path:
    """``mkdir(parents=True, exist_ok=True)`` wrapper that returns the
    path for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def known_benchmarks() -> Iterable[str]:
    """The three first-class benchmarks shipped in v2.1 / v2.2."""
    return ("forecastbench", "gdelt_cameo", "earnings")
