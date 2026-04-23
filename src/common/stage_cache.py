"""src.common.stage_cache: reuse-contract primitives for pipeline stages
(v2.2 [G1]).

Implements the per-stage reuse keys, invalidation triggers, and resume
invariants spelled out in docs/V2_2_ARCHITECTURE.md Section 4b. Each
stage exposes :meth:`StageCache.cache_key`, :meth:`is_valid`,
:meth:`invalidate`. The build orchestrator queries these instead of the
current ad-hoc ``--skip-*`` flag soup.

This module is the primitive library. The wiring into
``scripts/build_benchmark.py`` is DEFERRED because that file is in-flight;
this ships the substrate so the wiring PR is a minimal diff once
build_benchmark.py stabilizes.

Storage: one JSON file per stage at
``data/stage_meta/{cutoff}/{stage}.json`` with fields::

    {
      "stage": str,
      "cutoff": str,
      "cache_key": str,       # stage_cache_key(config, stage)
      "completed_at": iso8601,
      "outputs": [str, ...],  # absolute paths the stage produced
      "n_rows": int,          # heuristic payload count for sanity checks
    }

A stage is valid iff (a) the meta file exists, (b) its ``cache_key``
matches the live one, and (c) every listed output path still exists.
Any single failure invalidates the stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.common.config_slices import stage_cache_key
from src.common.paths import DATA_DIR

__all__ = [
    "StageMeta",
    "StageCache",
    "default_meta_root",
]


def default_meta_root() -> Path:
    """``<repo_root>/data/stage_meta/``."""
    return DATA_DIR / "stage_meta"


@dataclass
class StageMeta:
    """Persistent per-stage record."""

    stage: str
    cutoff: str
    cache_key: str
    completed_at: str
    outputs: list[str] = field(default_factory=list)
    n_rows: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "StageMeta":
        return cls(
            stage=str(obj["stage"]),
            cutoff=str(obj["cutoff"]),
            cache_key=str(obj["cache_key"]),
            completed_at=str(obj["completed_at"]),
            outputs=[str(p) for p in obj.get("outputs", [])],
            n_rows=int(obj.get("n_rows", 0)),
        )


class StageCache:
    """Per-(stage, cutoff) cache handle. Cheap to construct; does I/O
    only on :meth:`record` and :meth:`is_valid`.
    """

    def __init__(
        self,
        stage: str,
        cutoff: str,
        *,
        meta_root: Path | None = None,
    ) -> None:
        self.stage = stage
        self.cutoff = cutoff
        self._root = meta_root or default_meta_root()

    @property
    def meta_path(self) -> Path:
        return self._root / self.cutoff / f"{self.stage}.json"

    def cache_key(self, config: Mapping[str, Any]) -> str:
        return stage_cache_key(config, self.stage)

    def load(self) -> StageMeta | None:
        p = self.meta_path
        if not p.exists():
            return None
        try:
            return StageMeta.from_json(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None

    def is_valid(self, config: Mapping[str, Any]) -> bool:
        meta = self.load()
        if meta is None:
            return False
        if meta.cache_key != self.cache_key(config):
            return False
        # Every recorded output must still exist; catches the 2026-04-22
        # earnings-articles deletion failure mode at resume time.
        for out in meta.outputs:
            if not Path(out).exists():
                return False
        return True

    def invalidate(self) -> None:
        p = self.meta_path
        if p.exists():
            p.unlink()

    def record(
        self,
        config: Mapping[str, Any],
        *,
        outputs: list[Path] | None = None,
        n_rows: int = 0,
    ) -> StageMeta:
        """Atomically write a fresh StageMeta for this stage."""
        meta = StageMeta(
            stage=self.stage,
            cutoff=self.cutoff,
            cache_key=self.cache_key(config),
            completed_at=datetime.now(tz=timezone.utc).isoformat(),
            outputs=[str(p) for p in (outputs or [])],
            n_rows=n_rows,
        )
        p = self.meta_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(meta.to_json(), indent=2, sort_keys=True))
        tmp.replace(p)
        return meta
