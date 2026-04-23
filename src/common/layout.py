"""src.common.layout: typed layout for per-cutoff publish outputs (v2.2 [B9]).

Collects the per-script knowledge of where files live inside
``benchmark/data/{cutoff}/`` into a single frozen dataclass. Used by
``step_publish``, the preflight checks, and the audit scripts to avoid
string-building bugs (e.g. ``"forecasts.jsonl"`` vs
``"forecasts_filtered.jsonl"`` typos that caused the 2026-04-22
stale-publish incident).

Pure library; no I/O except where the methods explicitly say so.
Migration to all callers is incremental (see B9b, deferred to v2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.paths import BENCHMARK_DIR

__all__ = ["CutoffLayout", "layout_for"]


@dataclass(frozen=True)
class CutoffLayout:
    """Canonical layout for ``benchmark/data/{cutoff}/``. Every path
    method returns a :class:`pathlib.Path`; call ``.exists()`` yourself
    if you care about presence.
    """

    cutoff: str
    root: Path

    # ------------------------------------------------------------------
    # Deliverable files
    # ------------------------------------------------------------------

    @property
    def forecasts(self) -> Path:
        return self.root / "forecasts.jsonl"

    @property
    def forecasts_change(self) -> Path:
        return self.root / "forecasts_change.jsonl"

    @property
    def forecasts_stability(self) -> Path:
        return self.root / "forecasts_stability.jsonl"

    @property
    def articles(self) -> Path:
        return self.root / "articles.jsonl"

    @property
    def benchmark_yaml(self) -> Path:
        return self.root / "benchmark.yaml"

    @property
    def build_manifest(self) -> Path:
        return self.root / "build_manifest.json"

    @property
    def checksums(self) -> Path:
        return self.root / "checksums.sha256"

    # ------------------------------------------------------------------
    # Per-benchmark article files (copies of data/{bench}/{bench}_articles.jsonl)
    # ------------------------------------------------------------------

    def per_benchmark_articles(self, benchmark: str) -> Path:
        return self.root / f"{benchmark}_articles.jsonl"

    def per_benchmark_articles_checksum(self, benchmark: str) -> Path:
        return self.root / f"{benchmark}_articles.checksums.json"

    # ------------------------------------------------------------------
    # Audit / diagnostic files
    # ------------------------------------------------------------------

    @property
    def quality_meta(self) -> Path:
        return self.root / "quality_meta.json"

    @property
    def relevance_meta(self) -> Path:
        return self.root / "relevance_meta.json"

    @property
    def etd_facts_production(self) -> Path:
        return self.root / f"facts.v1_production_{self.cutoff}.jsonl"

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def all_deliverable_files(self) -> tuple[Path, ...]:
        """The files whose integrity ``step_publish`` should assert."""
        return (
            self.forecasts,
            self.forecasts_change,
            self.forecasts_stability,
            self.articles,
            self.benchmark_yaml,
            self.build_manifest,
        )


def layout_for(cutoff: str, *, benchmark_root: Path | None = None) -> CutoffLayout:
    """Build a :class:`CutoffLayout` for the given cutoff. Defaults to
    ``<repo>/benchmark/data/{cutoff}`` unless ``benchmark_root`` overrides.
    """
    root = (benchmark_root or BENCHMARK_DIR) / cutoff
    return CutoffLayout(cutoff=cutoff, root=root)
