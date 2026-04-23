"""src.common.config_slices: per-stage config-slice declarations and hashing
(v2.2 [B5]).

Each pipeline stage in scripts/build_benchmark.py only depends on a
narrow subset of keys in configs/default_config.yaml. The orchestrator's
resume logic should treat a stage's cache as valid iff *its slice* is
unchanged, not the entire config blob. Today the resume code rehashes
the whole config and invalidates everything on any edit; this module
provides the declarative slice map plus a stable hash so a stage can
ask "did my inputs change?" cheaply.

This is a pure library: no I/O, no side effects. Called by
``stage_cache.py`` (G1) and by the resume protocol invariants test (C5).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

__all__ = [
    "STAGE_SLICES",
    "slice_config",
    "hash_slice",
    "stage_cache_key",
    "known_stages",
]


# Declarative map: stage_name -> sequence of dotted config keys whose
# values determine the stage's output. Adding a key here is a
# semver-bumping action because it can invalidate caches across the
# fleet. Keys reference configs/default_config.yaml; "*" means "any
# child of this prefix".
#
# Source of truth: docs/V2_2_ARCHITECTURE.md Section 4b "Reuse contract".
STAGE_SLICES: dict[str, tuple[str, ...]] = {
    "fetch_forecastbench": (
        "cutoff",
        "fetch.lookback_days",
        "fetch.spam_blocklist_revision",
        "fetch.forecastbench.eligible_sources",
    ),
    "fetch_gdelt_cameo": (
        "cutoff",
        "fetch.lookback_days",
        "fetch.spam_blocklist_revision",
        "fetch.gdelt_cameo.aggregator_blocklist_revision",
    ),
    "fetch_earnings": (
        "cutoff",
        "fetch.lookback_days",
        "fetch.spam_blocklist_revision",
        "fetch.earnings.tickers_revision",
    ),
    "unify_articles": (
        "cutoff",
        "unify.dedup_strategy",
    ),
    "unify_forecasts": (
        "cutoff",
        "unify.subject_filter",
        "unify.horizon_days",
    ),
    "compute_relevance": (
        "cutoff",
        "relevance.encoder.model",
        "relevance.encoder.batch_size",
        "relevance.encoder.fp16",
        "relevance.encoder.backend",
        "relevance.top_k_per_fd",
        "relevance.lookback_days",
    ),
    "annotate_prior_state": (
        "cutoff",
        "prior_state.window_days",
    ),
    "articles_to_facts": (
        "cutoff",
        "etd.prompt_version",
        "etd.model",
        "etd.strict_dates",
        "etd.strict_quotes",
    ),
    "etd_dedup": (
        "cutoff",
        "etd.dedup.threshold",
        "etd.dedup.encoder",
    ),
    "etd_link": (
        "cutoff",
        "etd.link.threshold",
    ),
    "etd_filter": (
        "cutoff",
        "etd.filter.min_confidence",
        "etd.filter.polarity",
        "etd.filter.no_future",
        "etd.filter.require_linked_fd",
        "etd.filter.source_blocklist",
    ),
    "quality_filter": (
        "cutoff",
        "quality.min_articles_per_fd",
    ),
    "step_publish": (
        "cutoff",
        "publish.bench_set",
    ),
}


def known_stages() -> Iterable[str]:
    """Stable iteration order for tests and reuse-check CLI."""
    return tuple(STAGE_SLICES.keys())


def _resolve(config: Mapping[str, Any], dotted: str) -> Any:
    """Resolve a dotted key against a nested mapping.

    Returns ``None`` if the key is absent at any level. ``None`` is the
    canonical sentinel for "key unset"; callers must distinguish from a
    legitimate ``None`` value via explicit defaults in their schema.
    """
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def slice_config(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Return the subset of ``config`` whose dotted keys are listed in
    ``STAGE_SLICES[stage]``. Unknown stages raise KeyError so a typo
    surfaces immediately rather than producing a permissive empty slice.
    """
    if stage not in STAGE_SLICES:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGE_SLICES)!r}")
    return {dotted: _resolve(config, dotted) for dotted in STAGE_SLICES[stage]}


def hash_slice(slice_dict: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a slice. Uses canonical JSON (sorted keys, no
    whitespace) so insertion order does not matter.
    """
    blob = json.dumps(slice_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stage_cache_key(config: Mapping[str, Any], stage: str) -> str:
    """Convenience: ``hash_slice(slice_config(config, stage))``.

    Suitable for use as the cache-key column in stage_meta.json.
    """
    return hash_slice(slice_config(config, stage))
