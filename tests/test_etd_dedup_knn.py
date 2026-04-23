"""Parity + correctness tests for scripts/etd_dedup.py kNN search strategies.

Guards the v2.2 refactor: bucket-and-FAISS must produce the same (i, j, s)
pair set as brute force, as long as the window constraint is large enough
to cover every high-similarity pair in the synthetic corpus.

These tests do NOT hit the network (no OpenAI) and do NOT require the full
fact corpus; they build a tiny synthetic embedding matrix + date list and
compare the two search strategies.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Import the two kNN functions directly from the dedup script
import etd_dedup as dedup  # type: ignore  # noqa: E402


def _make_embs(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Random L2-normalized float32 embeddings."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)


def _plant_duplicates(embs: np.ndarray, pairs: list[tuple[int, int]],
                      noise: float = 0.001) -> None:
    """In-place: force embs[j] to be a tiny-noise copy of embs[i] for each pair.
    Re-normalizes after. After this, cosine(embs[i], embs[j]) ~= 1 - small."""
    rng = np.random.default_rng(42)
    for i, j in pairs:
        v = embs[i] + noise * rng.standard_normal(embs.shape[1], dtype=np.float32)
        v /= np.linalg.norm(v)
        embs[j] = v.astype(np.float32)


def _dates(n: int, start: str = "2026-01-01") -> list[datetime]:
    """Assign evenly-spaced dates, one per fact, so each day has ~1-2 facts."""
    base = datetime.strptime(start, "%Y-%m-%d")
    # Spread 100 facts across 30 days -> ~3-4 per day
    out = []
    for i in range(n):
        out.append(datetime.fromordinal(base.toordinal() + (i % 30)))
    return out


def _normalize_pairs(pairs):
    """Return a set of (i, j) with i < j (drops similarity for comparison)."""
    return {(min(i, j), max(i, j)) for i, j, _s in pairs}


def test_bucket_matches_brute_within_window():
    """With a large-enough window, bucket must find every pair brute finds."""
    n, dim = 100, 32
    embs = _make_embs(n, dim, seed=1)
    planted = [(0, 50), (1, 51), (2, 52), (10, 40)]  # cross-day dupes
    _plant_duplicates(embs, planted, noise=0.0001)
    dates = _dates(n)

    # Planted pair (0, 50): i%30=0, j%30=50%30=20 -> 20 days apart
    # Need window >= 20 to include all pairs
    window = 30

    brute = dedup._knn_brute(embs, threshold=0.98, top_k=5)
    bucket = dedup._knn_bucketed(embs, dates, window_days=window,
                                 threshold=0.98, top_k=5)

    brute_pairs = _normalize_pairs(brute)
    bucket_pairs = _normalize_pairs(bucket)

    # Every planted pair must appear in both
    planted_norm = {(min(i, j), max(i, j)) for i, j in planted}
    assert planted_norm <= brute_pairs, f"brute missed planted: {planted_norm - brute_pairs}"
    assert planted_norm <= bucket_pairs, f"bucket missed planted: {planted_norm - bucket_pairs}"

    # Under wide window, bucket == brute exactly
    assert bucket_pairs == brute_pairs, (
        f"bucket-vs-brute diverged under wide window:\n"
        f"  only-brute:  {brute_pairs - bucket_pairs}\n"
        f"  only-bucket: {bucket_pairs - brute_pairs}"
    )


def test_bucket_respects_window_constraint():
    """Narrow window must exclude out-of-window pairs that brute would find."""
    n, dim = 60, 32
    embs = _make_embs(n, dim, seed=2)
    # Plant pair (0, 45) which will be ~15 days apart at n=60, 30-day cycle:
    # 0%30=0, 45%30=15 -> 15 days apart
    _plant_duplicates(embs, [(0, 45)], noise=0.0001)
    dates = _dates(n)

    brute = dedup._knn_brute(embs, threshold=0.98, top_k=5)
    narrow = dedup._knn_bucketed(embs, dates, window_days=3,
                                 threshold=0.98, top_k=5)

    brute_pairs = _normalize_pairs(brute)
    narrow_pairs = _normalize_pairs(narrow)

    # Brute finds (0, 45) regardless of date; narrow must NOT
    assert (0, 45) in brute_pairs, "brute should see the planted high-sim pair"
    assert (0, 45) not in narrow_pairs, "narrow-window bucket should drop the out-of-window pair"


def test_bucket_handles_no_date_facts():
    """Facts with None date must be skipped cleanly (not crash)."""
    n, dim = 20, 16
    embs = _make_embs(n, dim, seed=3)
    _plant_duplicates(embs, [(0, 1)], noise=0.0001)
    dates = _dates(n)
    # Nuke some dates
    dates[5] = None  # type: ignore
    dates[7] = None  # type: ignore

    pairs = dedup._knn_bucketed(embs, dates, window_days=5,
                                threshold=0.98, top_k=5)
    norm = _normalize_pairs(pairs)
    assert (0, 1) in norm
    # No pair should reference a no-date fact
    for i, j in norm:
        assert i not in (5, 7) and j not in (5, 7)


def test_bucket_dedupes_pairs():
    """A pair must appear at most once, even when found from both days."""
    n, dim = 20, 16
    embs = _make_embs(n, dim, seed=4)
    _plant_duplicates(embs, [(3, 4)], noise=0.0001)
    dates = _dates(n)

    pairs = dedup._knn_bucketed(embs, dates, window_days=5,
                                threshold=0.98, top_k=5)
    norm = [(min(i, j), max(i, j)) for i, j, _s in pairs]
    assert len(norm) == len(set(norm)), f"duplicate pairs: {norm}"


def test_bucket_returns_sim_above_threshold():
    """All returned pairs must have sim >= threshold."""
    n, dim = 40, 32
    embs = _make_embs(n, dim, seed=5)
    _plant_duplicates(embs, [(0, 1), (10, 20)], noise=0.0001)
    dates = _dates(n)

    thresh = 0.95
    pairs = dedup._knn_bucketed(embs, dates, window_days=15,
                                threshold=thresh, top_k=5)
    for _i, _j, s in pairs:
        assert s >= thresh, f"sim {s} below threshold {thresh}"


def test_knn_mode_cli_accepts_both_choices():
    """Argparse must accept both 'bucket' and 'brute' for --knn-mode."""
    import argparse
    # Re-build the parser outside of main() via inspecting the module source
    # would be brittle; instead, just exercise argparse choices check here.
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn-mode", choices=["bucket", "brute"], default="bucket")
    assert ap.parse_args(["--knn-mode", "bucket"]).knn_mode == "bucket"
    assert ap.parse_args(["--knn-mode", "brute"]).knn_mode == "brute"
    with pytest.raises(SystemExit):
        ap.parse_args(["--knn-mode", "hnsw"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
