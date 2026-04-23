"""Per-row fingerprint collision tripwire (v2.2 [C8]).

The unified article pool uses MD5-based fingerprints to dedup rows
across the three fetchers. MD5 is cryptographically broken but that
does not matter here; the concern is accidental collisions on
adversarially-shaped but legitimate input. This test exercises the
fingerprinter over a synthetic pool of ~20k rows whose shapes mimic
the production pool and asserts zero collisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.news_fetcher import art_id_for


def test_synthetic_pool_no_collisions():
    seen: dict[str, tuple[str, str, str]] = {}
    collisions: list[tuple[str, tuple, tuple]] = []
    for bench in ("forecastbench", "gdelt_cameo", "earnings"):
        for url_tail in range(7_000):
            url = f"https://example.com/path/{bench}/{url_tail}"
            date = f"2026-{(url_tail % 12) + 1:02d}-{(url_tail % 28) + 1:02d}"
            aid = art_id_for(bench, url, date)
            key = (bench, url, date)
            if aid in seen and seen[aid] != key:
                collisions.append((aid, seen[aid], key))
            seen[aid] = key
    assert not collisions, f"{len(collisions)} collisions: first {collisions[:3]!r}"


def test_same_inputs_always_same_id():
    a = art_id_for("forecastbench", "https://x.com/1", "2026-01-01")
    b = art_id_for("forecastbench", "https://x.com/1", "2026-01-01")
    assert a == b


def test_different_inputs_give_different_ids():
    a = art_id_for("forecastbench", "https://x.com/1", "2026-01-01")
    b = art_id_for("forecastbench", "https://x.com/2", "2026-01-01")
    c = art_id_for("forecastbench", "https://x.com/1", "2026-01-02")
    d = art_id_for("gdelt_cameo", "https://x.com/1", "2026-01-01")
    assert len({a, b, c, d}) == 4
