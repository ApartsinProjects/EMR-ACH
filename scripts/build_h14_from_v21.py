"""
v2.2 reuse-first rebuild: apply horizon=14 leakage filter to the v2.1 publish.

Reads:
  benchmark/data/2026-01-01/forecasts.jsonl
  benchmark/data/2026-01-01/articles.jsonl

Produces (atomic):
  benchmark/data/2026-01-01-h14/forecasts.jsonl
  benchmark/data/2026-01-01-h14/articles.jsonl
  benchmark/data/2026-01-01-h14/benchmark.yaml
  benchmark/data/2026-01-01-h14/build_manifest.json
  benchmark/data/2026-01-01-h14/meta/dangling_article_ids.txt

Scope: forecastbench + earnings only (gdelt-cameo deferred).

For every surviving FD:
  - forecast_point = resolution_date - 14d
  - default_horizon_days = 14
  - lookback_days = 30
  - article_ids filtered to publish_date <= forecast_point
Drops FDs where len(article_ids) == 0 after the filter.

Usage:
  /c/Python314/python scripts/build_h14_from_v21.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "benchmark" / "data" / "2026-01-01"
OUT_DIR = REPO_ROOT / "benchmark" / "data" / "2026-01-01-h14"

HORIZON_DAYS = 14
LOOKBACK_DAYS = 30
BENCHMARKS_IN_SCOPE = {"forecastbench", "earnings"}


def _parse_date(value):
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    s = str(value)
    # Accept full ISO timestamps too.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s[: len(fmt) if "T" in fmt else 10], fmt).date() if fmt != "%Y-%m-%d" else datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    fd_in = SRC_DIR / "forecasts.jsonl"
    art_in = SRC_DIR / "articles.jsonl"
    if not fd_in.exists() or not art_in.exists():
        print(f"[fatal] missing source files under {SRC_DIR}", file=sys.stderr)
        return 2

    # ---------- Load articles ----------
    articles: dict[str, dict] = {}
    with fd_in.open(encoding="utf-8") as fh:
        pass
    with art_in.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            aid = a.get("id")
            if aid:
                articles[aid] = a
    print(f"[load] v2.1 articles: {len(articles):,}")

    # ---------- Process FDs ----------
    counts_in = Counter()
    counts_out = Counter()
    drop_reasons: Counter = Counter()
    referenced_aids: set[str] = set()
    dangling_aids: set[str] = set()

    leakage_violations = 0
    horizon_violations = 0

    kept_fds: list[dict] = []

    with fd_in.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fd = json.loads(line)
            bench = fd.get("benchmark")
            counts_in[bench] += 1

            if bench not in BENCHMARKS_IN_SCOPE:
                drop_reasons[f"out_of_scope:{bench}"] += 1
                continue

            rd = _parse_date(fd.get("resolution_date"))
            if rd is None:
                drop_reasons["bad_resolution_date"] += 1
                continue

            new_fp = rd - timedelta(days=HORIZON_DAYS)

            orig_ids = list(fd.get("article_ids") or [])
            filtered_ids: list[str] = []
            for aid in orig_ids:
                a = articles.get(aid)
                if a is None:
                    dangling_aids.add(aid)
                    continue
                pd = _parse_date(a.get("publish_date"))
                if pd is None:
                    # No-date articles: drop per H6 leakage-safe default.
                    continue
                if pd <= new_fp:
                    filtered_ids.append(aid)

            if not filtered_ids:
                drop_reasons[f"empty_after_leakage:{bench}"] += 1
                continue

            fd["forecast_point"] = new_fp.isoformat()
            fd["default_horizon_days"] = HORIZON_DAYS
            fd["lookback_days"] = LOOKBACK_DAYS
            fd["article_ids"] = filtered_ids

            # Invariants.
            if (rd - new_fp).days != HORIZON_DAYS:
                horizon_violations += 1
            for aid in filtered_ids:
                pd = _parse_date(articles[aid].get("publish_date"))
                if pd is None or pd > new_fp:
                    leakage_violations += 1

            referenced_aids.update(filtered_ids)
            counts_out[bench] += 1
            kept_fds.append(fd)

    # ---------- Write outputs ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

    fd_text = "\n".join(json.dumps(fd, ensure_ascii=False) for fd in kept_fds) + ("\n" if kept_fds else "")
    _atomic_write(OUT_DIR / "forecasts.jsonl", fd_text)

    art_text_parts: list[str] = []
    for aid in sorted(referenced_aids):
        a = articles.get(aid)
        if a is None:
            continue
        art_text_parts.append(json.dumps(a, ensure_ascii=False))
    _atomic_write(OUT_DIR / "articles.jsonl", "\n".join(art_text_parts) + ("\n" if art_text_parts else ""))

    dangling_text = "\n".join(sorted(dangling_aids))
    if dangling_text:
        dangling_text += "\n"
    _atomic_write(OUT_DIR / "meta" / "dangling_article_ids.txt", dangling_text)

    # benchmark.yaml (documentation copy; not a build driver here).
    yaml_text = (
        "# v2.2 reuse-first rebuild config (h14 leakage filter on v2.1 publish).\n"
        "# Built by scripts/build_h14_from_v21.py; NOT a build_benchmark.py driver.\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        "\n"
        "model_cutoff: '2026-01-01'\n"
        "cutoff_buffer_days: 0\n"
        f"default_forecast_horizon_days: {HORIZON_DAYS}\n"
        f"default_lookback_days: {LOOKBACK_DAYS}\n"
        "benchmarks:\n"
        "  forecastbench:\n"
        "    enabled: true\n"
        f"    forecast_horizon_days: {HORIZON_DAYS}\n"
        f"    lookback_days: {LOOKBACK_DAYS}\n"
        "  earnings:\n"
        "    enabled: true\n"
        f"    forecast_horizon_days: {HORIZON_DAYS}\n"
        f"    lookback_days: {LOOKBACK_DAYS}\n"
        "  gdelt_cameo:\n"
        "    enabled: false   # deferred per PROJECT_SPEC §10.1\n"
        "source:\n"
        "  parent_cutoff: '2026-01-01'\n"
        "  strategy: reuse-first (strategy A)\n"
    )
    _atomic_write(OUT_DIR / "benchmark.yaml", yaml_text)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "script": "scripts/build_h14_from_v21.py",
        "strategy": "reuse-first (PROJECT_SPEC §6.1A)",
        "parent_cutoff": "2026-01-01",
        "output_cutoff": "2026-01-01-h14",
        "horizon_days": HORIZON_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "benchmarks_in_scope": sorted(BENCHMARKS_IN_SCOPE),
        "fd_in": dict(counts_in),
        "fd_out": dict(counts_out),
        "fd_total_in": sum(counts_in.values()),
        "fd_total_out": len(kept_fds),
        "articles_in": len(articles),
        "articles_out": len(referenced_aids),
        "dangling_article_ids": sorted(dangling_aids)[:50],
        "dangling_article_ids_count": len(dangling_aids),
        "drop_reasons": dict(drop_reasons),
        "leakage_violations": leakage_violations,
        "horizon_violations": horizon_violations,
    }
    _atomic_write(OUT_DIR / "build_manifest.json", json.dumps(manifest, indent=2) + "\n")

    # ---------- Stats + assertion summary ----------
    print("")
    print("=== v2.2 h14 reuse-first rebuild ===")
    print(f"out dir: {OUT_DIR}")
    print(f"FD in (all benchmarks): {sum(counts_in.values())}  breakdown={dict(counts_in)}")
    print(f"FD out (in scope):      {len(kept_fds)}  breakdown={dict(counts_out)}")
    print(f"articles out: {len(referenced_aids)} / {len(articles)} reused")
    print(f"dangling article ids (referenced but missing in pool): {len(dangling_aids)}")
    print(f"drop reasons: {dict(drop_reasons)}")
    print("")
    print("leakage assertion: every kept FD article has publish_date <= forecast_point")
    print(f"  leakage violations: {leakage_violations}")
    print(f"  horizon violations: {horizon_violations}")
    if leakage_violations or horizon_violations:
        print("[FAIL] invariant violations detected")
        return 3
    print("[OK] invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
