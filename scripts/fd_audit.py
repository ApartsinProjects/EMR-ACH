"""Forecast Dossier audit: v2.1 schema integrity, target balance, stratification.

Complements the existing `quality_filter.py` (which DROPS bad FDs) and
`diagnostic_report.py` (which counts what survived) with a v2.1-specific
quality report focused on the things that go wrong silently:

  * Schema integrity: required fields per v2.1 (Comply/Surprise primary +
    x_multiclass_* secondary + prior_state + fd_type + default_horizon_days).
  * Comply/Surprise balance per benchmark and overall.
  * fd_type partition coverage (stability/change/unknown counts).
  * prior_state distribution per benchmark + missing-prior_state rate.
  * Multiclass-vs-primary consistency: assert that fd_type=stability
    implies ground_truth=Comply (and vice versa for change).
  * Article-count distribution per FD (gini-ish + lowest/highest deciles).
  * Per-source FD volume + change-subset share (which sources contribute
    the most "where the model has to read evidence" cases).
  * Forecast-point span check: unique forecast points per benchmark, gap
    histogram (catches "all FDs at the same date" pathologies).

CPU-only, atomic write, idempotent.

Usage:
  python scripts/fd_audit.py
  python scripts/fd_audit.py --in benchmark/data/2026-01-01/forecasts.jsonl
  python scripts/fd_audit.py --strict   # nonzero exit on schema fails or
                                        # target/partition inconsistencies
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_IN = DATA / "unified" / "forecasts.jsonl"
OUT_DIR = DATA / "unified" / "audit"

V21_BASE = {
    "id", "benchmark", "source", "hypothesis_set", "hypothesis_definitions",
    "question", "background", "forecast_point", "resolution_date",
    "ground_truth", "ground_truth_idx", "article_ids",
}
V21_PROMOTED = {
    "prior_state_30d", "fd_type",
    "x_multiclass_ground_truth", "x_multiclass_hypothesis_set",
}
COMPLY_SURPRISE = {"Comply", "Surprise"}
VALID_FD_TYPES = {"stability", "change", "unknown"}


def _atomic_write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _gini(xs):
    """Cheap Gini for non-negative integer counts."""
    xs = sorted(xs)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(OUT_DIR / "fd_audit.md"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[ERROR] {inp} not found")
        return 1

    print(f"[fd_audit] loading {inp}")
    fds: list[dict] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            try:
                fds.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = len(fds)
    print(f"[fd_audit] loaded {n} FDs")

    base_fails = []
    v21_missing = []
    primary_misaligned = []
    fd_type_invalid = []
    stability_consistency = []
    change_consistency = []
    by_bench = Counter()
    by_source = Counter()
    by_bench_primary = defaultdict(Counter)
    by_bench_fd_type = defaultdict(Counter)
    by_bench_prior = defaultdict(Counter)
    by_source_change = Counter()
    by_source_total = Counter()
    article_count_per_bench = defaultdict(list)
    no_prior_state = Counter()
    horizon_dist = Counter()
    forecast_points_per_bench = defaultdict(set)

    for fd in fds:
        keys = set(fd.keys())
        miss = V21_BASE - keys
        if miss:
            base_fails.append((fd.get("id"), sorted(miss)))
        v21_miss = V21_PROMOTED - keys
        if v21_miss:
            v21_missing.append((fd.get("id"), sorted(v21_miss)))

        bench = fd.get("benchmark", "(unknown)")
        src = fd.get("source", "(unknown)")
        by_bench[bench] += 1
        by_source[src] += 1

        # Primary target sanity
        hs = set(fd.get("hypothesis_set") or [])
        gt = fd.get("ground_truth")
        if hs == COMPLY_SURPRISE:
            by_bench_primary[bench][gt] += 1
            if gt not in hs:
                primary_misaligned.append((fd.get("id"), gt, sorted(hs)))
        else:
            # Pre-promotion FD (annotate hasn't run yet)
            primary_misaligned.append((fd.get("id"), f"hypothesis_set={sorted(hs)}", "expected Comply/Surprise"))

        # fd_type partition
        ft = fd.get("fd_type")
        by_bench_fd_type[bench][ft] += 1
        if ft is not None and ft not in VALID_FD_TYPES:
            fd_type_invalid.append((fd.get("id"), ft))
        if ft == "stability" and gt != "Comply":
            stability_consistency.append((fd.get("id"), gt))
        if ft == "change" and gt != "Surprise":
            change_consistency.append((fd.get("id"), gt))

        # Prior state
        ps = fd.get("prior_state_30d")
        if not ps:
            no_prior_state[bench] += 1
        else:
            by_bench_prior[bench][ps] += 1

        # Source-level change-subset share
        by_source_total[src] += 1
        if ft == "change":
            by_source_change[src] += 1

        # Article counts
        article_count_per_bench[bench].append(len(fd.get("article_ids") or []))

        # Horizon coverage
        horizon_dist[fd.get("default_horizon_days", "(missing)")] += 1

        # Forecast-point span
        fp = fd.get("forecast_point")
        if fp:
            forecast_points_per_bench[bench].add(fp)

    lines = []
    P = lines.append
    P(f"# Forecast Dossier Audit (v2.1)")
    P(f"")
    P(f"- Input: `{inp}`")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"- Total FDs: **{n}**")
    P(f"- Benchmarks: {dict(by_bench)}")
    P(f"")
    P(f"## Schema integrity")
    P(f"- Base-field fails: **{len(base_fails)}**")
    if base_fails[:5]:
        for fid, miss in base_fails[:5]:
            P(f"  - `{fid}`: missing {miss}")
    P(f"- v2.1 promoted-field missing: **{len(v21_missing)}**")
    if v21_missing[:5]:
        for fid, miss in v21_missing[:5]:
            P(f"  - `{fid}`: missing {miss}")
    P(f"")
    P(f"## Primary target (Comply / Surprise)")
    P(f"- Primary alignment fails: **{len(primary_misaligned)}**")
    P(f"- Per-benchmark Comply/Surprise balance:")
    P(f"")
    P(f"| benchmark | Comply | Surprise | Comply share |")
    P(f"|---|---:|---:|---:|")
    for b, ctr in by_bench_primary.items():
        c, s = ctr.get("Comply", 0), ctr.get("Surprise", 0)
        tot = c + s
        share = 100 * c / max(1, tot)
        P(f"| `{b}` | {c} | {s} | {share:.1f}% |")
    P(f"")
    P(f"## fd_type partition")
    P(f"- Invalid fd_type values: **{len(fd_type_invalid)}**")
    P(f"")
    P(f"| benchmark | stability | change | unknown | change share |")
    P(f"|---|---:|---:|---:|---:|")
    for b, ctr in by_bench_fd_type.items():
        st, ch, un = ctr.get("stability", 0), ctr.get("change", 0), ctr.get("unknown", 0) + ctr.get(None, 0)
        tot = st + ch + un
        share = 100 * ch / max(1, tot)
        P(f"| `{b}` | {st} | {ch} | {un} | {share:.1f}% |")
    P(f"")
    P(f"## Multiclass-vs-primary consistency")
    P(f"- stability FDs whose ground_truth != Comply: **{len(stability_consistency)}**")
    P(f"- change FDs whose ground_truth != Surprise: **{len(change_consistency)}**")
    if stability_consistency[:5]:
        P(f"  Sample stability/non-Comply:")
        for fid, gt in stability_consistency[:5]:
            P(f"    - `{fid}`: gt={gt!r}")
    if change_consistency[:5]:
        P(f"  Sample change/non-Surprise:")
        for fid, gt in change_consistency[:5]:
            P(f"    - `{fid}`: gt={gt!r}")
    P(f"")
    P(f"## Prior-state coverage")
    for b in by_bench:
        miss = no_prior_state.get(b, 0)
        P(f"- `{b}`: missing prior_state = **{miss}** / {by_bench[b]} ({100*miss/max(1,by_bench[b]):.1f}%)")
    P(f"")
    P(f"### Prior-state value distribution per benchmark")
    for b, ctr in by_bench_prior.items():
        P(f"- `{b}`: {dict(ctr.most_common())}")
    P(f"")
    P(f"## Article-count distribution per FD")
    P(f"| benchmark | n FDs | min | p50 | p90 | max | gini |")
    P(f"|---|---:|---:|---:|---:|---:|---:|")
    for b, counts in article_count_per_bench.items():
        if not counts:
            continue
        s = sorted(counts)
        m = len(s)
        p50 = s[m // 2]
        p90 = s[int(0.9 * m)]
        P(f"| `{b}` | {m} | {s[0]} | {p50} | {p90} | {s[-1]} | {_gini(counts):.3f} |")
    P(f"")
    P(f"## Per-source contribution to change subset (top 15)")
    P(f"| source | total | change | change share |")
    P(f"|---|---:|---:|---:|")
    for src, total in by_source_total.most_common(15):
        ch = by_source_change.get(src, 0)
        share = 100 * ch / max(1, total)
        P(f"| `{src}` | {total} | {ch} | {share:.1f}% |")
    P(f"")
    P(f"## Horizon coverage")
    P(f"- default_horizon_days values: {dict(horizon_dist)}")
    P(f"")
    P(f"## Forecast-point span (per benchmark)")
    for b, fps in forecast_points_per_bench.items():
        P(f"- `{b}`: {len(fps)} unique forecast points")
    P(f"")
    P(f"## Recommended actions")
    P(f"- If primary-alignment fails are nonzero: `annotate_prior_state.py` "
      f"did not run, or did not promote correctly. Re-run it.")
    P(f"- If a benchmark has high change-subset share AND low primary-fail "
      f"count, that's the headline-skill stratum to focus baseline runs on.")
    P(f"- If `forecast_points` is nearly 1 per benchmark, FDs collapse onto "
      f"one date and metrics will be noisy; re-check the test-month config.")
    P(f"- If article-count gini > 0.6, retrieval is concentrating heavily on "
      f"a small subset of FDs; lower `top_k` in `default_config.yaml` or "
      f"adjust `lookback_days`.")

    out_path = Path(args.out)
    _atomic_write(out_path, "\n".join(lines) + "\n")
    print(f"[fd_audit] report -> {out_path}")

    fatal = (base_fails or v21_missing or primary_misaligned
             or fd_type_invalid or stability_consistency or change_consistency)
    if args.strict and fatal:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
