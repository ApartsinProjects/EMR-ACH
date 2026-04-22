"""Annotate every FD with (a) its ground-truth label, (b) the domain-appropriate
`prior_state`, and (c) a `fd_type` partition into `stability` or `change`.

The partitioning exposes **where forecasting skill actually matters**: stability
FDs are status-quo continuations (chronic beaters beating again; perennial-peace
pairs staying peaceful; markets correctly predicting their own majority call);
change FDs are the cases where status quo is wrong and the model must read
evidence to win. Paper headline metrics are reported on the `change` subset.

### Prior-state source per benchmark

| Benchmark | prior_state comes from                        | Stability means                            |
|-----------|-----------------------------------------------|--------------------------------------------|
| gdelt-cameo | modal intensity over prior 30 days          | ground_truth == prior modal intensity      |
| earnings  | mode of prior 4 quarters' surprise class      | current-quarter class == historical mode   |
| forecastbench | crowd_probability: ≥0.5 → Yes, <0.5 → No  | ground_truth == crowd's implied majority   |

The GDELT path also **re-labels ground_truth** to the 3-class Peace/Tension/
Violence intensity taxonomy (see src/common/cameo_intensity.py).

Reads:
  data/unified/forecasts.jsonl               (all benchmarks)
  data/gdelt_cameo/data_kg.csv               (for GDELT prior-state)
  data/earnings/earnings_forecasts.jsonl     (for earnings historical quarters)

Writes (atomically):
  data/unified/forecasts.jsonl               (updated in place)
  data/unified/prior_state_meta.json         (audit: class + slice counts per benchmark)

Usage:
  python scripts/annotate_prior_state.py                       # all benchmarks
  python scripts/annotate_prior_state.py --benchmarks gdelt_cameo,forecastbench
  python scripts/annotate_prior_state.py --gdelt-lookback 30   # GDELT window
  python scripts/annotate_prior_state.py --earnings-quarters 4 # Earnings history depth
  python scripts/annotate_prior_state.py --fb-strength 0.0     # FB 0 = use bare majority;
                                                               #    >0 = require |p-0.5| > τ for stability

Idempotent: re-running with the same inputs produces the same output;
already-annotated FDs are re-annotated from scratch (no accumulation).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.common.cameo_intensity import (
    INTENSITY_CLASSES, INTENSITY_RANK, INTENSITY_DEFINITIONS,
    event_to_intensity, root_code,
)

# Primary task (v2.1, 2026-04-22): Comply vs Surprise — binary, unified across
# all three benchmarks. The legacy domain-specific multiclass label is moved to
# `x_multiclass_hypothesis_set` / `x_multiclass_ground_truth` as a secondary
# ablation target. See docs/FORECAST_DOSSIER.md for the full contract.
BINARY_HYPOTHESIS_SET: list[str] = ["Comply", "Surprise"]
BINARY_HYPOTHESIS_DEFS: dict[str, str] = {
    "Comply":   ("The outcome MATCHES the status-quo expectation derived from "
                 "the prior-state window. A status-quo predictor would be right; "
                 "the evidence did not need to be consulted."),
    "Surprise": ("The outcome BREAKS the status-quo expectation. A status-quo "
                 "predictor would be wrong; correctly predicting this FD "
                 "requires the forecaster to read and reason over the evidence."),
}


def _promote_to_binary(fc: dict, is_stability: bool) -> None:
    """Move the existing domain-specific hypothesis set to x_multiclass_*
    fields and install Comply/Surprise as the primary target."""
    fc["x_multiclass_hypothesis_set"] = list(fc.get("hypothesis_set", []))
    fc["x_multiclass_hypothesis_definitions"] = dict(fc.get("hypothesis_definitions", {}))
    fc["x_multiclass_ground_truth"] = fc.get("ground_truth")
    fc["x_multiclass_ground_truth_idx"] = fc.get("ground_truth_idx")
    fc["hypothesis_set"] = list(BINARY_HYPOTHESIS_SET)
    fc["hypothesis_definitions"] = dict(BINARY_HYPOTHESIS_DEFS)
    fc["ground_truth"] = "Comply" if is_stability else "Surprise"
    fc["ground_truth_idx"] = 0 if is_stability else 1

DATA = ROOT / "data"
UNIFIED = DATA / "unified"
FC_FILE = UNIFIED / "forecasts.jsonl"
KG_FILE = DATA / "gdelt_cameo" / "data_kg.csv"
META_FILE = UNIFIED / "prior_state_meta.json"

# data_kg.csv can have very long Docids fields — bump the CSV limit for reads.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_kg_events() -> dict[tuple[str, str], list[tuple[datetime, str]]]:
    """Index data_kg.csv by (sorted actor pair) -> sorted list of (date, intensity).

    Actor pair is sorted alphabetically so (ISR, PAL) and (PAL, ISR) index
    under the same key — events are usually symmetric at the actor-pair
    granularity we care about.
    """
    if not KG_FILE.exists():
        print(f"[ERROR] {KG_FILE} not found — build_gdelt_cameo.py must run first.")
        sys.exit(2)

    print(f"[kg] loading {KG_FILE} ...")
    pair_index: dict[tuple[str, str], list[tuple[datetime, str]]] = defaultdict(list)
    n_rows = 0
    n_skipped = 0
    with KG_FILE.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            a1, a2 = (row.get("Actor1CountryCode") or "").upper(), (row.get("Actor2CountryCode") or "").upper()
            if not a1 or not a2 or a1 == a2:
                n_skipped += 1; continue
            key = tuple(sorted([a1, a2]))
            intensity = event_to_intensity(row.get("EventBaseCode"))
            if intensity is None:
                n_skipped += 1; continue
            try:
                d = datetime.strptime((row.get("DateStr") or "")[:10], "%Y-%m-%d")
            except ValueError:
                n_skipped += 1; continue
            pair_index[key].append((d, intensity))

    # Sort each pair's events by date for efficient window queries
    for k in pair_index:
        pair_index[k].sort(key=lambda t: t[0])
    print(f"[kg] rows read: {n_rows}  indexed: {n_rows - n_skipped}  skipped: {n_skipped}")
    print(f"[kg] distinct actor pairs: {len(pair_index)}")
    return pair_index


def prior_state(
    actor_pair: tuple[str, str],
    forecast_point: datetime,
    kg_pair_index: dict[tuple[str, str], list[tuple[datetime, str]]],
    lookback_days: int,
) -> tuple[str | None, float, int]:
    """Compute (mode_intensity, stability_fraction, n_events) for the 30d
    window [forecast_point - lookback_days, forecast_point).

    Returns (None, 0.0, 0) if no events are indexed in the window.
    """
    key = tuple(sorted(actor_pair))
    series = kg_pair_index.get(key, [])
    if not series:
        return None, 0.0, 0
    window_start = forecast_point - timedelta(days=lookback_days)
    in_window = [intensity for d, intensity in series if window_start <= d < forecast_point]
    if not in_window:
        return None, 0.0, 0
    counter = Counter(in_window)
    mode_intensity, mode_count = counter.most_common(1)[0]
    stability = mode_count / len(in_window)
    return mode_intensity, stability, len(in_window)


def annotate_gdelt(forecasts: list[dict], kg_index, lookback_days: int) -> dict:
    """Annotate GDELT-CAMEO FDs in place. Returns meta summary."""
    intensity_counter: Counter = Counter()
    slice_counter: Counter = Counter()
    prior_counter: Counter = Counter()
    gt_changed = 0
    n_touched = 0
    n_no_prior = 0

    for fc in forecasts:
        if fc.get("benchmark") != "gdelt-cameo":
            continue
        n_touched += 1
        md = fc.get("metadata", {}) or {}
        ebc = md.get("event_base_code")
        gt_intensity = event_to_intensity(ebc) if ebc else None
        if gt_intensity is None:
            from src.common.cameo_intensity import quad_to_intensity_maybe
            gt_intensity = quad_to_intensity_maybe(fc.get("ground_truth_idx", -1) + 1)
        if gt_intensity is None:
            continue
        old_gt = fc.get("ground_truth")
        fc["hypothesis_set"] = list(INTENSITY_CLASSES)
        fc["hypothesis_definitions"] = dict(INTENSITY_DEFINITIONS)
        fc["ground_truth"] = gt_intensity
        fc["ground_truth_idx"] = INTENSITY_RANK[gt_intensity]
        if old_gt != gt_intensity:
            gt_changed += 1
        intensity_counter[gt_intensity] += 1

        actors = md.get("actors") or []
        if len(actors) < 2:
            fc["prior_state_30d"] = None; fc["prior_state_stability"] = 0.0
            fc["fd_type"] = "unknown"; slice_counter["unknown"] += 1; n_no_prior += 1
            continue
        try:
            fp_dt = datetime.strptime(fc.get("forecast_point", "")[:10], "%Y-%m-%d")
        except ValueError:
            continue
        mode_intensity, stability, n_events = prior_state(
            (actors[0], actors[1]), fp_dt, kg_index, lookback_days)
        fc["prior_state_30d"] = mode_intensity
        fc["prior_state_stability"] = round(stability, 4)
        fc["prior_state_n_events"] = n_events
        if mode_intensity is None:
            fd_type = "unknown"; n_no_prior += 1
        elif mode_intensity == gt_intensity:
            fd_type = "stability"
        else:
            fd_type = "change"
        fc["fd_type"] = fd_type
        slice_counter[fd_type] += 1
        if mode_intensity:
            prior_counter[mode_intensity] += 1
        # Promote fd_type to the primary binary target Comply/Surprise.
        # Multiclass intensity label is preserved under x_multiclass_*.
        if fd_type != "unknown":
            _promote_to_binary(fc, is_stability=(fd_type == "stability"))

    return {
        "n_fds":                          n_touched,
        "lookback_days":                  lookback_days,
        "ground_truth_class_dist":        dict(intensity_counter),
        "prior_state_class_dist":         dict(prior_counter),
        "fd_type_dist":                   dict(slice_counter),
        "ground_truth_relabeled":         gt_changed,
        "missing_prior_context":          n_no_prior,
    }


def annotate_earnings(forecasts: list[dict], quarters: int) -> dict:
    """For each earnings FD, compute prior state from the previous N quarters'
    surprise class (Beat/Meet/Miss) for the same ticker.

    Data source: other earnings FDs in the forecasts list with same ticker
    and earlier forecast_point (earnings_forecasts.jsonl retains all quarters).
    """
    # Group FDs by ticker, sorted by forecast_point
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for fc in forecasts:
        if fc.get("benchmark") != "earnings":
            continue
        by_ticker.setdefault(fc.get("metadata", {}).get("ticker", "?"), []).append(fc)
    for tk, lst in by_ticker.items():
        lst.sort(key=lambda f: f.get("forecast_point", ""))

    slice_counter: Counter = Counter()
    prior_counter: Counter = Counter()
    n_touched = 0
    n_no_prior = 0

    for tk, lst in by_ticker.items():
        for i, fc in enumerate(lst):
            n_touched += 1
            history = [h.get("ground_truth") for h in lst[max(0, i - quarters): i] if h.get("ground_truth")]
            if not history:
                fc["prior_state_30d"] = None; fc["prior_state_stability"] = 0.0
                fc["fd_type"] = "unknown"; slice_counter["unknown"] += 1; n_no_prior += 1
                continue
            counter = Counter(history)
            mode_cls, mode_cnt = counter.most_common(1)[0]
            stability = mode_cnt / len(history)
            fc["prior_state_30d"] = mode_cls
            fc["prior_state_stability"] = round(stability, 4)
            fc["prior_state_n_events"] = len(history)
            if mode_cls == fc.get("ground_truth"):
                fd_type = "stability"
            else:
                fd_type = "change"
            fc["fd_type"] = fd_type
            slice_counter[fd_type] += 1
            prior_counter[mode_cls] += 1
            _promote_to_binary(fc, is_stability=(fd_type == "stability"))

    return {
        "n_fds":                n_touched,
        "quarters_history":     quarters,
        "prior_state_class_dist": dict(prior_counter),
        "fd_type_dist":         dict(slice_counter),
        "missing_prior_context": n_no_prior,
    }


def annotate_forecastbench(forecasts: list[dict], strength: float) -> dict:
    """For each ForecastBench FD, derive prior_state from crowd_probability.

    crowd_probability ≥ 0.5 → market-expected majority "Yes"; else "No".
    If `strength > 0`, require |crowd_probability - 0.5| > strength to
    classify as stability/change; uncertain markets (near 0.5) get
    fd_type = "unknown" (excluded from headline subsets).
    """
    slice_counter: Counter = Counter()
    prior_counter: Counter = Counter()
    n_touched = 0
    n_no_prior = 0
    for fc in forecasts:
        if fc.get("benchmark") != "forecastbench":
            continue
        n_touched += 1
        crowd = fc.get("crowd_probability")
        if crowd is None:
            fc["prior_state_30d"] = None; fc["prior_state_stability"] = 0.0
            fc["fd_type"] = "unknown"; slice_counter["unknown"] += 1; n_no_prior += 1
            continue
        try:
            p = float(crowd)
        except (TypeError, ValueError):
            fc["prior_state_30d"] = None; fc["fd_type"] = "unknown"
            slice_counter["unknown"] += 1; n_no_prior += 1
            continue
        if strength > 0 and abs(p - 0.5) <= strength:
            fc["prior_state_30d"] = None
            fc["prior_state_stability"] = float(abs(p - 0.5) * 2)
            fc["fd_type"] = "unknown"       # crowd too uncertain to anchor
            slice_counter["unknown"] += 1
            continue
        expected = "Yes" if p >= 0.5 else "No"
        stability = float(abs(p - 0.5) * 2)   # 0 at 50/50, 1 at 0/100
        fc["prior_state_30d"] = expected
        fc["prior_state_stability"] = round(stability, 4)
        fc["prior_state_n_events"] = 1        # single crowd signal
        fc["fd_type"] = "stability" if fc.get("ground_truth") == expected else "change"
        slice_counter[fc["fd_type"]] += 1
        prior_counter[expected] += 1
        _promote_to_binary(fc, is_stability=(fc["fd_type"] == "stability"))
    return {
        "n_fds":                 n_touched,
        "strength_threshold":    strength,
        "prior_state_class_dist": dict(prior_counter),
        "fd_type_dist":          dict(slice_counter),
        "missing_crowd_prob":    n_no_prior,
    }


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        try: os.fsync(f.fileno())
        except Exception: pass
    os.replace(str(tmp), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="gdelt_cameo,earnings,forecastbench",
                    help="Comma-separated list of benchmarks to annotate.")
    ap.add_argument("--gdelt-lookback", type=int, default=30,
                    help="GDELT prior-state window in days (default 30).")
    ap.add_argument("--earnings-quarters", type=int, default=4,
                    help="Earnings prior-state depth in quarters (default 4).")
    ap.add_argument("--fb-strength", type=float, default=0.0,
                    help="ForecastBench: require |crowd_prob - 0.5| > strength to "
                         "classify as stability/change; FDs below threshold are "
                         "marked fd_type='unknown' (default 0 = use bare majority).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the meta dict without writing outputs.")
    args = ap.parse_args()

    benchmarks = {b.strip() for b in args.benchmarks.split(",") if b.strip()}

    if not FC_FILE.exists():
        print(f"[ERROR] {FC_FILE} not found; run unify_forecasts.py first.")
        sys.exit(2)

    print(f"[fds] loading {FC_FILE} ...")
    forecasts = [json.loads(l) for l in FC_FILE.open(encoding="utf-8")]
    by_bench = Counter(f.get("benchmark") for f in forecasts)
    print(f"[fds] total={len(forecasts)}  by_benchmark={dict(by_bench)}")

    meta_all: dict = {}
    if "gdelt_cameo" in benchmarks and by_bench.get("gdelt-cameo", 0) > 0:
        kg_index = load_kg_events()
        meta_all["gdelt_cameo"] = annotate_gdelt(forecasts, kg_index, args.gdelt_lookback)
    if "earnings" in benchmarks and by_bench.get("earnings", 0) > 0:
        meta_all["earnings"] = annotate_earnings(forecasts, args.earnings_quarters)
    if "forecastbench" in benchmarks and by_bench.get("forecastbench", 0) > 0:
        meta_all["forecastbench"] = annotate_forecastbench(forecasts, args.fb_strength)

    print("\n[meta]", json.dumps(meta_all, indent=2))
    if args.dry_run:
        print("\n[dry-run] no output written.")
        return

    atomic_write_jsonl(FC_FILE, forecasts)
    META_FILE.write_text(json.dumps(meta_all, indent=2), encoding="utf-8")
    print(f"\n[done] annotated -> {FC_FILE.relative_to(ROOT)}")
    print(f"[done] meta      -> {META_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
