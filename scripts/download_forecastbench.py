"""
Build forecastbench_geopolitics.jsonl from the locally cloned repo.

Reads question sets and resolution sets from data/forecastbench_repo/datasets/,
filters for geopolitics/conflict content, matches with resolutions, and writes
data/forecastbench_geopolitics.jsonl.

Usage:
  python scripts/download_forecastbench.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
REPO = DATA / "forecastbench_repo" / "datasets"
OUT  = DATA / "forecastbench_geopolitics.jsonl"

# Only include genuine prediction market / human forecaster sources.
# ACLED stores algorithmic event-count frequencies (not calibrated probabilities),
# and FRED stores financial indicator comparisons. Both have crowd Brier > 0.45.
PREDICTION_MARKET_SOURCES = {"polymarket", "metaculus", "manifold", "infer"}

GEOPOLITICS_KEYWORDS = [
    "china", "taiwan", "ukraine", "russia", "conflict", "war", "invasion",
    "election", "diplomacy", "sanction", "israel", "palestine", "gaza",
    "middle east", "iran", "north korea", "missile", "nuclear", "nato",
    "military", "border", "coup", "ceasefire", "armed", "troops",
    "government", "protest", "president", "prime minister", "referendum",
    "geopolit", "territory", "occupation", "rebel", "insurgent",
]


def is_geopolitics(q: dict) -> bool:
    text = (
        q.get("question", "") + " " +
        q.get("background", "") + " " +
        q.get("resolution_criteria", "")
    ).lower()
    return any(kw in text for kw in GEOPOLITICS_KEYWORDS)


def _filter_all(q: dict) -> bool:
    """No subject filter — include every resolved FB question (v2.1 growth
    config: drops the geopolitics-only restriction, grows from ~530 to
    ~1000–1500 FDs). Set EMRACH_FB_SUBJECT_FILTER=geopolitics to restore
    the legacy filter."""
    return True


# Selectable subject filter, controlled by env var EMRACH_FB_SUBJECT_FILTER.
# Default: "all" (no filter). Pass "geopolitics" for the legacy subset.
import os as _os
_SUBJECT_FILTER_MODE = _os.environ.get("EMRACH_FB_SUBJECT_FILTER", "all").lower()
if _SUBJECT_FILTER_MODE == "geopolitics":
    subject_filter = is_geopolitics
else:
    subject_filter = _filter_all


def load_all() -> tuple[dict, dict]:
    q_dir = REPO / "question_sets"
    r_dir = REPO / "resolution_sets"

    if not REPO.exists():
        print(f"[ERROR] Repo not found at {REPO}")
        print("  Run: git clone https://github.com/forecastingresearch/forecastbench-datasets data/forecastbench_repo")
        sys.exit(1)

    all_questions: dict[str, dict] = {}
    all_resolutions: dict[str, dict] = {}

    # Load questions from all LLM question sets (skip human sets, skip latest)
    q_files = sorted(f for f in q_dir.glob("*-llm.json") if "latest" not in f.name)
    print(f"Found {len(q_files)} question set files")
    for qf in q_files:
        data = json.loads(qf.read_text(encoding="utf-8"))
        qs = data if isinstance(data, list) else data.get("questions", [])
        added = 0
        for q in qs:
            qid = q.get("id")
            if not isinstance(qid, str):  # skip combination questions (list IDs)
                continue
            if qid and qid not in all_questions:
                all_questions[qid] = q
                added += 1
        print(f"  {qf.stem}: {added} new (total {len(all_questions)})")

    # Load all resolution sets
    r_files = sorted(r_dir.glob("*_resolution_set.json"))
    print(f"\nFound {len(r_files)} resolution set files")
    for rf in r_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        rs = data if isinstance(data, list) else data.get("resolutions", [])
        for res in rs:
            rid = res.get("id")
            if not isinstance(rid, str):
                continue
            if rid:
                # Prefer resolved entries
                if rid not in all_resolutions or res.get("resolved"):
                    all_resolutions[rid] = res
        print(f"  {rf.stem}: {len(rs)} resolutions")

    print(f"\nTotal unique questions: {len(all_questions)}")
    print(f"Total resolution entries: {len(all_resolutions)}")
    return all_questions, all_resolutions


def build_dataset(all_questions: dict, all_resolutions: dict) -> list[dict]:
    results = []
    seen_ids: set[str] = set()
    skipped_no_res = 0
    skipped_unresolved = 0
    skipped_no_value = 0
    skipped_no_crowd = 0
    skipped_not_geo = 0

    for qid, q in all_questions.items():
        if not subject_filter(q):
            skipped_not_geo += 1
            continue

        res = all_resolutions.get(qid)
        if not res:
            skipped_no_res += 1
            continue
        if not res.get("resolved", False):
            skipped_unresolved += 1
            continue

        resolved_to = res.get("resolved_to")
        if resolved_to is None:
            skipped_no_value += 1
            continue

        # Only use questions from genuine prediction markets (calibrated human forecasts).
        # ACLED and FRED store algorithmic event counts / financial comparisons, not probabilities.
        if q.get("source") not in PREDICTION_MARKET_SOURCES:
            skipped_no_crowd += 1
            continue

        crowd_prob = q.get("freeze_datetime_value")
        if crowd_prob is None:
            skipped_no_crowd += 1
            continue

        try:
            crowd_prob = float(crowd_prob)
            ground_truth = int(float(resolved_to) >= 0.5)
        except (ValueError, TypeError):
            continue

        # Sanity-check: prediction markets should stay in [0,1]
        if not (0.0 <= crowd_prob <= 1.0):
            skipped_no_crowd += 1
            continue

        if qid in seen_ids:
            continue
        seen_ids.add(qid)

        results.append({
            "id":                f"fb_{qid[:12]}",
            "question":          q.get("question", ""),
            "background":        q.get("background", ""),
            "resolution_date":   res.get("resolution_date", ""),
            "ground_truth":      ground_truth,
            "crowd_probability": round(crowd_prob, 4),
            "source":            q.get("source", ""),
            "category":          "geopolitics",
        })

    print(f"\nFilter stats:")
    print(f"  Not geopolitics: {skipped_not_geo}")
    print(f"  No resolution entry: {skipped_no_res}")
    print(f"  Not yet resolved: {skipped_unresolved}")
    print(f"  No resolved_to value: {skipped_no_value}")
    print(f"  No crowd probability: {skipped_no_crowd}")
    print(f"  Kept: {len(results)}")

    results.sort(key=lambda x: x["resolution_date"])
    return results


def main():
    if not REPO.exists():
        print(f"[ERROR] Repo not cloned. Run:")
        print(f"  git clone https://github.com/forecastingresearch/forecastbench-datasets data/forecastbench_repo")
        sys.exit(1)

    all_questions, all_resolutions = load_all()
    results = build_dataset(all_questions, all_resolutions)

    if not results:
        print("\n[ERROR] No matching questions found.")
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    gt1 = sum(1 for r in results if r["ground_truth"] == 1)
    print(f"\nWritten {len(results)} geopolitics questions -> {OUT.relative_to(ROOT)}")
    print(f"  Resolved Yes: {gt1} ({gt1/len(results)*100:.0f}%)")
    print(f"  Resolved No:  {len(results)-gt1} ({(len(results)-gt1)/len(results)*100:.0f}%)")

    print("\nSample questions:")
    for r in results[:5]:
        gt = "YES" if r["ground_truth"] == 1 else "NO"
        print(f"  [{gt}] [{r['source']}] {r['question'][:90]}")


if __name__ == "__main__":
    main()
