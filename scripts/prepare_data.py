"""
**DEPRECATED (2026-04-22).** This is a legacy downloader used during the very
first pilot of this project to clone MIRAI's original dataset
(https://github.com/yecchen/MIRAI) for the Nov-2023 test set. The production
pipeline no longer depends on MIRAI's repo or dataset — our Geopolitics
subset is built from scratch by `scripts/build_gdelt_cameo.py` against the
public GDELT 2.0 Knowledge Graph.

MIRAI is cited in the paper as a prior-work external anchor only. Do NOT run
this script as part of a publication build; it is retained for reproducibility
of the very-early pilot numbers only.

For the current production build, run:

    python scripts/build_benchmark.py --cutoff YYYY-MM-DD

which invokes the three domain-builders (GDELT-CAMEO + ForecastBench + Earnings)
end-to-end without touching the MIRAI repo.

────────────────────────────────────────────────────────────────────────────
ORIGINAL HEADER (kept for historical reference):

Download and prepare benchmark data for EMR-ACH experiments.

MIRAI benchmark:
  - Clones/pulls https://github.com/yecchen/MIRAI
  - Converts their test split to the expected JSONL format

ForecastBench:
  - Downloads geopolitics/conflict questions resolved Oct-Dec 2024
  - Filters to N=300 questions with known resolutions

Usage:
  python scripts/prepare_data.py               # download everything
  python scripts/prepare_data.py --mirai-only
  python scripts/prepare_data.py --fb-only
  python scripts/prepare_data.py --check        # verify data is present

The script falls back to generating mock data when network access fails,
so smoke tests always work regardless of data availability.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

MIRAI_QUERIES_PATH   = DATA / "mirai_test_queries.jsonl"
MIRAI_ARTICLES_PATH  = DATA / "mirai_articles.jsonl"
FORECASTBENCH_PATH   = DATA / "forecastbench_geopolitics.jsonl"
MIRAI_CLONE_DIR      = DATA / "mirai_repo"

HYPOTHESES = ["VC", "MC", "VK", "MK"]
HYPOTHESIS_NAMES = {
    "VC": "Verbal Cooperation",
    "MC": "Material Cooperation",
    "VK": "Verbal Conflict",
    "MK": "Material Conflict",
}


# ── MIRAI ─────────────────────────────────────────────────────────────────────

def clone_or_update_mirai():
    """Clone MIRAI repo if not present, else git pull."""
    if not MIRAI_CLONE_DIR.exists():
        print("Cloning MIRAI repo (this may take a few minutes)...")
        result = subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/yecchen/MIRAI.git",
             str(MIRAI_CLONE_DIR)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [WARN] Clone failed: {result.stderr.strip()}")
            return False
        print("  Clone complete.")
    else:
        print("MIRAI repo already present, pulling latest...")
        result = subprocess.run(
            ["git", "-C", str(MIRAI_CLONE_DIR), "pull"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [WARN] Pull failed: {result.stderr.strip()}")
    return True


def _cameo_to_label(cameo_code: str) -> str:
    """Map CAMEO verb code to quad-class label."""
    try:
        code = int(str(cameo_code).split(".")[0])
    except (ValueError, AttributeError):
        return "VK"
    if 1 <= code <= 4:
        return "VC"
    elif 5 <= code <= 8:
        return "MC"
    elif 9 <= code <= 16:
        return "VK"
    elif 17 <= code <= 20:
        return "MK"
    return "VK"


def convert_mirai_queries():
    """Convert MIRAI test split to expected JSONL format."""
    print("\nConverting MIRAI test queries...")

    # Try several common locations in the repo
    candidates = [
        MIRAI_CLONE_DIR / "data" / "MIRAI" / "test_mirai.json",
        MIRAI_CLONE_DIR / "data" / "test_mirai.json",
        MIRAI_CLONE_DIR / "dataset" / "test.json",
        MIRAI_CLONE_DIR / "data" / "test.json",
    ]
    test_file = next((p for p in candidates if p.exists()), None)
    if test_file is None:
        # List what's available
        for p in sorted(MIRAI_CLONE_DIR.rglob("*.json"))[:10]:
            print(f"  Found: {p.relative_to(MIRAI_CLONE_DIR)}")
        print("[WARN] Could not locate MIRAI test file. Manual setup required.")
        print("  Please place test queries as: data/mirai_test_queries.jsonl")
        print("  See data/README.md for the expected format.")
        return False

    print(f"  Reading from: {test_file.relative_to(ROOT)}")
    with open(test_file) as f:
        raw = json.load(f)

    queries = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = list(raw.values())
    else:
        print("[ERROR] Unexpected MIRAI JSON structure.")
        return False

    for item in items:
        # Handle both flat and nested structures
        label = item.get("label") or item.get("relation_type") or ""
        if not label and item.get("cameo_code"):
            label = _cameo_to_label(item["cameo_code"])
        if label not in HYPOTHESES:
            label = _cameo_to_label(item.get("cameo_code", "12"))

        queries.append({
            "id":         item.get("id") or item.get("event_id", f"q{len(queries):04d}"),
            "timestamp":  item.get("timestamp") or item.get("date", "2023-01-01"),
            "subject":    item.get("subject") or item.get("subject_country", "Unknown"),
            "relation":   item.get("relation") or item.get("verb", ""),
            "object":     item.get("object") or item.get("object_country", "Unknown"),
            "label":      label,
            "label_full": HYPOTHESIS_NAMES.get(label, ""),
            "doc_ids":    item.get("doc_ids") or item.get("article_ids", []),
        })

    with open(MIRAI_QUERIES_PATH, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    label_dist = {}
    for q in queries:
        label_dist[q["label"]] = label_dist.get(q["label"], 0) + 1

    print(f"  Written {len(queries)} queries -> {MIRAI_QUERIES_PATH.relative_to(ROOT)}")
    print(f"  Label distribution: {label_dist}")
    return True


def convert_mirai_articles():
    """Convert MIRAI article corpus to expected JSONL format."""
    print("\nConverting MIRAI articles...")

    candidates = [
        MIRAI_CLONE_DIR / "data" / "MIRAI" / "news_articles.json",
        MIRAI_CLONE_DIR / "data" / "news_articles.json",
        MIRAI_CLONE_DIR / "data" / "articles.json",
        MIRAI_CLONE_DIR / "dataset" / "articles.json",
    ]
    art_file = next((p for p in candidates if p.exists()), None)
    if art_file is None:
        for p in sorted(MIRAI_CLONE_DIR.rglob("*.json"))[:15]:
            print(f"  Found: {p.relative_to(MIRAI_CLONE_DIR)}")
        print("[WARN] Could not locate MIRAI articles file.")
        print("  Please place articles as: data/mirai_articles.jsonl")
        return False

    print(f"  Reading from: {art_file.relative_to(ROOT)}")
    with open(art_file) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = list(raw.values())
    else:
        print("[ERROR] Unexpected article JSON structure.")
        return False

    with open(MIRAI_ARTICLES_PATH, "w") as f:
        for art in items:
            record = {
                "id":               art.get("id") or art.get("article_id", f"art_{len(items)}"),
                "title":            art.get("title", ""),
                "abstract":         art.get("abstract") or art.get("summary", ""),
                "text":             art.get("text") or art.get("content", ""),
                "date":             art.get("date") or art.get("publish_date", ""),
                "source":           art.get("source") or art.get("outlet", ""),
                "country_mentions": art.get("country_mentions") or art.get("countries", []),
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Written {len(items)} articles -> {MIRAI_ARTICLES_PATH.relative_to(ROOT)}")
    return True


def prepare_mirai():
    if MIRAI_QUERIES_PATH.exists() and MIRAI_ARTICLES_PATH.exists():
        n_q = sum(1 for _ in open(MIRAI_QUERIES_PATH))
        n_a = sum(1 for _ in open(MIRAI_ARTICLES_PATH))
        print(f"MIRAI already prepared: {n_q} queries, {n_a} articles")
        return True

    ok = clone_or_update_mirai()
    if not ok:
        return False

    ok_q = convert_mirai_queries()
    ok_a = convert_mirai_articles()
    return ok_q and ok_a


# ── ForecastBench ─────────────────────────────────────────────────────────────

def prepare_forecastbench():
    if FORECASTBENCH_PATH.exists():
        n = sum(1 for _ in open(FORECASTBENCH_PATH))
        print(f"ForecastBench already prepared: {n} questions")
        return True

    print("\nPreparing ForecastBench data...")
    print("  ForecastBench data must be downloaded manually.")
    print("  1. Visit https://forecastbench.org/datasets")
    print("  2. Download the geopolitics/conflict subset (Oct-Dec 2024 resolved questions)")
    print("  3. Convert to JSONL with fields: id, question, resolution_date,")
    print("     ground_truth (0/1), crowd_probability, category")
    print(f"  4. Place as: {FORECASTBENCH_PATH.relative_to(ROOT)}")
    print()
    print("  Alternatively, use the API if you have an account:")
    print("    python scripts/download_forecastbench.py")
    print()
    print("  Mock data will be used for smoke tests (no download needed).")
    return False


# ── Verify ────────────────────────────────────────────────────────────────────

def check_data():
    print("\nData availability check:")
    print("-" * 50)

    ok = True

    if MIRAI_QUERIES_PATH.exists():
        n = sum(1 for _ in open(MIRAI_QUERIES_PATH))
        print(f"  [OK] MIRAI queries:   {n:>5} records")
    else:
        print(f"  [--] MIRAI queries:   NOT FOUND  ({MIRAI_QUERIES_PATH.relative_to(ROOT)})")
        ok = False

    if MIRAI_ARTICLES_PATH.exists():
        n = sum(1 for _ in open(MIRAI_ARTICLES_PATH))
        print(f"  [OK] MIRAI articles:  {n:>5} records")
    else:
        print(f"  [--] MIRAI articles:  NOT FOUND  ({MIRAI_ARTICLES_PATH.relative_to(ROOT)})")
        ok = False

    if FORECASTBENCH_PATH.exists():
        n = sum(1 for _ in open(FORECASTBENCH_PATH))
        print(f"  [OK] ForecastBench:   {n:>5} records")
    else:
        print(f"  [--] ForecastBench:   NOT FOUND  ({FORECASTBENCH_PATH.relative_to(ROOT)})")
        ok = False

    env_file = ROOT / ".env"
    api_key = ""
    if env_file.exists():
        for line in open(env_file):
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if api_key and api_key != "your-key-here":
        print(f"  [OK] OPENAI_API_KEY:  set")
    else:
        import os
        if os.environ.get("OPENAI_API_KEY"):
            print(f"  [OK] OPENAI_API_KEY:  set (env)")
        else:
            print(f"  [!!] OPENAI_API_KEY:  NOT SET (copy .env.example -> .env)")
            ok = False

    print()
    if ok:
        print("All data ready. You can run smoke tests:")
        print("  python experiments/00_smoke/smoke_pipeline_e2e.py --mode smoke")
    else:
        print("Some data missing. Smoke tests will use mock data (still works).")
        print("Full experiments require real data.")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data for EMR-ACH")
    parser.add_argument("--mirai-only",    action="store_true")
    parser.add_argument("--fb-only",       action="store_true")
    parser.add_argument("--check",         action="store_true",
                        help="Only verify data presence, do not download")
    args = parser.parse_args()

    if args.check:
        check_data()
        return

    if args.fb_only:
        prepare_forecastbench()
        check_data()
        return

    if args.mirai_only:
        prepare_mirai()
        check_data()
        return

    prepare_mirai()
    prepare_forecastbench()
    check_data()


if __name__ == "__main__":
    main()
