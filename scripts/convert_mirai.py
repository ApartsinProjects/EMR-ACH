"""
Convert MIRAI CSV files to the JSONL format expected by src/data/mirai.py.

Input:
  data/mirai_extracted/MIRAI/test/relation_query.csv   (705 test queries)
  data/mirai_extracted/MIRAI/data_news.csv             (5.8M articles)

Output:
  data/mirai_test_queries.jsonl
  data/mirai_articles.jsonl   (only articles referenced by test queries)

CAMEO quad-class mapping:
  01-04 -> VC (Verbal Cooperation)
  05-08 -> MC (Material Cooperation)
  09-16 -> VK (Verbal Conflict)
  17-20 -> MK (Material Conflict)
"""

import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
EXTRACTED = DATA / "mirai_extracted" / "MIRAI"

QUERIES_CSV   = EXTRACTED / "test" / "relation_query.csv"
NEWS_CSV      = DATA / "mirai_extracted" / "MIRAI" / "data_news.csv"
OUT_QUERIES   = DATA / "mirai_test_queries.jsonl"
OUT_ARTICLES  = DATA / "mirai_articles.jsonl"

HYPOTHESIS_NAMES = {
    "VC": "Verbal Cooperation",
    "MC": "Material Cooperation",
    "VK": "Verbal Conflict",
    "MK": "Material Conflict",
}


def cameo_to_quad(code: str) -> str:
    try:
        root = int(str(code).split(".")[0][:2])
    except (ValueError, TypeError):
        return "VK"
    if 1 <= root <= 4:
        return "VC"
    elif 5 <= root <= 8:
        return "MC"
    elif 9 <= root <= 16:
        return "VK"
    elif 17 <= root <= 20:
        return "MK"
    return "VK"


def dominant_quad(answer_list: list[str]) -> str:
    """Pick the most common quad class across all answer CAMEO codes."""
    if not answer_list:
        return "VK"
    counts = {"VC": 0, "MC": 0, "VK": 0, "MK": 0}
    for code in answer_list:
        q = cameo_to_quad(code)
        counts[q] += 1
    return max(counts, key=lambda k: counts[k])


def convert_queries() -> tuple[list[dict], set[int]]:
    print(f"Reading queries from {QUERIES_CSV.relative_to(ROOT)}...")
    queries = []
    all_doc_ids: set[int] = set()

    with open(QUERIES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Parse doc IDs (stored as a Python list literal)
            try:
                raw_docids = ast.literal_eval(row["Docids"])
                doc_ids = [str(d) for d in raw_docids]
                all_doc_ids.update(int(d) for d in raw_docids)
            except Exception:
                doc_ids = []

            # Parse answer list for label
            try:
                answer_list = ast.literal_eval(row["AnswerList"])
            except Exception:
                answer_list = [row.get("EventBaseCode", "111")]

            label = dominant_quad(answer_list)

            queries.append({
                "id":         f"q{row['QueryId']}",
                "timestamp":  row["DateStr"],
                "subject":    row["Actor1CountryName"],
                "relation":   row["RelName"],
                "object":     row["Actor2CountryName"],
                "label":      label,
                "label_full": HYPOTHESIS_NAMES[label],
                "doc_ids":    doc_ids,
            })

    print(f"  {len(queries)} queries, {len(all_doc_ids)} unique article IDs")
    return queries, all_doc_ids


def write_queries(queries: list[dict]):
    with open(OUT_QUERIES, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"Written: {OUT_QUERIES.relative_to(ROOT)}")

    dist = {}
    for q in queries:
        dist[q["label"]] = dist.get(q["label"], 0) + 1
    print(f"  Label distribution: {dist}")


def convert_articles(keep_ids: set[int]):
    print(f"\nReading articles from {NEWS_CSV.relative_to(ROOT)}...")
    print(f"  Will keep {len(keep_ids)} referenced articles out of ~5.8M rows")

    written = 0
    scanned = 0

    with open(NEWS_CSV, newline="", encoding="utf-8", errors="replace") as f_in, \
         open(OUT_ARTICLES, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            scanned += 1
            if scanned % 500_000 == 0:
                pct = written / len(keep_ids) * 100 if keep_ids else 0
                print(f"  Scanned {scanned:,} rows, found {written}/{len(keep_ids)} ({pct:.0f}%)")

            try:
                doc_id = int(row["Docid"])
            except (ValueError, KeyError):
                continue

            if doc_id not in keep_ids:
                continue

            record = {
                "id":       str(doc_id),
                "title":    row.get("Title", ""),
                "abstract": row.get("Abstract", ""),
                "text":     row.get("Text", "")[:3000],  # cap at 3k chars
                "date":     row.get("Date", ""),
                "source":   "",
                "country_mentions": [],
            }
            f_out.write(json.dumps(record) + "\n")
            written += 1

            if written == len(keep_ids):
                print(f"  All {written} articles found. Stopping early.")
                break

    print(f"\nScanned {scanned:,} rows total.")
    print(f"Written: {written} articles -> {OUT_ARTICLES.relative_to(ROOT)}")
    if written < len(keep_ids):
        print(f"  [WARN] {len(keep_ids) - written} article IDs not found in news CSV.")
    return written


def main():
    if not QUERIES_CSV.exists():
        print(f"[ERROR] Not found: {QUERIES_CSV}")
        print("  Run: python scripts/prepare_data.py --mirai-only first.")
        sys.exit(1)

    queries, doc_ids = convert_queries()
    write_queries(queries)
    convert_articles(doc_ids)

    print("\nConversion complete.")
    print(f"  Queries: {OUT_QUERIES.relative_to(ROOT)}")
    print(f"  Articles: {OUT_ARTICLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
