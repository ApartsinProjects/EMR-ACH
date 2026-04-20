"""
Download news articles from GDELT DOC API for ForecastBench questions.

For each question, searches GDELT for relevant articles from the 30 days
before the question's resolution date and stores them in data/gdelt_articles.jsonl.

Usage:
  python scripts/download_gdelt_news.py --n 50      # first 50 ForecastBench questions
  python scripts/download_gdelt_news.py --all        # all 2050 questions
  python scripts/download_gdelt_news.py --date-from 2024-07-01 --date-to 2024-12-31
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FB_FILE = DATA / "forecastbench_geopolitics.jsonl"
OUT = DATA / "gdelt_articles.jsonl"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def load_fb_questions(n=None, date_from=None, date_to=None) -> list[dict]:
    questions = []
    with open(FB_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if date_from and q["resolution_date"] < date_from:
                continue
            if date_to and q["resolution_date"] > date_to:
                continue
            questions.append(q)
    if n:
        questions = questions[:n]
    return questions


def extract_keywords(question: str, background: str = "") -> str:
    """Extract 3-5 key terms from question for GDELT search."""
    text = question + " " + background
    # Remove common stop words and keep geopolitically relevant terms
    stop = {"will", "the", "a", "an", "be", "at", "in", "of", "to", "and",
             "or", "is", "are", "was", "were", "has", "have", "had", "by",
             "between", "from", "for", "with", "on", "that", "this", "than",
             "more", "than", "least", "most", "before", "after", "during",
             "whether", "would", "could", "when", "which", "who", "what"}
    words = re.findall(r'\b[A-Za-z][a-z]{2,}\b', text)
    filtered = [w for w in words if w.lower() not in stop]
    # Prefer proper nouns (country names, etc.)
    proper = [w for w in filtered if w[0].isupper()]
    common = [w.lower() for w in filtered if not w[0].isupper()]
    query_words = proper[:4] + common[:2]
    return " ".join(query_words[:5])


def gdelt_search(query: str, start_date: str, end_date: str, max_records: int = 10) -> list[dict]:
    """Search GDELT DOC API and return article list."""
    # GDELT datetime format: YYYYMMDDHHMMSS
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d000000")
        ed = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d235959")
    except ValueError:
        return []

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "startdatetime": sd,
        "enddatetime": ed,
        "format": "json",
        "sort": "DateDesc",
    }

    try:
        r = requests.get(GDELT_DOC_API, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        articles = data.get("articles", [])
        return articles
    except Exception as e:
        print(f"  [WARN] GDELT search failed: {e}")
        return []


def articles_to_records(articles: list[dict], question_id: str, question: str) -> list[dict]:
    """Convert GDELT article objects to our standard format."""
    records = []
    for art in articles:
        url = art.get("url", "")
        title = art.get("title", "")
        seendate = art.get("seendate", "")
        # Parse GDELT date: YYYYMMDDTHHMMSSZ
        date_str = ""
        if seendate:
            try:
                dt = datetime.strptime(seendate[:8], "%Y%m%d")
                date_str = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        records.append({
            "id": f"gdelt_{abs(hash(url)) % 10**10}",
            "question_id": question_id,
            "title": title,
            "abstract": art.get("socialimage", ""),
            "text": "",
            "date": date_str,
            "url": url,
            "source": art.get("domain", ""),
            "tone": art.get("tone", 0.0),
            "country_mentions": [],
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of questions to process")
    parser.add_argument("--all", action="store_true", help="Process all questions")
    parser.add_argument("--date-from", default=None, help="Filter questions from date (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Filter questions to date (YYYY-MM-DD)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="Days before resolution to search for news (default: 30)")
    parser.add_argument("--max-articles", type=int, default=10,
                        help="Max articles per question (default: 10)")
    args = parser.parse_args()

    if not FB_FILE.exists():
        print(f"[ERROR] {FB_FILE} not found. Run: python scripts/download_forecastbench.py first.")
        sys.exit(1)

    n = None if args.all else args.n
    questions = load_fb_questions(n=n, date_from=args.date_from, date_to=args.date_to)
    print(f"Processing {len(questions)} ForecastBench questions")
    print(f"Output: {OUT}")

    # Load existing article IDs to skip duplicates
    existing_qids: set[str] = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                existing_qids.add(d.get("question_id", ""))
        print(f"Already have articles for {len(existing_qids)} questions, skipping those")

    written = 0
    with open(OUT, "a", encoding="utf-8") as f_out:
        for i, q in enumerate(questions):
            qid = q["id"]
            if qid in existing_qids:
                continue

            resolution_date = q["resolution_date"]
            try:
                res_dt = datetime.strptime(resolution_date, "%Y-%m-%d")
                end_dt = res_dt - timedelta(days=1)
                start_dt = res_dt - timedelta(days=args.lookback_days)
            except ValueError:
                continue

            keywords = extract_keywords(q["question"], q.get("background", ""))
            if not keywords:
                continue

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] {qid}: {q['question'][:60]}...")

            articles = gdelt_search(
                query=keywords,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                max_records=args.max_articles,
            )

            records = articles_to_records(articles, qid, q["question"])
            for rec in records:
                f_out.write(json.dumps(rec) + "\n")
            written += len(records)

            # Rate limiting: GDELT allows ~100 queries/min
            time.sleep(0.7)

    print(f"\nWritten {written} article records for {len(questions)} questions -> {OUT.relative_to(ROOT)}")
    if OUT.exists():
        total = sum(1 for _ in open(OUT))
        print(f"Total in file: {total} articles")


if __name__ == "__main__":
    main()
