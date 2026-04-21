"""
Download resolved binary Metaculus questions with crowd probabilities.

Fetches all resolved binary questions with resolution_date > 2024-04-01
(post GPT-4o cutoff), filters for geopolitics and multi-domain coverage,
then fetches GDELT news headlines for each.

NOTE (2026-04): After the Metaculus 2.0 platform migration, the API no longer
exposes resolution values or community prediction history for resolved questions.
The v2/v3 API returns `resolution: null` and empty aggregations for all questions.
This script may yield 0 valid records until Metaculus restores resolution access.
Our ForecastBench dataset already includes 82 Metaculus questions with community
probabilities captured before the migration, which provides sufficient coverage.

Outputs:
  data/metaculus_questions.jsonl  - question records
  data/metaculus_news.jsonl       - GDELT article metadata per question

Usage:
  python scripts/download_metaculus.py --n 200          # first 200 questions
  python scripts/download_metaculus.py --all             # all available
  python scripts/download_metaculus.py --domain politics # filter by domain
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import os

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
Q_OUT   = DATA / "metaculus_questions.jsonl"
NEWS_OUT = DATA / "metaculus_news.jsonl"

METACULUS_API = "https://www.metaculus.com/api2/questions/"
# Get token from env (METACULUS_TOKEN) or pass via --token arg
_ENV_TOKEN = os.getenv("METACULUS_TOKEN", "")
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
CUTOFF = "2024-04-01"

# Category tags to collect; keep broad for multi-domain coverage
DOMAIN_TAGS = {
    "geopolitics":  ["geopolitics", "international-politics", "war-and-military",
                     "ukraine", "china", "middle-east", "elections"],
    "economics":    ["economics", "finance", "inflation", "markets", "trade"],
    "science":      ["science", "technology", "ai", "space", "climate"],
    "health":       ["health", "medicine", "pandemics", "epidemiology"],
}


def fetch_metaculus_page(url: str, params: dict, token: str = "") -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 403:
            print("[ERROR] Metaculus API requires authentication.")
            print("  Get a free token at: https://www.metaculus.com/accounts/profile/")
            print("  Then add METACULUS_TOKEN=<your_token> to your .env file")
        else:
            print(f"  [WARN] HTTP {r.status_code} for {url}")
    except Exception as e:
        print(f"  [WARN] {e}")
    return {}


def iter_questions(domain: str | None = None, max_n: int | None = None, token: str = ""):
    """Yield resolved binary Metaculus questions post-cutoff.

    Uses the v3 /api/posts/ endpoint with statuses=resolved. Note: after the
    Metaculus 2.0 migration, resolution values and community predictions may not
    be populated in the API response. This function skips questions without both.
    """
    # v3 API endpoint
    url = "https://www.metaculus.com/api/posts/"
    params = {
        "forecast_type": "binary",
        "statuses": "resolved",
        "order_by": "close_time",
        "limit": 100,
    }
    seen = 0
    while url:
        data = fetch_metaculus_page(url, params if "metaculus.com/api/posts/" in url else {}, token=token)
        if not data:
            break
        results = data.get("results", [])
        if not results:
            break
        for post in results:
            if max_n and seen >= max_n:
                return
            q = post.get("question") or {}
            # extract resolution
            resolution = q.get("resolution")
            if resolution not in ("yes", "no", 0, 1):
                continue
            ground_truth = 1 if resolution in ("yes", 1) else 0
            # extract community prediction from aggregations
            agg = q.get("aggregations") or {}
            rw = agg.get("recency_weighted") or {}
            latest = rw.get("latest") or {}
            centers = latest.get("centers") or []
            prob = centers[0] if centers else None
            if prob is None:
                continue
            res_date = (str(q.get("actual_resolve_time") or ""))[:10]
            if not res_date or res_date < CUTOFF:
                continue
            yield post, float(prob), int(ground_truth), res_date
            seen += 1
        url = data.get("next")
        if url:
            params = {}
        time.sleep(0.3)


def extract_keywords(title: str, body: str = "") -> str:
    text = title + " " + body
    stop = {"will", "the", "a", "an", "be", "at", "in", "of", "to", "and", "or",
            "is", "are", "was", "were", "has", "have", "had", "by", "from", "for",
            "with", "on", "that", "this", "whether", "would", "could", "when", "which"}
    words = re.findall(r'\b[A-Za-z][a-z]{2,}\b', text)
    filtered = [w for w in words if w.lower() not in stop]
    proper = [w for w in filtered if w[0].isupper()]
    common = [w.lower() for w in filtered if not w[0].isupper()]
    return " ".join((proper[:4] + common[:2])[:5])


def gdelt_search(query: str, start_date: str, end_date: str, max_records: int = 8) -> list[dict]:
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d000000")
        ed = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d235959")
    except ValueError:
        return []
    params = {
        "query": query, "mode": "ArtList", "maxrecords": max_records,
        "startdatetime": sd, "enddatetime": ed, "format": "json", "sort": "DateDesc",
    }
    try:
        r = requests.get(GDELT_DOC_API, params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get("articles", [])
    except Exception:
        pass
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--domain", default=None, choices=list(DOMAIN_TAGS) + [None])
    parser.add_argument("--token", default=_ENV_TOKEN, help="Metaculus API token (or set METACULUS_TOKEN in .env)")
    parser.add_argument("--lookback", type=int, default=30)
    args = parser.parse_args()

    n = None if args.all else args.n

    # load existing to skip
    existing_qids: set[str] = set()
    if Q_OUT.exists():
        for line in open(Q_OUT, encoding="utf-8"):
            existing_qids.add(json.loads(line)["id"])
    if existing_qids:
        print(f"Skipping {len(existing_qids)} already-downloaded questions")

    existing_news_qids: set[str] = set()
    if NEWS_OUT.exists():
        for line in open(NEWS_OUT, encoding="utf-8"):
            existing_news_qids.add(json.loads(line)["question_id"])

    if not args.token:
        print("[ERROR] No Metaculus token. Get one free at https://www.metaculus.com/accounts/profile/")
        print("  Add METACULUS_TOKEN=<token> to .env, or pass --token <token>")
        return

    q_written = news_written = 0
    print(f"Downloading Metaculus binary questions (cutoff={CUTOFF}, n={n or 'all'})")

    with open(Q_OUT, "a", encoding="utf-8") as fq, \
         open(NEWS_OUT, "a", encoding="utf-8") as fn:

        for post, crowd_prob, ground_truth, res_date in iter_questions(args.domain, n, token=args.token):
            post_id = post.get("id")
            qid = f"meta_{post_id}"
            if qid in existing_qids:
                continue

            title = post.get("title", "") or ""
            q_obj = post.get("question") or {}
            body  = q_obj.get("description", "") or post.get("description", "") or ""

            record = {
                "id":               qid,
                "question":         title,
                "background":       body[:500],
                "resolution_date":  res_date,
                "ground_truth":     ground_truth,
                "crowd_probability": round(crowd_prob, 4),
                "source":           "metaculus",
                "category":         "general",
                "url":              f"https://www.metaculus.com/questions/{post_id}/",
            }
            fq.write(json.dumps(record) + "\n")
            q_written += 1

            # fetch GDELT news if not already done
            if qid not in existing_news_qids:
                try:
                    res_dt = datetime.strptime(res_date, "%Y-%m-%d")
                    end_dt = res_dt - timedelta(days=1)
                    start_dt = res_dt - timedelta(days=args.lookback)
                except ValueError:
                    continue
                keywords = extract_keywords(title, body)
                if keywords:
                    articles = gdelt_search(
                        keywords,
                        start_dt.strftime("%Y-%m-%d"),
                        end_dt.strftime("%Y-%m-%d"),
                    )
                    for art in articles:
                        url = art.get("url", "")
                        seendate = art.get("seendate", "")
                        date_str = ""
                        if seendate:
                            try:
                                date_str = datetime.strptime(seendate[:8], "%Y%m%d").strftime("%Y-%m-%d")
                            except ValueError:
                                pass
                        fn.write(json.dumps({
                            "id":           f"gdelt_{abs(hash(url)) % 10**10}",
                            "question_id":  qid,
                            "title":        art.get("title", ""),
                            "text":         "",
                            "date":         date_str,
                            "url":          url,
                            "source":       art.get("domain", ""),
                            "tone":         art.get("tone", 0.0),
                        }) + "\n")
                        news_written += 1
                    time.sleep(0.6)

            if q_written % 25 == 0:
                print(f"  [{q_written}] {title[:70]}")

    print(f"\nDone. Questions: {q_written}, news articles: {news_written}")
    if q_written == 0:
        print("[WARN] 0 questions written. The Metaculus API (post-2024 migration) no longer")
        print("  exposes resolution values or community predictions for resolved questions.")
        print("  ForecastBench already includes 82 Metaculus questions with crowd probabilities.")
    # stats
    if Q_OUT.exists():
        qs = [json.loads(l) for l in open(Q_OUT, encoding="utf-8")]
        if qs:
            yes = sum(q["ground_truth"] for q in qs)
            brier = sum((q["crowd_probability"] - q["ground_truth"])**2 for q in qs) / len(qs)
            print(f"Total questions: {len(qs)}, Yes={yes} ({yes/len(qs)*100:.0f}%), crowd Brier={brier:.4f}")


if __name__ == "__main__":
    main()
