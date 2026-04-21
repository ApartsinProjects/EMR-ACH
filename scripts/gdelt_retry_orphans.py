"""
Step B of coverage recovery: LLM keyword rewriting + GDELT DOC API retry
for forecasts that still have zero relevant articles after Step A.

For each orphan forecast:
  1. Ask gpt-4o-mini to produce 3 keyword-search-friendly variants of the
     question plus a list of named entities.
  2. Issue GDELT DOC API (with each variant, OR-joined terms) over the
     lookback window [t* - lookback_days, t*).
  3. Deduplicate returned URLs against the unified article pool.
  4. Append new article metadata to data/gdelt_articles_supplement.jsonl
     with the orphan's question_id.

Downstream: re-run unify_articles.py then compute_relevance.py to let the
new articles participate in SBERT matching.

Usage:
  python scripts/gdelt_retry_orphans.py                  # all orphans
  python scripts/gdelt_retry_orphans.py --limit 10       # debug
  python scripts/gdelt_retry_orphans.py --dry-run        # no API calls
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

FC_FILE = ROOT / "data" / "unified" / "forecasts.jsonl"
ART_FILE = ROOT / "data" / "unified" / "articles.jsonl"
GDELT_OUT = ROOT / "data" / "gdelt_articles_supplement.jsonl"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT = "https://api.openai.com/v1/chat/completions"

REWRITE_SYS = (
    "You rewrite forecasting questions into GDELT-searchable news query variants. "
    "Output strict JSON only."
)

REWRITE_USER = """Question: {question}
Background: {background}
Resolution date: {resolution_date}

Produce three short news-search queries that would find news articles relevant
to this question. Each query should be a 3-6 word noun phrase suitable for a
keyword search engine (use proper nouns; avoid 'will', 'by', future dates, or
punctuation). Also list up to 5 key named entities (people, organizations,
countries, events, products) that appear in the question.

Return strict JSON:
{{
  "queries":  ["query1", "query2", "query3"],
  "entities": ["ent1", "ent2", ...]
}}"""


def gdelt_search(query: str, start: str, end: str, max_records: int = 15) -> list[dict]:
    try:
        sd = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d000000")
        ed = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d235959")
    except ValueError:
        return []
    params = {
        "query": query, "mode": "ArtList", "maxrecords": max_records,
        "startdatetime": sd, "enddatetime": ed,
        "format": "json", "sort": "HybridRel",  # relevance-sorted
    }
    try:
        r = requests.get(GDELT_DOC_API, params=params, timeout=20)
        if r.status_code == 200:
            return r.json().get("articles", [])
    except Exception:
        pass
    return []


def rewrite_query(question: str, background: str, resolution_date: str,
                  dry_run: bool = False) -> dict:
    if dry_run or not OPENAI_KEY:
        # fallback: naive keyword extraction (first 4 capitalized words)
        import re
        caps = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question)
        q1 = " ".join(caps[:4]) if caps else question[:40]
        return {"queries": [q1], "entities": caps[:5]}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": REWRITE_SYS},
            {"role": "user", "content": REWRITE_USER.format(
                question=question, background=(background or "")[:400],
                resolution_date=resolution_date,
            )},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 200,
    }
    try:
        r = requests.post(OPENAI_CHAT,
                          headers={"Authorization": f"Bearer {OPENAI_KEY}",
                                   "Content-Type": "application/json"},
                          json=body, timeout=30)
        if r.status_code != 200:
            print(f"    [WARN] OpenAI HTTP {r.status_code}: {r.text[:150]}")
            return {"queries": [question[:40]], "entities": []}
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"    [WARN] rewrite failed: {e}")
        return {"queries": [question[:40]], "entities": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    forecasts = [json.loads(l) for l in open(FC_FILE, encoding="utf-8")]
    orphans = [f for f in forecasts if not f.get("article_ids")]
    if args.limit:
        orphans = orphans[: args.limit]
    print(f"Orphans: {len(orphans)} (of {len(forecasts)} total)")

    # build URL dedup set against existing articles
    existing_urls = set()
    if ART_FILE.exists():
        for l in open(ART_FILE, encoding="utf-8"):
            existing_urls.add(json.loads(l).get("url", ""))
    print(f"Existing article URL count: {len(existing_urls)}")

    # open supplement file (append mode, dedup across runs via URL check)
    already_supplied = set()
    if GDELT_OUT.exists():
        for l in open(GDELT_OUT, encoding="utf-8"):
            already_supplied.add(json.loads(l).get("url", ""))

    new_art_count = 0
    new_q2art = {}  # for report
    with open(GDELT_OUT, "a", encoding="utf-8") as fout:
        for i, fc in enumerate(orphans, 1):
            qid = fc["id"]
            question = fc.get("question", "")
            bg = fc.get("background", "") or ""
            res = fc.get("resolution_date", "") or fc.get("forecast_point", "")[:10]
            lookback = int(fc.get("lookback_days", 30))
            try:
                rd = datetime.strptime(res[:10], "%Y-%m-%d")
            except Exception:
                continue
            start_d = (rd - timedelta(days=lookback)).strftime("%Y-%m-%d")
            end_d = (rd - timedelta(days=1)).strftime("%Y-%m-%d")

            rw = rewrite_query(question, bg, res, dry_run=args.dry_run)
            queries = rw.get("queries", [])[:3]
            print(f"  [{i}/{len(orphans)}] {qid[:20]:<20} queries={queries}")

            added_here = 0
            for q in queries:
                if not q or len(q) < 3:
                    continue
                arts = gdelt_search(q, start_d, end_d, max_records=10)
                for art in arts:
                    url = art.get("url", "")
                    if not url or url in existing_urls or url in already_supplied:
                        continue
                    existing_urls.add(url)
                    already_supplied.add(url)
                    seendate = art.get("seendate", "")
                    try:
                        date_str = datetime.strptime(seendate[:8], "%Y%m%d").strftime("%Y-%m-%d")
                    except Exception:
                        date_str = ""
                    fout.write(json.dumps({
                        "id": f"gdeltb_{abs(hash(url)) % 10**10}",
                        "question_id": qid,
                        "title": art.get("title", "") or "",
                        "abstract": "",
                        "text": "",
                        "date": date_str,
                        "url": url,
                        "source": art.get("domain", "") or "",
                        "tone": float(art.get("tone", 0.0) or 0.0),
                        "country_mentions": [],
                        "retry_query": q,
                    }, ensure_ascii=False) + "\n")
                    added_here += 1
                    new_art_count += 1
                time.sleep(0.4)
            new_q2art[qid] = added_here

    recovered = sum(1 for v in new_q2art.values() if v > 0)
    print(f"\nDone. New articles: {new_art_count}")
    print(f"Orphans now with >=1 new article: {recovered} / {len(orphans)}")
    print(f"Supplement written to: {GDELT_OUT}")
    print("Next: re-run scripts/unify_articles.py then scripts/compute_relevance.py")


if __name__ == "__main__":
    main()
