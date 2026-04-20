"""
Fetch full article text for MIRAI-2024 news articles.

Reads data/mirai_2024/data_news.csv (URLs from GDELT KG pipeline),
downloads and extracts text with trafilatura, and writes
data/mirai_2024/data_news_full.csv with 'Text' and 'Abstract' filled.

Run AFTER build_mirai_2024.py steps 1-4.

Usage:
  python scripts/fetch_mirai_2024_text.py --workers 12 --n 5000
  python scripts/fetch_mirai_2024_text.py --all
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import trafilatura

ROOT     = Path(__file__).parent.parent
NEWS_IN  = ROOT / "data" / "mirai_2024" / "data_news.csv"
NEWS_OUT = ROOT / "data" / "mirai_2024" / "data_news_full.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 12


def fetch_one(row: dict) -> dict:
    url = row.get("URL", "")
    if not url:
        return {**row, "Text": "", "Title_fetched": "", "Abstract": ""}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return {**row, "Text": "", "Title_fetched": "", "Abstract": ""}
        text = trafilatura.extract(resp.content, include_comments=False,
                                   include_tables=False, no_fallback=False) or ""
        meta = trafilatura.extract_metadata(resp.content)
        title = (meta.title if meta else "") or row.get("Title", "")
        paragraphs = text.split("\n")
        abstract = title + "\n" + paragraphs[0] if paragraphs else title
        for par in paragraphs[1:]:
            if len(abstract) > 200:
                break
            abstract += "\n" + par
        return {**row, "Text": text, "Title": title, "Abstract": abstract}
    except Exception:
        return {**row, "Text": "", "Abstract": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if not NEWS_IN.exists():
        print(f"[ERROR] {NEWS_IN} not found. Run build_mirai_2024.py steps 1-4 first.")
        return

    df = pd.read_csv(NEWS_IN, sep="\t", dtype=str)
    if not args.all and args.n:
        df = df.head(args.n)

    # skip already-fetched
    fetched_ids: set[int] = set()
    if NEWS_OUT.exists():
        df_done = pd.read_csv(NEWS_OUT, sep="\t", dtype=str)
        fetched_ids = set(df_done[df_done["Text"].notna() & (df_done["Text"] != "")]["Docid"].astype(int))
        print(f"Skipping {len(fetched_ids)} already-fetched articles")

    df_pending = df[~df["Docid"].astype(int).isin(fetched_ids)]
    print(f"Fetching text for {len(df_pending)} articles ({args.workers} workers)")

    rows_done = []
    records   = df_pending.to_dict("records")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, r) for r in records]
        for i, fut in enumerate(as_completed(futures), 1):
            rows_done.append(fut.result())
            if i % 200 == 0:
                good = sum(1 for r in rows_done if r.get("Text"))
                print(f"  [{i}/{len(records)}] with_text={good}")

    df_new = pd.DataFrame(rows_done)
    if NEWS_OUT.exists():
        df_old = pd.read_csv(NEWS_OUT, sep="\t", dtype=str)
        df_out = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates("Docid")
    else:
        df_out = df_new

    df_out.to_csv(NEWS_OUT, index=False, sep="\t")
    with_text = df_out["Text"].notna() & (df_out["Text"] != "")
    print(f"\nDone. Total={len(df_out)}, with_text={with_text.sum()} ({with_text.mean()*100:.0f}%)")
    print(f"Output: {NEWS_OUT}")


if __name__ == "__main__":
    main()
