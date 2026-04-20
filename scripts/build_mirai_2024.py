"""
Build MIRAI-2024: a leakage-free CAMEO event forecasting benchmark.

Replicates the MIRAI dataset construction pipeline (Ye et al., arXiv:2407.01231)
for Aug-Dec 2024 data, which is entirely post GPT-4o training cutoff (Apr 2024).

Pipeline:
  Step 1 - Download GDELT 2.0 KG export CSVs for the date range
  Step 2 - Merge, clean, filter (same-date, country codes, CAMEO codes)
  Step 3 - Filter by source reliability (>= MIN_DAILY_MENTIONS)
  Step 4 - Generate test queries for TEST_MONTH
  Step 5 - Write MIRAI-format output files

Output: data/mirai_2024/
  data_kg.csv          - knowledge graph (events with article links)
  data_news.csv        - article metadata (title, URL, date)
  test/relation_query.csv  - test queries (date, s, ?, o) with ground truth

Usage:
  python scripts/build_mirai_2024.py              # full run
  python scripts/build_mirai_2024.py --steps 1    # only download KG
  python scripts/build_mirai_2024.py --steps 2,3  # only clean+filter
  python scripts/build_mirai_2024.py --steps 4,5  # only generate queries
"""

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
INFO = ROOT / "data" / "mirai_extracted" / "info"
OUT  = ROOT / "data" / "mirai_2024"

# Date range: context + test
CONTEXT_START  = "202408"  # Aug 2024
CONTEXT_END    = "202410"  # Oct 2024 (training context for RAG)
TEST_MONTH     = "202411"  # Nov 2024 (evaluation queries)
ALL_END        = "202412"  # include Dec for full context

MIN_DAILY_MENTIONS = 50    # MIRAI's source reliability threshold
MAX_DOWNLOAD_WORKERS = 16

GDELT_MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

COL_NAMES = [
    'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate',
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
    'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
    'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
    'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
    'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass',
    'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone',
    'Actor1Geo_Type', 'Actor1Geo_Fullname', 'Actor1Geo_CountryCode',
    'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code', 'Actor1Geo_Lat',
    'Actor1Geo_Long', 'Actor1Geo_FeatureID',
    'Actor2Geo_Type', 'Actor2Geo_Fullname', 'Actor2Geo_CountryCode',
    'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat',
    'Actor2Geo_Long', 'Actor2Geo_FeatureID',
    'EventGeo_Type', 'EventGeo_Fullname', 'EventGeo_CountryCode',
    'EventGeo_ADM1Code', 'EventGeo_ADM2Code', 'EventGeo_Lat',
    'EventGeo_Long', 'EventGeo_FeatureID',
    'DATEADDED', 'SOURCEURL',
]

QUAD_MAP = {"1": "VC", "2": "MC", "3": "VK", "4": "MK"}


def load_info():
    """Load country and CAMEO lookup tables from MIRAI info files."""
    iso2country, cameo2name = {}, {}
    iso_file = INFO / "ISO_country_GeoNames.txt"
    cameo_file = INFO / "CAMEO_relation.txt"
    if iso_file.exists():
        for line in iso_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                iso2country[parts[0]] = parts[1]
    if cameo_file.exists():
        for line in cameo_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                cameo2name[parts[0]] = parts[1]
    return iso2country, cameo2name


# ── Step 1: Download GDELT KG exports ────────────────────────────────────────

def download_gdelt_master() -> list[str]:
    """Return list of (size, hash, url) lines from GDELT master file."""
    print("Fetching GDELT master file list...")
    resp = requests.get(GDELT_MASTER, timeout=30)
    resp.raise_for_status()
    return resp.text.splitlines()


def filter_urls(lines: list[str]) -> list[str]:
    """Filter master file to export CSVs in our date range."""
    urls = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        url = parts[2]
        if ".export.CSV.zip" not in url:
            continue
        fname = url.split("/")[-1]
        # fname starts with YYYYMMDDHHMMSS
        if len(fname) < 6:
            continue
        ym = fname[:6]
        if ym < CONTEXT_START or ym > ALL_END:
            continue
        urls.append(url)
    print(f"Found {len(urls)} GDELT export files for {CONTEXT_START}-{ALL_END}")
    return urls


def download_one(url: str, out_dir: Path) -> int:
    """Download one GDELT zip, extract CSV, return row count (0 on failure)."""
    fname = url.split("/")[-1].replace(".zip", "")
    out_path = out_dir / fname
    if out_path.exists():
        return 0  # already done
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return 0
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extract(fname, out_dir)
        return 1
    except Exception:
        return 0


def step1_download(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    already = len(list(raw_dir.glob("*.CSV")))
    print(f"KG raw dir: {raw_dir} ({already} files already present)")

    lines = download_gdelt_master()
    urls  = filter_urls(lines)
    remaining = [u for u in urls
                 if not (raw_dir / u.split("/")[-1].replace(".zip","")).exists()]
    print(f"Downloading {len(remaining)} new files ({len(urls)-len(remaining)} cached)...")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_one, u, raw_dir): u for u in remaining}
        for fut in tqdm(as_completed(futures), total=len(remaining), unit="file"):
            done += fut.result()
    print(f"Step 1 done: {done + already} CSV files total")


# ── Step 2: Merge + clean ──────────────────────────────────────────────────

def step2_clean(raw_dir: Path, tmp_dir: Path, iso2country: dict, cameo2name: dict):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "kg_cleaned.csv"
    if out_path.exists():
        print(f"Step 2: {out_path} already exists, skipping")
        return

    csv_files = sorted(raw_dir.glob("*.CSV"))
    print(f"Step 2: merging {len(csv_files)} CSV files...")

    dfs = []
    for f in tqdm(csv_files, unit="file"):
        try:
            df = pd.read_csv(f, sep="\t", header=None, dtype=str, low_memory=False)
        except Exception:
            continue
        if len(df.columns) != 61:
            continue
        dfs.append(df)

    if not dfs:
        print("[ERROR] No valid CSV files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df.columns = COL_NAMES
    df.drop_duplicates(subset="GlobalEventID", inplace=True)
    print(f"  Raw rows: {len(df)}")

    # same-date filter: event date == news date
    df["NewsDate"] = df["DATEADDED"].str[:8]
    df = df[df["Day"] == df["NewsDate"]]

    # URL dedup: keep earliest occurrence of each URL
    url_first_date: dict[str, str] = {}
    for url, nd in zip(df["SOURCEURL"], df["NewsDate"]):
        if url not in url_first_date or nd < url_first_date[url]:
            url_first_date[url] = nd
    df["URLDay"] = df["SOURCEURL"].map(url_first_date)
    df = df[df["NewsDate"] == df["URLDay"]]

    # country filter
    df = df[df[["Actor1CountryCode", "Actor2CountryCode"]].notnull().all(axis=1)]
    df = df[df["Actor1CountryCode"].isin(iso2country)]
    df = df[df["Actor2CountryCode"].isin(iso2country)]
    df = df[df["Actor1CountryCode"] != df["Actor2CountryCode"]]

    # CAMEO code filter
    df = df[df["EventRootCode"] != "--"]
    df = df[df["EventBaseCode"].isin(cameo2name)]

    # enrich
    df["Actor1CountryName"] = df["Actor1CountryCode"].map(iso2country)
    df["Actor2CountryName"] = df["Actor2CountryCode"].map(iso2country)
    df["RelName"] = df["EventBaseCode"].map(cameo2name)
    df["DateStr"]  = pd.to_datetime(df["Day"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df["QuadEventCode"] = (
        df["DateStr"] + "_" + df["Actor1CountryCode"] + "_" +
        df["Actor2CountryCode"] + "_" + df["EventBaseCode"]
    )

    df.to_csv(out_path, index=False, sep="\t")
    print(f"Step 2 done: {len(df)} rows -> {out_path}")


# ── Step 3: Filter by source reliability ─────────────────────────────────────

def step3_filter(tmp_dir: Path):
    in_path  = tmp_dir / "kg_cleaned.csv"
    out_path = tmp_dir / "kg_source.csv"
    if out_path.exists():
        print(f"Step 3: {out_path} already exists, skipping")
        return

    print("Step 3: filtering by source reliability...")
    df = pd.read_csv(in_path, sep="\t", dtype=str, low_memory=False)
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0).astype(int)

    # daily mention count per quad event
    daily: dict[str, int] = defaultdict(int)
    for qc, nm in zip(df["QuadEventCode"], df["NumMentions"]):
        daily[qc] += nm
    df["NumDailyMentions"] = df["QuadEventCode"].map(daily)
    df = df[df["NumDailyMentions"] >= MIN_DAILY_MENTIONS]

    df.to_csv(out_path, index=False, sep="\t")
    print(f"Step 3 done: {len(df)} rows -> {out_path}")


# ── Step 4: Build news index + test queries ───────────────────────────────────

def step4_build_dataset(tmp_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    kg_path = tmp_dir / "kg_source.csv"
    print("Step 4: building dataset...")

    df = pd.read_csv(kg_path, sep="\t", dtype=str, low_memory=False)

    # assign doc IDs
    urls = df["SOURCEURL"].unique().tolist()
    url2docid = {url: i for i, url in enumerate(urls)}
    df["Docid"] = df["SOURCEURL"].map(url2docid)

    # data_news.csv
    df_news = pd.DataFrame({
        "Docid":    range(len(urls)),
        "URL":      urls,
        "Title":    "",     # filled if full text available
        "Abstract": "",
        "Text":     "",
        "Date":     [df.loc[df["SOURCEURL"]==u, "DateStr"].iloc[0]
                     if (df["SOURCEURL"]==u).any() else "" for u in tqdm(urls[:len(urls)], desc="news index")],
    })
    df_news.to_csv(out_dir / "data_news.csv", index=False, sep="\t")
    print(f"  data_news.csv: {len(df_news)} articles")

    # data_kg.csv (dedup by quad event)
    quad2docids: dict[str, list[int]] = defaultdict(list)
    for qc, did in zip(df["QuadEventCode"], df["Docid"]):
        quad2docids[qc].append(int(did))

    df["Docids"] = df["QuadEventCode"].map(quad2docids)
    cols = ["DateStr", "Actor1CountryCode", "Actor2CountryCode", "EventBaseCode",
            "Actor1CountryName", "Actor2CountryName", "RelName",
            "QuadEventCode", "Docid", "Docids"]
    df_kg = df[cols].drop_duplicates(subset=["QuadEventCode", "Docid"])
    df_kg.to_csv(out_dir / "data_kg.csv", index=False, sep="\t")
    print(f"  data_kg.csv: {len(df_kg)} rows")

    # test queries: test month only
    test_dir = out_dir / "test"
    test_dir.mkdir(exist_ok=True)
    df_test = df_kg[df_kg["DateStr"].str.startswith(TEST_MONTH[:4] + "-" + TEST_MONTH[4:])]
    if df_test.empty:
        # fallback: use last available month
        last_month = df_kg["DateStr"].dropna().str[:7].max()
        df_test = df_kg[df_kg["DateStr"].str[:7] == last_month]
        print(f"  [WARN] No data for {TEST_MONTH}, using {last_month}")

    df_test.to_csv(test_dir / "test_kg.csv", index=False, sep="\t")
    print(f"  test_kg.csv: {len(df_test)} rows")

    # relation_query.csv
    step5_relation_query(df_test, df_kg, test_dir)


def step5_relation_query(df_test: pd.DataFrame, df_full: pd.DataFrame, test_dir: Path):
    """Generate (date, s, ?, o) query format matching MIRAI's relation_query.csv."""

    # build RelQueryCode = (date, s, ?, o)
    df_test = df_test.copy()
    df_test["RelQueryCode"] = (
        "(" + df_test["DateStr"] + ", " +
        df_test["Actor1CountryCode"] + ", ?, " +
        df_test["Actor2CountryCode"] + ")"
    )

    df_full["RelQueryCode"] = (
        "(" + df_full["DateStr"] + ", " +
        df_full["Actor1CountryCode"] + ", ?, " +
        df_full["Actor2CountryCode"] + ")"
    )

    # for each query, collect all ground-truth CAMEO codes
    unique_queries = df_test["RelQueryCode"].unique()
    df_answers = df_full[df_full["RelQueryCode"].isin(unique_queries)]
    query2rels: dict[str, list[str]] = defaultdict(list)
    for qc, ec in zip(df_answers["RelQueryCode"], df_answers["EventBaseCode"]):
        if ec not in query2rels[qc]:
            query2rels[qc].append(ec)

    query2reldict: dict[str, dict] = {}
    for q, rels in query2rels.items():
        rd: dict[str, list] = {}
        for r in rels:
            root = r[:2]
            rd.setdefault(root, []).append(r)
        query2reldict[q] = rd

    df_q = df_test.drop_duplicates(subset=["RelQueryCode"]).copy()
    df_q["AnswerList"] = df_q["RelQueryCode"].map(lambda q: sorted(query2rels.get(q, [])))
    df_q["AnswerDict"] = df_q["RelQueryCode"].map(lambda q: query2reldict.get(q, {}))
    df_q["DateNLP"]    = pd.to_datetime(df_q["DateStr"], errors="coerce").dt.strftime("%B %d, %Y")
    df_q["QueryId"]    = range(1, len(df_q) + 1)

    # quad class label (VC/MC/VK/MK) for the top answer
    def top_quad(rels):
        for r in (rels or []):
            rc = int(r[:2]) if r[:2].isdigit() else 0
            if rc <= 4:   return "VC"
            if rc <= 8:   return "MC"
            if rc <= 16:  return "VK"
            return "MK"
        return "VK"

    df_q["QuadLabel"] = df_q["AnswerList"].map(top_quad)

    out_cols = ["QueryId", "DateStr", "DateNLP", "RelQueryCode",
                "Actor1CountryCode", "Actor1CountryName",
                "Actor2CountryCode", "Actor2CountryName",
                "AnswerList", "AnswerDict", "QuadLabel",
                "Docids"]
    df_q[[c for c in out_cols if c in df_q.columns]].to_csv(
        test_dir / "relation_query.csv", index=False, sep="\t"
    )

    # stats
    label_counts = df_q["QuadLabel"].value_counts().to_dict()
    print(f"Step 5 done: {len(df_q)} test queries -> {test_dir / 'relation_query.csv'}")
    print(f"  Label distribution: {label_counts}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="1,2,3,4,5",
                        help="Comma-separated steps to run (default: all)")
    args = parser.parse_args()
    steps = set(args.steps.split(","))

    raw_dir = OUT / "kg_raw"
    tmp_dir = OUT / "kg_tmp"

    iso2country, cameo2name = load_info()
    if not iso2country:
        print(f"[ERROR] Could not load ISO country file from {INFO}. "
              "Ensure data/mirai_extracted/info/ exists.")
        return

    print(f"MIRAI-2024 builder: {CONTEXT_START}-{ALL_END}, test={TEST_MONTH}")
    print(f"Output: {OUT}")
    print(f"Steps: {args.steps}")
    print("=" * 60)

    if "1" in steps:
        step1_download(raw_dir)
    if "2" in steps:
        step2_clean(raw_dir, tmp_dir, iso2country, cameo2name)
    if "3" in steps:
        step3_filter(tmp_dir)
    if "4" in steps or "5" in steps:
        step4_build_dataset(tmp_dir, OUT)


if __name__ == "__main__":
    main()
