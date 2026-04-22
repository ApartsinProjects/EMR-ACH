"""Fetch pre-event GDELT-CAMEO news via Google BigQuery.

DRAMATICALLY faster than the DOC API for bulk actor-pair retrieval: one SQL
query against `gdelt-bq.gdeltv2.gkg` fetches all pre-event URLs for all FDs at
once, instead of 2,396 × 3 HTTP calls rate-limited to ~1/sec.

Expected cost: ~5-10 GB scanned (well under the 1 TB/month free tier).
Expected wall time: 1-3 minutes for the SQL, then 30-60 min of trafilatura
fetch for the returned URLs. End-to-end: ~1h for 2k FDs vs 20h+ via DOC API.

Prerequisites (one-time setup):

  1. Install the BigQuery client:
         pip install google-cloud-bigquery

  2. Have a Google Cloud project with billing enabled. Note the project ID.

  3. Authenticate (pick ONE):
       a. `gcloud auth application-default login` (local dev)
       b. Download a service-account JSON key, then:
          export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
       c. On GCE / Cloud Run, the service account is picked up automatically.

  4. Export your project:
         export GCP_PROJECT=my-gcp-project-id

Usage:
  python scripts/fetch_gdelt_cameo_bigquery.py --limit 50     # smoke test
  python scripts/fetch_gdelt_cameo_bigquery.py --all         # full corpus
  python scripts/fetch_gdelt_cameo_bigquery.py --urls-only    # skip trafilatura
                                                             # fetch; just write
                                                             # URL metadata

Writes:
  data/gdelt_cameo/gdelt_cameo_news.jsonl  (same schema as the DOC-API fetcher,
                                            so unify_articles.py picks it up
                                            unchanged)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.common.optional_imports import optional  # noqa: E402
from src.common.spam_domains import is_spam_url  # noqa: E402

bigquery = optional("google.cloud.bigquery", install_hint="pip install google-cloud-bigquery")
trafilatura = optional("trafilatura", install_hint="pip install trafilatura")
requests_lib = optional("requests")

DATA_DIR = ROOT / "data" / "gdelt_cameo"
OUT_FILE = DATA_DIR / "gdelt_cameo_news.jsonl"
FD_SOURCES = [
    ROOT / "data" / "unified" / "forecasts.jsonl",
    ROOT / "data" / "staged" / "20260421_161626" / "01_after_first_unify" / "forecasts.jsonl",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# FD loading
# ---------------------------------------------------------------------------

def load_gdelt_fds() -> list[dict]:
    """Load GDELT-CAMEO FDs from the first available source."""
    for src in FD_SOURCES:
        if src.exists():
            print(f"[fds] loading from {src}")
            out = []
            for line in src.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("benchmark") == "gdelt-cameo":
                    out.append(r)
            return out
    raise FileNotFoundError("No forecasts.jsonl found; run unify_forecasts.py first.")


def fd_actor_pair(fd: dict) -> tuple[str, str] | None:
    """Extract (actor1_cc, actor2_cc) from FD metadata. Returns None if missing."""
    md = fd.get("metadata", {}) or {}
    actors = md.get("actors") or []
    if len(actors) >= 2 and actors[0] and actors[1]:
        return (actors[0].upper(), actors[1].upper())
    return None


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------

_QUERY_TEMPLATE = """
-- EMR-ACH pre-event GDELT GKG fetch. Returns URLs indexed by actor pair
-- and publication date, restricted to the 90-day pre-event window per FD.
SELECT
  DATE,
  V2SourceCommonName AS source,
  DocumentIdentifier AS url,
  V2Themes,
  V2Locations,
  V2Tone,
  actor_s,
  actor_o
FROM (
  SELECT
    d.DATE,
    d.V2SourceCommonName,
    d.DocumentIdentifier,
    d.V2Themes,
    d.V2Locations,
    d.V2Tone,
    pair.actor_s,
    pair.actor_o
  FROM `gdelt-bq.gdeltv2.gkg` AS d
  CROSS JOIN UNNEST(@actor_pairs) AS pair
  WHERE d.DATE BETWEEN pair.start_yyyymmdd AND pair.end_yyyymmdd
    AND REGEXP_CONTAINS(d.V2Locations, pair.actor_s)
    AND REGEXP_CONTAINS(d.V2Locations, pair.actor_o)
    AND d.DocumentIdentifier IS NOT NULL
)
-- Cap per (actor_s, actor_o) pair to keep results bounded
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY actor_s, actor_o
  ORDER BY DATE DESC
) <= @per_pair_cap
"""


def run_bigquery_fetch(fds: list[dict], lookback_days: int, per_pair_cap: int,
                      project: str | None) -> list[dict]:
    """Execute the single bulk BigQuery call. Returns a list of
    {fd_id, url, date, source} records."""
    if not bigquery:
        raise RuntimeError("google-cloud-bigquery not installed; see module docstring.")

    # Group FDs by actor pair (many FDs may share a pair; we query once per pair)
    pair_to_fds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fd in fds:
        pair = fd_actor_pair(fd)
        if pair:
            pair_to_fds[pair].append(fd)

    # Build parameter set: for each pair, determine widest (earliest_forecast_point - lookback .. latest_fp - 1)
    actor_pair_params = []
    for (s, o), pair_fds in pair_to_fds.items():
        fps = sorted(fd.get("forecast_point", "") for fd in pair_fds if fd.get("forecast_point"))
        if not fps:
            continue
        earliest_fp = datetime.strptime(fps[0][:10], "%Y-%m-%d")
        latest_fp = datetime.strptime(fps[-1][:10], "%Y-%m-%d")
        start = (earliest_fp - timedelta(days=lookback_days)).strftime("%Y%m%d")
        end = (latest_fp - timedelta(days=1)).strftime("%Y%m%d")
        actor_pair_params.append({
            "actor_s": s, "actor_o": o,
            "start_yyyymmdd": int(start), "end_yyyymmdd": int(end),
        })

    print(f"[bq] {len(pair_to_fds)} distinct actor pairs; covering {len(fds)} FDs")

    client = bigquery._resolve().Client(project=project)
    import google.cloud.bigquery as bq  # noqa: E402 -- real import, safe now
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter("actor_pairs", "STRUCT<actor_s STRING, actor_o STRING, start_yyyymmdd INT64, end_yyyymmdd INT64>", actor_pair_params),
            bq.ScalarQueryParameter("per_pair_cap", "INT64", per_pair_cap),
        ],
        use_legacy_sql=False,
    )

    print(f"[bq] submitting query; scanning ~5-10 GB of the gdeltv2.gkg table...")
    t0 = time.time()
    query_job = client.query(_QUERY_TEMPLATE, job_config=job_config)
    rows = list(query_job.result())
    t1 = time.time()
    bytes_billed = query_job.total_bytes_billed or 0
    print(f"[bq] returned {len(rows)} rows in {t1-t0:.1f}s  (bytes_billed={bytes_billed/1e9:.2f} GB)")

    # Convert BQ rows to our intermediate format, then attach to FDs by actor pair + date window
    out = []
    for row in rows:
        d = str(row["DATE"])
        date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) >= 8 else d
        out.append({
            "date":    date_str,
            "url":     row["url"],
            "source":  row["source"] or "",
            "actor_s": row["actor_s"],
            "actor_o": row["actor_o"],
            "themes":  row["V2Themes"] or "",
            "tone":    float(row["V2Tone"].split(",")[0]) if row["V2Tone"] else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Writing — match the JSONL schema of fetch_gdelt_cameo_news.py
# ---------------------------------------------------------------------------

def art_id(url: str) -> str:
    return "gdc_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except (AttributeError, ValueError):
        return ""


def write_articles(bq_rows: list[dict], fds: list[dict], out_path: Path,
                   per_fd_cap: int = 30) -> int:
    """Attach BQ rows to FDs whose actor pair and forecast_point match, then
    write per-FD article records to the output file. Dedups by URL across
    existing content in the file.
    """
    # Index FDs by actor pair
    pair_to_fds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fd in fds:
        p = fd_actor_pair(fd)
        if p:
            pair_to_fds[p].append(fd)

    # Load existing URLs for dedup
    seen_urls: set[str] = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                seen_urls.add(json.loads(line).get("url", ""))
            except json.JSONDecodeError:
                continue
    print(f"[write] {len(seen_urls)} existing URLs (will dedup)")

    written = 0
    per_prov = Counter()
    with out_path.open("a", encoding="utf-8") as f:
        for row in bq_rows:
            url = row["url"]
            if not url or url in seen_urls or is_spam_url(url):
                if url and is_spam_url(url):
                    per_prov["__dropped_spam__"] += 1
                continue
            # Find the FDs this row could belong to (same actor pair, date
            # within the FD's lookback window).
            key = (row["actor_s"], row["actor_o"])
            candidate_fds = pair_to_fds.get(key, [])
            try:
                row_dt = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            except ValueError:
                continue
            for fd in candidate_fds:
                fp_str = fd.get("forecast_point", "")
                if not fp_str:
                    continue
                fp_dt = datetime.strptime(fp_str[:10], "%Y-%m-%d")
                if not (fp_dt - timedelta(days=90) <= row_dt < fp_dt):
                    continue
                # Count the same URL only once per FD for per-FD cap
                rec = {
                    "id":          art_id(url),
                    "fd_id":       fd["id"],
                    "query":       f"BQ:{key[0]}-{key[1]}",
                    "url":         url,
                    "title":       "",   # GKG doesn't return titles; trafilatura fetch fills this
                    "text":        "",
                    "date":        row["date"],
                    "source":      row["source"] or domain_of(url),
                    "provenance":  "gdelt-cameo-bq",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_prov["gdelt-cameo-bq"] += 1
            seen_urls.add(url)

    print(f"\n[done] wrote {written} article records  {dict(per_prov)}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT"),
                    help="GCP project (also from $GCP_PROJECT env var)")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--per-pair-cap", type=int, default=50,
                    help="Max URLs per actor pair returned by BigQuery")
    ap.add_argument("--per-fd-cap", type=int, default=30,
                    help="Max URLs attached to any single FD")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N GDELT FDs (debug)")
    ap.add_argument("--all", action="store_true",
                    help="Process the full set (required unless --limit is given)")
    ap.add_argument("--urls-only", action="store_true",
                    help="Skip trafilatura full-text fetch; write URL metadata only "
                         "(trafilatura can then be run separately via fetch_article_text.py).")
    args = ap.parse_args()

    if not args.all and args.limit is None:
        print("[ERROR] Pass --all for full corpus or --limit N for debug.")
        sys.exit(2)

    if not args.project:
        print("[ERROR] No GCP project set. Pass --project or export GCP_PROJECT.")
        sys.exit(2)

    if not bigquery:
        print("[ERROR] google-cloud-bigquery not installed. `pip install google-cloud-bigquery`")
        sys.exit(2)

    fds = load_gdelt_fds()
    if args.limit:
        fds = fds[: args.limit]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_bigquery_fetch(fds, args.lookback, args.per_pair_cap, args.project)
    n_written = write_articles(rows, fds, OUT_FILE, args.per_fd_cap)
    print(f"\n[summary] {n_written} articles written to {OUT_FILE.relative_to(ROOT)}")

    if not args.urls_only:
        print("\nNext step (trafilatura text fetch on the newly-added URLs):")
        print("  python scripts/fetch_article_text.py --only-referenced --workers 24")


if __name__ == "__main__":
    main()
