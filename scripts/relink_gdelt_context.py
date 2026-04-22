"""
Re-link GDELT-CAMEO FDs to leakage-free context articles.

The GDELT-CAMEO oracle Docids point to articles that describe the Nov 2024
test event itself, so using them for prediction leaks the answer. This
script replaces each query's article_ids with Aug-Oct 2024 context articles
from data_kg.csv that share the same actor pair (s, o).

Strategy per query (t=Nov 2024 date, s, o):
  1. Scan data_kg.csv for rows where
       DateStr < t  AND
       ((Actor1CountryCode=s AND Actor2CountryCode=o) OR
        (Actor1CountryCode=o AND Actor2CountryCode=s))
  2. Collect the Docid / Docids fields
  3. Resolve Docid -> URL via data_news.csv, URL -> unified art_id
  4. Take the most recent up-to-10 distinct article_ids
  5. Overwrite fc.article_ids for GDELT-CAMEO entries

Reads:
  data/unified/forecasts.jsonl
  data/unified/articles.jsonl
  data/gdelt_cameo/data_kg.csv
  data/gdelt_cameo/data_news.csv

Writes:
  data/unified/forecasts.jsonl   (in place, only GDELT-CAMEO entries modified)
  data/unified/gdelt_cameo_relink_meta.json

Usage:
  python scripts/relink_gdelt_context.py
"""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
UNI = ROOT / "data" / "unified"
FC_FILE = UNI / "forecasts.jsonl"
ART_FILE = UNI / "articles.jsonl"
KG_CSV = ROOT / "data" / "gdelt_cameo" / "data_kg.csv"
NEWS_CSV = ROOT / "data" / "gdelt_cameo" / "data_news.csv"
META = UNI / "gdelt_cameo_relink_meta.json"

TOP_K = 10


def art_id(url: str) -> str:
    return "art_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def parse_date(s: str):
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d")
    except Exception:
        return None


def parse_docids_field(raw: str):
    """Docids can be '[123, 456, 123]' or just a single int or empty."""
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    if raw.startswith("["):
        try:
            return [str(d) for d in json.loads(raw.replace("'", '"'))]
        except Exception:
            return []
    return [raw]


def main():
    forecasts = [json.loads(l) for l in open(FC_FILE, encoding="utf-8")]
    mirai_fcs = [f for f in forecasts if f.get("benchmark") == "gdelt-cameo"]
    print(f"Loaded {len(forecasts)} forecasts; {len(mirai_fcs)} GDELT-CAMEO entries")

    # Build Docid -> art_id map via data_news.csv
    print(f"Building Docid -> art_id map from {NEWS_CSV}...")
    docid_to_aid = {}
    docid_to_date = {}
    with open(NEWS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            did = row.get("Docid", "")
            url = row.get("URL", "")
            d = (row.get("Date", "") or "")[:10]
            if did and url:
                docid_to_aid[did] = art_id(url)
                docid_to_date[did] = d
    print(f"  {len(docid_to_aid)} Docid->art_id mappings")

    # Build an index from (s, o) actor-pair -> list of (date, [article_ids])
    # by scanning data_kg.csv
    print(f"Building actor-pair index from {KG_CSV}...")
    pair_idx: dict[tuple[str, str], list[tuple[str, list[str]]]] = defaultdict(list)
    total_rows = 0
    with open(KG_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            total_rows += 1
            s = row.get("Actor1CountryCode", "")
            o = row.get("Actor2CountryCode", "")
            d = row.get("DateStr", "")
            if not (s and o and d):
                continue
            # Docids may be multi-valued list; Docid is the primary
            dids = parse_docids_field(row.get("Docids", "")) or [row.get("Docid", "")]
            aids = [docid_to_aid[d_] for d_ in dids if d_ in docid_to_aid]
            if not aids:
                continue
            pair_idx[(s, o)].append((d, aids))
    print(f"  scanned {total_rows} KG rows, {len(pair_idx)} distinct actor pairs indexed")

    # now relink each GDELT-CAMEO FD
    articles = {json.loads(l)["id"]: json.loads(l) for l in open(ART_FILE, encoding="utf-8")}
    relinked = 0
    still_empty = 0
    per_pair_hits = []
    for fc in mirai_fcs:
        # Actors now live in metadata.actors = [subject_code, object_code].
        # The question text is natural-language and no longer a parseable tuple.
        try:
            date_str = fc.get("forecast_point", "")[:10]
            rd = parse_date(date_str)
            actors = fc.get("metadata", {}).get("actors") or []
            if len(actors) < 2:
                continue
            s, o = actors[0], actors[1]
        except Exception:
            continue

        # Gather pre-event rows for this (s,o) and (o,s) order
        cand_rows = []
        for pair in ((s, o), (o, s)):
            for d_str, aids in pair_idx.get(pair, []):
                d = parse_date(d_str)
                if not d or not rd or d >= rd:
                    continue
                cand_rows.append((d, aids))

        # Sort by recency (most recent first), flatten, dedupe while preserving order
        cand_rows.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        ordered_aids = []
        for _, aids in cand_rows:
            for aid in aids:
                if aid in articles and aid not in seen:
                    ordered_aids.append(aid)
                    seen.add(aid)
                if len(ordered_aids) >= TOP_K:
                    break
            if len(ordered_aids) >= TOP_K:
                break

        fc["article_ids"] = ordered_aids
        if ordered_aids:
            relinked += 1
            per_pair_hits.append(len(ordered_aids))
        else:
            still_empty += 1

    # Save — atomic write to protect against corruption on mid-way crash.
    # Write to .tmp sibling file, fsync, then os.replace (POSIX + Windows safe).
    tmp_path = FC_FILE.with_suffix(FC_FILE.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for fc in forecasts:
            f.write(json.dumps(fc, ensure_ascii=False) + "\n")
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except Exception:
            pass
    import os as _os
    _os.replace(str(tmp_path), str(FC_FILE))

    import statistics
    meta = {
        "gdelt_cameo_total": len(mirai_fcs),
        "relinked": relinked,
        "still_empty": still_empty,
        "median_articles_per_fd": statistics.median(per_pair_hits) if per_pair_hits else 0,
        "mean_articles_per_fd": round(sum(per_pair_hits) / len(per_pair_hits), 2) if per_pair_hits else 0,
        "articles_saturated_at_10": sum(1 for k in per_pair_hits if k == 10),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nRelinking complete:")
    print(f"  GDELT-CAMEO FDs:       {len(mirai_fcs)}")
    print(f"  Successfully relinked: {relinked}")
    print(f"  Still zero articles:   {still_empty}")
    print(f"  Median art/FD:         {meta['median_articles_per_fd']}")
    print(f"  Saturated at 10 art:   {meta['articles_saturated_at_10']}")


if __name__ == "__main__":
    main()
