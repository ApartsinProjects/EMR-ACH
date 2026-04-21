"""
Unify forecast records from ForecastBench and MIRAI-2024 into a single schema.

Reads (read-only):
  data/forecastbench_geopolitics.jsonl          - ForecastBench binary questions
  data/mirai_2024/relation_query.csv            - MIRAI-2024 4-class queries (if exists)
  data/unified/articles.jsonl                   - needed to build url->article_id map

Writes:
  data/unified/forecasts.jsonl                  - unified forecast records
  data/unified/forecasts_meta.json              - summary stats

Forecast schema:
  { id, benchmark, source, hypothesis_set, question, background, forecast_point,
    resolution_date, ground_truth, ground_truth_idx, crowd_probability,
    lookback_days, article_ids }

`article_ids` is populated from the existing question_id <-> article linkage in
gdelt_articles.jsonl for ForecastBench and from oracle Docid for MIRAI-2024.
compute_relevance.py can later refine or replace these.

Usage:
  python scripts/unify_forecasts.py
"""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
UNIFIED = DATA / "unified"
UNIFIED.mkdir(parents=True, exist_ok=True)

FB_IN        = DATA / "forecastbench_geopolitics.jsonl"
FB_ARTS      = DATA / "gdelt_articles.jsonl"
FB_ARTS_SUPP = DATA / "gdelt_articles_supplement.jsonl"
MIRAI_Q      = DATA / "mirai_2024" / "test" / "relation_query.csv"
MIRAI_KG     = DATA / "mirai_2024" / "data_kg.csv"
UNI_ARTS     = UNIFIED / "articles.jsonl"
OUT_FC       = UNIFIED / "forecasts.jsonl"
OUT_META     = UNIFIED / "forecasts_meta.json"

FB_HYP       = ["Yes", "No"]
MIRAI_HYP    = ["VC", "MC", "VK", "MK"]
FB_LOOKBACK  = 30
MIRAI_LOOKBACK = 90  # Aug-Oct context for Nov test month


def art_id(url: str) -> str:
    return "art_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def load_url_to_art_id() -> dict[str, str]:
    m: dict[str, str] = {}
    if not UNI_ARTS.exists():
        return m
    with open(UNI_ARTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["url"]] = r["id"]
    return m


def build_fb_question_to_urls() -> dict[str, list[str]]:
    """Map ForecastBench question_id -> list of article URLs (main + supplement)."""
    q2urls: dict[str, list[str]] = defaultdict(list)
    for path in (FB_ARTS, FB_ARTS_SUPP):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                qid = r.get("question_id", "")
                url = r.get("url", "")
                if qid and url:
                    q2urls[qid].append(url)
    return q2urls


def load_forecastbench(url2aid: dict[str, str]) -> list[dict]:
    if not FB_IN.exists():
        print(f"[WARN] {FB_IN} not found; skipping ForecastBench")
        return []
    q2urls = build_fb_question_to_urls()
    out = []
    for line in open(FB_IN, encoding="utf-8"):
        q = json.loads(line)
        qid = q.get("id", "")
        gt = int(q.get("ground_truth", 0))
        gt_label = FB_HYP[0] if gt == 1 else FB_HYP[1]
        res_date = q.get("resolution_date", "")
        # forecast_point = resolution_date minus some small offset; ForecastBench
        # uses freeze_datetime which is typically the question's freeze date.
        # Use resolution_date as a conservative stand-in — articles with a
        # stricter pub_date < forecast_point filter will be enforced by the
        # relevance/quality steps and the per-query protocol runs.
        forecast_point = res_date
        urls = q2urls.get(qid, [])
        article_ids = sorted({url2aid[u] for u in urls if u in url2aid})
        out.append({
            "id": f"fb_{qid}" if not qid.startswith("fb_") else qid,
            "benchmark": "forecastbench",
            "source": q.get("source", ""),
            "hypothesis_set": FB_HYP,
            "question": q.get("question", ""),
            "background": q.get("background", ""),
            "forecast_point": forecast_point,
            "resolution_date": res_date,
            "ground_truth": gt_label,
            "ground_truth_idx": 0 if gt == 1 else 1,  # Yes=idx0, No=idx1
            "crowd_probability": q.get("crowd_probability"),
            "lookback_days": FB_LOOKBACK,
            "article_ids": article_ids,
        })
    return out


def cameo_to_quad(code: str) -> int | None:
    """Map CAMEO 2-digit root code (01-20) to quad-class index 0..3."""
    try:
        root = int(str(code)[:2])
    except Exception:
        return None
    if 1 <= root <= 4:
        return 0  # VC
    if 5 <= root <= 8:
        return 1  # MC
    if 9 <= root <= 16:
        return 2  # VK
    if 17 <= root <= 20:
        return 3  # MK
    return None


def load_mirai_2024(url2aid: dict[str, str]) -> list[dict]:
    if not MIRAI_Q.exists():
        print(f"[WARN] {MIRAI_Q} not found; skipping MIRAI-2024 (build pending)")
        return []
    # Build Docid -> URL lookup from data_news.csv so we can map oracle doc
    # references into unified article IDs.
    docid_to_url: dict[str, str] = {}
    news_csv = DATA / "mirai_2024" / "data_news_full.csv"
    if not news_csv.exists():
        news_csv = DATA / "mirai_2024" / "data_news.csv"
    if news_csv.exists():
        with open(news_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                did = row.get("Docid", "")
                url = row.get("URL", "")
                if did and url:
                    docid_to_url[did] = url
    out = []
    with open(MIRAI_Q, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qid = row.get("QueryId", "") or row.get("QueryID", "")
            s = row.get("Actor1CountryCode", "")
            o = row.get("Actor2CountryCode", "")
            date = (row.get("DateStr", "") or "")[:10]
            # QuadLabel is precomputed VC/MC/VK/MK — use it directly.
            quad = row.get("QuadLabel", "")
            if quad not in MIRAI_HYP:
                continue
            gt_idx = MIRAI_HYP.index(quad)
            docids_raw = row.get("Docids", "")
            try:
                docids = [str(d) for d in json.loads(docids_raw.replace("'", '"'))] if docids_raw.startswith("[") else []
            except Exception:
                docids = []
            urls = [docid_to_url[d] for d in docids if d in docid_to_url]
            article_ids = sorted({url2aid[u] for u in urls if u in url2aid})
            out.append({
                "id": f"mirai24_{qid}",
                "benchmark": "mirai-2024",
                "source": "gdelt-kg",
                "hypothesis_set": MIRAI_HYP,
                "question": f"({date}, {s}, ?, {o})",
                "background": "",
                "forecast_point": date,
                "resolution_date": date,
                "ground_truth": quad,
                "ground_truth_idx": gt_idx,
                "crowd_probability": None,
                "lookback_days": MIRAI_LOOKBACK,
                "article_ids": article_ids,
            })
    return out


def main():
    print("Loading URL -> unified article_id map...")
    url2aid = load_url_to_art_id()
    print(f"  {len(url2aid)} URL-keyed articles")

    print("Loading ForecastBench forecasts...")
    fb = load_forecastbench(url2aid)
    print(f"  {len(fb)} records")

    print("Loading MIRAI-2024 forecasts...")
    m24 = load_mirai_2024(url2aid)
    print(f"  {len(m24)} records")

    all_fc = fb + m24
    with open(OUT_FC, "w", encoding="utf-8") as f:
        for r in all_fc:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stats
    from collections import Counter
    by_src = Counter(r["source"] for r in all_fc)
    by_bench = Counter(r["benchmark"] for r in all_fc)
    avg_arts = sum(len(r["article_ids"]) for r in all_fc) / len(all_fc) if all_fc else 0
    art_hist = Counter(len(r["article_ids"]) for r in all_fc)
    gt_dist_fb = Counter(r["ground_truth"] for r in all_fc if r["benchmark"] == "forecastbench")
    gt_dist_m24 = Counter(r["ground_truth"] for r in all_fc if r["benchmark"] == "mirai-2024")

    meta = {
        "total_forecasts": len(all_fc),
        "by_benchmark": dict(by_bench),
        "by_source": dict(by_src),
        "avg_articles_per_forecast": round(avg_arts, 2),
        "article_count_histogram": dict(sorted(art_hist.items())),
        "ground_truth_distribution": {
            "forecastbench": dict(gt_dist_fb),
            "mirai-2024": dict(gt_dist_m24),
        },
        "output_path": str(OUT_FC),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Wrote {len(all_fc)} forecasts to {OUT_FC}")
    print(f"  By benchmark: {dict(by_bench)}")
    print(f"  By source:    {dict(by_src)}")
    print(f"  Avg articles per forecast: {avg_arts:.2f}")
    print(f"  Articles-per-forecast histogram (0/1/2/3+): "
          f"{art_hist.get(0,0)}/{art_hist.get(1,0)}/{art_hist.get(2,0)}/"
          f"{sum(v for k,v in art_hist.items() if k>=3)}")


if __name__ == "__main__":
    main()
