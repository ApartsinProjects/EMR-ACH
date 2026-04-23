"""Unify forecast records from all three benchmarks into a single Forecast Dossier.

This script is the **producer** of the canonical FD record. The contract is
defined in [`docs/FORECAST_DOSSIER.md`](../docs/FORECAST_DOSSIER.md) (v2.1) and
the per-field reference at [`benchmark/schema/forecast_dossier.md`].
Downstream consumers (annotate_prior_state, compute_relevance, quality_filter,
baselines runner) treat the FD as immutable schema; any new field added here
must be reflected in both schema docs and the JSON Schema validator.

Reads (read-only):
  data/forecastbench_geopolitics.jsonl   ForecastBench binary questions
  data/gdelt_cameo/relation_query.csv    GDELT-CAMEO queries with EventBaseCode
  data/earnings/earnings_forecasts.jsonl Earnings Beat/Meet/Miss FDs
  data/unified/articles.jsonl            url -> article_id resolution

Writes:
  data/unified/forecasts.jsonl           one FD per line
  data/unified/forecasts_meta.json       summary stats by benchmark + source

Per-benchmark `forecast_horizon_days` is read from the env vars
EMRACH_FB_HORIZON_DAYS / EMRACH_GDELT_HORIZON_DAYS / EMRACH_EARN_HORIZON_DAYS
(set by scripts/build_benchmark.py from `default_config.yaml`). The horizon
is emitted as `default_horizon_days` on each FD; the actual horizon filter
runs at experiment time in the baselines runner.

GDELT ground-truth labels go through `src/common/cameo_intensity.event_to_intensity()`
to produce the Peace/Tension/Violence 3-class ordinal target. The lossy
`quad_to_intensity_maybe()` fallback runs only if EventBaseCode is missing.

Comply/Surprise binary promotion happens **downstream** in
`scripts/annotate_prior_state.py`, NOT here. This script produces
domain-multiclass FDs; the annotator stashes the multiclass label as
`x_multiclass_*` and sets the primary `hypothesis_set` to ["Comply", "Surprise"].

Usage:
  python scripts/unify_forecasts.py
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
# Fast JSONL I/O (orjson if available, stdlib json fallback) — see _fast_jsonl.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from _fast_jsonl import loads as _j_loads, dumps as _j_dumps

# The GDELT-CAMEO relation_query.csv has Docids fields that can exceed Python's default
# 128k CSV field limit. Bump to the max.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
UNIFIED = DATA / "unified"
UNIFIED.mkdir(parents=True, exist_ok=True)

FB_IN        = DATA / "forecastbench_geopolitics.jsonl"
FB_ARTS      = DATA / "gdelt_articles.jsonl"
FB_ARTS_SUPP = DATA / "gdelt_articles_supplement.jsonl"
GDELT_CAMEO_Q      = DATA / "gdelt_cameo" / "test" / "relation_query.csv"
GDELT_CAMEO_KG     = DATA / "gdelt_cameo" / "data_kg.csv"
UNI_ARTS     = UNIFIED / "articles.jsonl"
OUT_FC       = UNIFIED / "forecasts.jsonl"
OUT_META     = UNIFIED / "forecasts_meta.json"

FB_HYP       = ["Yes", "No"]

# GDELT-CAMEO hypothesis set (2026-04-22 redesign):
#   3-class intensity taxonomy — Peace / Tension / Violence — mapped from
#   CAMEO root codes. See src/common/cameo_intensity.py for the full mapping
#   and scripts/annotate_gdelt_prior_state.py for the stability/change slice
#   computation that accompanies this label set.
#
# Motivation: the prior 4-class VC/MC/VK/MK framing had two semantically-close
# pairs that even competent LLMs struggled to distinguish (VC vs MC, VK vs MK).
# The Peace/Tension/Violence reduction has clearly distinguishable class
# boundaries, is ordinally ordered (Peace < Tension < Violence), and maps
# cleanly to the underlying CAMEO root taxonomy.
from src.common.cameo_intensity import (
    INTENSITY_CLASSES as _GDC_INTENSITY,
    INTENSITY_DEFINITIONS as _GDC_INTENSITY_DEFS,
)
GDELT_CAMEO_HYP        = list(_GDC_INTENSITY)                  # ["Peace","Tension","Violence"]
GDELT_CAMEO_HYP_CODES  = ["P", "T", "V"]                       # compact labels
GDELT_CAMEO_HYP_DEFS   = dict(_GDC_INTENSITY_DEFS)

FB_LOOKBACK  = 90   # unified 'forecast from prior news' analysis window
GDELT_CAMEO_LOOKBACK = 90   # matches FB + earnings for uniform task framing

# Forecast horizon h — evidence cutoff is resolution_date − h days.
# Default 14 (2 weeks): the system's last permissible article is 2 weeks
# before the outcome, so the task is genuine forecasting (not nowcasting).
# Overridable per-benchmark via configs/default_config.yaml -> benchmarks.<name>.forecast_horizon_days
# and injected into this module at build time via environment variables set
# by scripts/build_benchmark.py.
import os as _os
FB_FORECAST_HORIZON_DAYS      = int(_os.environ.get("EMRACH_FB_HORIZON_DAYS", "14"))
GDELT_FORECAST_HORIZON_DAYS   = int(_os.environ.get("EMRACH_GDELT_HORIZON_DAYS", "14"))
EARN_FORECAST_HORIZON_DAYS    = int(_os.environ.get("EMRACH_EARN_HORIZON_DAYS", "14"))

FB_HYP_DEFS = {
    "Yes": "The event or outcome described in the question occurs by the resolution date.",
    "No":  "The event or outcome described in the question does not occur by the resolution date.",
}

EARN_HYP_DEFS = {
    "Beat":  "Reported EPS exceeds analyst consensus by at least the beat threshold.",
    "Meet":  "Reported EPS is within the beat/miss threshold of analyst consensus.",
    "Miss":  "Reported EPS falls below analyst consensus by at least the miss threshold.",
}


def art_id(url: str) -> str:
    return "art_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def load_url_to_art_id() -> dict[str, str]:
    m: dict[str, str] = {}
    if not UNI_ARTS.exists():
        return m
    with open(UNI_ARTS, encoding="utf-8") as f:
        for line in f:
            r = _j_loads(line)
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
                r = _j_loads(line)
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
        q = _j_loads(line)
        qid = q.get("id", "")
        gt = int(q.get("ground_truth", 0))
        gt_label = FB_HYP[0] if gt == 1 else FB_HYP[1]
        res_date = q.get("resolution_date", "")
        # forecast_point stores the max-evidence-date (= resolution_date). The
        # actual experimental cutoff is applied at runtime: the baseline runner
        # filters article_ids to those with publish_date < (resolution_date −
        # experiment_horizon_days). Default horizon is emitted into the FD as
        # `default_horizon_days` (see v2.1 design, docs/FORECAST_DOSSIER.md §3).
        forecast_point = res_date
        urls = q2urls.get(qid, [])
        article_ids = sorted({url2aid[u] for u in urls if u in url2aid})
        out.append({
            "id": f"fb_{qid}" if not qid.startswith("fb_") else qid,
            "benchmark": "forecastbench",
            "source": q.get("source", ""),
            "hypothesis_set": FB_HYP,
            "hypothesis_definitions": FB_HYP_DEFS,
            "question": q.get("question", ""),
            "background": q.get("background", ""),
            "forecast_point": forecast_point,
            "resolution_date": res_date,
            "ground_truth": gt_label,
            "ground_truth_idx": 0 if gt == 1 else 1,
            "crowd_probability": q.get("crowd_probability"),
            "lookback_days": FB_LOOKBACK,
            "default_horizon_days": FB_FORECAST_HORIZON_DAYS,
            "article_ids": article_ids,
            "metadata": {
                "category":       q.get("category", ""),
                "original_id":    qid,
            },
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


def load_gdelt_cameo(url2aid: dict[str, str]) -> list[dict]:
    if not GDELT_CAMEO_Q.exists():
        print(f"[WARN] {GDELT_CAMEO_Q} not found; skipping GDELT-CAMEO (build pending)")
        return []
    # Build Docid -> URL lookup from data_news.csv so we can map oracle doc
    # references into unified article IDs.
    docid_to_url: dict[str, str] = {}
    news_csv = DATA / "gdelt_cameo" / "data_news_full.csv"
    if not news_csv.exists():
        news_csv = DATA / "gdelt_cameo" / "data_news.csv"
    if news_csv.exists():
        with open(news_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                did = row.get("Docid", "")
                url = row.get("URL", "")
                if did and url:
                    docid_to_url[did] = url
    from src.common.cameo_intensity import event_to_intensity, INTENSITY_RANK
    out = []
    with open(GDELT_CAMEO_Q, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qid = row.get("QueryId", "") or row.get("QueryID", "")
            s = row.get("Actor1CountryCode", "")
            o = row.get("Actor2CountryCode", "")
            s_name = row.get("Actor1CountryName", "") or s
            o_name = row.get("Actor2CountryName", "") or o
            date = (row.get("DateStr", "") or "")[:10]
            # EventBaseCode is the 3- or 4-digit CAMEO code; we derive the
            # 3-class intensity label (Peace/Tension/Violence) from its root.
            event_base_code = row.get("EventBaseCode", "") or row.get("QuadLabel", "")
            intensity = event_to_intensity(event_base_code)
            if intensity is None:
                # If EventBaseCode absent (older relation_query.csv format), fall
                # back to the QuadLabel + lossy mapping, keeping the FD.
                from src.common.cameo_intensity import quad_to_intensity_maybe
                intensity = quad_to_intensity_maybe(row.get("QuadLabel", ""))
            if intensity is None:
                continue
            gt_idx = INTENSITY_RANK[intensity]
            gt_label = intensity  # already the human-readable class name
            docids_raw = row.get("Docids", "")
            try:
                docids = [str(d) for d in json.loads(docids_raw.replace("'", '"'))] if docids_raw.startswith("[") else []
            except Exception:
                docids = []
            urls = [docid_to_url[d] for d in docids if d in docid_to_url]
            article_ids = sorted({url2aid[u] for u in urls if u in url2aid})

            # Natural-language question framing: forecast the conflict-intensity
            # level from prior news. No CAMEO codes surfaced in the prompt.
            question = (f"Based on news from the preceding months, what is the dominant "
                        f"intensity of interaction between {s_name} and {o_name} "
                        f"on or around {date}: peace, tension, or violence?")

            # forecast_point = resolution date (max evidence date). The
            # experimental horizon filter is applied at runtime; default
            # horizon is emitted below as `default_horizon_days`.
            forecast_point_str = date

            out.append({
                "id": f"gdc_{qid}",
                "benchmark": "gdelt-cameo",
                "source": "gdelt-kg",
                "hypothesis_set": GDELT_CAMEO_HYP,
                "hypothesis_definitions": GDELT_CAMEO_HYP_DEFS,
                "question": question,
                "background": (f"Forecast the conflict-intensity level (Peace, Tension, or "
                               f"Violence) between {s_name} ({s}) and {o_name} ({o}) on "
                               f"{date}, using only news published at least "
                               f"{GDELT_FORECAST_HORIZON_DAYS} days before that date."),
                "forecast_point": forecast_point_str,
                "resolution_date": date,
                "ground_truth": gt_label,
                "ground_truth_idx": gt_idx,
                "crowd_probability": None,
                "lookback_days": GDELT_CAMEO_LOOKBACK,
                "default_horizon_days": GDELT_FORECAST_HORIZON_DAYS,
                "article_ids": article_ids,
                "metadata": {
                    # Retrieval + leakage-check + prior-state annotation side
                    # rely on these structured fields; downstream prompt
                    # construction is natural-language only.
                    "actors":              [s, o],
                    "actor_names":         [s_name, o_name],
                    "event_base_code":     event_base_code,
                    "intensity_class":     intensity,
                    "hypothesis_codes":    GDELT_CAMEO_HYP_CODES,
                    "original_query_id":   qid,
                },
            })
    return out


EARN_IN = DATA / "earnings" / "earnings_forecasts.jsonl"


def load_earnings(url2aid: dict[str, str]) -> list[dict]:
    """Load earnings-surprise FDs from build_earnings_benchmark.py output.
    Strips the private `_earnings_meta` field before emitting; the salient
    numbers (consensus EPS, sector) are already folded into `background`.
    """
    if not EARN_IN.exists():
        return []
    out = []
    for line in open(EARN_IN, encoding="utf-8"):
        r = _j_loads(line)
        # normalize benchmark identifier to bare "earnings" (year-agnostic)
        r["benchmark"] = "earnings"
        r["hypothesis_definitions"] = EARN_HYP_DEFS
        r.pop("_earnings_meta", None)
        out.append(r)
    return out


def main():
    print("Loading URL -> unified article_id map...")
    url2aid = load_url_to_art_id()
    print(f"  {len(url2aid)} URL-keyed articles")

    print("Loading ForecastBench forecasts...")
    fb = load_forecastbench(url2aid)
    print(f"  {len(fb)} records")

    print("Loading GDELT-CAMEO forecasts...")
    gdc = load_gdelt_cameo(url2aid)
    print(f"  {len(gdc)} records")

    print("Loading Earnings forecasts...")
    earn = load_earnings(url2aid)
    print(f"  {len(earn)} records")

    all_fc = fb + gdc + earn
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
    gt_dist_gdc = Counter(r["ground_truth"] for r in all_fc if r["benchmark"] == "gdelt-cameo")

    meta = {
        "total_forecasts": len(all_fc),
        "by_benchmark": dict(by_bench),
        "by_source": dict(by_src),
        "avg_articles_per_forecast": round(avg_arts, 2),
        "article_count_histogram": dict(sorted(art_hist.items())),
        "ground_truth_distribution": {
            "forecastbench": dict(gt_dist_fb),
            "gdelt-cameo": dict(gt_dist_gdc),
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
