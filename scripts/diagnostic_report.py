"""
Pre-experiment diagnostic report on the unified Forecast Dossier set.

Reads:
  data/unified/forecasts_filtered.jsonl
  data/unified/articles.jsonl

Writes:
  data/unified/diagnostic_report.md       - human-readable report
  data/unified/diagnostic_report.json     - machine-readable stats

Run before Phase 1. Surfaces:
  - per-source FD counts and resolution-date spans
  - articles-per-FD histogram
  - article date-spread (stddev in days) distribution
  - ground-truth class balance
  - 20 random FDs spot-check (id + question + n_articles + crowd_prob + GT)
  - leakage audit confirmation (no article date >= forecast_point in any FD)

Usage:
  python scripts/diagnostic_report.py
"""
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
UNI = ROOT / "data" / "unified"
FC = UNI / "forecasts_filtered.jsonl"
AR = UNI / "articles.jsonl"
OUT_MD = UNI / "diagnostic_report.md"
OUT_JSON = UNI / "diagnostic_report.json"


def parse_date(s: str):
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d")
    except Exception:
        return None


def main():
    random.seed(42)
    fds = [json.loads(l) for l in open(FC, encoding="utf-8")]
    arts = {json.loads(l)["id"]: json.loads(l) for l in open(AR, encoding="utf-8")}

    n = len(fds)
    print(f"Loaded {n} filtered Forecast Dossiers, {len(arts)} articles in pool")

    if n == 0:
        # Empty FD set (e.g., cutoff drops everything). Write a minimal report
        # so downstream steps don't crash.
        OUT_MD.write_text(
            "# Unified Forecast Dossier - Diagnostic Report\n\n"
            f"Generated: {datetime.now().isoformat(timespec='minutes')}\n\n"
            "**NO ACCEPTED FORECAST DOSSIERS.** The quality filter dropped every "
            "candidate (most likely due to the model_cutoff + buffer_days guard). "
            "Check `forecasts_dropped.jsonl` and `quality_meta.json` for reasons.\n",
            encoding="utf-8",
        )
        OUT_JSON.write_text(json.dumps({
            "n_filtered_fds": 0, "n_articles_pool": len(arts),
            "leakage_violations": 0, "note": "empty FD set after quality filter",
        }, indent=2), encoding="utf-8")
        print(f"Wrote empty-state report to {OUT_MD} and {OUT_JSON}")
        return

    # per-source breakdown
    per_src = defaultdict(list)
    for fd in fds:
        per_src[fd["source"]].append(fd)

    # global stats
    n_arts = [len(fd.get("article_ids", [])) for fd in fds]
    arts_hist = Counter(n_arts)

    # date spread per FD (in days)
    spreads = []
    char_totals = []
    for fd in fds:
        dates = [parse_date(arts[a].get("publish_date", "")) for a in fd["article_ids"] if a in arts]
        dates = sorted(d for d in dates if d)
        if len(dates) >= 2:
            spreads.append((dates[-1] - dates[0]).days)
        char_totals.append(sum(arts[a].get("char_count", 0) for a in fd["article_ids"] if a in arts))

    # ground-truth distribution per benchmark
    gt_by_bench = defaultdict(Counter)
    for fd in fds:
        gt_by_bench[fd["benchmark"]][fd["ground_truth"]] += 1

    # crowd probability distribution (FB only)
    crowd = [fd["crowd_probability"] for fd in fds if fd.get("crowd_probability") is not None]

    # LEAKAGE AUDIT
    leakage_violations = 0
    leakage_examples = []
    for fd in fds:
        t_star = parse_date(fd.get("forecast_point", ""))
        if not t_star:
            continue
        for aid in fd["article_ids"]:
            d = parse_date(arts.get(aid, {}).get("publish_date", ""))
            if d and d >= t_star:
                leakage_violations += 1
                if len(leakage_examples) < 5:
                    leakage_examples.append({"fd": fd["id"], "art": aid,
                                             "art_date": arts[aid]["publish_date"],
                                             "t_star": fd["forecast_point"]})

    # 20 random FD samples
    samples = random.sample(fds, min(20, len(fds)))
    sample_view = [{
        "id": s["id"],
        "source": s["source"],
        "question": s["question"][:80],
        "n_articles": len(s["article_ids"]),
        "crowd_prob": s.get("crowd_probability"),
        "ground_truth": s["ground_truth"],
        "forecast_point": s["forecast_point"],
    } for s in samples]

    # build report
    md = []
    md.append("# Unified Forecast Dossier — Diagnostic Report\n")
    md.append(f"Generated: {datetime.now().isoformat(timespec='minutes')}\n")
    md.append(f"\n## Headline numbers\n")
    md.append(f"- **Filtered FDs**: {n}\n")
    md.append(f"- **Articles in unified pool**: {len(arts)}\n")
    md.append(f"- **Avg articles per FD**: {sum(n_arts)/n:.2f}\n")
    md.append(f"- **Median articles per FD**: {statistics.median(n_arts):.0f}\n")
    md.append(f"- **Median article-date spread per FD**: {statistics.median(spreads) if spreads else 0:.0f} days\n")
    md.append(f"- **Median total chars of evidence per FD**: {statistics.median(char_totals):.0f}\n")
    md.append(f"- **Leakage audit (must be 0)**: {leakage_violations} violations\n")

    md.append(f"\n## Per-source breakdown\n")
    md.append("| Source | N FDs | Date range | Avg articles | Median day-spread | Median chars |\n|---|---|---|---|---|---|\n")
    for src in sorted(per_src):
        lst = per_src[src]
        dates = sorted(d for d in (parse_date(f["resolution_date"]) for f in lst) if d)
        lst_arts = [len(f["article_ids"]) for f in lst]
        lst_spread = []
        lst_chars = []
        for f in lst:
            ds = sorted(d for d in (parse_date(arts[a].get("publish_date", "")) for a in f["article_ids"] if a in arts) if d)
            if len(ds) >= 2:
                lst_spread.append((ds[-1] - ds[0]).days)
            lst_chars.append(sum(arts[a].get("char_count", 0) for a in f["article_ids"] if a in arts))
        md.append(f"| {src} | {len(lst)} | "
                  f"{dates[0].date() if dates else '?'} \u2192 {dates[-1].date() if dates else '?'} | "
                  f"{sum(lst_arts)/len(lst_arts):.1f} | "
                  f"{statistics.median(lst_spread) if lst_spread else 0:.0f} | "
                  f"{statistics.median(lst_chars):.0f} |\n")

    md.append(f"\n## Articles-per-FD histogram\n")
    for k in sorted(arts_hist):
        md.append(f"- {k} articles: {arts_hist[k]} FDs\n")

    md.append(f"\n## Ground-truth class balance\n")
    for bench, c in gt_by_bench.items():
        total = sum(c.values())
        md.append(f"- **{bench}**: {dict(c)}  (Yes-rate = {100*c.get('Yes', 0)/total:.1f}%)\n" if bench == "forecastbench"
                  else f"- **{bench}**: {dict(c)}\n")

    if crowd:
        md.append(f"\n## Crowd-probability distribution (ForecastBench)\n")
        md.append(f"- N = {len(crowd)}, mean = {statistics.mean(crowd):.3f}, median = {statistics.median(crowd):.3f}\n")
        # histogram
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        bin_counts = Counter()
        for p in crowd:
            for i in range(len(bins) - 1):
                if bins[i] <= p < bins[i + 1]:
                    bin_counts[i] += 1
                    break
        for i in range(len(bins) - 1):
            md.append(f"  - [{bins[i]:.1f}, {bins[i+1]:.2f}): {bin_counts[i]}\n")

    md.append(f"\n## Random 20-FD spot-check\n")
    md.append("| ID | Source | n_arts | Crowd | GT | t* | Question |\n|---|---|---|---|---|---|---|\n")
    for s in sample_view:
        md.append(f"| {s['id'][:18]} | {s['source']} | {s['n_articles']} | "
                  f"{s['crowd_prob']} | {s['ground_truth']} | {s['forecast_point']} | "
                  f"{s['question'][:60]} |\n")

    if leakage_examples:
        md.append(f"\n## Leakage examples (these must NOT exist after filter)\n")
        for ex in leakage_examples:
            md.append(f"- FD {ex['fd']}: art {ex['art']} dated {ex['art_date']} >= t* {ex['t_star']}\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")

    js = {
        "n_filtered_fds": n,
        "n_articles_pool": len(arts),
        "leakage_violations": leakage_violations,
        "per_source_counts": {k: len(v) for k, v in per_src.items()},
        "ground_truth": {k: dict(v) for k, v in gt_by_bench.items()},
        "articles_per_fd_histogram": dict(arts_hist),
        "median_day_spread": statistics.median(spreads) if spreads else 0,
        "median_chars": statistics.median(char_totals),
        "crowd_n": len(crowd),
        "crowd_mean": statistics.mean(crowd) if crowd else None,
    }
    OUT_JSON.write_text(json.dumps(js, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(f"\nHeadline: {n} FDs accepted | leakage_violations={leakage_violations} (must be 0)")


if __name__ == "__main__":
    main()
