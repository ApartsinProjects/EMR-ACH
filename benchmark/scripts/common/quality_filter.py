"""
Quality filter over unified Forecast Dossiers (FDs).

A forecast is ACCEPTED iff ALL of the following hold:
  1. >= 3 articles linked (article_ids length)
  2. Article publication dates span at least 2 distinct days
     (avoids all-same-day clusters that indicate stale batch reprocessing)
  3. Sum of char_count across linked articles >= 1500 chars (non-trivial evidence)
  4. Ground truth is NOT observable in any linked article. Leakage guard: drop if any
     article.publish_date >= forecast.forecast_point.
  5. Question length (after whitespace normalization) >= 20 chars

Reads:
  data/unified/forecasts.jsonl
  data/unified/articles.jsonl

Writes:
  data/unified/forecasts_filtered.jsonl       - accepted FDs
  data/unified/forecasts_dropped.jsonl        - rejected FDs with reasons
  data/unified/quality_meta.json              - summary stats

Usage:
  python benchmark/scripts/common/quality_filter.py
  python benchmark/scripts/common/quality_filter.py --min-arts 3 --min-days 2 --min-chars 1500
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
UNI = ROOT / "data" / "unified"
FC_FILE = UNI / "forecasts.jsonl"
ART_FILE = UNI / "articles.jsonl"
OUT_OK = UNI / "forecasts_filtered.jsonl"
OUT_BAD = UNI / "forecasts_dropped.jsonl"
META = UNI / "quality_meta.json"


def parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-arts", type=int, default=3)
    ap.add_argument("--min-days", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--min-q-chars", type=int, default=20)
    ap.add_argument("--model-cutoff", default=None,
                    help="ISO date (YYYY-MM-DD) of the evaluator LLM's training cutoff. "
                         "Any FD with resolution_date <= cutoff is dropped as model-training leakage. "
                         "Example: --model-cutoff 2024-04-01 for GPT-4o. "
                         "Defaults to None (no cutoff check) for backwards compatibility.")
    ap.add_argument("--cutoff-buffer-days", type=int, default=0,
                    help="Require resolution_date > cutoff + N days, to hedge against "
                         "continued-training / RLHF drift past the stated cutoff.")
    args = ap.parse_args()

    cutoff_dt = None
    if args.model_cutoff:
        cutoff_dt = parse_date(args.model_cutoff)
        if not cutoff_dt:
            print(f"[ERROR] --model-cutoff '{args.model_cutoff}' not parseable as YYYY-MM-DD")
            return
        from datetime import timedelta
        cutoff_dt = cutoff_dt + timedelta(days=args.cutoff_buffer_days)
        print(f"Enforcing resolution_date > {cutoff_dt.date()} "
              f"(cutoff + {args.cutoff_buffer_days}d buffer) "
              f"- model-training leakage guard")

    forecasts = [json.loads(l) for l in open(FC_FILE, encoding="utf-8")]
    articles = {json.loads(l)["id"]: json.loads(l) for l in open(ART_FILE, encoding="utf-8")}

    accepted = []
    dropped = []
    drop_reasons = Counter()
    per_src_kept = defaultdict(int)
    per_src_drop = defaultdict(int)

    leakage_pruned = 0
    for fc in forecasts:
        reasons = []
        aids = fc.get("article_ids") or []
        linked = [articles[a] for a in aids if a in articles]

        # PRE-FILTER: drop any article dated on/after forecast_point (leakage guard)
        t_star = parse_date(fc.get("forecast_point", ""))
        if t_star:
            kept_linked = []
            removed = 0
            for a in linked:
                d = parse_date(a.get("publish_date", ""))
                if d is None or d < t_star:
                    kept_linked.append(a)
                else:
                    removed += 1
            if removed:
                leakage_pruned += removed
            linked = kept_linked
            fc["article_ids"] = [a["id"] for a in linked]

        # 1. min articles
        if len(linked) < args.min_arts:
            reasons.append(f"n_articles<{args.min_arts}")

        # 2. date spread
        dates = [parse_date(a.get("publish_date", "")) for a in linked]
        dates = [d for d in dates if d]
        day_set = {d.date() for d in dates}
        if len(day_set) < args.min_days:
            reasons.append(f"day_spread<{args.min_days}")

        # 3. min chars
        total_chars = sum(a.get("char_count", 0) for a in linked)
        if total_chars < args.min_chars:
            reasons.append(f"char_count<{args.min_chars}")

        # 4. question length
        qlen = len(" ".join((fc.get("question", "") or "").split()))
        if qlen < args.min_q_chars:
            reasons.append(f"question_len<{args.min_q_chars}")

        # 5. Model-training leakage guard.
        # The answer becomes publicly observable in news at ~resolution_date.
        # If the LLM's training cutoff is before that, the model cannot have
        # trained on the outcome. We require: resolution_date > cutoff + buffer.
        # (Article-level retrieval leakage is handled separately by the
        #  publish_date < forecast_point pre-prune in step 0 above.)
        res_date = parse_date(fc.get("resolution_date", ""))
        if cutoff_dt and res_date and res_date <= cutoff_dt:
            reasons.append(f"resolution_date<=cutoff({args.model_cutoff}+{args.cutoff_buffer_days}d)")

        src = fc.get("source", "?")
        if reasons:
            dropped.append({**fc, "_drop_reasons": reasons})
            drop_reasons[tuple(reasons)] += 1
            per_src_drop[src] += 1
        else:
            accepted.append(fc)
            per_src_kept[src] += 1

    # SHIPPED-DATASET SCHEMA: strip pipeline-internal fields so forecasts_filtered.jsonl
    # is self-contained for an evaluator. Fields not in this whitelist are removed
    # from each accepted FD before writing. `metadata` is dropped entirely.
    shipped_fields = {
        "id", "benchmark", "source",
        "hypothesis_set", "hypothesis_definitions",
        "question", "background",
        "forecast_point", "resolution_date",
        "ground_truth", "ground_truth_idx",
        "crowd_probability", "lookback_days",
        "article_ids",
    }
    with open(OUT_OK, "w", encoding="utf-8") as f:
        for r in accepted:
            clean = {k: v for k, v in r.items() if k in shipped_fields}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    with open(OUT_BAD, "w", encoding="utf-8") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "total":    len(forecasts),
        "accepted": len(accepted),
        "dropped":  len(dropped),
        "leakage_articles_pruned": leakage_pruned,
        "per_source_accepted": dict(per_src_kept),
        "per_source_dropped":  dict(per_src_drop),
        "top_drop_patterns": {"+".join(k): v for k, v in drop_reasons.most_common(10)},
        "thresholds": {
            "min_articles":      args.min_arts,
            "min_distinct_days": args.min_days,
            "min_total_chars":   args.min_chars,
            "min_question_chars": args.min_q_chars,
        },
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Total:    {len(forecasts)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Dropped:  {len(dropped)}")
    print(f"Leakage articles pruned (kept FD): {leakage_pruned}")
    print(f"\nPer-source kept: {dict(per_src_kept)}")
    print(f"Per-source drop: {dict(per_src_drop)}")
    print(f"\nTop drop patterns:")
    for pat, n in drop_reasons.most_common(10):
        print(f"  {n:>4}  {'+'.join(pat)}")


if __name__ == "__main__":
    main()
