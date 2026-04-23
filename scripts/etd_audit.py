"""ETD audit: schema validity, distributions, date sanity, duplicate detection.

Runs on `data/etd/facts.v1.jsonl` (or any later stage output) and emits a
single Markdown report covering the failure modes that warrant action:

  * Schema validity   -- required-field coverage and type sanity
  * Field distributions -- extraction_confidence / polarity / kind / language
  * Per-article density -- histogram of facts/article (catches over- and
                           under-extraction outliers)
  * Date sanity       -- fact.time vs article_date skew (catches hallucinated
                         dates and post-publish dates)
  * Exact duplicates  -- same fact text + same primary article (Stage-1
                         shouldn't emit these; flags re-extraction loops)
  * Near-dupe shortlist -- normalized-text duplicates within the same article
                         (Stage-2 dedup is the real solution; this is an
                         early-warning signal at Stage-1 quality)
  * Entity coverage    -- unique entity count, top-50, type breakdown
  * Per-source rate    -- facts-per-article rate by article source domain
                         (catches extractor drift on specific outlets)

CPU-only, no API calls. Idempotent. Safe to run while the build is GPU-busy.

Usage:
  python scripts/etd_audit.py
  python scripts/etd_audit.py --in data/etd/facts.v1_canonical.jsonl
  python scripts/etd_audit.py --strict       # nonzero exit on schema fails
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_IN = DATA / "etd" / "facts.v1.jsonl"
ARTICLES_IN = DATA / "unified" / "articles.jsonl"
OUT_DIR = DATA / "etd" / "audit"

REQUIRED_FIELDS = {
    "id", "schema_version", "time", "fact",
    "article_ids", "primary_article_id", "article_date",
    "entities", "polarity", "extraction_confidence",
    "extractor", "extracted_at",
}
# Mirror docs/etd.schema.json. Update both together if the enum changes.
ENUM_POLARITY = {"asserted", "negated", "hypothetical", "reported"}
ENUM_CONFIDENCE = {"high", "medium", "low"}


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _norm_text(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _bucket(n, bounds):
    for hi in bounds:
        if n < hi:
            return f"<{hi}"
    return f">={bounds[-1]}"


def _atomic_write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--articles", default=str(ARTICLES_IN))
    ap.add_argument("--out", default=str(OUT_DIR / "audit_report.md"))
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if any schema-fail or duplicate is detected.")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[ERROR] {inp} not found")
        return 1

    print(f"[etd_audit] loading {inp}")
    facts: list[dict] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            try:
                facts.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: skip malformed line: {e}")
    n = len(facts)
    print(f"[etd_audit] loaded {n} facts")

    # Article source map for per-source breakdown
    art_source: dict[str, str] = {}
    if Path(args.articles).exists():
        with open(args.articles, encoding="utf-8") as f:
            for line in f:
                try:
                    a = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = a.get("id")
                if aid:
                    art_source[aid] = a.get("source", "(unknown)")

    # Schema validity
    schema_fails = []
    bad_polarity = 0
    bad_conf = 0
    for fact in facts:
        missing = REQUIRED_FIELDS - set(fact.keys())
        if missing:
            schema_fails.append((fact.get("id"), f"missing fields: {sorted(missing)}"))
        if fact.get("polarity") and fact["polarity"] not in ENUM_POLARITY:
            bad_polarity += 1
        if fact.get("extraction_confidence") and fact["extraction_confidence"] not in ENUM_CONFIDENCE:
            bad_conf += 1

    # Field distributions
    conf_dist = Counter(f.get("extraction_confidence") for f in facts)
    pol_dist = Counter(f.get("polarity") for f in facts)
    kind_dist = Counter(f.get("kind") for f in facts)
    lang_dist = Counter(f.get("language") for f in facts)

    # Per-article density
    per_article = Counter(f.get("primary_article_id") for f in facts)
    density_hist = Counter()
    for n_facts in per_article.values():
        density_hist[_bucket(n_facts, [1, 2, 5, 10, 20, 50])] += 1

    # Date sanity
    future_facts = 0
    far_past_facts = 0
    no_date = 0
    for fact in facts:
        ft = _parse_date(fact.get("time"))
        ad = _parse_date(fact.get("article_date"))
        if ft is None or ad is None:
            no_date += 1
            continue
        if ft > ad:
            future_facts += 1
        elif ad - ft > timedelta(days=365):
            far_past_facts += 1

    # Exact + within-article near-duplicates
    seen_pair: dict[tuple, int] = defaultdict(int)
    seen_article_norm: dict[tuple, int] = defaultdict(int)
    for fact in facts:
        pid = fact.get("primary_article_id")
        text = fact.get("fact", "")
        seen_pair[(pid, text)] += 1
        seen_article_norm[(pid, _norm_text(text))] += 1
    exact_dupes = sum(c - 1 for c in seen_pair.values() if c > 1)
    near_dupes = sum(c - 1 for c in seen_article_norm.values() if c > 1) - exact_dupes

    # Entity stats
    ent_names = Counter()
    ent_types = Counter()
    facts_with_ents = 0
    for fact in facts:
        ents = fact.get("entities") or []
        if ents:
            facts_with_ents += 1
        for e in ents:
            if isinstance(e, dict):
                if e.get("name"):
                    ent_names[e["name"]] += 1
                if e.get("type"):
                    ent_types[e["type"]] += 1

    # Per-source extraction rate
    src_facts = Counter()
    src_articles = Counter()
    seen_articles: set[str] = set()
    for fact in facts:
        pid = fact.get("primary_article_id")
        s = art_source.get(pid, "(unknown)")
        src_facts[s] += 1
        if pid not in seen_articles:
            src_articles[s] += 1
            seen_articles.add(pid)

    # Build the report
    lines = []
    P = lines.append
    P(f"# ETD Stage-1 Audit Report")
    P(f"")
    P(f"- Input: `{inp}`")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"- Total facts: **{n}**")
    P(f"- Unique articles producing facts: **{len(per_article)}**")
    P(f"- Avg facts per article: **{n / max(1, len(per_article)):.2f}**")
    P(f"")
    P(f"## Schema validity")
    P(f"- Schema fails: **{len(schema_fails)}** ({100*len(schema_fails)/max(1,n):.1f}%)")
    P(f"- Bad `polarity` values: **{bad_polarity}**")
    P(f"- Bad `extraction_confidence` values: **{bad_conf}**")
    if schema_fails[:5]:
        P(f"- Sample schema fails:")
        for fid, msg in schema_fails[:5]:
            P(f"  - `{fid}`: {msg}")
    P(f"")
    P(f"## Field distributions")
    P(f"- `extraction_confidence`: {dict(conf_dist)}")
    P(f"- `polarity`: {dict(pol_dist)}")
    P(f"- `language` top: {dict(lang_dist.most_common(8))}")
    P(f"- `kind` top: {dict(kind_dist.most_common(15))}")
    P(f"")
    P(f"## Date sanity")
    P(f"- Facts with no parseable date: **{no_date}**")
    P(f"- Facts dated AFTER article publish (impossible): **{future_facts}**")
    P(f"- Facts dated >365d BEFORE article publish (likely hallucinated): **{far_past_facts}**")
    P(f"")
    P(f"## Per-article fact density")
    for bucket, count in sorted(density_hist.items()):
        P(f"- `{bucket}` facts/article: **{count}** articles")
    P(f"")
    P(f"## Duplicates (within Stage-1)")
    P(f"- Exact duplicates (same article + same fact text): **{exact_dupes}**")
    P(f"- Near duplicates (same article + normalized-equal text): **{near_dupes}**")
    P(f"  - These should be 0 after Stage-2 dedup (`scripts/etd_dedup.py`).")
    P(f"")
    P(f"## Entity coverage")
    P(f"- Facts with >=1 entity: **{facts_with_ents}** ({100*facts_with_ents/max(1,n):.1f}%)")
    P(f"- Unique entity names: **{len(ent_names)}**")
    P(f"- Entity type distribution: {dict(ent_types.most_common(10))}")
    P(f"- Top-20 entities by mention count:")
    for name, cnt in ent_names.most_common(20):
        P(f"  - `{name}`: {cnt}")
    P(f"")
    P(f"## Per-source extraction rate (top 20 by article volume)")
    src_stats = []
    for s, ac in src_articles.most_common(20):
        rate = src_facts[s] / max(1, ac)
        src_stats.append((s, ac, src_facts[s], rate))
    P(f"| Source | Articles | Facts | Facts/article |")
    P(f"|---|---:|---:|---:|")
    for s, ac, fc, rate in src_stats:
        P(f"| `{s}` | {ac} | {fc} | {rate:.2f} |")
    P(f"")
    P(f"## Recommended next steps")
    P(f"- If `Date sanity` shows >1% future facts -> Stage-1 prompt needs explicit "
      f"\"fact.time must be on or before publish_date\" reminder.")
    P(f"- If `Duplicates` shows non-trivial near-dupes -> run "
      f"`python scripts/etd_dedup.py` (Stage 2).")
    P(f"- If `Per-article density` shows many 0-fact articles -> run "
      f"`python scripts/etd_debug_empty.py`.")
    P(f"- If `Per-source extraction rate` shows >2x variance -> a specific "
      f"outlet may have parser issues; sample those articles for spot-check.")

    body = "\n".join(lines) + "\n"
    out_path = Path(args.out)
    _atomic_write(out_path, body)
    print(f"[etd_audit] report -> {out_path}")

    # Strict-mode exit code
    fatal = len(schema_fails) > 0 or exact_dupes > 0
    if args.strict and fatal:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
