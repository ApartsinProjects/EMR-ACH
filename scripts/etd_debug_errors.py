"""ETD Stage-1 error triage: categorize and cross-tabulate failed extractions.

Reads `data/etd/facts.errors.jsonl` (the sidecar file emitted by
`scripts/articles_to_facts.py` whenever a Stage-1 extraction is rejected at
write time) and produces a Markdown report covering:

  * Error type breakdown (validation_failed / parse_error / api_error /
    refusal / timeout / token_limit / other).
  * Root-cause shortlist for each error type, by error_detail signature.
  * Article-feature correlations: failure rate by char length bucket,
    language, source domain. This is what tells you "long articles fail",
    "Russian-language articles fail", "wsj.com articles fail" -- the levers
    a prompt revision can target.
  * Sample of the worst-offending articles for direct prompt-review.

CPU-only, no API calls. Joins with `data/unified/articles.jsonl` to get
article features. Idempotent.

Usage:
  python scripts/etd_debug_errors.py
  python scripts/etd_debug_errors.py --in data/etd/facts.errors.jsonl --top 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_IN = DATA / "etd" / "facts.errors.jsonl"
ARTICLES_IN = DATA / "unified" / "articles.jsonl"
OUT_DIR = DATA / "etd" / "audit"

LEN_BUCKETS = [1000, 5000, 20000, 50000]


def _bucket(n, bounds):
    for hi in bounds:
        if n < hi:
            return f"<{hi:>5d}"
    return f">={bounds[-1]:>4d}"


def _atomic_write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _signature(detail: str) -> str:
    """Collapse error_detail to a comparable signature: strip per-fact indices,
    file paths, dates, hashes, numeric IDs. Two errors with the same
    signature have the same root cause."""
    s = detail or ""
    s = re.sub(r"fact\[\d+\]", "fact[N]", s)
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", s)
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<HASH>", s)
    s = re.sub(r"\b\d+\b", "<N>", s)
    return s[:160]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--articles", default=str(ARTICLES_IN))
    ap.add_argument("--out", default=str(OUT_DIR / "error_triage.md"))
    ap.add_argument("--top", type=int, default=15,
                    help="Top-N signatures + offenders to surface.")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[ERROR] {inp} not found")
        return 1

    print(f"[etd_debug_errors] loading {inp}")
    errors: list[dict] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            try:
                errors.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: skip malformed line: {e}")
    n = len(errors)
    print(f"[etd_debug_errors] loaded {n} errors")

    # Article feature lookup
    art_meta: dict[str, dict] = {}
    if Path(args.articles).exists():
        with open(args.articles, encoding="utf-8") as f:
            for line in f:
                try:
                    a = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = a.get("id")
                if aid:
                    art_meta[aid] = {
                        "len": len(a.get("text", "") or ""),
                        "source": a.get("source", "(unknown)"),
                        "language": a.get("language", "und"),
                    }

    # Type breakdown
    type_counts = Counter(e.get("error_type", "(unspecified)") for e in errors)

    # Per-type signature breakdown
    sig_by_type: dict[str, Counter] = defaultdict(Counter)
    for e in errors:
        sig_by_type[e.get("error_type", "(unspecified)")][_signature(e.get("error_detail", ""))] += 1

    # Article-feature correlations
    by_len = Counter()
    by_lang = Counter()
    by_source = Counter()
    by_len_total = Counter()
    by_lang_total = Counter()
    by_source_total = Counter()
    for e in errors:
        m = art_meta.get(e.get("article_id"))
        if not m:
            continue
        by_len[_bucket(m["len"], LEN_BUCKETS)] += 1
        by_lang[m["language"]] += 1
        by_source[m["source"]] += 1
    # Denominators (total articles in those buckets) for rate calc
    for aid, m in art_meta.items():
        by_len_total[_bucket(m["len"], LEN_BUCKETS)] += 1
        by_lang_total[m["language"]] += 1
        by_source_total[m["source"]] += 1

    # Worst offenders by article (one article producing many error rows
    # signals an extractor loop or a repeatedly-broken article)
    worst_articles = Counter(e.get("article_id") for e in errors)

    lines = []
    P = lines.append
    P(f"# ETD Stage-1 Error Triage")
    P(f"")
    P(f"- Input: `{inp}`")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"- Total error rows: **{n}**")
    P(f"")
    P(f"## Error-type breakdown")
    P(f"| error_type | count | share |")
    P(f"|---|---:|---:|")
    for t, c in type_counts.most_common():
        P(f"| `{t}` | {c} | {100*c/max(1,n):.1f}% |")
    P(f"")
    P(f"## Top root-cause signatures per error type (top {args.top})")
    for t, sigs in sig_by_type.items():
        P(f"### `{t}` ({sum(sigs.values())} rows)")
        for sig, c in sigs.most_common(args.top):
            P(f"- {c:>5d}  `{sig}`")
        P(f"")
    P(f"## Failure rate by article length")
    P(f"| char-length bucket | failures | total articles | failure rate |")
    P(f"|---|---:|---:|---:|")
    for bk in [_bucket(b, LEN_BUCKETS) for b in [0, 1500, 8000, 30000, 60000]]:
        f_, t_ = by_len.get(bk, 0), by_len_total.get(bk, 0)
        rate = 100 * f_ / t_ if t_ else 0
        P(f"| `{bk}` | {f_} | {t_} | {rate:.1f}% |")
    P(f"")
    P(f"## Failure rate by language (top 10 by failures)")
    P(f"| language | failures | total | failure rate |")
    P(f"|---|---:|---:|---:|")
    for lang, f_ in by_lang.most_common(10):
        t_ = by_lang_total.get(lang, 0)
        rate = 100 * f_ / t_ if t_ else 0
        P(f"| `{lang}` | {f_} | {t_} | {rate:.1f}% |")
    P(f"")
    P(f"## Failure rate by source domain (top 15 by failures)")
    P(f"| source | failures | total | failure rate |")
    P(f"|---|---:|---:|---:|")
    for src, f_ in by_source.most_common(15):
        t_ = by_source_total.get(src, 0)
        rate = 100 * f_ / t_ if t_ else 0
        P(f"| `{src}` | {f_} | {t_} | {rate:.1f}% |")
    P(f"")
    P(f"## Worst-offending articles (>=3 error rows)")
    P(f"| article_id | n_errors | source | language | char_len |")
    P(f"|---|---:|---|---|---:|")
    for aid, c in worst_articles.most_common(args.top):
        if c < 3:
            break
        m = art_meta.get(aid, {})
        P(f"| `{aid}` | {c} | `{m.get('source','?')}` | `{m.get('language','?')}` | {m.get('len','?')} |")
    P(f"")
    P(f"## Recommended actions")
    P(f"- Fix the top-1 signature per error_type first; it usually accounts "
      f"for >40% of that bucket.")
    P(f"- If failure rate climbs sharply with length, add chunked-extraction "
      f"to the prompt (split articles >20k chars into 2-3 windows).")
    P(f"- If a single language dominates failures, the extractor prompt may "
      f"not be robust to non-English text; add a translate-first pass or "
      f"language-specific instructions.")
    P(f"- A specific source domain dominating failures usually means a "
      f"trafilatura extraction artifact (paywall stub, JSON-LD-only body); "
      f"sample those articles before blaming the LLM.")

    out_path = Path(args.out)
    _atomic_write(out_path, "\n".join(lines) + "\n")
    print(f"[etd_debug_errors] report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
