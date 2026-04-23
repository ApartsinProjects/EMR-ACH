"""ETD missing-extraction audit: articles that produced ZERO facts.

This catches a failure mode the errors file does not record: when the
Stage-1 LLM returned a syntactically valid empty list (e.g. `{"facts":[]}`).
Those articles silently disappear from the ETD layer with no error trail,
yet the article had real content the model should have extracted.

Reads `data/unified/articles.jsonl` and `data/etd/facts.v1.jsonl`, takes
the set difference on `primary_article_id`, then characterizes the
silently-empty bucket by source / language / length.

CPU-only. Idempotent. Fast (~5 sec on 200k articles, 70k facts).

Outputs:
  data/etd/audit/empty_extractions.md      summary + recommendations
  data/etd/audit/empty_sample.jsonl        100-article eyeball sample with
                                            url + first 800 chars of text

Usage:
  python scripts/etd_debug_empty.py
  python scripts/etd_debug_empty.py --sample 50 --min-chars 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_FACTS = DATA / "etd" / "facts.v1.jsonl"
ARTICLES_IN = DATA / "unified" / "articles.jsonl"
OUT_DIR = DATA / "etd" / "audit"

LEN_BUCKETS = [200, 1000, 5000, 20000, 50000]


def _bucket(n, bounds):
    for hi in bounds:
        if n < hi:
            return f"<{hi:>5d}"
    return f">={bounds[-1]:>4d}"


def _atomic_write_text(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(DEFAULT_FACTS))
    ap.add_argument("--articles", default=str(ARTICLES_IN))
    ap.add_argument("--out", default=str(OUT_DIR / "empty_extractions.md"))
    ap.add_argument("--sample", type=int, default=100,
                    help="N silently-empty articles to dump for eyeball.")
    ap.add_argument("--sample-out", default=str(OUT_DIR / "empty_sample.jsonl"))
    ap.add_argument("--min-chars", type=int, default=300,
                    help="Only count articles with >= this many chars in the "
                         "silently-empty bucket. Below this, an empty fact "
                         "list is plausibly correct (paywall stubs, etc.).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    facts_path = Path(args.facts)
    arts_path = Path(args.articles)
    if not facts_path.exists() or not arts_path.exists():
        print(f"[ERROR] missing inputs: {facts_path} or {arts_path}")
        return 1

    print(f"[etd_debug_empty] loading facts {facts_path}")
    extracted_aids: set[str] = set()
    with open(facts_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                pid = d.get("primary_article_id")
                if pid:
                    extracted_aids.add(pid)
            except json.JSONDecodeError:
                continue
    print(f"[etd_debug_empty] {len(extracted_aids)} articles produced >=1 fact")

    print(f"[etd_debug_empty] loading articles {arts_path}")
    all_articles: list[dict] = []
    with open(arts_path, encoding="utf-8") as f:
        for line in f:
            try:
                all_articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[etd_debug_empty] {len(all_articles)} articles total")

    empty = [a for a in all_articles
             if a.get("id") not in extracted_aids
             and len(a.get("text", "") or "") >= args.min_chars]
    silently_empty_n = len(empty)

    by_len = Counter()
    by_lang = Counter()
    by_src = Counter()
    by_len_total = Counter()
    by_lang_total = Counter()
    by_src_total = Counter()
    for a in empty:
        ln = len(a.get("text", "") or "")
        by_len[_bucket(ln, LEN_BUCKETS)] += 1
        by_lang[a.get("language", "und")] += 1
        by_src[a.get("source", "(unknown)")] += 1
    for a in all_articles:
        ln = len(a.get("text", "") or "")
        if ln < args.min_chars:
            continue
        by_len_total[_bucket(ln, LEN_BUCKETS)] += 1
        by_lang_total[a.get("language", "und")] += 1
        by_src_total[a.get("source", "(unknown)")] += 1

    # Eyeball sample
    rng = random.Random(args.seed)
    sample = rng.sample(empty, min(args.sample, len(empty)))
    sample_dump = [{
        "id": a.get("id"),
        "url": a.get("url"),
        "title": a.get("title"),
        "source": a.get("source"),
        "language": a.get("language"),
        "date": a.get("date"),
        "text_preview": (a.get("text", "") or "")[:800],
        "char_len": len(a.get("text", "") or ""),
    } for a in sample]
    _atomic_write_jsonl(Path(args.sample_out), sample_dump)

    lines = []
    P = lines.append
    P(f"# ETD Stage-1 Missing-Extraction Audit")
    P(f"")
    P(f"- Articles file: `{arts_path}`")
    P(f"- Facts file:    `{facts_path}`")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"- Articles with >= {args.min_chars} chars: **{sum(by_len_total.values())}**")
    P(f"- Articles producing >=1 fact: **{len(extracted_aids)}**")
    P(f"- Silently-empty articles (>= {args.min_chars} chars + zero facts): "
      f"**{silently_empty_n}** "
      f"({100*silently_empty_n/max(1,sum(by_len_total.values())):.1f}% of in-scope)")
    P(f"- Eyeball sample: `{args.sample_out}` ({len(sample)} articles)")
    P(f"")
    P(f"## Empty rate by article length")
    P(f"| char-length | empty | total | empty rate |")
    P(f"|---|---:|---:|---:|")
    for bk in [_bucket(b, LEN_BUCKETS) for b in [0, 500, 2000, 10000, 30000, 60000]]:
        e_, t_ = by_len.get(bk, 0), by_len_total.get(bk, 0)
        rate = 100 * e_ / t_ if t_ else 0
        P(f"| `{bk}` | {e_} | {t_} | {rate:.1f}% |")
    P(f"")
    P(f"## Empty rate by language (top 10 by empty count)")
    P(f"| language | empty | total | empty rate |")
    P(f"|---|---:|---:|---:|")
    for lang, e_ in by_lang.most_common(10):
        t_ = by_lang_total.get(lang, 0)
        rate = 100 * e_ / t_ if t_ else 0
        P(f"| `{lang}` | {e_} | {t_} | {rate:.1f}% |")
    P(f"")
    P(f"## Empty rate by source (top 20 by empty count)")
    P(f"| source | empty | total | empty rate |")
    P(f"|---|---:|---:|---:|")
    for src, e_ in by_src.most_common(20):
        t_ = by_src_total.get(src, 0)
        rate = 100 * e_ / t_ if t_ else 0
        P(f"| `{src}` | {e_} | {t_} | {rate:.1f}% |")
    P(f"")
    P(f"## Recommended actions")
    P(f"- If empty rate is uniform across lengths: extractor is sometimes "
      f"refusing or returning [] on valid input; add a one-shot example to "
      f"the prompt and re-extract the silently-empty bucket.")
    P(f"- If empty rate is high only on short articles: those are usually "
      f"paywall stubs or 404 pages; raise `--min-chars` or refilter at "
      f"text-fetch time.")
    P(f"- If a specific source dominates: trafilatura body extraction may be "
      f"failing on that domain; spot-check `text` field manually.")
    P(f"- After the source/length pattern is understood, re-run `articles_to_"
      f"facts.py` with `--retry-only data/etd/audit/empty_sample.jsonl`.")

    _atomic_write_text(Path(args.out), "\n".join(lines) + "\n")
    print(f"[etd_debug_empty] report -> {args.out}")
    print(f"[etd_debug_empty] sample -> {args.sample_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
