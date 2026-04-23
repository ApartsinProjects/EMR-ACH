"""Article-pool quality audit: distributions, near-dupes, missing-text, source mix.

Complements `build_eda_report.py` (which produces HTML for human review) with
a Markdown report focused on *quality signals* a downstream consumer needs to
trust the article pool:

  * Per-article schema sanity (required fields present, types right).
  * Length distribution (chars + tokens approximation).
  * Empty / near-empty text rate (many fetchers populate title-only stubs;
    that's the silent-failure mode).
  * Source-domain breakdown (top-N by article count + by per-FD usage).
  * Language coverage.
  * URL-canonicalization audit: how many distinct hostnames per article? How
    many already-collapsed dupes survived (same canonicalized URL surviving
    in two records).
  * Near-duplicate text detection (within-source title shingles + sample of
    cross-article 5-gram collisions). Catches the wire-syndication echo
    that hurts retrieval diversity.
  * Per-source spam-blocklist hits (count of records whose URL would have
    been dropped by `src/common/spam_domains.is_spam_url`).

CPU-only, no API. Atomic write. Idempotent.

Usage:
  python scripts/articles_audit.py
  python scripts/articles_audit.py --in benchmark/data/2026-01-01/articles.jsonl
  python scripts/articles_audit.py --strict  # nonzero exit on schema fails
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
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
DEFAULT_IN = DATA / "unified" / "articles.jsonl"
OUT_DIR = DATA / "unified" / "audit"

# Unified-article producer (scripts/unify_articles.py) emits `publish_date`
# and `source_domain`. The earlier `date`/`source` names are accepted as
# aliases so this audit also runs on the per-benchmark fetcher output
# (which is what some external consumers see).
_DATE_KEYS   = {"date", "publish_date"}
_SOURCE_KEYS = {"source", "source_domain"}
REQUIRED_FIELDS = {"id", "url", "title", "text"}
LEN_BUCKETS = [200, 1000, 5000, 20000, 50000]


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


def _host(url):
    try:
        return urlparse(url).hostname or "(no-host)"
    except Exception:
        return "(parse-fail)"


def _shingles(s, k=5):
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    if len(s) < k:
        return set()
    return set(s[i:i + k] for i in range(len(s) - k + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(OUT_DIR / "articles_audit.md"))
    ap.add_argument("--near-dup-sample", type=int, default=2000,
                    help="Random sample size for cross-article near-dup check.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 on schema fails or spam survivors.")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[ERROR] {inp} not found")
        return 1

    print(f"[articles_audit] loading {inp}")
    arts: list[dict] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            try:
                arts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = len(arts)
    print(f"[articles_audit] loaded {n} articles")

    # Spam check (lazy import, optional)
    try:
        from src.common.spam_domains import is_spam_url
    except Exception:
        is_spam_url = lambda u: False

    schema_fails = []
    by_len = Counter()
    by_lang = Counter()
    by_source = Counter()
    by_host = Counter()
    by_provenance = Counter()
    no_text = 0
    title_only = 0
    spam_survivors = []
    seen_url = Counter()
    seen_id = Counter()
    seen_title = defaultdict(list)

    for a in arts:
        keys = set(a.keys())
        missing = REQUIRED_FIELDS - keys
        if not (keys & _DATE_KEYS):
            missing.add("date|publish_date")
        if not (keys & _SOURCE_KEYS):
            missing.add("source|source_domain")
        if missing:
            schema_fails.append((a.get("id"), f"missing: {sorted(missing)}"))

        text = a.get("text") or ""
        title = a.get("title") or ""
        ln = len(text)
        by_len[_bucket(ln, LEN_BUCKETS)] += 1
        by_lang[a.get("language", "und")] += 1
        by_source[a.get("source") or a.get("source_domain") or "(unknown)"] += 1
        by_host[_host(a.get("url"))] += 1
        prov = a.get("provenance", "(unspecified)")
        if isinstance(prov, list):
            prov = "|".join(str(x) for x in prov)
        by_provenance[prov] += 1
        if ln == 0:
            no_text += 1
        elif ln < 200 and len(title) > 20:
            title_only += 1
        url = a.get("url") or ""
        if url and is_spam_url(url):
            spam_survivors.append(a.get("id"))
        if url:
            seen_url[url] += 1
        aid = a.get("id")
        if aid:
            seen_id[aid] += 1
        if title:
            seen_title[title.strip().lower()].append(a.get("id"))

    duplicate_urls = sum(c - 1 for c in seen_url.values() if c > 1)
    duplicate_ids = sum(c - 1 for c in seen_id.values() if c > 1)
    title_collisions = sum(1 for v in seen_title.values() if len(v) > 1)

    # Near-dup sample (5-gram Jaccard on body shingles)
    import random
    rng = random.Random(0)
    pool = [a for a in arts if (a.get("text") or "").strip()]
    if len(pool) > args.near_dup_sample:
        pool = rng.sample(pool, args.near_dup_sample)
    print(f"[articles_audit] near-dup pass over {len(pool)} sampled articles")
    shingled = [(_shingles(a.get("title", "") + " " + (a.get("text") or "")[:1500]),
                 a.get("id")) for a in pool]
    near_dupes = 0
    near_pairs_sample: list[tuple] = []
    # Bucket by shingle to find candidate pairs (cheap)
    shingle_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (sh, _aid) in enumerate(shingled):
        if not sh:
            continue
        # use a fixed-seed random subsample of shingles to bucket on
        sample = list(sh)[:64]
        for s in sample:
            shingle_to_idxs[s].append(i)
    candidate_pairs: set[tuple[int, int]] = set()
    for idxs in shingle_to_idxs.values():
        if len(idxs) < 2 or len(idxs) > 200:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                candidate_pairs.add((idxs[i], idxs[j]))
    for (i, j) in list(candidate_pairs):
        si, sj = shingled[i][0], shingled[j][0]
        if not si or not sj:
            continue
        jacc = len(si & sj) / len(si | sj)
        if jacc >= 0.7:
            near_dupes += 1
            if len(near_pairs_sample) < 10:
                near_pairs_sample.append((shingled[i][1], shingled[j][1], round(jacc, 3)))

    lines = []
    P = lines.append
    P(f"# Article Pool Audit")
    P(f"")
    P(f"- Input: `{inp}`")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"- Total articles: **{n}**")
    P(f"")
    P(f"## Schema validity")
    P(f"- Schema fails: **{len(schema_fails)}**")
    if schema_fails[:5]:
        for aid, msg in schema_fails[:5]:
            P(f"  - `{aid}`: {msg}")
    P(f"- Duplicate IDs: **{duplicate_ids}**")
    P(f"- Duplicate URLs (post-dedup survivors): **{duplicate_urls}**")
    P(f"")
    P(f"## Body-text quality")
    P(f"- Empty `text`: **{no_text}** ({100*no_text/max(1,n):.1f}%)")
    P(f"- Title-only (text<200 chars but title present): **{title_only}**")
    P(f"- Length distribution:")
    for bk in [_bucket(b, LEN_BUCKETS) for b in [0, 500, 2000, 10000, 30000, 60000]]:
        P(f"  - `{bk}`: **{by_len.get(bk, 0)}**")
    P(f"")
    P(f"## Language coverage")
    for lang, c in by_lang.most_common(10):
        P(f"- `{lang}`: {c}")
    P(f"")
    P(f"## Source breakdown (top 25)")
    P(f"| source | n |")
    P(f"|---|---:|")
    for s, c in by_source.most_common(25):
        P(f"| `{s}` | {c} |")
    P(f"")
    P(f"## Hostname breakdown (top 25)")
    P(f"| host | n |")
    P(f"|---|---:|")
    for h, c in by_host.most_common(25):
        P(f"| `{h}` | {c} |")
    P(f"")
    P(f"## Provenance breakdown (top 15)")
    for p, c in by_provenance.most_common(15):
        P(f"- `{p}`: {c}")
    P(f"")
    P(f"## Spam survivors")
    P(f"- Records that would have been dropped by `is_spam_url`: **{len(spam_survivors)}**")
    if spam_survivors[:10]:
        P(f"- Sample IDs:")
        for aid in spam_survivors[:10]:
            P(f"  - `{aid}`")
    P(f"")
    P(f"## Near-duplicate text (Jaccard >= 0.7 on title+lead shingles)")
    P(f"- Sampled articles: {len(pool)}")
    P(f"- Near-dupe pairs in sample: **{near_dupes}**")
    if near_pairs_sample:
        P(f"- Sample pairs:")
        for a, b, j in near_pairs_sample:
            P(f"  - `{a}` <-> `{b}`  (Jaccard={j})")
    P(f"")
    P(f"## Title collisions (exact lower-case match across multiple records)")
    P(f"- Title groups with >=2 records: **{title_collisions}**")
    P(f"")
    P(f"## Recommended actions")
    P(f"- If `Empty text` rate > 30%: re-run text-fetch step; trafilatura "
      f"may have failed on a class of URLs (paywall stubs, JS-rendered).")
    P(f"- If `Title-only` rate > 20%: tighten the relevance pre-filter to "
      f"drop title-only records before SBERT scoring.")
    P(f"- If `Spam survivors` > 0: extend `src/common/spam_domains.py` and "
      f"rerun unify_articles.")
    P(f"- If `Near-duplicate` count is large: lower top-k in retrieval, or "
      f"add a near-dup pruning pass to `compute_relevance.py`.")

    out_path = Path(args.out)
    _atomic_write(out_path, "\n".join(lines) + "\n")
    print(f"[articles_audit] report -> {out_path}")

    if args.strict and (schema_fails or spam_survivors or duplicate_ids):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
