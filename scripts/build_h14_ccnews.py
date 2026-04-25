"""v2.2 strategy 6.1B: rebuild h14 with CC-News pool + v2.1 carryover.

Reads:
  benchmark/data/2026-01-01-h14/forecasts.jsonl    (drives FD set: 89 FDs;
                                                    H6 forecast_point already applied)
  benchmark/data/2026-01-01/articles.jsonl         (v2.1 carry-over pool, READ-ONLY)
  data/cc_news/index_2025-12.sqlite                (FTS5 + metadata index from B6.1B-1)

Produces (atomic):
  benchmark/data/2026-01-01-h14-ccnews/forecasts.jsonl
  benchmark/data/2026-01-01-h14-ccnews/articles.jsonl
  benchmark/data/2026-01-01-h14-ccnews/benchmark.yaml
  benchmark/data/2026-01-01-h14-ccnews/build_manifest.json
  benchmark/data/2026-01-01-h14-ccnews/meta/dangling_article_ids.txt
  benchmark/data/2026-01-01-h14-ccnews/meta/source_distribution.json

Per-FD article-pool selection
-----------------------------
For each FD (forecastbench or earnings) we union three candidate pools:

  1. CC-News retrieval over the [fp - lookback_days, fp] window via the
     SQLite index built by B6.1B-1.
       * forecastbench: FTS5 MATCH on question keywords (top-K by recency).
       * earnings: FTS5 MATCH on ticker symbol AND/OR company name; if the
         FD has a known finance host (fool.com is the only finance domain
         in the 2025-12 CC-News slice) those rows are boosted.

  2. v2.1 carry-over: the FD's existing article_ids whose articles are in
     the v2.1 publish AND pass publish_date <= forecast_point.

  3. Dedup by URL hash. CC-News provenance wins ties.

We cap at --top-k (default 30). Articles are written to articles.jsonl
in the unified schema; CC-News rows are tagged provenance=["cc-news",
"v2.2-h14-ccnews"]; carry-over rows keep their v2.1 provenance plus
"v2.1-carryover".

Leakage guard
-------------
Every emitted (fd, article_id) pair satisfies article.publish_date <=
fd.forecast_point. The script aborts with non-zero exit if any pair fails
the assertion.

Usage
-----
    /c/Python314/python scripts/build_h14_ccnews.py
    /c/Python314/python scripts/build_h14_ccnews.py --top-k 30
    /c/Python314/python scripts/build_h14_ccnews.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
H14_DIR = REPO_ROOT / "benchmark" / "data" / "2026-01-01-h14"
V21_DIR = REPO_ROOT / "benchmark" / "data" / "2026-01-01"
OUT_DIR = REPO_ROOT / "benchmark" / "data" / "2026-01-01-h14-ccnews"
DEFAULT_INDEX = REPO_ROOT / "data" / "cc_news" / "index_2025-12.sqlite"

HORIZON_DAYS = 14
LOOKBACK_DAYS = 30
DEFAULT_TOP_K = 30
BENCHMARKS_IN_SCOPE = {"forecastbench", "earnings"}

# FTS5 query helpers — strip punctuation, lowercase, split on whitespace.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "from",
    "by", "with", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "should", "could", "can", "may", "might", "shall",
    "do", "does", "did", "have", "has", "had", "this", "that", "these",
    "those", "as", "at", "it", "its", "they", "their", "them", "we", "us",
    "our", "you", "your", "than", "then", "but", "if", "so", "not", "no",
    "yes", "all", "any", "some", "more", "most", "other", "such",
    "what", "when", "where", "which", "who", "whom", "whose", "why",
    "how", "about",
}


def _parse_date(value):
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    s = str(value)
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _art_id(url: str) -> str:
    return "art_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except (AttributeError, ValueError):
        return ""


def _fts_query_from_question(question: str, max_terms: int = 10) -> str:
    """Build an FTS5 OR query from question keywords.

    FTS5 syntax: terms separated by space => AND. We want OR so
    semantically related articles surface. Form: "term1 OR term2 OR ...".
    Quotes around each term protect against operator chars.
    """
    if not question:
        return ""
    toks = _TOKEN_RE.findall(question.lower())
    seen = []
    for t in toks:
        if t in _STOPWORDS:
            continue
        if t in seen:
            continue
        seen.append(t)
        if len(seen) >= max_terms:
            break
    if not seen:
        return ""
    return " OR ".join(f'"{t}"' for t in seen)


def _fts_query_from_earnings(meta: dict) -> tuple[str, str]:
    """Return (primary_query, ticker) for an earnings FD.

    Primary query OR-joins ticker symbol + significant company-name tokens.
    """
    ticker = (meta.get("ticker") or "").strip().upper()
    company = (meta.get("company") or "").strip()
    company_toks = []
    for t in _TOKEN_RE.findall(company):
        tl = t.lower()
        if tl in _STOPWORDS or tl in {"inc", "corp", "corporation", "company", "co", "ltd", "plc", "the"}:
            continue
        company_toks.append(t)
        if len(company_toks) >= 4:
            break
    parts = []
    if ticker:
        parts.append(f'"{ticker}"')
    parts.extend(f'"{t}"' for t in company_toks)
    if not parts:
        return "", ticker
    return " OR ".join(parts), ticker


def _query_cc_news(con: sqlite3.Connection, fts: str, lo_iso: str, hi_iso: str,
                   top_k: int, host_boost: str | None = None) -> list[dict]:
    """Run an FTS5 MATCH with a date prefilter; return up to top_k rows."""
    if not fts:
        return []
    # bm25() returns a relevance score (lower = better). We re-sort by
    # publish_date desc within the top-K*3 to bias toward recency.
    sql = """
        SELECT a.url, a.host, a.publish_date, a.publish_ts, a.title, a.text,
               a.shard, bm25(articles_fts) AS rank
        FROM articles_fts
        JOIN articles a ON a.rowid = articles_fts.rowid
        WHERE articles_fts MATCH ?
          AND a.publish_date >= ?
          AND a.publish_date <= ?
        ORDER BY rank ASC
        LIMIT ?
    """
    try:
        rows = con.execute(sql, (fts, lo_iso, hi_iso, top_k * 3)).fetchall()
    except sqlite3.OperationalError as exc:
        # FTS5 syntax error (e.g. malformed token) — log and return empty.
        print(f"[cc-news-unify]   WARN FTS5 query failed for '{fts[:80]}...': {exc}")
        return []
    out = []
    for url, host, pdate, pts, title, text, shard, rank in rows:
        out.append(
            {
                "url": url, "host": host, "publish_date": pdate, "publish_ts": pts,
                "title": title or "", "text": text or "",
                "shard": shard, "rank": rank,
            }
        )
    if host_boost:
        # Stable sort: boosted hosts first, then by rank then recency.
        out.sort(key=lambda r: (r["host"] != host_boost, r["rank"], -_date_key(r["publish_date"])))
    else:
        out.sort(key=lambda r: (r["rank"], -_date_key(r["publish_date"])))
    return out[:top_k]


def _date_key(s: str) -> int:
    # YYYY-MM-DD lex sort works; convert to int for tiebreak in sort.
    if not s or len(s) < 10:
        return 0
    try:
        return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])
    except ValueError:
        return 0


def _ccnews_to_article(r: dict) -> dict:
    """Convert a CC-News row to a unified article record."""
    title = r.get("title") or ""
    text = r.get("text") or ""
    title_text = (title + "\n" + text).strip()
    pdate = r.get("publish_date") or ""
    return {
        "id": _art_id(r["url"]),
        "url": r["url"],
        "title": title,
        "text": text,
        "title_text": title_text,
        "publish_date": pdate,
        "source_domain": (r.get("host") or _domain_of(r["url"])).lower(),
        "gdelt_themes": [],
        "gdelt_tone": 0.0,
        "actors": [],
        "cameo_code": "",
        "char_count": len(title_text),
        "provenance": ["cc-news", "v2.2-h14-ccnews"],
    }


def _carryover_article(a: dict) -> dict:
    """Tag a v2.1 article as carry-over and re-emit (no schema mutation)."""
    out = dict(a)
    prov = list(out.get("provenance") or [])
    if "v2.1-carryover" not in prov:
        prov.append("v2.1-carryover")
    out["provenance"] = prov
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(DEFAULT_INDEX), help="CC-News SQLite index")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only; do not write outputs")
    args = ap.parse_args()

    fd_in = H14_DIR / "forecasts.jsonl"
    art_v21 = V21_DIR / "articles.jsonl"
    if not fd_in.exists() or not art_v21.exists():
        print(f"[fatal] missing inputs: {fd_in}, {art_v21}", file=sys.stderr)
        return 2
    if not Path(args.index).exists():
        print(f"[fatal] missing CC-News SQLite index: {args.index}", file=sys.stderr)
        return 2

    # Load v2.1 article pool keyed by id.
    v21_articles: dict[str, dict] = {}
    with art_v21.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            aid = a.get("id")
            if aid:
                v21_articles[aid] = a
    print(f"[load] v2.1 articles: {len(v21_articles):,}")

    con = sqlite3.connect(args.index)
    n_idx = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"[load] CC-News index rows: {n_idx:,}")

    # ---------- Process FDs ----------
    counts_in: Counter = Counter()
    counts_out: Counter = Counter()
    drop_reasons: Counter = Counter()
    referenced_aids: set[str] = set()
    dangling_aids: set[str] = set()
    leakage_violations = 0
    horizon_violations = 0

    src_dist: Counter = Counter()
    new_articles: dict[str, dict] = {}     # id -> article (CC-News + carryover)
    per_fd_counts: list[tuple[str, str, int, int, int]] = []  # (fd_id, bench, ccnews, carry, total)

    kept_fds: list[dict] = []

    with fd_in.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fd = json.loads(line)
            bench = fd.get("benchmark", "?")
            counts_in[bench] += 1

            if bench not in BENCHMARKS_IN_SCOPE:
                drop_reasons[f"out_of_scope:{bench}"] += 1
                continue

            fp = _parse_date(fd.get("forecast_point"))
            rd = _parse_date(fd.get("resolution_date"))
            if fp is None or rd is None:
                drop_reasons["bad_dates"] += 1
                continue

            lo = fp - timedelta(days=args.lookback_days)
            lo_iso, hi_iso = lo.isoformat(), fp.isoformat()

            # ---- Build CC-News candidate pool ----
            ccnews_rows: list[dict] = []
            if bench == "forecastbench":
                fts = _fts_query_from_question(fd.get("question") or "")
                ccnews_rows = _query_cc_news(con, fts, lo_iso, hi_iso, args.top_k)
            elif bench == "earnings":
                meta = fd.get("_earnings_meta") or fd.get("metadata") or {}
                fts, ticker = _fts_query_from_earnings(meta)
                ccnews_rows = _query_cc_news(
                    con, fts, lo_iso, hi_iso, args.top_k, host_boost="www.fool.com"
                )

            # ---- Build v2.1 carryover pool ----
            orig_ids = list(fd.get("article_ids") or [])
            carry_rows: list[dict] = []
            for aid in orig_ids:
                a = v21_articles.get(aid)
                if a is None:
                    dangling_aids.add(aid)
                    continue
                pd = _parse_date(a.get("publish_date"))
                if pd is None or pd > fp:
                    continue
                carry_rows.append(a)

            # ---- Merge: CC-News first (priority), then carryover; dedup by id ----
            picked_ids: list[str] = []
            seen: set[str] = set()
            n_cc = 0
            n_carry = 0
            for r in ccnews_rows:
                pdate = _parse_date(r.get("publish_date"))
                if pdate is None or pdate > fp:
                    continue
                aid = _art_id(r["url"])
                if aid in seen:
                    continue
                seen.add(aid)
                picked_ids.append(aid)
                if aid not in new_articles:
                    new_articles[aid] = _ccnews_to_article(r)
                src_dist[(r.get("host") or "?").lower()] += 1
                n_cc += 1
                if len(picked_ids) >= args.top_k:
                    break

            for a in carry_rows:
                if len(picked_ids) >= args.top_k:
                    break
                aid = a.get("id")
                if not aid or aid in seen:
                    continue
                seen.add(aid)
                picked_ids.append(aid)
                if aid not in new_articles:
                    new_articles[aid] = _carryover_article(a)
                src_dist[(a.get("source_domain") or "?").lower()] += 1
                n_carry += 1

            if not picked_ids:
                drop_reasons[f"empty_pool:{bench}"] += 1
                continue

            fd["article_ids"] = picked_ids
            fd["default_horizon_days"] = HORIZON_DAYS
            fd["lookback_days"] = args.lookback_days

            # Invariants.
            if (rd - fp).days != HORIZON_DAYS:
                horizon_violations += 1
            for aid in picked_ids:
                a = new_articles.get(aid)
                if a is None:
                    leakage_violations += 1
                    continue
                pd = _parse_date(a.get("publish_date"))
                if pd is None or pd > fp:
                    leakage_violations += 1

            referenced_aids.update(picked_ids)
            counts_out[bench] += 1
            kept_fds.append(fd)
            per_fd_counts.append((fd.get("id"), bench, n_cc, n_carry, len(picked_ids)))

    con.close()

    # ---------- Emit summary ----------
    print("")
    print("=== v2.2 h14-ccnews build ===")
    print(f"FD in (h14):  {sum(counts_in.values())}  breakdown={dict(counts_in)}")
    print(f"FD out:       {len(kept_fds)}  breakdown={dict(counts_out)}")
    print(f"Articles out: {len(referenced_aids)}")
    print(f"Drop reasons: {dict(drop_reasons)}")
    print(f"Leakage violations: {leakage_violations}")
    print(f"Horizon violations: {horizon_violations}")

    if per_fd_counts:
        cc_per_fd = [n_cc for _, _, n_cc, _, _ in per_fd_counts]
        carry_per_fd = [n_carry for _, _, _, n_carry, _ in per_fd_counts]
        print(f"per-FD CC-News articles: mean={sum(cc_per_fd)/len(cc_per_fd):.1f}  "
              f"median={sorted(cc_per_fd)[len(cc_per_fd)//2]}  "
              f"max={max(cc_per_fd)}  min={min(cc_per_fd)}")
        print(f"per-FD carryover articles: mean={sum(carry_per_fd)/len(carry_per_fd):.1f}  "
              f"median={sorted(carry_per_fd)[len(carry_per_fd)//2]}  "
              f"max={max(carry_per_fd)}  min={min(carry_per_fd)}")

    print("\nTop-20 source domains in selected pool:")
    for host, c in src_dist.most_common(20):
        print(f"  {c:5d}  {host}")

    if leakage_violations or horizon_violations:
        print("[FAIL] invariant violations detected")
        return 3

    if args.dry_run:
        print("[dry-run] no outputs written")
        return 0

    # ---------- Write outputs ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

    fd_text = "\n".join(json.dumps(fd, ensure_ascii=False) for fd in kept_fds) + ("\n" if kept_fds else "")
    _atomic_write(OUT_DIR / "forecasts.jsonl", fd_text)

    art_lines: list[str] = []
    for aid in sorted(referenced_aids):
        a = new_articles.get(aid)
        if a is None:
            continue
        art_lines.append(json.dumps(a, ensure_ascii=False))
    _atomic_write(OUT_DIR / "articles.jsonl", "\n".join(art_lines) + ("\n" if art_lines else ""))

    dangling_text = "\n".join(sorted(dangling_aids))
    if dangling_text:
        dangling_text += "\n"
    _atomic_write(OUT_DIR / "meta" / "dangling_article_ids.txt", dangling_text)

    src_dist_json = {h: c for h, c in src_dist.most_common()}
    _atomic_write(OUT_DIR / "meta" / "source_distribution.json",
                  json.dumps(src_dist_json, indent=2))

    yaml_text = (
        "# v2.2 strategy 6.1B build (CC-News + v2.1 carryover, h14 leakage filter).\n"
        "# Built by scripts/build_h14_ccnews.py.\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        "\n"
        "model_cutoff: '2026-01-01'\n"
        "cutoff_buffer_days: 0\n"
        f"default_forecast_horizon_days: {HORIZON_DAYS}\n"
        f"default_lookback_days: {args.lookback_days}\n"
        "benchmarks:\n"
        "  forecastbench:\n"
        "    enabled: true\n"
        f"    forecast_horizon_days: {HORIZON_DAYS}\n"
        f"    lookback_days: {args.lookback_days}\n"
        "  earnings:\n"
        "    enabled: true\n"
        f"    forecast_horizon_days: {HORIZON_DAYS}\n"
        f"    lookback_days: {args.lookback_days}\n"
        "  gdelt_cameo:\n"
        "    enabled: false   # deferred per PROJECT_SPEC §10.1\n"
        "source:\n"
        "  parent_cutoff: '2026-01-01'\n"
        "  parent_h14_cutoff: '2026-01-01-h14'\n"
        "  strategy: CC-News rebuild + v2.1 carryover (strategy B / 6.1B)\n"
        "  cc_news_index: 'data/cc_news/index_2025-12.sqlite'\n"
        f"  top_k_per_fd: {args.top_k}\n"
    )
    _atomic_write(OUT_DIR / "benchmark.yaml", yaml_text)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "script": "scripts/build_h14_ccnews.py",
        "strategy": "CC-News rebuild + v2.1 carryover (PROJECT_SPEC §6.1B)",
        "parent_cutoff": "2026-01-01",
        "parent_h14_cutoff": "2026-01-01-h14",
        "output_cutoff": "2026-01-01-h14-ccnews",
        "horizon_days": HORIZON_DAYS,
        "lookback_days": args.lookback_days,
        "top_k_per_fd": args.top_k,
        "cc_news_index": args.index,
        "cc_news_index_rows": n_idx,
        "benchmarks_in_scope": sorted(BENCHMARKS_IN_SCOPE),
        "fd_in": dict(counts_in),
        "fd_out": dict(counts_out),
        "fd_total_in": sum(counts_in.values()),
        "fd_total_out": len(kept_fds),
        "v21_pool_articles": len(v21_articles),
        "articles_out": len(referenced_aids),
        "drop_reasons": dict(drop_reasons),
        "dangling_article_ids_count": len(dangling_aids),
        "leakage_violations": leakage_violations,
        "horizon_violations": horizon_violations,
        "top_source_domains": dict(src_dist.most_common(20)),
    }
    _atomic_write(OUT_DIR / "build_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print(f"\n[OK] wrote {len(kept_fds)} FDs and {len(referenced_aids)} articles to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
