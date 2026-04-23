"""Build a self-contained "gold" subset of a published v2.1 benchmark.

Curates FDs from the parent `benchmark/data/{cutoff}/` with strict quality
filters (article count, source diversity, horizon, predictability) to
produce a small, balanced, defensible benchmark ready for headline runs.
The output folder is a complete, standalone deliverable: a consumer can
download just `benchmark/data/{cutoff}-gold/` and use it without any
other file from this repo.

## Selection cascade (applied in order)

1. Parent FD passes v2.1 quality filter (article_ids >= 3 already enforced upstream).
2. article_ids count >= --min-articles (default 8): strong evidence base.
3. Distinct article dates >= --min-distinct-days (default 5): multi-day coverage.
4. forecast_point <= resolution_date - --horizon-days (default 14): true 2-week horizon.
5. All retained article dates <= forecast_point - --horizon-days: no nowcasting.
6. fd_type in {stability, change}; drop "unknown" by default
   (--keep-unknown to retain).
7. Per-article text length avg >= --min-avg-chars (default 1500): no title-only stubs.
8. Source diversity >= --min-source-diversity (default 3 distinct domains).
9. Per-benchmark predictability filters:
     - forecastbench: drop sports / lottery / pure-entertainment markets
       via a keyword denylist (configurable in the script).
     - gdelt-cameo: prefer FDs where at least one actor is in
       `GEOPOLITICALLY_SIGNIFICANT_ACTORS` (G20 + NATO + ME principals).
     - earnings: restrict to S&P 100 (or top-N by market cap; configurable).
10. Stratified sampling toward per-benchmark / per-fd_type quotas
    specified in --target-quotas-json.

## Output layout (self-contained)

  benchmark/data/{cutoff}-gold/
    README.md                       what + how + cite
    forecasts.jsonl                 curated FDs
    articles.jsonl                  only referenced articles
    benchmark.yaml                  effective config (drop-in reproducer)
    build_manifest.json             provenance + counts + git_sha + selection rules
    checksums.sha256                SHA256 for forecasts + articles
    selection_criteria.md           human-readable rule list
    LICENSE                         data license (MIT)
    CITATION.cff                    academic citation file
    schema/
      forecast_dossier.schema.json  full JSON Schema Draft 2020-12 (no $ref)
      article.schema.json           same
      FIELD_REFERENCE.md            plain-English per-field guide
    examples/
      load.py                       zero-dep loader (stdlib only)
      validate.py                   checksum + schema validator (zero-dep)
      usage.md                      5 example queries
      eval_template.py              pick-only eval skeleton (no repo imports)
    meta/
      selection_audit.json          per-FD why-kept
      distribution.md               source / topic / horizon / fd_type breakdown
      parent_manifest_sha256.txt    SHA of parent v2.1 forecasts.jsonl

Usage:
  python scripts/build_gold_subset.py --cutoff 2026-01-01
  python scripts/build_gold_subset.py --cutoff 2026-01-01 --target-fds 400
  python scripts/build_gold_subset.py --cutoff 2026-01-01 --dry-run --report-only
  python scripts/build_gold_subset.py --cutoff 2026-01-01 --min-articles 10 --keep-unknown
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Per-benchmark predictability filters
# ---------------------------------------------------------------------------

# FB topic denylist: drop markets where forecasting skill ceiling is low
# (random sports, lottery, weather-on-a-day, pure-entertainment).
FB_TOPIC_DENYLIST = [
    r"\b(super\s*bowl|world\s*cup|nba\s*finals|olympics?\s+(gold|medal))\b",
    r"\bwill\s+\w+\s+win\s+(the\s+)?(grammy|oscar|emmy|tony)\b",
    r"\b(eurovision|met\s*gala)\b",
    r"\blottery\b",
    r"\bcoin(?:\s+toss| flip)\b",
    r"\b(?:nba|nfl|nhl|mlb)\b.*\b(rebounds?|assists?|points?)\s+(?:per\s+)?game\b",
    r"\bplayoff\s+series\b",
    r"\bchampionship\s+(game|match)\b",
]
FB_TOPIC_DENY_RE = re.compile("|".join(FB_TOPIC_DENYLIST), re.IGNORECASE)

# GDELT-CAMEO: prefer pairs where at least one actor is in this set.
# G20 + NATO + ME principals + East Asia powers.
GEOPOLITICALLY_SIGNIFICANT_ACTORS = {
    # Permanent UNSC + G7
    "USA", "GBR", "FRA", "RUS", "CHN", "DEU", "JPN", "ITA", "CAN",
    # G20 plus
    "IND", "BRA", "MEX", "KOR", "AUS", "TUR", "IDN", "ARG", "ZAF", "SAU", "ESP",
    # ME principals
    "ISR", "IRN", "IRQ", "EGY", "JOR", "LBN", "SYR", "QAT", "ARE", "YEM", "PSE",
    # NATO + EU significant
    "POL", "UKR", "NLD", "BEL", "SWE", "FIN", "NOR", "GRC", "ROU", "PRT", "CZE",
    # East Asia
    "PRK", "TWN", "VNM", "PHL", "THA", "SGP", "MYS",
    # Other powers / hot spots
    "PAK", "AFG", "BLR", "VEN", "CUB", "NGA", "ETH",
}

# Earnings: top-100 by market-cap proxy (S&P 100 list as of 2024). Restrict
# to these for predictability + coverage. List can be widened with
# --earnings-universe-file.
SP100_TICKERS = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "BRK.B", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "XOM", "AVGO", "LLY",
    "CVX", "ABBV", "PEP", "KO", "MRK", "BAC", "TMO", "PFE", "COST", "ABT",
    "DIS", "MCD", "CSCO", "ACN", "ADBE", "NKE", "WFC", "VZ", "CRM", "DHR",
    "TXN", "NEE", "BMY", "PM", "RTX", "AMGN", "T", "QCOM", "UPS", "HON",
    "UNP", "ORCL", "INTC", "LIN", "IBM", "LOW", "MS", "INTU", "AMD", "GS",
    "CAT", "AMAT", "C", "AXP", "BLK", "GE", "MDT", "SBUX", "ELV", "DE",
    "ISRG", "BKNG", "GILD", "TGT", "ADP", "MMC", "MO", "VRTX", "REGN", "CI",
    "F", "SCHW", "DUK", "BA", "CB", "ZTS", "PLD", "EOG", "SO", "BDX",
    "MU", "ITW", "CL", "EQIX", "NSC", "ICE", "USB", "AON", "CME", "ETN",
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _atomic_write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _domain(url: str) -> str:
    if not url:
        return ""
    # Cheap host extraction without urllib import overhead per row
    m = re.search(r"https?://([^/]+)/?", url)
    return (m.group(1).lower() if m else "").lstrip("www.")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_fd(fd: dict, articles_idx: dict, args) -> tuple[bool, str]:
    """Return (kept, reason). reason is short tag for the audit log."""
    bench = fd.get("benchmark", "")
    aids = fd.get("article_ids") or []

    # Filter 1: article count
    if len(aids) < args.min_articles:
        return (False, f"min_articles({len(aids)}<{args.min_articles})")

    # Pull article records actually retrievable
    art_recs = [articles_idx[a] for a in aids if a in articles_idx]
    if len(art_recs) < args.min_articles:
        return (False, f"resolved_articles<{args.min_articles}")

    # Filter 2: distinct dates
    dates = sorted({a.get("publish_date", "")[:10] for a in art_recs if a.get("publish_date")})
    if len(dates) < args.min_distinct_days:
        return (False, f"distinct_days({len(dates)}<{args.min_distinct_days})")

    # Filter 3 + 4: horizon
    fp = _parse_date(fd.get("forecast_point"))
    rd = _parse_date(fd.get("resolution_date"))
    if fp is None or rd is None:
        return (False, "no_dates")
    if (rd - fp).days < args.horizon_days:
        return (False, f"horizon_short({(rd - fp).days}<{args.horizon_days})")
    horizon_cutoff = fp - timedelta(days=args.horizon_days)
    horizon_violators = [a for a in art_recs
                         if (_parse_date(a.get("publish_date")) or fp) > horizon_cutoff]
    if horizon_violators:
        # Allow if MOST articles pass; soft-strict mode could keep these
        if len(horizon_violators) > len(art_recs) * 0.2:
            return (False, f"horizon_violators({len(horizon_violators)}/{len(art_recs)})")

    # Filter 5: fd_type
    ft = fd.get("fd_type")
    if ft == "unknown" and not args.keep_unknown:
        return (False, "fd_type_unknown")
    if ft not in {"stability", "change", "unknown"}:
        return (False, f"fd_type_invalid({ft})")

    # Filter 6: avg article text length
    lens = [len(a.get("text") or "") for a in art_recs]
    avg = sum(lens) / max(1, len(lens))
    if avg < args.min_avg_chars:
        return (False, f"avg_text({int(avg)}<{args.min_avg_chars})")

    # Filter 7: source diversity
    domains = {_domain(a.get("url", "")) for a in art_recs if a.get("url")}
    domains.discard("")
    if len(domains) < args.min_source_diversity:
        return (False, f"src_diversity({len(domains)}<{args.min_source_diversity})")

    # Filter 8: per-benchmark predictability
    if bench == "forecastbench":
        question = fd.get("question", "") + " " + (fd.get("background") or "")
        if FB_TOPIC_DENY_RE.search(question):
            return (False, "fb_topic_denied")
    elif bench == "gdelt-cameo":
        meta = fd.get("metadata") or {}
        actors = set(meta.get("actors") or [])
        if not (actors & GEOPOLITICALLY_SIGNIFICANT_ACTORS):
            return (False, "gdelt_no_significant_actor")
    elif bench == "earnings":
        em = fd.get("_earnings_meta") or {}
        ticker = (em.get("ticker") or "").upper().split(".")[0]
        if ticker not in {t.split(".")[0] for t in SP100_TICKERS}:
            return (False, "earnings_not_sp100")

    return (True, "kept")


def _stratified_sample(kept: list[dict], target_per_stratum: dict[tuple, int],
                       seed: int = 42) -> list[dict]:
    """Sample to hit per-(benchmark, fd_type) targets.
    Returns the union of strata-capped samples, deterministic by seed.
    """
    import random
    rng = random.Random(seed)
    by_stratum: dict[tuple, list[dict]] = collections.defaultdict(list)
    for fd in kept:
        by_stratum[(fd.get("benchmark"), fd.get("fd_type"))].append(fd)

    out: list[dict] = []
    used_strata = []
    for stratum, fds in by_stratum.items():
        target = target_per_stratum.get(stratum, len(fds))
        rng.shuffle(fds)
        out.extend(fds[:target])
        used_strata.append((stratum, target, len(fds), min(target, len(fds))))
    return out, used_strata


# ---------------------------------------------------------------------------
# Self-contained output writers
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = {
    ("forecastbench", "stability"): 60,
    ("forecastbench", "change"):    40,
    ("gdelt-cameo",  "stability"): 100,
    ("gdelt-cameo",  "change"):    100,
    ("earnings",     "stability"):  60,
    ("earnings",     "change"):     60,
}


def _load_etd_facts_for_articles(article_ids_set: set[str]) -> list[dict]:
    """Load ETD facts filtered to those whose primary_article_id is in the
    gold-articles set. Prefers the Stage-3 linked output, then Stage-2
    canonical, then Stage-1 raw. Returns [] if nothing is available."""
    candidates = [
        ROOT / "data" / "etd" / "facts.v1_production_2026-01-01.jsonl",  # post-Stage-3 + filter
        ROOT / "data" / "etd" / "facts.v1_linked.jsonl",                  # Stage-3 link
        ROOT / "data" / "etd" / "facts.v1_canonical.jsonl",               # Stage-2 dedup
        ROOT / "data" / "etd" / "facts.v1.jsonl",                         # Stage-1 raw
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("[gold] no ETD facts file found; gold facts.jsonl will be empty")
        return []
    print(f"[gold] ETD source: {src.name}")
    out = []
    seen_ids = set()
    for line in src.open(encoding="utf-8"):
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = f.get("primary_article_id")
        if pid in article_ids_set and f.get("id") not in seen_ids:
            out.append(f)
            seen_ids.add(f.get("id"))
    return out


def _write_self_contained(out_dir: Path, fds: list[dict], articles: list[dict],
                          parent_dir: Path, args, audit: dict, used_strata: list,
                          parent_sha: str, git_sha: str) -> None:
    """Write the complete self-contained gold-folder payload."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary deliverables (FDs + articles + ETD facts)
    _atomic_write_jsonl(out_dir / "forecasts.jsonl", fds)
    _atomic_write_jsonl(out_dir / "articles.jsonl", articles)
    article_ids_set = {a["id"] for a in articles if "id" in a}
    facts = _load_etd_facts_for_articles(article_ids_set)
    _atomic_write_jsonl(out_dir / "facts.jsonl", facts)
    print(f"[gold] facts.jsonl: {len(facts)} ETD facts linked to {len(article_ids_set)} gold articles")

    # 2. Manifest
    bench_counts = collections.Counter(d.get("benchmark") for d in fds)
    fdtype_counts = collections.Counter((d.get("benchmark"), d.get("fd_type")) for d in fds)
    n_facts = sum(1 for _ in (out_dir / "facts.jsonl").open(encoding="utf-8")) if (out_dir / "facts.jsonl").exists() else 0
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "kind": "gold_subset",
        "parent_cutoff": args.cutoff,
        "parent_forecasts_sha256": parent_sha,
        "git_sha": git_sha,
        "n_fds": len(fds),
        "n_articles": len(articles),
        "n_facts": n_facts,
        "n_fds_by_benchmark": dict(bench_counts),
        "n_fds_by_stratum": {f"{b}|{t}": n for (b, t), n in fdtype_counts.items()},
        "selection_thresholds": {
            "min_articles": args.min_articles,
            "min_distinct_days": args.min_distinct_days,
            "horizon_days": args.horizon_days,
            "min_avg_chars": args.min_avg_chars,
            "min_source_diversity": args.min_source_diversity,
            "keep_unknown_fd_type": args.keep_unknown,
            "earnings_universe": "S&P 100",
            "gdelt_cameo_actor_significance_filter": True,
            "fb_topic_denylist_count": len(FB_TOPIC_DENYLIST),
        },
        "deliverable_layout": {
            "forecasts.jsonl":   "curated FD subset, one record per line",
            "articles.jsonl":    "only articles referenced by the subset, full text",
            "facts.jsonl":       "ETD atomic facts whose primary_article_id is in this subset's articles.jsonl (may be empty if Stage 1 ETD has not been run for the parent cutoff)",
            "benchmark.yaml":    "effective build config (drop-in reproducer)",
            "build_manifest.json": "this file",
            "checksums.sha256":  "SHA256 for forecasts + articles + facts",
            "selection_criteria.md": "human-readable selection rules",
            "schema/":           "self-contained JSON Schema for FD + article + ETD fact",
            "examples/":         "zero-dependency loader, validator, eval template",
            "meta/":              "per-FD audit + distribution stats + parent SHA",
        },
    }
    _atomic_write_json(out_dir / "build_manifest.json", manifest)

    # 3. Effective config (copy parent benchmark.yaml + annotate as gold)
    src_yaml = parent_dir / "benchmark.yaml"
    if src_yaml.exists():
        body = src_yaml.read_text(encoding="utf-8")
        body = "# Gold subset of " + str(parent_dir) + "\n" + \
               "# See selection_criteria.md and build_manifest.json for the gold-specific filters.\n\n" + body
        _atomic_write_text(out_dir / "benchmark.yaml", body)

    # 4. Checksums (now covers FDs + articles + ETD facts)
    sha_lines = [f"{_sha256(out_dir / n)}  {n}"
                 for n in ("forecasts.jsonl", "articles.jsonl", "facts.jsonl")]
    _atomic_write_text(out_dir / "checksums.sha256", "\n".join(sha_lines) + "\n")

    # 5. Selection criteria doc
    crit_md = _selection_criteria_doc(args)
    _atomic_write_text(out_dir / "selection_criteria.md", crit_md)

    # 6. README.md (entry point)
    readme = _readme_doc(args, len(fds), len(articles), bench_counts, fdtype_counts, parent_sha)
    _atomic_write_text(out_dir / "README.md", readme)

    # 7. LICENSE + CITATION.cff
    _atomic_write_text(out_dir / "LICENSE", _license_text())
    _atomic_write_text(out_dir / "CITATION.cff", _citation_cff(args.cutoff, len(fds), git_sha))

    # 8. Schema folder (self-contained JSON Schema for FD + article + ETD fact)
    schema_dir = out_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(schema_dir / "forecast_dossier.schema.json", _fd_schema())
    _atomic_write_json(schema_dir / "article.schema.json", _article_schema())
    _atomic_write_json(schema_dir / "etd_fact.schema.json", _etd_fact_schema())
    _atomic_write_text(schema_dir / "FIELD_REFERENCE.md", _field_reference_doc())

    # 9. Examples folder (zero-dep loader + validator + eval template)
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(ex_dir / "load.py", _example_load_py())
    _atomic_write_text(ex_dir / "validate.py", _example_validate_py())
    _atomic_write_text(ex_dir / "usage.md", _example_usage_md())
    _atomic_write_text(ex_dir / "eval_template.py", _example_eval_template_py())

    # 10. Meta folder
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(meta_dir / "selection_audit.json", audit)
    dist = _distribution_doc(fds, articles, bench_counts, fdtype_counts, used_strata)
    _atomic_write_text(meta_dir / "distribution.md", dist)
    _atomic_write_text(meta_dir / "parent_manifest_sha256.txt", parent_sha + "\n")


# ---------------------------------------------------------------------------
# Doc generators (kept terse; expand offline if needed)
# ---------------------------------------------------------------------------

def _selection_criteria_doc(args) -> str:
    return f"""# EMR-ACH Gold Subset, selection criteria

This document records the exact filter cascade applied to produce the gold
subset from the parent v2.1 publish at `benchmark/data/{args.cutoff}/`.
Every threshold is recorded here AND in `build_manifest.json` for
reproducibility.

## Cascade (applied in order; first failure drops the FD)

1. **Article count**: `len(article_ids) >= {args.min_articles}`
2. **Distinct article dates**: `>= {args.min_distinct_days}` (no single-news-cycle FDs)
3. **Forecast horizon**: `resolution_date - forecast_point >= {args.horizon_days} days`
4. **Strict horizon enforcement**: at most 20% of an FD's articles may be
   dated after `forecast_point - {args.horizon_days} days` (no nowcasting)
5. **fd_type**: in `{{stability, change}}`{', plus `unknown`' if args.keep_unknown else ' (unknown is dropped)'}
6. **Avg article text length**: `>= {args.min_avg_chars} chars` (no title-only stubs)
7. **Source diversity**: `>= {args.min_source_diversity}` distinct hostnames per FD
8. **Per-benchmark predictability**:
   - `forecastbench`: drop FDs whose question or background matches a
     denylist of pure sports / lottery / pure-entertainment topics
     ({len(FB_TOPIC_DENYLIST)} regex patterns; see `scripts/build_gold_subset.py`).
   - `gdelt-cameo`: at least one of the two actors must be in
     `GEOPOLITICALLY_SIGNIFICANT_ACTORS` (G20 + NATO + ME principals +
     East Asia powers; ~70 country codes).
   - `earnings`: ticker must be in S&P 100.
9. **Stratified sampling** to per-(benchmark, fd_type) quotas; default
   targets: `{json.dumps({f"{k[0]}::{k[1]}": v for k, v in DEFAULT_TARGETS.items()})}`. The actual
   sample sizes per stratum are in `meta/distribution.md`.

## Why these thresholds

- **8 articles, 5 distinct days** filter out FDs with shallow evidence
  pools where forecasting reduces to luck.
- **14-day horizon** matches the v2.1 design (`docs/FORECAST_DOSSIER.md`)
  and rules out nowcasting confounds.
- **1500 chars / 3 distinct domains** filter out title-only stubs and
  echo-chamber FDs where multiple recordings are the same wire story.
- **S&P 100 only for earnings**: large caps have strong analyst coverage
  and well-defined consensus EPS expectations; mid-cap earnings have
  much sparser news coverage and noisier EPS estimates.

## Reproducibility

Re-run with the recorded thresholds in `build_manifest.json` to regenerate
this exact subset bytes-for-bytes. The selection script is at
`scripts/build_gold_subset.py` in the parent repository.
"""


def _readme_doc(args, n_fds, n_articles, bench_counts, fdtype_counts, parent_sha) -> str:
    return f"""# EMR-ACH Gold Subset, cutoff `{args.cutoff}`

A small, balanced, high-quality forecasting benchmark curated from the
EMR-ACH v2.1 release. Self-contained: nothing in this folder references
files outside the folder.

## TL;DR

- **{n_fds} Forecast Dossiers** across 3 benchmarks (geopolitics, finance, prediction markets), unified primary target **Comply vs Surprise**.
- **{n_articles} retrieved news articles**, full body text inline.
- **ETD atomic facts** (`facts.jsonl`) linked to those articles, providing a structured-evidence channel alongside the raw text. May be empty if Stage 1 ETD has not been run for the parent cutoff.
- **2-week forecast horizon** strictly enforced: every article is dated at least 14 days before the resolution date.
- **fd_type stratified** ({{stability, change}}) so the headline metric on the change subset measures real forecasting skill, not status-quo persistence.

## Files

- `forecasts.jsonl` ({n_fds} records). Each line is one FD.
- `articles.jsonl` ({n_articles} records). Each line is one article.
- `facts.jsonl`. ETD atomic facts whose `primary_article_id` is in `articles.jsonl`. One line per fact.
- `schema/`: full JSON Schema (Draft 2020-12) for all three record types (FD + article + ETD fact).
- `examples/load.py`: 50-line stdlib-only loader (loads FDs + articles + facts).
- `examples/validate.py`: SHA + schema validator (covers all three).
- `examples/eval_template.py`: pick-only evaluation skeleton.
- `selection_criteria.md`: exact filter cascade.
- `build_manifest.json`: counts + git_sha + thresholds + parent SHA.
- `meta/`: selection audit + distribution stats.

## Quickstart

```python
import json
from pathlib import Path

fds = [json.loads(l) for l in open("forecasts.jsonl", encoding="utf-8")]
arts = {{a["id"]: a for a in (json.loads(l) for l in open("articles.jsonl", encoding="utf-8"))}}

for fd in fds[:3]:
    print(fd["id"], fd["benchmark"], fd["hypothesis_set"], "ground_truth:", fd["ground_truth"])
    for aid in fd["article_ids"][:2]:
        a = arts[aid]
        print(f"   - {{a['publish_date']}} {{a['url']}}")
```

See `examples/usage.md` for richer queries and `examples/eval_template.py` for a baseline-runner skeleton.

## Data card

| Benchmark | n FDs | stability | change |
|---|---:|---:|---:|
""" + "\n".join(
        f"| `{b}` | {bench_counts[b]} | {fdtype_counts.get((b, 'stability'), 0)} | {fdtype_counts.get((b, 'change'), 0)} |"
        for b in sorted(bench_counts)
    ) + f"""

## Provenance

- Parent: `benchmark/data/{args.cutoff}/forecasts.jsonl` (SHA256 in `meta/parent_manifest_sha256.txt`).
- Parent SHA256: `{parent_sha[:16]}...`
- Build: see `build_manifest.json` for git_sha + timestamps.

## License

MIT. See `LICENSE`. Citation: `CITATION.cff`.

## Schema versioning

Schema version `2.1-gold` (see `schema/FIELD_REFERENCE.md`). Backwards-compatible additions only; field removals would bump to `3.0`.
"""


def _license_text() -> str:
    return """MIT License

Copyright (c) 2026 EMR-ACH Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this dataset and associated documentation files (the "Dataset"), to deal
in the Dataset without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Dataset, and to permit persons to whom the Dataset is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Dataset.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

Third-party article content (NYT, The Guardian, GDELT, SEC EDGAR,
yfinance, ForecastBench upstream sources) retains its original license;
users are responsible for respecting source terms.
"""


def _citation_cff(cutoff: str, n_fds: int, git_sha: str) -> str:
    return f"""cff-version: 1.2.0
title: EMR-ACH Gold Subset, cutoff {cutoff}
type: dataset
authors:
  - name: EMR-ACH Project
abstract: >
  A {n_fds}-FD curated subset of the EMR-ACH v2.1 forecasting benchmark
  spanning geopolitics (GDELT-CAMEO), finance (S&P 100 earnings), and
  prediction-market questions (ForecastBench), unified under a binary
  Comply vs Surprise primary target with strict 14-day forecast horizon.
keywords:
  - forecasting
  - benchmark
  - LLM
  - evidence retrieval
  - GDELT
  - earnings
  - prediction markets
date-released: {datetime.utcnow().date().isoformat()}
license: MIT
identifiers:
  - type: other
    value: emr-ach-gold-{cutoff}
    description: cutoff identifier
  - type: other
    value: git-{git_sha[:12]}
    description: source-tree commit
"""


def _fd_schema() -> dict:
    """Self-contained JSON Schema for the FD record."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "emr-ach-gold/forecast_dossier.schema.json",
        "title": "Forecast Dossier (gold subset)",
        "description": "v2.1-gold FD record. Self-contained: hypothesis_definitions are inline; downstream consumers do not need any external doc.",
        "type": "object",
        "required": ["id", "benchmark", "source", "hypothesis_set", "hypothesis_definitions",
                     "question", "forecast_point", "resolution_date", "ground_truth",
                     "ground_truth_idx", "article_ids", "fd_type"],
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string"},
            "benchmark": {"type": "string", "enum": ["forecastbench", "gdelt-cameo", "earnings"]},
            "source": {"type": "string"},
            "hypothesis_set": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "hypothesis_definitions": {"type": "object"},
            "question": {"type": "string"},
            "background": {"type": ["string", "null"]},
            "forecast_point": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
            "resolution_date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
            "ground_truth": {"type": "string"},
            "ground_truth_idx": {"type": "integer", "minimum": 0},
            "article_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "fd_type": {"type": "string", "enum": ["stability", "change", "unknown"]},
            "prior_state_30d": {"type": ["string", "null"]},
            "prior_state_stability": {"type": ["number", "null"]},
            "prior_state_n_events": {"type": ["integer", "null"]},
            "x_multiclass_hypothesis_set": {"type": ["array", "null"]},
            "x_multiclass_hypothesis_definitions": {"type": ["object", "null"]},
            "x_multiclass_ground_truth": {"type": ["string", "null"]},
            "x_multiclass_ground_truth_idx": {"type": ["integer", "null"]},
            "default_horizon_days": {"type": ["integer", "null"]},
            "lookback_days": {"type": ["integer", "null"]},
            "crowd_probability": {"type": ["number", "null"]},
            "metadata": {"type": ["object", "null"]},
            "_earnings_meta": {"type": ["object", "null"]},
        },
    }


def _article_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "emr-ach-gold/article.schema.json",
        "title": "Article (gold subset)",
        "type": "object",
        "required": ["id", "url", "title", "publish_date"],
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string", "pattern": "^art_[0-9a-f]{12}$"},
            "url": {"type": "string"},
            "title": {"type": "string"},
            "text": {"type": "string"},
            "publish_date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
            "source_domain": {"type": "string"},
            "language": {"type": "string"},
            "provenance": {"type": ["array", "string"]},
        },
    }


def _etd_fact_schema() -> dict:
    """Self-contained JSON Schema for the ETD atomic-fact record."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "emr-ach-gold/etd_fact.schema.json",
        "title": "ETD Atomic Fact (gold subset)",
        "description": (
            "v2.1-gold ETD fact record. One atomic dated event extracted from "
            "one or more articles by an LLM (gpt-4o-mini, v3 prompt with "
            "anchor-table + verbatim-quote requirements). The `primary_article_id` "
            "links each fact to its source article; the article must be present "
            "in articles.jsonl for the gold subset to be self-contained."
        ),
        "type": "object",
        "required": ["id", "fact", "time", "primary_article_id",
                     "polarity", "extraction_confidence"],
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string", "pattern": "^f_[0-9a-f]{12}$"},
            "schema_version": {"type": "string"},
            "time": {"type": "string",
                     "description": "ISO date (YYYY-MM-DD), YYYY-MM, YYYY, or 'unknown'."},
            "time_end": {"type": ["string", "null"]},
            "time_precision": {"type": "string",
                               "enum": ["day", "week", "month", "quarter", "year", "unknown"]},
            "time_type": {"type": "string",
                          "enum": ["point", "interval", "ongoing", "periodic"]},
            "fact": {"type": "string", "minLength": 1},
            "evidence_quote": {"type": ["string", "null"],
                               "description": "Verbatim substring of the article body that grounds the fact (v3 prompt)."},
            "language": {"type": "string"},
            "article_ids": {"type": "array", "items": {"type": "string"}},
            "primary_article_id": {"type": "string", "pattern": "^art_[0-9a-f]{12}$"},
            "article_date": {"type": "string"},
            "source": {"type": ["string", "null"]},
            "entities": {"type": "array",
                         "items": {"type": "object",
                                   "properties": {"name": {"type": "string"},
                                                  "type": {"type": "string"}}}},
            "location": {"type": ["string", "object", "null"]},
            "metrics": {"type": "array"},
            "kind": {"type": ["string", "null"]},
            "tags": {"type": "array"},
            "polarity": {"type": "string",
                         "enum": ["asserted", "negated", "hypothetical", "reported"]},
            "attribution": {"type": ["string", "null"]},
            "extraction_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "canonical_id": {"type": ["string", "null"]},
            "variant_ids": {"type": "array"},
            "linked_fd_ids": {"type": "array",
                              "items": {"type": "string"},
                              "description": "Stage-3 link: FD IDs whose article_ids include this fact's primary_article_id."},
            "extractor": {"type": "string"},
            "extracted_at": {"type": "string"},
        },
    }


def _field_reference_doc() -> str:
    return """# Per-field reference, EMR-ACH Gold Subset

## Forecast Dossier (`forecasts.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable per-FD identifier. |
| `benchmark` | enum | One of `forecastbench`, `gdelt-cameo`, `earnings`. |
| `source` | string | Origin source within the benchmark. |
| `hypothesis_set` | string[] | Primary hypotheses; v2.1-gold uses `["Comply", "Surprise"]`. |
| `hypothesis_definitions` | object | Plain-English definition per hypothesis. Inline so the FD is self-contained. |
| `question` | string | The forecasting question. |
| `background` | string | Context / resolution criteria; may be empty. |
| `forecast_point` | date | Cutoff for evidence; predictions are made at this date. |
| `resolution_date` | date | When the outcome is known. |
| `ground_truth` | string | The realized hypothesis (one of `hypothesis_set`). |
| `ground_truth_idx` | int | Index into `hypothesis_set`. |
| `article_ids` | string[] | Article IDs that constitute the evidence pool; resolve via `articles.jsonl`. |
| `fd_type` | enum | One of `stability` (status-quo holds; ground_truth=Comply), `change` (status-quo breaks; ground_truth=Surprise), `unknown` (insufficient prior history). |
| `prior_state_30d` | string | Domain-specific prior expectation: modal CAMEO intensity (GDELT) / mode of prior 4 quarters Beat-Meet-Miss (earnings) / crowd majority Yes-No (FB). |
| `prior_state_stability` | float | Confidence in the prior expectation, 0-1. |
| `prior_state_n_events` | int | History sample size used to compute the prior. |
| `x_multiclass_hypothesis_set` | string[] | Original domain-multiclass labels (3-class CAMEO, Beat/Meet/Miss, Yes/No), preserved for ablation. |
| `x_multiclass_ground_truth` | string | Original multiclass label that resolved. |
| `default_horizon_days` | int | Recommended experiment horizon h; the published bundle was built so every article is dated <= forecast_point - h. |
| `lookback_days` | int | The retrieval lookback window used at build time. |
| `crowd_probability` | number | ForecastBench markets only: prediction-market freeze-time probability. |
| `metadata` | object | Per-benchmark metadata (e.g. CAMEO actor codes, event_base_code). |
| `_earnings_meta` | object | Earnings only: ticker, sector, industry, eps_estimate, eps_actual, surprise_pct, report_date. |

## Article (`articles.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable URL-hash identifier; format `art_<12-hex>`. |
| `url` | string | Canonical URL. |
| `title` | string | Article title. |
| `text` | string | Trafilatura-extracted body. May be empty for title-only records. |
| `publish_date` | date | ISO publication date (YYYY-MM-DD). |
| `source_domain` | string | Hostname (lowercase, `www.` stripped). |
| `language` | string | BCP-47 language tag (default: `en`). |
| `provenance` | string|string[] | Pipe-separated tags from each fetcher that surfaced this URL. |

## Comply vs Surprise contract

Every FD's primary target is binary:
- `Comply` if the resolved outcome matches the prior-state expectation.
- `Surprise` if it breaks the expectation.

Headline metrics in this dataset should be reported on the **change** subset (where the prior is wrong and the model must read evidence to win).

## ETD Atomic Fact (`facts.jsonl`)

ETD = Event Timeline Dossier. Each fact is one atomic dated event extracted by an LLM (gpt-4o-mini, v3 prompt with anchor table + verbatim-quote requirements) from one or more articles. Facts in this file are filtered to those whose `primary_article_id` is present in `articles.jsonl`, so the gold subset stays self-contained.

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable per-fact identifier; format `f_<12-hex>`. |
| `time` | string | The event's date: ISO date (YYYY-MM-DD), YYYY-MM, YYYY, or `unknown`. Bounded above by the source article's `publish_date`. |
| `time_precision` | enum | `day` / `week` / `month` / `quarter` / `year` / `unknown`. |
| `time_type` | enum | `point` / `interval` / `ongoing` / `periodic`. |
| `fact` | string | One-sentence atomic claim. |
| `evidence_quote` | string | Verbatim substring of the article body that grounds the fact (v3 prompt requirement). |
| `entities` | object[] | Named actors involved: `[{name, type}, ...]`. |
| `metrics` | object[] | Numeric quantities: `[{name, value, unit}, ...]`. |
| `kind` | string | Short tag, e.g. `military-deployment`, `earnings-release`, `policy-statement`. |
| `polarity` | enum | `asserted` (default) / `negated` / `hypothetical` / `reported` (cited; requires `attribution`). |
| `attribution` | string | Named source if `polarity == reported`. |
| `extraction_confidence` | enum | `high` / `medium` / `low`. Production filter typically requires `high`. |
| `primary_article_id` | string | Article ID; resolves to a record in `articles.jsonl`. |
| `article_date` | string | The source article's publish_date (denormalized for convenience). |
| `linked_fd_ids` | string[] | If Stage-3 link ran, the FD IDs whose `article_ids` include this fact's `primary_article_id`. May be empty if Stage 1 or Stage 2 only. |
| `canonical_id` | string | If Stage-2 dedup ran, points at the canonical fact for a near-duplicate cluster (null if this fact IS the canonical or no dedup ran). |
| `extractor` | string | Model identifier, e.g. `gpt-4o-mini-2024-07-18`. |
| `extracted_at` | string | ISO timestamp of the extraction. |

### How to use ETD facts as evidence

The default baseline runner uses `articles.jsonl` text as evidence. ETD facts are an **additive** channel, useful for:
- Building a "facts-only" baseline (B10b in the v2.2 design): pass the per-FD fact subset as the evidence block.
- Building a "hybrid" baseline (B10): include both article snippets AND structured facts in the prompt.
- Time-travel queries: filter `facts.jsonl` by `time` to reconstruct what was known at any past date.

See `examples/load.py:fd_facts()` for a one-call helper that returns the facts linked to a given FD via its `article_ids`.
"""


def _example_load_py() -> str:
    return '''#!/usr/bin/env python
"""Zero-dependency loader for the EMR-ACH Gold Subset.

Uses only Python stdlib. Run from the gold-folder directory.
Loads three record types: FDs, articles, and ETD atomic facts.
"""
import json
from pathlib import Path
from collections import defaultdict


def load_fds(path: str = "forecasts.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_articles(path: str = "articles.jsonl") -> dict[str, dict]:
    """Returns {article_id: article_record}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            out[a["id"]] = a
    return out


def load_facts(path: str = "facts.jsonl") -> list[dict]:
    """Returns the ETD atomic-fact records. May be empty if Stage 1 ETD
    has not been run for the parent cutoff."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_facts_by_article(facts: list[dict]) -> dict[str, list[dict]]:
    """Group facts by primary_article_id for fast per-article lookup."""
    out = defaultdict(list)
    for f in facts:
        pid = f.get("primary_article_id")
        if pid:
            out[pid].append(f)
    return out


def fd_evidence(fd: dict, articles: dict[str, dict], top_k: int | None = None) -> list[dict]:
    """Return the article records cited in fd.article_ids, in order."""
    aids = fd.get("article_ids") or []
    if top_k is not None:
        aids = aids[:top_k]
    return [articles[a] for a in aids if a in articles]


def fd_facts(fd: dict, facts_by_article: dict[str, list[dict]],
             top_k: int | None = None) -> list[dict]:
    """Return ETD facts whose primary_article_id is in fd.article_ids,
    sorted by fact.time descending."""
    out = []
    for aid in fd.get("article_ids") or []:
        out.extend(facts_by_article.get(aid, []))
    out.sort(key=lambda f: f.get("time") or "", reverse=True)
    return out[:top_k] if top_k else out


if __name__ == "__main__":
    fds = load_fds()
    arts = load_articles()
    facts = load_facts()
    print(f"Loaded {len(fds)} FDs, {len(arts)} articles, {len(facts)} ETD facts.")
    by_bench = {}
    for fd in fds:
        by_bench[fd["benchmark"]] = by_bench.get(fd["benchmark"], 0) + 1
    print(f"By benchmark: {by_bench}")
    sample = fds[0]
    print(f"\\nSample FD: {sample['id']} ({sample['benchmark']})")
    print(f"  Q: {sample['question'][:90]}")
    print(f"  Hypotheses: {sample['hypothesis_set']}")
    print(f"  Ground truth: {sample['ground_truth']}  (fd_type={sample['fd_type']})")
    ev = fd_evidence(sample, arts, top_k=3)
    for a in ev:
        print(f"  Article: {a['publish_date']} {a['url'][:70]}")
    if facts:
        idx = index_facts_by_article(facts)
        for fact in fd_facts(sample, idx, top_k=3):
            print(f"  Fact: ({fact.get('time','?')}) {fact['fact'][:90]}")
'''


def _example_validate_py() -> str:
    return '''#!/usr/bin/env python
"""Standalone validator: SHA256 + minimal schema check.

No external dependencies. Run from the gold-folder directory.
Exits non-zero on any mismatch; prints a short report.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    expected = {}
    with open("checksums.sha256", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]

    failed = 0
    for name, want in expected.items():
        got = sha256(name)
        ok = got == want
        print(f"{'OK ' if ok else 'BAD'}  {name}  (sha256 {got[:16]}{'==' if ok else ' != '}{want[:16]})")
        if not ok:
            failed += 1

    # Minimal schema check: every FD has the required v2.1-gold fields.
    REQUIRED_FD = {"id", "benchmark", "source", "hypothesis_set", "hypothesis_definitions",
                   "question", "forecast_point", "resolution_date", "ground_truth",
                   "ground_truth_idx", "article_ids", "fd_type"}
    n = 0; bad_fd = 0
    with open("forecasts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n += 1
            if REQUIRED_FD - set(d.keys()):
                bad_fd += 1
    print(f"\\nFDs: {n}; missing-required: {bad_fd}")
    if bad_fd:
        failed += 1

    # Article schema spot-check
    REQUIRED_ART = {"id", "url", "title", "publish_date"}
    n = 0; bad_art = 0
    with open("articles.jsonl", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            n += 1
            if REQUIRED_ART - set(a.keys()):
                bad_art += 1
    print(f"Articles: {n}; missing-required: {bad_art}")
    if bad_art:
        failed += 1

    # ETD fact schema spot-check (file may be empty)
    if Path("facts.jsonl").exists():
        REQUIRED_FACT = {"id", "fact", "time", "primary_article_id",
                         "polarity", "extraction_confidence"}
        n = 0; bad_f = 0
        with open("facts.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n += 1
                if REQUIRED_FACT - set(d.keys()):
                    bad_f += 1
        print(f"ETD facts: {n}; missing-required: {bad_f}")
        if bad_f:
            failed += 1

    if failed:
        print(f"\\nFAIL: {failed} integrity issue(s).")
        return 1
    print("\\nOK: integrity verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _example_usage_md() -> str:
    return """# Five example queries (zero-dep, stdlib only)

## 1. Count FDs per benchmark and per fd_type

```python
import json, collections
fds = [json.loads(l) for l in open("forecasts.jsonl", encoding="utf-8")]
ctr = collections.Counter((d["benchmark"], d["fd_type"]) for d in fds)
for k, n in sorted(ctr.items()): print(f"{k}: {n}")
```

## 2. Find the change subset for one benchmark

```python
change = [d for d in fds if d["benchmark"] == "earnings" and d["fd_type"] == "change"]
print(f"Earnings change subset: {len(change)} FDs")
```

## 3. Look up the evidence pool for a specific FD

```python
arts = {a["id"]: a for a in (json.loads(l) for l in open("articles.jsonl", encoding="utf-8"))}
fd = next(d for d in fds if d["benchmark"] == "earnings")
for aid in fd["article_ids"]:
    a = arts[aid]
    print(a["publish_date"], a["url"])
```

## 4. Compute majority-class baseline accuracy on the change subset

```python
correct = sum(1 for d in change if d["ground_truth"] == "Surprise")
print(f"All-Surprise baseline accuracy: {correct/len(change):.2f}")
```

## 5. Render a baseline prompt without the EMR-ACH codebase

See `eval_template.py` for the full pattern. Sketch:

```python
def render_prompt(fd, articles, max_chars_per_article=600):
    arts = [articles[a] for a in fd["article_ids"][:10] if a in articles]
    hyp_block = "\\n".join(f"  - {h}: {fd['hypothesis_definitions'].get(h, '')}"
                           for h in fd["hypothesis_set"])
    art_block = "\\n\\n".join(f"[A{i+1}] {a['publish_date']} {a.get('source_domain', '')}: "
                              f"{a['title']}\\n{a.get('text', '')[:max_chars_per_article]}"
                              for i, a in enumerate(arts))
    return f"Question: {fd['question']}\\n\\nHypotheses:\\n{hyp_block}\\n\\nEvidence:\\n{art_block}"
```
"""


def _example_eval_template_py() -> str:
    return '''#!/usr/bin/env python
"""Baseline pick-only evaluation skeleton, gold subset, zero EMR-ACH dependency.

Loads the gold subset, formats a prompt per FD, calls a user-supplied
LLM client, and computes accuracy + per-fd_type breakdown. No reference
to any module outside this folder.

Configure the LLM call by editing `call_llm()`; the rest is inert.
"""
import json
from pathlib import Path
from collections import Counter


def load_gold():
    fds = [json.loads(l) for l in open("forecasts.jsonl", encoding="utf-8")]
    arts = {a["id"]: a for a in (json.loads(l) for l in open("articles.jsonl", encoding="utf-8"))}
    return fds, arts


def render_prompt(fd, arts, max_arts=10, max_chars_per_article=600):
    hyp_block = "\\n".join(f"  - {h}: {fd['hypothesis_definitions'].get(h, '')}" for h in fd["hypothesis_set"])
    selected = [arts[a] for a in fd["article_ids"][:max_arts] if a in arts]
    art_block = "\\n\\n".join(
        f"[A{i+1}] {a['publish_date']} {a.get('source_domain', '?')} -- {a['title']}\\n"
        f"{(a.get('text') or '')[:max_chars_per_article]}"
        for i, a in enumerate(selected)
    )
    return (
        f"Forecasting question: {fd['question']}\\n\\n"
        f"Background: {fd.get('background') or '(none)'}\\n\\n"
        f"Hypotheses (pick exactly one):\\n{hyp_block}\\n\\n"
        f"Evidence:\\n{art_block}\\n\\n"
        f"Forecast point: {fd['forecast_point']}\\n"
        f"Resolution date: {fd['resolution_date']}\\n\\n"
        f"Return JSON only, no prose:\\n"
        f'{{"prediction": "<exactly one of: {", ".join(fd["hypothesis_set"])}>"}}'
    )


def call_llm(prompt: str) -> str:
    """Replace this stub with your provider call. The default returns
    the FD's first hypothesis (a degenerate baseline)."""
    raise NotImplementedError("Edit call_llm() to call your model. The default raises by design.")


def main():
    fds, arts = load_gold()
    print(f"Loaded {len(fds)} FDs and {len(arts)} articles.")

    correct_total = 0
    by_ft = Counter()
    correct_by_ft = Counter()
    skipped = 0

    for i, fd in enumerate(fds, 1):
        try:
            prompt = render_prompt(fd, arts)
            response_text = call_llm(prompt)
            pred = json.loads(response_text).get("prediction")
        except Exception as e:
            skipped += 1
            continue
        ft = fd.get("fd_type", "unknown")
        by_ft[ft] += 1
        if pred == fd["ground_truth"]:
            correct_total += 1
            correct_by_ft[ft] += 1

    n = sum(by_ft.values())
    print(f"\\nResults (skipped={skipped}):")
    print(f"  overall: {correct_total}/{n} = {100*correct_total/max(1, n):.1f}%")
    for ft, count in by_ft.items():
        c = correct_by_ft[ft]
        print(f"  fd_type={ft}: {c}/{count} = {100*c/max(1, count):.1f}%")


if __name__ == "__main__":
    main()
'''


def _distribution_doc(fds, articles, bench_counts, fdtype_counts, used_strata) -> str:
    n = len(fds)
    lines = [
        "# Gold subset distribution",
        "",
        f"Total FDs: **{n}**.  Total referenced articles: **{len(articles)}**.",
        "",
        "## By benchmark x fd_type",
        "",
        "| benchmark | stability | change | unknown | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in sorted(bench_counts):
        st = fdtype_counts.get((b, "stability"), 0)
        ch = fdtype_counts.get((b, "change"), 0)
        un = fdtype_counts.get((b, "unknown"), 0)
        lines.append(f"| `{b}` | {st} | {ch} | {un} | {bench_counts[b]} |")

    lines += ["", "## Per-stratum sampling (target vs available vs taken)", "",
              "| (benchmark, fd_type) | target | available | taken |",
              "|---|---:|---:|---:|"]
    for (stratum, target, available, taken) in used_strata:
        lines.append(f"| `{stratum}` | {target} | {available} | {taken} |")

    # Article date span
    dates = sorted(a["publish_date"] for a in articles if "publish_date" in a)
    if dates:
        lines += ["", f"## Article date span", "",
                  f"- Earliest: `{dates[0]}`", f"- Latest: `{dates[-1]}`",
                  f"- Distinct dates: {len(set(dates))}"]

    # Source diversity
    by_source = collections.Counter()
    for a in articles:
        by_source[a.get("source_domain", "(unknown)")] += 1
    lines += ["", "## Top 20 article source domains", "", "| domain | n |", "|---|---:|"]
    for src, c in by_source.most_common(20):
        lines.append(f"| `{src}` | {c} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_target_quotas(args) -> dict[tuple, int]:
    if args.target_quotas_json:
        raw = json.loads(args.target_quotas_json)
        return {tuple(k.split("|")): v for k, v in raw.items()}
    return DEFAULT_TARGETS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--cutoff", required=True, help="Parent published cutoff (e.g. 2026-01-01).")
    ap.add_argument("--out-suffix", default="-gold",
                    help="Output dir = benchmark/data/{cutoff}{suffix}/.")
    ap.add_argument("--min-articles", type=int, default=8)
    ap.add_argument("--min-distinct-days", type=int, default=5)
    ap.add_argument("--horizon-days", type=int, default=14)
    ap.add_argument("--min-avg-chars", type=int, default=1500)
    ap.add_argument("--min-source-diversity", type=int, default=3)
    ap.add_argument("--keep-unknown", action="store_true", help="Keep fd_type=unknown FDs.")
    ap.add_argument("--target-quotas-json", default=None,
                    help='JSON string mapping "benchmark|fd_type" -> int target.')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute selection but do not write the gold folder.")
    ap.add_argument("--report-only", action="store_true",
                    help="Print headline counts then exit.")
    args = ap.parse_args()

    parent_dir = ROOT / "benchmark" / "data" / args.cutoff
    fc_in = parent_dir / "forecasts.jsonl"
    art_in = parent_dir / "articles.jsonl"
    if not fc_in.exists() or not art_in.exists():
        print(f"[ERROR] parent missing: {parent_dir}")
        return 1

    print(f"[gold] parent: {parent_dir}")
    fds = [json.loads(l) for l in open(fc_in, encoding="utf-8")]
    arts = {a["id"]: a for a in (json.loads(l) for l in open(art_in, encoding="utf-8")) if "id" in a}
    print(f"[gold] parent: {len(fds)} FDs, {len(arts)} articles")

    drop_audit = collections.Counter()
    kept_audit: dict[str, str] = {}
    kept: list[dict] = []
    for fd in fds:
        ok, reason = _select_fd(fd, arts, args)
        if ok:
            kept.append(fd)
            kept_audit[fd["id"]] = reason
        else:
            drop_audit[reason] += 1

    print(f"[gold] after filter cascade: {len(kept)} FDs")
    print(f"[gold] drop reasons (top 15):")
    for r, c in drop_audit.most_common(15):
        print(f"    {r}: {c}")

    targets = _resolve_target_quotas(args)
    sampled, used_strata = _stratified_sample(kept, targets, seed=args.seed)
    print(f"[gold] after stratified sampling: {len(sampled)} FDs")
    for s in used_strata:
        print(f"    {s}")

    if args.report_only:
        return 0

    referenced_ids = set()
    for fd in sampled:
        for aid in fd.get("article_ids") or []:
            referenced_ids.add(aid)
    sampled_arts = [arts[a] for a in referenced_ids if a in arts]
    print(f"[gold] referenced articles: {len(sampled_arts)}")

    if args.dry_run:
        print("[gold] dry-run; not writing.")
        return 0

    out_dir = ROOT / "benchmark" / "data" / f"{args.cutoff}{args.out_suffix}"
    parent_sha = _sha256(fc_in)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()

    audit = {
        "parent_n_fds": len(fds),
        "parent_n_articles": len(arts),
        "kept_after_filter": len(kept),
        "kept_after_sample": len(sampled),
        "drop_reasons": dict(drop_audit),
        "kept_sample_ids": [fd["id"] for fd in sampled],
    }

    _write_self_contained(out_dir, sampled, sampled_arts, parent_dir, args, audit,
                          used_strata, parent_sha, git_sha)
    print(f"[gold] wrote self-contained gold subset to {out_dir}")
    print(f"[gold] verify: cd {out_dir} && python examples/validate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
