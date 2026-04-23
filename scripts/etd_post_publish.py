"""ETD post-publish orchestrator: run the full ETD pipeline against a
freshly-published v2.1 benchmark in one CLI call.

Sequences:
  1. Compute delta = (articles in benchmark/data/{cutoff}/articles.jsonl)
                     minus (articles already covered by data/etd/facts.v1.jsonl).
     Writes data/etd/audit/delta_{cutoff}.jsonl.
  2. Phase D: launch articles_to_facts.py on the delta with the v3 prompt
     and --strict-dates. OpenAI Batch API; resumable; safe to interrupt.
     SKIPPED if --skip-extract or no delta articles.
  3. Stage 2: scripts/etd_dedup.py over the union (Stage-1 + recovered +
     delta). GPU-bound; ~5-10 min on RTX 2060 for ~80-90k facts.
     SKIPPED if --skip-dedup.
  4. Stage 3: scripts/etd_link.py --cutoff {cutoff}. CPU-only; ~2 min.
  5. Apply etd_filter.py with the production preset (medium+ confidence,
     asserted polarity, no future, linked, source-blocklist for known
     stale-republish outlets). Output: facts.v1_production.jsonl.
  6. Run etd_audit.py on the production set. Logs to
     data/etd/audit/audit_post_publish_{cutoff}.md.
  7. Optional: run etd_compare_facts_vs_articles.py per benchmark
     (--n 50 --bench {forecastbench,gdelt-cameo,earnings}) to validate
     forecasting parity. Costs ~$0.05 / benchmark.

Each stage is independently skippable so the orchestrator can be re-run
to pick up where a previous run failed.

Usage:
  python scripts/etd_post_publish.py --cutoff 2026-01-01
  python scripts/etd_post_publish.py --cutoff 2026-01-01 --skip-extract --skip-dedup
  python scripts/etd_post_publish.py --cutoff 2026-01-01 --skip-compare
  python scripts/etd_post_publish.py --cutoff 2026-01-01 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ETD_DIR = DATA / "etd"
AUDIT_DIR = ETD_DIR / "audit"
PY = sys.executable
SCRIPTS = ROOT / "scripts"

DEFAULT_BLOCKLIST = "news.fjsen.com,world.people.com.cn"


def _log(msg: str) -> None:
    print(f"[post_publish] {msg}", flush=True)


def _atomic_write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _run(cmd: list, dry_run: bool = False) -> int:
    pretty = " ".join(str(c) for c in cmd)
    _log(f"RUN: {pretty}")
    if dry_run:
        return 0
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())
    _log(f"     -> exit={rc}  ({time.time()-t0:.1f}s)")
    return rc


def _step_compute_delta(cutoff: str, out_path: Path, dry_run: bool) -> int:
    """Articles in the published bundle minus articles already covered by
    Stage-1 facts. Writes a JSONL of {id, ...} for use with
    `articles_to_facts.py --only-articles`."""
    fd_path = ROOT / "benchmark" / "data" / cutoff / "articles.jsonl"
    facts_path = ETD_DIR / "facts.v1.jsonl"
    if not fd_path.exists():
        _log(f"[ERROR] published articles missing: {fd_path}; build the benchmark first")
        return 1
    if not facts_path.exists():
        _log(f"[ERROR] Stage-1 facts missing: {facts_path}")
        return 1

    if dry_run:
        _log(f"[delta] would compute {fd_path} \\ {facts_path} -> {out_path}")
        return 0

    covered: set[str] = set()
    for line in open(facts_path, encoding="utf-8"):
        try:
            d = json.loads(line)
            pid = d.get("primary_article_id")
            if pid: covered.add(pid)
        except json.JSONDecodeError:
            continue

    delta_records: list[dict] = []
    n_total = 0
    for line in open(fd_path, encoding="utf-8"):
        try:
            a = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_total += 1
        if a.get("id") and a["id"] not in covered:
            delta_records.append({"id": a["id"], "url": a.get("url"),
                                   "title": a.get("title"),
                                   "publish_date": a.get("publish_date") or a.get("date")})

    _atomic_write_jsonl(out_path, delta_records)
    _log(f"[delta] published articles: {n_total}")
    _log(f"[delta] already covered:    {sum(1 for x in covered)}")
    _log(f"[delta] new to extract:     {len(delta_records)} -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="Published benchmark cutoff (YYYY-MM-DD).")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip Phase D delta extract (use existing Stage-1 facts).")
    ap.add_argument("--skip-dedup",   action="store_true",
                    help="Skip Stage 2 etd_dedup.py.")
    ap.add_argument("--skip-link",    action="store_true",
                    help="Skip Stage 3 etd_link.py.")
    ap.add_argument("--skip-filter",  action="store_true",
                    help="Skip etd_filter.py production-preset apply.")
    ap.add_argument("--skip-audit",   action="store_true",
                    help="Skip etd_audit.py on the production set.")
    ap.add_argument("--skip-compare", action="store_true",
                    help="Skip etd_compare_facts_vs_articles.py per benchmark.")
    ap.add_argument("--source-blocklist", default=DEFAULT_BLOCKLIST,
                    help=f"Pass through to etd_filter.py (default: {DEFAULT_BLOCKLIST}).")
    ap.add_argument("--compare-n", type=int, default=50,
                    help="Sample size per benchmark for facts-vs-articles compare.")
    ap.add_argument("--prompt", default="docs/prompts/etd_extraction_v3.txt",
                    help="Stage-1 prompt for delta extract.")
    ap.add_argument("--chunk-size", type=int, default=5000,
                    help="OpenAI Batch chunk size for delta extract.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing.")
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    delta_path = AUDIT_DIR / f"delta_{args.cutoff}.jsonl"

    # ----- 1. delta -----
    rc = _step_compute_delta(args.cutoff, delta_path, args.dry_run)
    if rc != 0:
        return rc

    # ----- 2. Phase D delta extract -----
    if not args.skip_extract:
        if not args.dry_run and not delta_path.exists():
            _log("[extract] no delta file; skipping")
        else:
            n = sum(1 for _ in open(delta_path, encoding="utf-8")) if delta_path.exists() else 0
            if n == 0 and not args.dry_run:
                _log("[extract] delta is empty; skipping Phase D")
            else:
                cmd = [PY, str(SCRIPTS / "articles_to_facts.py"),
                       "--prompt", args.prompt,
                       "--strict-dates",
                       "--only-articles", str(delta_path),
                       "--run-id", f"delta_{args.cutoff.replace('-','')}",
                       "--chunk-size", str(args.chunk_size)]
                rc = _run(cmd, args.dry_run)
                if rc != 0:
                    _log(f"[extract] failed (exit {rc}); halting orchestrator")
                    return rc
    else:
        _log("[extract] --skip-extract")

    # ----- 3. Stage 2 dedup -----
    if not args.skip_dedup:
        rc = _run([PY, str(SCRIPTS / "etd_dedup.py")], args.dry_run)
        if rc != 0:
            _log(f"[dedup] failed (exit {rc}); halting")
            return rc
    else:
        _log("[dedup] --skip-dedup")

    # ----- 4. Stage 3 link -----
    if not args.skip_link:
        rc = _run([PY, str(SCRIPTS / "etd_link.py"), "--cutoff", args.cutoff], args.dry_run)
        if rc != 0:
            _log(f"[link] failed (exit {rc}); halting")
            return rc
    else:
        _log("[link] --skip-link")

    # ----- 5. Production filter -----
    if not args.skip_filter:
        production_out = ETD_DIR / f"facts.v1_production_{args.cutoff}.jsonl"
        cmd = [PY, str(SCRIPTS / "etd_filter.py"),
               "--in", str(ETD_DIR / "facts.v1_linked.jsonl") if (ETD_DIR / "facts.v1_linked.jsonl").exists()
                       else str(ETD_DIR / "facts.v1.jsonl"),
               "--out", str(production_out),
               "--source-blocklist", args.source_blocklist,
               "--min-confidence", "high",
               "--polarity", "asserted",
               "--no-future"]
        if not args.skip_link:
            cmd.append("--require-linked-fd")
        rc = _run(cmd, args.dry_run)
        if rc != 0:
            _log(f"[filter] failed (exit {rc}); halting")
            return rc
    else:
        _log("[filter] --skip-filter")

    # ----- 6. Audit production set -----
    if not args.skip_audit:
        production_out = ETD_DIR / f"facts.v1_production_{args.cutoff}.jsonl"
        if production_out.exists() or args.dry_run:
            audit_out = AUDIT_DIR / f"audit_post_publish_{args.cutoff}.md"
            rc = _run([PY, str(SCRIPTS / "etd_audit.py"),
                       "--in", str(production_out),
                       "--out", str(audit_out)], args.dry_run)
            if rc != 0:
                _log(f"[audit] failed (exit {rc}); continuing")
    else:
        _log("[audit] --skip-audit")

    # ----- 7. Facts-vs-articles compare per benchmark -----
    if not args.skip_compare:
        for bench in ("forecastbench", "gdelt-cameo", "earnings"):
            cmd = [PY, str(SCRIPTS / "etd_compare_facts_vs_articles.py"),
                   "--cutoff", args.cutoff,
                   "--n", str(args.compare_n),
                   "--bench", bench,
                   "--out", str(AUDIT_DIR / f"facts_vs_articles_{bench}_{args.cutoff}.md"),
                   "--diff-out", str(AUDIT_DIR / f"facts_vs_articles_{bench}_{args.cutoff}_diffs.jsonl")]
            rc = _run(cmd, args.dry_run)
            if rc != 0:
                _log(f"[compare/{bench}] failed (exit {rc}); continuing")
    else:
        _log("[compare] --skip-compare")

    _log(f"DONE. cutoff={args.cutoff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
