"""
Single-entry orchestrator for building the EMR-ACH Forecast Dossier benchmark.

Reads `configs/default_config.yaml` for all construction parameters, optionally
overrides `model_cutoff` from the CLI, and produces a final FD bundle at
`{output_root}/{model_cutoff}/`.

Usage:
  # Default: GPT-4o cutoff from config
  python scripts/build_benchmark.py

  # Different cutoff
  python scripts/build_benchmark.py --cutoff 2025-04-01

  # Subset of benchmarks only
  python scripts/build_benchmark.py --benchmarks forecastbench,earnings

  # Skip raw-data rebuild steps (use when intermediate files already exist)
  python scripts/build_benchmark.py --skip-raw

  # Dry-run: print commands without executing
  python scripts/build_benchmark.py --dry-run

Pipeline steps (always run in order):
  1. Per-benchmark raw build (MIRAI KG download + filter; earnings via yfinance;
     ForecastBench uses the static repo clone — no raw build, just filter)
  2. Per-benchmark article-text fetch (trafilatura for MIRAI + ForecastBench
     URLs; earnings news fetch TBD)
  3. unify_articles.py -> data/unified/articles.jsonl
  4. unify_forecasts.py -> data/unified/forecasts.jsonl
  5. compute_relevance.py per benchmark (SBERT cross-match within-scope)
  6. relink_gdelt_context.py (MIRAI-specific: replace oracle event articles
     with pre-event context)
  7. quality_filter.py --model-cutoff X --cutoff-buffer-days Y
  8. diagnostic_report.py
  9. build_eda_report.py
  10. Copy final artifacts to {output_root}/{cutoff}/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
CONFIG_DEFAULT = ROOT / "configs" / "default_config.yaml"
UNIFIED = ROOT / "data" / "unified"
STAGED  = ROOT / "data" / "staged"   # snapshot root: data/staged/{run_id}/stepNN_{name}/

# Hardlink threshold: files >= this go via os.link() (zero-cost) instead of copy
SNAPSHOT_HARDLINK_BYTES = 20_000_000   # 20 MB

_RUN_ID_CURRENT: str = ""   # set by start_run_id(), used by snapshot()


def start_run_id() -> str:
    """Generate a run_id and create the staging folder. Idempotent."""
    global _RUN_ID_CURRENT
    if not _RUN_ID_CURRENT:
        _RUN_ID_CURRENT = datetime.now().strftime("%Y%m%d_%H%M%S")
        (STAGED / _RUN_ID_CURRENT).mkdir(parents=True, exist_ok=True)
    return _RUN_ID_CURRENT


def snapshot(step_name: str, dry_run: bool = False) -> Path | None:
    """Copy every file in data/unified/ into data/staged/{run_id}/{step_name}/.
    Large files (>=20 MB, e.g. articles.jsonl, *.npy) are hardlinked for zero-cost
    preservation; small files are fully copied. Previous snapshots are never
    modified — nothing overwrites across step_names.
    """
    if dry_run:
        log(f"[snapshot] {step_name} (dry-run, skipped)")
        return None
    if not UNIFIED.exists():
        return None
    run_id = start_run_id()
    dst = STAGED / run_id / step_name
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    linked = 0
    for src in UNIFIED.iterdir():
        if not src.is_file():
            continue
        tgt = dst / src.name
        if tgt.exists():
            continue  # step_name already snapshotted
        try:
            if src.stat().st_size >= SNAPSHOT_HARDLINK_BYTES:
                os.link(src, tgt)
                linked += 1
            else:
                shutil.copy2(src, tgt)
                copied += 1
        except OSError:
            # cross-volume or filesystem limitation — fall back to copy
            try:
                shutil.copy2(src, tgt)
                copied += 1
            except Exception:
                pass
    log(f"[snapshot] {step_name}: copied={copied} hardlinked={linked} -> {dst.relative_to(ROOT)}")
    return dst

PY = sys.executable or "python"


def log(msg: str):
    print(f"[build_benchmark] {msg}", flush=True)


def run_cmd(cmd: list[str], env_overrides: dict = None, dry_run: bool = False) -> int:
    """Run a subprocess, merging env_overrides onto os.environ."""
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    log(f"RUN: {' '.join(cmd)}")
    if env_overrides:
        log(f"  env overrides: {env_overrides}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def step_gdelt_cameo(cfg_bm: dict, skip_raw: bool, dry_run: bool) -> None:
    if not cfg_bm.get("enabled", True):
        log("GDELT-CAMEO disabled in config; skipping")
        return
    env = {
        "GDELT_CONTEXT_START":     cfg_bm["context_start"],
        "GDELT_CONTEXT_END":       cfg_bm["context_end"],
        "GDELT_TEST_MONTH":        cfg_bm["test_month"],
        "GDELT_ALL_END":           cfg_bm["all_end"],
        "GDELT_MIN_DAILY_MENTIONS": cfg_bm.get("min_daily_mentions", 50),
        "GDELT_MAX_DOWNLOAD_WORKERS": cfg_bm.get("max_download_workers", 16),
    }
    if skip_raw:
        log("--skip-raw: assuming GDELT-CAMEO data_news.csv + data_kg.csv + relation_query.csv already present")
    else:
        log("GDELT-CAMEO: running build.py (all 5 steps). This may take ~60 min.")
        run_cmd([PY, str(SCRIPTS / "gdelt_cameo" / "build.py"), "--steps", "1,2,3,4,5"],
                env_overrides=env, dry_run=dry_run)
    # Full-text fetch is invoked from main() AFTER relink_gdelt_context.py
    # populates article_ids with the pre-event context articles. See
    # step_gdelt_text_fetch() below.
    log("GDELT-CAMEO: initial raw build done; full-text fetch will run after relink.")


def step_gdelt_text_fetch(dry_run: bool) -> None:
    """Fetch full article text for MIRAI FDs (only URLs referenced by FDs).
    Must run AFTER relink_gdelt_context.py populates article_ids with
    pre-event context (oracle-event articles are replaced by then)."""
    run_cmd([PY, str(SCRIPTS / "gdelt_cameo" / "fetch_text.py"),
             "--only-referenced", "--workers", "24"],
            dry_run=dry_run)


def step_fb_text_fetch(dry_run: bool) -> None:
    """Fetch full article text for ForecastBench GDELT URLs using trafilatura.
    Reuses existing cache if present; only fetches missing URLs."""
    run_cmd([PY, str(SCRIPTS / "forecastbench" / "fetch_article_text.py"), "--all",
             "--workers", "24"],
            dry_run=dry_run)


def step_forecastbench(cfg_bm: dict, skip_raw: bool, dry_run: bool) -> None:
    if not cfg_bm.get("enabled", True):
        log("ForecastBench disabled; skipping")
        return
    log("ForecastBench: using static repo clone (no raw rebuild). Filter applied in unify step.")
    # If a download_forecastbench.py exists, --skip-raw still bypasses it
    dl = SCRIPTS / "forecastbench" / "download_forecastbench.py"
    if dl.exists() and not skip_raw:
        run_cmd([PY, str(dl)], dry_run=dry_run)


def step_earnings(cfg_bm: dict, skip_raw: bool, dry_run: bool) -> None:
    if not cfg_bm.get("enabled", True):
        log("Earnings disabled; skipping")
        return
    if skip_raw:
        log("--skip-raw: assuming data/earnings/earnings_forecasts.jsonl already exists")
        return
    cmd = [PY, str(SCRIPTS / "earnings" / "build.py"),
           "--start", cfg_bm["start"], "--end", cfg_bm["end"],
           "--threshold", str(cfg_bm.get("threshold", 0.02))]
    if cfg_bm.get("tickers"):
        tickers = cfg_bm["tickers"]
        cmd += ["--tickers", ",".join(tickers) if isinstance(tickers, list) else tickers]
    run_cmd(cmd, dry_run=dry_run)


def step_unify(dry_run: bool) -> None:
    run_cmd([PY, str(SCRIPTS / "common" / "unify_articles.py")],    dry_run=dry_run)
    run_cmd([PY, str(SCRIPTS / "common" / "unify_forecasts.py")],   dry_run=dry_run)


def step_relink_gdelt(enabled: bool, dry_run: bool) -> None:
    if enabled:
        run_cmd([PY, str(SCRIPTS / "gdelt_cameo" / "relink_context.py")], dry_run=dry_run)


def step_annotate_prior_state(benchmarks: list[str], dry_run: bool) -> None:
    """Annotate every FD with prior_state + fd_type (stability/change) and
    promote ground_truth to the Comply/Surprise binary primary target.

    v2.1 framing: domain multiclass (Peace/Tension/Violence, Beat/Meet/Miss,
    Yes/No) is preserved as `x_multiclass_*` secondary ablation fields; the
    FD's primary `hypothesis_set` becomes `["Comply", "Surprise"]`. No new
    API calls; reads data/gdelt_cameo/data_kg.csv and data/earnings/* for
    the per-benchmark prior-state heuristics. See
    benchmark/scripts/common/annotate_prior_state.py.
    """
    run_cmd([PY, str(SCRIPTS / "common" / "annotate_prior_state.py"),
             "--benchmarks", ",".join(benchmarks)],
            dry_run=dry_run)


def step_relevance(benchmark: str, rebuild: bool, dry_run: bool) -> None:
    cmd = [PY, str(SCRIPTS / "common" / "compute_relevance.py"),
           "--benchmark-filter", benchmark]
    if rebuild:
        cmd.append("--rebuild")
    run_cmd(cmd, dry_run=dry_run)


def step_relevance_parallel(benchmarks: list[str], rebuild: bool,
                            dry_run: bool) -> None:
    """Run compute_relevance.py for each benchmark SEQUENTIALLY.
    (Name kept for backward compat.) Earlier version dispatched concurrently
    via ThreadPoolExecutor, but both subprocesses read/write the same
    `data/unified/forecasts.jsonl` + `article_embeddings.npy` + `.fp.txt`,
    producing a write-write race where the last run silently overwrote the
    first. With incremental SBERT caching (per-row fingerprints) the second
    pass is cheap; sequentialness costs ~1-2 min total, not 7."""
    for b in benchmarks:
        step_relevance(b, rebuild, dry_run)


def step_quality(cutoff: str, buffer_days: int, quality_cfg: dict, dry_run: bool) -> None:
    cmd = [PY, str(SCRIPTS / "common" / "quality_filter.py"),
           "--model-cutoff", cutoff,
           "--cutoff-buffer-days", str(buffer_days),
           "--min-arts",    str(quality_cfg.get("min_articles", 3)),
           "--min-days",    str(quality_cfg.get("min_distinct_days", 2)),
           "--min-chars",   str(quality_cfg.get("min_chars", 1500)),
           "--min-q-chars", str(quality_cfg.get("min_question_chars", 20))]
    run_cmd(cmd, dry_run=dry_run)


def step_diagnostic(dry_run: bool) -> None:
    run_cmd([PY, str(SCRIPTS / "common" / "diagnostic_report.py")], dry_run=dry_run)


def step_eda(dry_run: bool) -> None:
    run_cmd([PY, str(SCRIPTS / "common" / "build_eda_report.py")], dry_run=dry_run)


def step_publish(cutoff: str, output_root: Path, dry_run: bool) -> Path:
    """Publish two SEPARATE folder hierarchies:

      {output_root}/{cutoff}/              <- SHIPPED DELIVERABLE (clean data only)
        ├── forecasts.jsonl                    primary deliverable
        ├── articles.jsonl                     primary deliverable (subset to FD-referenced)
        ├── benchmark.yaml                     effective config used (reproducibility)
        └── build_manifest.json                timestamp + git sha (provenance)

      {output_root}/../audit/{cutoff}/      <- AUDIT MATERIAL (diagnostics, never shipped)
        ├── eda_report.html
        ├── diagnostic_report.md / .json
        ├── quality_meta.json
        └── relevance_meta.json

    Nothing else leaves the build pipeline.
    """
    dest = output_root / cutoff
    # audit tree sits as a sibling of the data tree: e.g. benchmark/audit/{cutoff}/
    audit_dir = output_root.parent / "audit" / cutoff
    log(f"Publishing deliverable to {dest}")
    log(f"Publishing audit to        {audit_dir}")
    if dry_run:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    # back-compat: the old meta_dir name still accessible but is now empty/unused
    meta_dir = audit_dir

    # 1. forecasts.jsonl — rename from internal filtered name for cleaner ship
    src_fc = UNIFIED / "forecasts_filtered.jsonl"
    if src_fc.exists():
        shutil.copy2(src_fc, dest / "forecasts.jsonl")

    # 2. articles.jsonl — subset to only those referenced by the filtered FDs
    src_arts = UNIFIED / "articles.jsonl"
    out_arts = dest / "articles.jsonl"
    referenced = set()
    if (dest / "forecasts.jsonl").exists():
        with open(dest / "forecasts.jsonl", encoding="utf-8") as f:
            for line in f:
                referenced.update(json.loads(line).get("article_ids", []))
    if src_arts.exists():
        kept = 0
        with open(out_arts, "w", encoding="utf-8") as out, \
             open(src_arts, encoding="utf-8") as src:
            for line in src:
                try:
                    if json.loads(line).get("id") in referenced:
                        out.write(line)
                        kept += 1
                except Exception:
                    continue
        log(f"articles.jsonl: {kept} articles (of {len(referenced)} referenced)")
    else:
        out_arts.write_text("", encoding="utf-8")
        log("articles.jsonl: empty (source pool missing)")

    # 3. meta/ — audit material (useful but not primary benchmark data)
    meta_files = [
        "eda_report.html",
        "diagnostic_report.md",
        "diagnostic_report.json",
        "quality_meta.json",
        "relevance_meta.json",
        "gdelt_cameo_relink_meta.json",
    ]
    for fname in meta_files:
        src = UNIFIED / fname
        if src.exists():
            shutil.copy2(src, meta_dir / fname)
    log(f"meta/: {len(list(meta_dir.iterdir()))} audit files")
    return dest


def write_run_manifest(dest: Path, cfg: dict, cutoff: str, benchmarks: list[str],
                        effective_cfg: dict) -> None:
    """Write two files: build_manifest.json (run metadata) and benchmark.yaml
    (the effective config used — a drop-in to reproduce this exact benchmark)."""
    # Try to capture the git sha at build time
    git_sha = ""
    try:
        import subprocess as _sp
        git_sha = _sp.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=_sp.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    manifest = {
        "generated_at":        datetime.now().isoformat(timespec="seconds"),
        "model_cutoff":        cutoff,
        "cutoff_buffer_days":  effective_cfg.get("cutoff_buffer_days", 0),
        "benchmarks_included": benchmarks,
        "git_sha":             git_sha,
        "python":              PY,
    }
    (dest / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Effective config — exactly what to pass to build_benchmark.py to reproduce.
    # model_cutoff is forced to the resolved value (overrides any default in the
    # template file).
    effective_cfg = {**effective_cfg, "model_cutoff": cutoff}
    yaml_header = (
        f"# Benchmark build config used to generate the sibling files.\n"
        f"# Reproduce with:\n"
        f"#   python scripts/build_benchmark.py --config <path-to-this-file>\n"
        f"# Generated: {manifest['generated_at']}  (git: {git_sha[:12] or 'unknown'})\n\n"
    )
    (dest / "benchmark.yaml").write_text(
        yaml_header + yaml.safe_dump(effective_cfg, sort_keys=False),
        encoding="utf-8")

    log(f"Wrote build manifest: {dest / 'build_manifest.json'}")
    log(f"Wrote effective config: {dest / 'benchmark.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_DEFAULT),
                    help="Path to YAML config (default: configs/default_config.yaml)")
    ap.add_argument("--cutoff", default=None,
                    help="Model training cutoff (YYYY-MM-DD). Overrides config.model_cutoff.")
    ap.add_argument("--benchmarks", default=None,
                    help="Comma-separated subset: forecastbench,gdelt_cameo,earnings. "
                         "Default: all enabled in config.")
    ap.add_argument("--skip-raw", action="store_true",
                    help="Skip raw-data rebuild (use when intermediate files already exist)")
    ap.add_argument("--rebuild-embeddings", action="store_true",
                    help="Pass --rebuild to compute_relevance.py (re-embeds all articles)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    ap.add_argument("--fresh", action="store_true",
                    help="Wipe data/unified/ and per-benchmark raw outputs before "
                         "rebuilding. Implies NOT --skip-raw. Use when you want a "
                         "truly from-scratch reproduction.")
    ap.add_argument("--keep-staging", action="store_true",
                    help="Keep data/unified/ (pipeline staging) after a successful "
                         "build. Default: wipe staging once the deliverable is "
                         "published.")
    ap.add_argument("--keep-snapshots", dest="keep_snapshots",
                    action="store_true", default=True,
                    help="Keep data/staged/ versioned snapshots (default).")
    ap.add_argument("--no-keep-snapshots", dest="keep_snapshots",
                    action="store_false",
                    help="Wipe data/staged/ versioned snapshots after publish.")
    args = ap.parse_args()

    if args.fresh:
        if args.skip_raw:
            log("[WARN] --fresh overrides --skip-raw")
            args.skip_raw = False
        from shutil import rmtree
        for wipe in [
            ROOT / "data" / "unified",
            ROOT / "data" / "earnings",
            ROOT / "data" / "gdelt_cameo" / "kg_raw",
            ROOT / "data" / "gdelt_cameo" / "kg_tmp",
            ROOT / "data" / "gdelt_cameo" / "test",
        ]:
            if wipe.exists() and not args.dry_run:
                rmtree(wipe, ignore_errors=True)
                log(f"--fresh: wiped {wipe}")
        for wipe_file in [
            ROOT / "data" / "gdelt_cameo" / "data_kg.csv",
            ROOT / "data" / "gdelt_cameo" / "data_news.csv",
            ROOT / "data" / "gdelt_cameo" / "data_news_full.csv",
        ]:
            if wipe_file.exists() and not args.dry_run:
                wipe_file.unlink()
                log(f"--fresh: removed {wipe_file}")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cutoff = args.cutoff or cfg["model_cutoff"]
    try:
        datetime.strptime(cutoff, "%Y-%m-%d")
    except ValueError:
        log(f"[ERROR] --cutoff '{cutoff}' must be YYYY-MM-DD")
        sys.exit(1)

    buffer_days = int(cfg.get("cutoff_buffer_days", 0))
    output_root = ROOT / cfg.get("output", {}).get("root", "benchmark/data")

    all_benchmarks = ["forecastbench", "gdelt_cameo", "earnings"]
    chosen = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else all_benchmarks
    for b in chosen:
        if b not in all_benchmarks:
            log(f"[ERROR] unknown benchmark '{b}'. Pick from {all_benchmarks}.")
            sys.exit(1)

    log(f"model_cutoff = {cutoff}  (buffer = {buffer_days}d)")
    log(f"benchmarks   = {chosen}")
    log(f"output_root  = {output_root}")
    log(f"dry_run      = {args.dry_run}")
    log("-" * 60)

    # 1. Raw builds
    if "forecastbench" in chosen:
        step_forecastbench(cfg["benchmarks"]["forecastbench"], args.skip_raw, args.dry_run)
    if "gdelt_cameo" in chosen:
        step_gdelt_cameo(cfg["benchmarks"]["gdelt_cameo"], args.skip_raw, args.dry_run)
    if "earnings" in chosen:
        step_earnings(cfg["benchmarks"]["earnings"], args.skip_raw, args.dry_run)

    # 2. Unify (first pass — article pool has titles only for gdelt_cameo)
    step_unify(args.dry_run)
    snapshot("01_after_first_unify", args.dry_run)

    # 3. Relevance (runs the two benchmark-filter passes IN PARALLEL —
    # see step_relevance_parallel). Saves ~7 min per build.
    rel_targets = [b for b in ("forecastbench", "earnings") if b in chosen]
    step_relevance_parallel(rel_targets, args.rebuild_embeddings, args.dry_run)
    snapshot("02_after_first_relevance", args.dry_run)

    # 4. GDELT-CAMEO context relink (must come BEFORE quality_filter since it
    # changes article_ids to pre-event context articles)
    if "gdelt_cameo" in chosen:
        step_relink_gdelt(True, args.dry_run)
        snapshot("03_after_first_relink", args.dry_run)
        # Fetch full text for referenced (relinked) articles
        step_gdelt_text_fetch(args.dry_run)
        snapshot("04_after_gdelt_text_fetch", args.dry_run)

    # 4b. Fetch ForecastBench article text (trafilatura on GDELT DOC URLs).
    if "forecastbench" in chosen:
        step_fb_text_fetch(args.dry_run)
        snapshot("05_after_fb_text_fetch", args.dry_run)

    # Re-run unify + relevance so newly-fetched text participates in scoring.
    # The incremental SBERT cache makes the re-relevance fast (~1-2 min vs 7)
    # since only rows with changed text get re-embedded.
    if "gdelt_cameo" in chosen or "forecastbench" in chosen:
        step_unify(args.dry_run)
        snapshot("06_after_reunify", args.dry_run)
        step_relevance_parallel(rel_targets, args.rebuild_embeddings, args.dry_run)
        if "gdelt_cameo" in chosen:
            step_relink_gdelt(True, args.dry_run)
        snapshot("07_after_re_relevance_and_relink", args.dry_run)

    # 4c. Annotate prior_state + fd_type and promote ground_truth to the
    # Comply/Surprise v2.1 primary target. Must run AFTER the final
    # unify_forecasts.py pass (which rebuilds FD records from per-benchmark
    # source files and would wipe earlier annotations). Must run BEFORE
    # quality_filter so the partition-aware slicing (forecasts_filtered_
    # change.jsonl / _stability.jsonl) is well-defined.
    step_annotate_prior_state(chosen, args.dry_run)
    snapshot("07b_after_prior_state_annotation", args.dry_run)

    # 5. Quality filter + diagnostic + EDA
    step_quality(cutoff, buffer_days, cfg.get("quality", {}), args.dry_run)
    snapshot("08_after_quality_filter", args.dry_run)
    step_diagnostic(args.dry_run)
    step_eda(args.dry_run)
    snapshot("09_final_before_publish", args.dry_run)

    # 6. Publish to {output_root}/{cutoff}/
    dest = step_publish(cutoff, output_root, args.dry_run)
    if not args.dry_run:
        write_run_manifest(dest, cfg, cutoff, chosen, effective_cfg=cfg)

    # 7. Optional staging cleanup (Fix #6)
    if not args.dry_run:
        from shutil import rmtree
        if not args.keep_staging:
            unified_dir = ROOT / "data" / "unified"
            if unified_dir.exists():
                rmtree(unified_dir, ignore_errors=True)
                log(f"Cleaned pipeline staging: {unified_dir}")
        if not args.keep_snapshots:
            staged_dir = ROOT / "data" / "staged"
            if staged_dir.exists():
                rmtree(staged_dir, ignore_errors=True)
                log(f"Cleaned versioned snapshots: {staged_dir}")

    log("-" * 60)
    log(f"DONE. Final benchmark at: {dest}")


if __name__ == "__main__":
    main()
