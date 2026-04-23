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
  1. Per-benchmark raw build (GDELT-CAMEO KG download + filter; earnings via yfinance;
     ForecastBench uses the static repo clone — no raw build, just filter)
  2. Per-benchmark article-text fetch (trafilatura for Geopolitics + ForecastBench
     URLs; earnings news fetch TBD)
  3. unify_articles.py -> data/unified/articles.jsonl
  4. unify_forecasts.py -> data/unified/forecasts.jsonl
  5. compute_relevance.py per benchmark (SBERT cross-match within-scope)
  6. relink_gdelt_context.py (Geopolitics-specific: replace oracle event articles
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

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
CONFIG_DEFAULT = ROOT / "configs" / "default_config.yaml"
UNIFIED = ROOT / "data" / "unified"
STAGED  = ROOT / "data" / "staged"   # snapshot root: data/staged/{run_id}/stepNN_{name}/

# Hardlink threshold: files >= this go via os.link() (zero-cost) instead of copy.
# Override via pipeline.snapshot_hardlink_bytes in default_config.yaml.
SNAPSHOT_HARDLINK_BYTES = 20_000_000   # 20 MB (default)

# Single-process state. The orchestrator is strictly sequential, so a module
# global is safe here. Exposed as get/set functions so future refactors can
# drop in a context object without changing call sites.
_RUN_ID_CURRENT: str = ""


def start_run_id() -> str:
    """Generate a run_id and create the staging folder. Idempotent — calling
    it multiple times in the same process returns the same id.

    Thread-safety: NOT thread-safe. The orchestrator runs steps sequentially;
    if a future multi-process pipeline is introduced, refactor this into a
    context object passed through function params.
    """
    global _RUN_ID_CURRENT
    if not _RUN_ID_CURRENT:
        _RUN_ID_CURRENT = datetime.now().strftime("%Y%m%d_%H%M%S")
        (STAGED / _RUN_ID_CURRENT).mkdir(parents=True, exist_ok=True)
    return _RUN_ID_CURRENT


def reset_run_id() -> None:
    """Clear the cached run_id. Only used by tests and by `--fresh` rebuild."""
    global _RUN_ID_CURRENT
    _RUN_ID_CURRENT = ""


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
        run_cmd([PY, str(SCRIPTS / "build_gdelt_cameo.py"), "--steps", "1,2,3,4,5"],
                env_overrides=env, dry_run=dry_run)
    # Full-text fetch is invoked from main() AFTER relink_gdelt_context.py
    # populates article_ids with the pre-event context articles. See
    # step_gdelt_text_fetch() below.
    log("GDELT-CAMEO: initial raw build done; full-text fetch will run after relink.")


def step_gdelt_text_fetch(dry_run: bool, workers: int = 24) -> None:
    """Fetch full article text for Geopolitics (GDELT-CAMEO) FDs — only URLs referenced by FDs.
    Must run AFTER relink_gdelt_context.py populates article_ids with
    pre-event context (oracle-event articles are replaced by then).

    Args:
        dry_run: print the command without executing.
        workers: concurrent HTTP workers. Tunable via
                 `pipeline.gdelt_text_fetch_workers` in default_config.yaml.
    """
    run_cmd([PY, str(SCRIPTS / "fetch_gdelt_text.py"),
             "--only-referenced", "--workers", str(workers)],
            dry_run=dry_run)


def step_fb_text_fetch(dry_run: bool, workers: int = 24) -> None:
    """Fetch full article text for ForecastBench GDELT URLs using trafilatura.
    Reuses existing cache if present; only fetches missing URLs.

    Args:
        dry_run: print the command without executing.
        workers: concurrent HTTP workers. Tunable via
                 `pipeline.text_fetch_workers` in default_config.yaml.
    """
    run_cmd([PY, str(SCRIPTS / "fetch_article_text.py"), "--all",
             "--workers", str(workers)],
            dry_run=dry_run)


def step_multi_source_news(cfg_bm_all: dict, dry_run: bool) -> None:
    """Fetch per-FD news from NYT / Guardian / Google News / GDELT DOC / Finnhub
    across all three benchmarks, using the unified 90d analysis window.
    Implements the 'forecast event outcome from prior news' task.
    Idempotent: each fetcher honors --skip-completed so reruns are cheap."""
    # ForecastBench
    run_cmd([PY, str(SCRIPTS / "fetch_forecastbench_news.py"),
             "--source", "all", "--lookback", "90", "--skip-completed"],
            dry_run=dry_run)
    # GDELT-CAMEO (pre-event news per actor pair; replaces same-day oracle)
    run_cmd([PY, str(SCRIPTS / "fetch_gdelt_cameo_news.py"),
             "--source", "all", "--lookback", "90", "--skip-completed"],
            dry_run=dry_run)
    # Earnings (Finnhub + others per ticker)
    run_cmd([PY, str(SCRIPTS / "fetch_earnings_news.py"),
             "--source", "all", "--lookback", "90", "--skip-completed"],
            dry_run=dry_run)


def step_forecastbench(cfg_bm: dict, skip_raw: bool, dry_run: bool) -> None:
    if not cfg_bm.get("enabled", True):
        log("ForecastBench disabled; skipping")
        return
    log("ForecastBench: using static repo clone (no raw rebuild). Filter applied in unify step.")
    # If a download_forecastbench.py exists, --skip-raw still bypasses it
    dl = SCRIPTS / "download_forecastbench.py"
    if dl.exists() and not skip_raw:
        run_cmd([PY, str(dl)], dry_run=dry_run)


def step_earnings(cfg_bm: dict, skip_raw: bool, dry_run: bool) -> None:
    if not cfg_bm.get("enabled", True):
        log("Earnings disabled; skipping")
        return
    if skip_raw:
        log("--skip-raw: assuming data/earnings/earnings_forecasts.jsonl already exists")
        return
    cmd = [PY, str(SCRIPTS / "build_earnings_benchmark.py"),
           "--start", cfg_bm["start"], "--end", cfg_bm["end"],
           "--threshold", str(cfg_bm.get("threshold", 0.02))]
    if cfg_bm.get("tickers"):
        tickers = cfg_bm["tickers"]
        cmd += ["--tickers", ",".join(tickers) if isinstance(tickers, list) else tickers]
    run_cmd(cmd, dry_run=dry_run)


def step_unify(dry_run: bool, horizons: dict | None = None) -> None:
    """Run article + forecast unifiers. `horizons` dict, when provided, is
    passed as env vars so unify_forecasts.py uses the configured horizon per
    benchmark (no hardcoded defaults)."""
    env = {}
    if horizons:
        if "forecastbench" in horizons:
            env["EMRACH_FB_HORIZON_DAYS"] = str(horizons["forecastbench"])
        if "gdelt_cameo" in horizons:
            env["EMRACH_GDELT_HORIZON_DAYS"] = str(horizons["gdelt_cameo"])
        if "earnings" in horizons:
            env["EMRACH_EARN_HORIZON_DAYS"] = str(horizons["earnings"])
    run_cmd([PY, str(SCRIPTS / "unify_articles.py")],   dry_run=dry_run)
    run_cmd([PY, str(SCRIPTS / "unify_forecasts.py")],
            env_overrides=env or None, dry_run=dry_run)


def step_relink_gdelt(enabled: bool, dry_run: bool) -> None:
    if enabled:
        run_cmd([PY, str(SCRIPTS / "relink_gdelt_context.py")], dry_run=dry_run)


def step_relevance(benchmark: str, rebuild: bool, dry_run: bool,
                   embedder: str = "sbert", openai_model: str = "text-embedding-3-small",
                   openai_mode: str = "batch") -> None:
    cmd = [PY, str(SCRIPTS / "compute_relevance.py"),
           "--benchmark-filter", benchmark]
    if rebuild:
        cmd.append("--rebuild")
    if embedder != "sbert":
        cmd += ["--embedder", embedder,
                "--openai-model", openai_model,
                "--openai-mode", openai_mode]
    run_cmd(cmd, dry_run=dry_run)


def step_relevance_parallel(benchmarks: list[str], rebuild: bool,
                            dry_run: bool, embedder: str = "sbert",
                            openai_model: str = "text-embedding-3-small",
                            openai_mode: str = "batch") -> None:
    """Run compute_relevance.py for each benchmark SEQUENTIALLY.
    (Name kept for backward compat.) Earlier version dispatched concurrently
    via ThreadPoolExecutor, but both subprocesses read/write the same
    `data/unified/forecasts.jsonl` + `article_embeddings.npy` + `.fp.txt`,
    producing a write-write race where the last run silently overwrote the
    first. With incremental SBERT caching (per-row fingerprints) the second
    pass is cheap; sequentialness costs ~1-2 min total, not 7.

    For --embedder=openai, the cost-dominant work is one OpenAI Batch upload
    of the full pool (encoded once, scored per benchmark from the cached
    .npy). Per-benchmark sequential calls still each load the cache.
    """
    for b in benchmarks:
        step_relevance(b, rebuild, dry_run, embedder, openai_model, openai_mode)


def step_quality(cutoff: str, buffer_days: int, quality_cfg: dict, dry_run: bool) -> None:
    cmd = [PY, str(SCRIPTS / "quality_filter.py"),
           "--model-cutoff", cutoff,
           "--cutoff-buffer-days", str(buffer_days),
           "--min-arts",    str(quality_cfg.get("min_articles", 3)),
           "--min-days",    str(quality_cfg.get("min_distinct_days", 2)),
           "--min-chars",   str(quality_cfg.get("min_chars", 1500)),
           "--min-q-chars", str(quality_cfg.get("min_question_chars", 20))]
    # Per-benchmark day-spread override. Under the unified "forecast from prior
    # news" design (2026-04-22), every source fetches a 90d analysis window, so
    # min_days=2 applies uniformly. The gdelt-cameo=1 oracle-era override is
    # retained as an opt-in config value only.
    mdpb = quality_cfg.get("min_distinct_days_per_benchmark", {})
    if mdpb:
        cmd.extend(["--min-days-per-benchmark",
                    ",".join(f"{k}={v}" for k, v in mdpb.items())])
    run_cmd(cmd, dry_run=dry_run)


def step_diagnostic(dry_run: bool) -> None:
    run_cmd([PY, str(SCRIPTS / "diagnostic_report.py")], dry_run=dry_run)


def step_eda(dry_run: bool) -> None:
    run_cmd([PY, str(SCRIPTS / "build_eda_report.py")], dry_run=dry_run)


def step_pool_audits(dry_run: bool) -> None:
    """Run the pool-level audits that complement build_eda_report.py:
      * articles_audit.py: spam survivors, near-dupes, source/length mix
      * fd_audit.py:       v2.1 schema integrity, Comply/Surprise balance,
                           fd_type partition consistency, prior-state coverage
    Both are CPU-only and idempotent. Reports land at
    data/unified/audit/{articles_audit.md, fd_audit.md}.
    """
    run_cmd([PY, str(SCRIPTS / "articles_audit.py")], dry_run=dry_run)
    run_cmd([PY, str(SCRIPTS / "fd_audit.py")], dry_run=dry_run)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_checksums(dest: Path, dry_run: bool) -> None:
    """Emit SHA256 sidecars for the two primary deliverables. Lets a
    consumer verify they have the exact same bytes the build produced
    without trusting the surrounding manifest."""
    if dry_run:
        log("[checksums] (dry-run)")
        return
    out = dest / "checksums.sha256"
    lines = []
    for name in ("forecasts.jsonl", "articles.jsonl"):
        p = dest / name
        if p.exists():
            lines.append(f"{_sha256(p)}  {name}")
    if not lines:
        return
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[checksums] wrote {out}")


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
    # Contract: benchmark/data/{cutoff}/ is self-contained, no intermediate data.
    # Scrub any legacy intermediate subdirs from earlier build versions.
    for legacy in ("meta", "intermediate", "staging", "audit"):
        legacy_dir = dest / legacy
        if legacy_dir.exists() and legacy_dir.is_dir():
            shutil.rmtree(legacy_dir)
            log(f"  scrubbed legacy subdir {legacy_dir} (no intermediate data in deliverable)")
    # back-compat alias: audit material goes to benchmark/audit/{cutoff}/
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
        pool_ids: set[str] = set()
        with open(out_arts, "w", encoding="utf-8") as out, \
             open(src_arts, encoding="utf-8") as src:
            for line in src:
                try:
                    aid = json.loads(line).get("id")
                    if aid is None:
                        continue
                    pool_ids.add(aid)
                    if aid in referenced:
                        out.write(line)
                        kept += 1
                except Exception:
                    continue
        log(f"articles.jsonl: {kept} articles (of {len(referenced)} referenced)")

        # v2.2 [H2] dangling-ref integrity check. FDs referencing
        # article_ids that are NOT present in the unified pool are
        # published as silently-broken records; downstream (relevance,
        # baselines, gold-subset filtering) treats them as zero-article
        # FDs. Common cause: an article-fetcher (e.g. earnings Finnhub /
        # Google News enrichment) wrote to its per-track JSONL AFTER
        # unify_articles.py built the pool but BEFORE the FD linker ran,
        # so the linker stored IDs that the pool never saw.
        dangling = referenced - pool_ids
        if dangling:
            pct = 100.0 * len(dangling) / max(1, len(referenced))
            log(f"articles.jsonl: WARN {len(dangling)} dangling refs "
                f"({pct:.1f}%); re-run unify_articles.py + link_* after "
                f"every fetcher that touches a per-track jsonl")
            # Write a diagnostic for the step_publish meta/
            dangling_path = dest / "meta" / "dangling_article_ids.txt"
            dangling_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dangling_path, "w", encoding="utf-8") as fh:
                for aid in sorted(dangling):
                    fh.write(aid + "\n")
            log(f"articles.jsonl: dangling ids listed -> {dangling_path}")
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

    # Count deliverable contents for manifest (self-contained provenance)
    fc_path = dest / "forecasts.jsonl"
    art_path = dest / "articles.jsonl"
    from collections import Counter
    bench_counts: Counter = Counter()
    n_fds = 0
    if fc_path.exists():
        with open(fc_path, encoding="utf-8") as f:
            for line in f:
                try:
                    bench_counts[json.loads(line).get("benchmark", "?")] += 1
                    n_fds += 1
                except Exception:
                    pass
    n_articles = 0
    if art_path.exists():
        with open(art_path, encoding="utf-8") as f:
            n_articles = sum(1 for _ in f)

    manifest = {
        "generated_at":        datetime.now().isoformat(timespec="seconds"),
        "model_cutoff":        cutoff,
        "cutoff_buffer_days":  effective_cfg.get("cutoff_buffer_days", 0),
        "benchmarks_included": benchmarks,
        "n_fds":               n_fds,
        "n_fds_by_benchmark":  dict(bench_counts),
        "n_articles":          n_articles,
        "git_sha":             git_sha,
        "python":              PY,
        "quality_thresholds":  effective_cfg.get("quality", {}),
        "deliverable_layout":  {
            "forecasts.jsonl":        "primary data — one FD per line",
            "articles.jsonl":         "primary data — articles referenced by FDs",
            "benchmark.yaml":         "effective build config (fully reproducible)",
            "build_manifest.json":    "this file — provenance + counts",
        },
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
    ap.add_argument("--skip-news-fetch", action="store_true",
                    help="Skip the step_multi_source_news() stage entirely. Use when "
                         "the per-benchmark news fetchers (fetch_*_news.py) have "
                         "already populated data/{forecastbench,gdelt_cameo,earnings}/ "
                         "from prior runs (cached). The pipeline proceeds straight to "
                         "annotate_prior_state + relevance using whatever articles are "
                         "currently in the unified pool.")
    ap.add_argument("--rebuild-embeddings", action="store_true",
                    help="Pass --rebuild to compute_relevance.py (re-embeds all articles)")
    ap.add_argument("--embedder", choices=["sbert", "openai"], default="sbert",
                    help="Embedding backend for the relevance step. 'sbert' uses local "
                         "GPU SentenceTransformer (default, free, ~2-3h on RTX 2060). "
                         "'openai' uses OpenAI Batch API (~$0.30 per full-pool encode, "
                         "~30-60 min wall-clock, no local GPU). Pass-through to "
                         "compute_relevance.py via --embedder.")
    ap.add_argument("--openai-model", default="text-embedding-3-small",
                    help="OpenAI embedding model when --embedder=openai.")
    ap.add_argument("--openai-mode", choices=["sync", "batch"], default="batch",
                    help="OpenAI execution mode when --embedder=openai. Batch is 50%% "
                         "cheaper but adds queue wait; sync is instant but rate-limited.")
    ap.add_argument("--horizon-days", type=int, default=None,
                    help="v2.2 forecast horizon in days. forecast_point = "
                         "resolution_date - horizon_days. Default 14 (from "
                         "config.temporal.horizon_days). Overrides YAML.")
    ap.add_argument("--lookback-days", type=int, default=None,
                    help="v2.2 article-pool lookback window in days. Fetchers "
                         "query [forecast_point - lookback_days, forecast_point]. "
                         "Default 90 (from config.temporal.lookback_days). "
                         "Overrides YAML.")
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

    # Validate config schema — catches typos / bad types BEFORE we start a rebuild.
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from src.common.config_validation import validate_config
        schema_path = ROOT / "configs" / "default_config.schema.json"
        if schema_path.exists():
            errs = validate_config(cfg, schema_path)
            if errs:
                log(f"[ERROR] {len(errs)} config validation error(s) in {args.config}:")
                for e in errs:
                    log(f"  {e}")
                sys.exit(2)
            log(f"[config] validated against {schema_path.relative_to(ROOT)}")
    except ImportError:
        log("[config] schema validator unavailable; install jsonschema to enable")

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

    # v2.2 temporal block — horizon_days / lookback_days. CLI overrides YAML.
    temporal_cfg = cfg.get("temporal", {}) or {}
    horizon_days = int(args.horizon_days if args.horizon_days is not None
                       else temporal_cfg.get("horizon_days", 14))
    lookback_days = int(args.lookback_days if args.lookback_days is not None
                        else temporal_cfg.get("lookback_days", 90))
    log(f"[temporal] horizon_days={horizon_days}  lookback_days={lookback_days}")
    # Export as env vars so downstream subprocesses (unify_forecasts.py,
    # build_earnings_benchmark.py, fetch_*_news.py, link_earnings_articles.py)
    # see the same values without needing individual CLI threading.
    os.environ["EMRACH_HORIZON_DAYS"] = str(horizon_days)
    os.environ["EMRACH_LOOKBACK_DAYS"] = str(lookback_days)

    # Build the per-benchmark forecast-horizon map from config, falling back
    # to `default_forecast_horizon_days` (14) when a specific benchmark omits it.
    # v2.2: the global temporal.horizon_days wins unless a benchmark explicitly
    # overrides it via `benchmarks.<name>.forecast_horizon_days`.
    _default_h = int(cfg.get("default_forecast_horizon_days", horizon_days))
    horizons = {
        b: int(cfg.get("benchmarks", {}).get(b, {}).get("forecast_horizon_days", _default_h))
        for b in ("forecastbench", "gdelt_cameo", "earnings")
    }
    log(f"[horizons] forecast_horizon_days: {horizons}")

    # 2. Unify (first pass — article pool has titles only for gdelt_cameo)
    step_unify(args.dry_run, horizons=horizons)
    snapshot("01_after_first_unify", args.dry_run)

    # 2b. Multi-source per-FD news fetch (unified 90d analysis window):
    #     NYT + Guardian + Google News + GDELT DOC + Finnhub (earnings).
    #     Populates data/{forecastbench,gdelt_cameo,earnings}/*_articles.jsonl
    #     for all three benchmarks. Idempotent via --skip-completed.
    if args.skip_news_fetch:
        log("--skip-news-fetch: skipping step_multi_source_news; using whatever "
            "articles are already in data/{forecastbench,gdelt_cameo,earnings}/")
    else:
        step_multi_source_news(cfg.get("benchmarks", {}), args.dry_run)
        # Re-unify so the newly fetched articles enter the unified article pool.
        step_unify(args.dry_run)
        snapshot("01b_after_multi_source_news", args.dry_run)

    # 2c. Annotate FDs with prior-state + stability/change partition across
    #     all benchmarks. Required for the headline "change" subset metrics:
    #       gdelt-cameo  : modal Peace/Tension/Violence intensity over prior 30d
    #       earnings     : mode of prior 4 quarters' Beat/Meet/Miss surprise class
    #       forecastbench: crowd_probability ≥ 0.5 → expected "Yes" else "No"
    #     No new API calls. See scripts/annotate_prior_state.py.
    run_cmd([PY, str(SCRIPTS / "annotate_prior_state.py"),
             "--benchmarks", ",".join(chosen)],
            dry_run=args.dry_run)
    snapshot("01c_after_prior_state_annotation", args.dry_run)

    # 3. Relevance (runs the two benchmark-filter passes IN PARALLEL —
    # see step_relevance_parallel). Saves ~7 min per build.
    rel_targets = [b for b in ("forecastbench", "earnings", "gdelt_cameo") if b in chosen]
    step_relevance_parallel(rel_targets, args.rebuild_embeddings, args.dry_run, args.embedder, args.openai_model, args.openai_mode)
    snapshot("02_after_first_relevance", args.dry_run)

    # 4. GDELT-CAMEO context relink (must come BEFORE quality_filter since it
    # changes article_ids to pre-event context articles)
    pipeline_cfg = cfg.get("pipeline", {}) or {}
    gd_workers = int(pipeline_cfg.get("gdelt_text_fetch_workers", 24))
    fb_workers = int(pipeline_cfg.get("text_fetch_workers", 24))
    if "gdelt_cameo" in chosen:
        step_relink_gdelt(True, args.dry_run)
        snapshot("03_after_first_relink", args.dry_run)
        # Fetch full text for referenced (relinked) articles
        step_gdelt_text_fetch(args.dry_run, workers=gd_workers)
        snapshot("04_after_gdelt_text_fetch", args.dry_run)

    # 4b. Fetch ForecastBench article text (trafilatura on GDELT DOC URLs).
    if "forecastbench" in chosen:
        step_fb_text_fetch(args.dry_run, workers=fb_workers)
        snapshot("05_after_fb_text_fetch", args.dry_run)

    # Re-run unify + relevance so newly-fetched text participates in scoring.
    # The incremental SBERT cache makes the re-relevance fast (~1-2 min vs 7)
    # since only rows with changed text get re-embedded.
    if "gdelt_cameo" in chosen or "forecastbench" in chosen:
        step_unify(args.dry_run)
        snapshot("06_after_reunify", args.dry_run)
        step_relevance_parallel(rel_targets, args.rebuild_embeddings, args.dry_run, args.embedder, args.openai_model, args.openai_mode)
        if "gdelt_cameo" in chosen:
            step_relink_gdelt(True, args.dry_run)
        snapshot("07_after_re_relevance_and_relink", args.dry_run)

    # 5. Quality filter + diagnostic + EDA + pool audits
    step_quality(cutoff, buffer_days, cfg.get("quality", {}), args.dry_run)
    snapshot("08_after_quality_filter", args.dry_run)
    step_diagnostic(args.dry_run)
    step_eda(args.dry_run)
    step_pool_audits(args.dry_run)
    snapshot("09_final_before_publish", args.dry_run)

    # 6. Publish to {output_root}/{cutoff}/  (+ SHA256 sidecar)
    dest = step_publish(cutoff, output_root, args.dry_run)
    if not args.dry_run:
        write_run_manifest(dest, cfg, cutoff, chosen, effective_cfg=cfg)
        step_checksums(dest, args.dry_run)

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
