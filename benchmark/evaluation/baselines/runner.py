"""
Single entry point for the baselines battery.

Usage:
  python -m evaluation.baselines.runner \
      --method b4 \
      --fds benchmark/data/2024-04-01/forecasts.jsonl \
      --articles benchmark/data/2024-04-01/articles.jsonl \
      --config benchmark/configs/baselines.yaml \
      [--dry-run] [--smoke N] [--sync] [--limit N]

Three-stage debug flow:
  1. --dry-run                      : build requests, print first 3, no API call
  2. --smoke 5 --sync               : 5 FDs, synchronous API calls (skip BatchClient)
  3. --smoke 20 --sync              : 20 FDs, synchronous; first calibration look
  4. (no debug flags)               : full Batch API flow on all FDs (production)

Outputs (Rule 1, versioned, never overwrites):
  benchmark/results/{cutoff}/{method}/{run_id}/
      run_manifest.json
      predictions_{method}.jsonl
      metrics_{method}.json
      debug/{fd_id}.json            # only in --smoke mode
  benchmark/results/{cutoff}/{method}/latest.txt    # pointer to newest run_id
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# runner.py lives at benchmark/evaluation/baselines/runner.py
#   parents[0] = baselines
#   parents[1] = evaluation
#   parents[2] = benchmark
#   parents[3] = repo root
_THIS_FILE = Path(__file__).resolve()
_BENCHMARK_ROOT = _THIS_FILE.parents[2]
_REPO_ROOT = _THIS_FILE.parents[3]

CONFIG_DEFAULT = _BENCHMARK_ROOT / "configs" / "baselines.yaml"

# The shim adds repo root to sys.path so src.* imports resolve.
from ._shim import (  # noqa: E402
    BatchClient,
    BatchRequest,
    BatchResult,
    brier_score,
    get_config,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def load_articles_index(path: str | Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx[rec["id"]] = rec
    return idx


# ---------------------------------------------------------------------------
# Class loading
# ---------------------------------------------------------------------------

def load_baseline_class(dotted: str):
    """'methods.b4_self_consistency.B4SelfConsistency' -> class."""
    module_path, cls_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(f"evaluation.baselines.{module_path}")
    return getattr(mod, cls_name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _norm_balacc(bal_acc: float, K: int) -> float:
    """Normalized balanced accuracy (a.k.a. Informedness / Bookmaker's Score):
      0 = random-guess level (BalAcc = 1/K)
      1 = perfect
     <0 = worse than random.
    K-invariant so binary (K=2) and multi-class (K=4) benchmarks are directly comparable."""
    if K <= 1:
        return 0.0
    return (bal_acc - 1.0/K) / (1.0 - 1.0/K)


def _point_metrics(preds: list[dict], hs: list[str]) -> tuple[float, float, float, float, float]:
    """Return (accuracy, balanced_accuracy, norm_balanced_accuracy, macro_f1, mcc).
    Used internally for bootstrap resampling."""
    import math
    n = len(preds)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    correct = sum(1 for p in preds if p["predicted_class"] == p["ground_truth"])
    acc = correct / n
    per_recall = []
    f1s = []
    for h in hs:
        gt_c = [p for p in preds if p["ground_truth"] == h]
        if not gt_c:
            continue
        tp = sum(1 for p in gt_c if p["predicted_class"] == h)
        fp = sum(1 for p in preds if p["predicted_class"] == h and p["ground_truth"] != h)
        fn = len(gt_c) - tp
        rec = tp / len(gt_c)
        per_recall.append(rec)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    # Macro-F1 averages over every class in hs (zeros for absent classes),
    # matching sklearn default behavior; per_recall skips zero-support classes.
    f1s_full = []
    for h in hs:
        tp = sum(1 for p in preds if p["predicted_class"] == h and p["ground_truth"] == h)
        fp = sum(1 for p in preds if p["predicted_class"] == h and p["ground_truth"] != h)
        fn = sum(1 for p in preds if p["predicted_class"] != h and p["ground_truth"] == h)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
        f1s_full.append(f1)
    bal = float(np.mean(per_recall)) if per_recall else 0.0
    nbal = _norm_balacc(bal, len(hs))
    mf1 = float(np.mean(f1s_full)) if f1s_full else 0.0
    # MCC (Gorodkin generalized)
    K = len(hs)
    idx = {h: i for i, h in enumerate(hs)}
    C = np.zeros((K, K), dtype=float)
    for p in preds:
        gt_i = idx.get(p["ground_truth"]); pr_i = idx.get(p["predicted_class"])
        if gt_i is not None and pr_i is not None:
            C[gt_i, pr_i] += 1
    s = C.sum()
    mcc = 0.0
    if s > 0:
        t_v = C.sum(axis=1); p_v = C.sum(axis=0); c_v = C.trace()
        num = c_v * s - (p_v @ t_v)
        den = math.sqrt(max(s*s - (p_v @ p_v), 0)) * math.sqrt(max(s*s - (t_v @ t_v), 0))
        mcc = float(num / den) if den > 0 else 0.0
    return acc, bal, nbal, mf1, mcc


def _bootstrap_cis(preds: list[dict], hs: list[str], B: int = 1000, seed: int = 42,
                    stratified: bool = True) -> dict:
    """95% bootstrap CIs (percentile method) for accuracy / balanced_accuracy / macro_f1 / mcc.
    Post-hoc: no new API calls.

    stratified=True (default): resample within each ground-truth class independently,
    preserving class support per draw. Appropriate for class-imbalanced benchmarks
    (esp. when minority class has <20 FDs) and for metrics that are themselves
    class-stratified (balanced accuracy, macro-F1, MCC).
    stratified=False: uniform resample with replacement across all FDs.
    """
    if not preds:
        return {}
    import random
    rng = random.Random(seed)
    n = len(preds)
    # Pre-compute per-class index lists for stratified mode
    per_class_idx: dict[str, list[int]] = {h: [] for h in hs}
    for i, p in enumerate(preds):
        gt = p.get("ground_truth")
        if gt in per_class_idx:
            per_class_idx[gt].append(i)
    accs, bals, nbals, mf1s, mccs = [], [], [], [], []
    for _ in range(B):
        if stratified:
            idxs: list[int] = []
            for h, pool in per_class_idx.items():
                if not pool:
                    continue
                for _ in range(len(pool)):
                    idxs.append(pool[rng.randrange(len(pool))])
            boot = [preds[i] for i in idxs]
        else:
            boot = [preds[rng.randrange(n)] for _ in range(n)]
        a, b, nb, f, m = _point_metrics(boot, hs)
        accs.append(a); bals.append(b); nbals.append(nb); mf1s.append(f); mccs.append(m)
    def ci(xs):
        xs = sorted(xs)
        lo = xs[int(0.025 * len(xs))]
        hi = xs[int(0.975 * len(xs)) - 1]
        return round(100 * lo, 2), round(100 * hi, 2)
    def ci_raw(xs):
        xs = sorted(xs)
        return round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)
    return {
        "accuracy_ci95":               list(ci(accs)),
        "balanced_accuracy_ci95":      list(ci(bals)),
        "norm_balanced_accuracy_ci95": list(ci_raw(nbals)),
        "macro_f1_ci95":               list(ci_raw(mf1s)),
        "mcc_ci95":                    list(ci_raw(mccs)),
        "bootstrap_B":                 B,
        "bootstrap_stratified":        stratified,
    }


def _metrics_single_group(preds: list[dict], hs: list[str]) -> dict:
    """Pick-only metrics: accuracy, macro-F1, per-class precision/recall/F1,
    confusion matrix, majority-class reference, parse-failure count, MCC,
    and 95% bootstrap CIs for accuracy/balanced_accuracy/macro_f1/mcc.
    Parse failures count as wrong (never silently fall back to hs[0])."""
    if not preds:
        return {"n": 0}
    n = len(preds)
    correct = sum(1 for p in preds if p["predicted_class"] == p["ground_truth"])
    n_parse_failed = sum(1 for p in preds if p.get("parse_failed"))
    # Per-class precision/recall/F1
    per_class = {}
    for h in hs:
        tp = sum(1 for p in preds if p["predicted_class"] == h and p["ground_truth"] == h)
        fp = sum(1 for p in preds if p["predicted_class"] == h and p["ground_truth"] != h)
        fn = sum(1 for p in preds if p["predicted_class"] != h and p["ground_truth"] == h)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[h] = {"precision": round(prec, 4),
                        "recall":    round(rec, 4),
                        "f1":        round(f1, 4),
                        "support":   sum(1 for p in preds if p["ground_truth"] == h)}
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    # Balanced accuracy: mean of per-class recall (robust to class imbalance).
    # For binary this is equivalent to the AUC of a one-point ROC.
    class_recalls = [v["recall"] for v in per_class.values() if v["support"] > 0]
    balanced_acc = float(np.mean(class_recalls)) if class_recalls else 0.0
    norm_bal_acc = _norm_balacc(balanced_acc, len(hs))
    # Confusion matrix (counts): C[gt][pred]
    cm = {gt: {pr: 0 for pr in hs} for gt in hs}
    for p in preds:
        if p["ground_truth"] in cm and p["predicted_class"] in cm[p["ground_truth"]]:
            cm[p["ground_truth"]][p["predicted_class"]] += 1
    # Majority-class reference: always predict argmax gt-frequency
    from collections import Counter
    gt_counter = Counter(p["ground_truth"] for p in preds)
    maj_class = max(hs, key=lambda h: (gt_counter[h], -hs.index(h)))
    maj_correct = sum(1 for p in preds if p["ground_truth"] == maj_class)
    # Matthews correlation coefficient (generalized Gorodkin 2004 form)
    import math
    K = len(hs)
    idx = {h: i for i, h in enumerate(hs)}
    C = np.zeros((K, K), dtype=float)
    for p in preds:
        gt_i = idx.get(p["ground_truth"])
        pr_i = idx.get(p["predicted_class"])
        if gt_i is None or pr_i is None:
            continue
        C[gt_i, pr_i] += 1
    s = C.sum()
    if s > 0:
        t_v = C.sum(axis=1)
        p_v = C.sum(axis=0)
        c_v = C.trace()
        num = c_v * s - (p_v @ t_v)
        den = math.sqrt(max(s*s - (p_v @ p_v), 0)) * math.sqrt(max(s*s - (t_v @ t_v), 0))
        mcc = float(num / den) if den > 0 else 0.0
    else:
        mcc = 0.0
    cis = _bootstrap_cis(preds, hs, B=1000, seed=42)
    return {
        "n":                       n,
        "n_classes":               len(hs),
        "n_parse_failed":          n_parse_failed,
        "accuracy":                round(100 * correct / n, 2),
        "balanced_accuracy":       round(100 * balanced_acc, 2),
        "norm_balanced_accuracy":  round(norm_bal_acc, 4),
        "macro_f1":                round(macro_f1, 4),
        "mcc":                     round(mcc, 4),
        **cis,
        "per_class":               per_class,
        "confusion_matrix":        cm,
        "majority_class":          maj_class,
        "majority_class_accuracy": round(100 * maj_correct / n, 2),
        "beats_majority":          (100 * correct / n) > (100 * maj_correct / n),
    }


def compute_metrics(preds: list[dict]) -> dict:
    """Group predictions by (benchmark, hypothesis_set) and compute per-group
    metrics. Handles FDs from different benchmarks (binary FB, 4-class GDELT,
    3-class earnings) mixed in one run — previously crashed when preds[0]'s
    hypothesis_set didn't apply to all rows."""
    if not preds:
        return {"n": 0}
    # Group by (benchmark, hypothesis_set_tuple) — tuple so dict-keyable
    groups: dict[tuple, list[dict]] = {}
    for p in preds:
        key = (p.get("benchmark", "?"), tuple(p.get("hypothesis_set", [])))
        groups.setdefault(key, []).append(p)

    # If only one group, return its metrics at top level (back-compat).
    if len(groups) == 1:
        (_, hs_tuple), rows = next(iter(groups.items()))
        return _metrics_single_group(rows, list(hs_tuple))

    # Otherwise emit nested per-benchmark metrics + an overall accuracy.
    out: dict[str, Any] = {
        "n": len(preds),
        "accuracy": round(100 * sum(
            1 for p in preds if p["predicted_class"] == p["ground_truth"]
        ) / len(preds), 2),
        "per_benchmark": {},
    }
    for (bench, hs_tuple), rows in groups.items():
        out["per_benchmark"][bench] = _metrics_single_group(rows, list(hs_tuple))
    return out


# ---------------------------------------------------------------------------
# Fairness banner
# ---------------------------------------------------------------------------

def emit_banner(method_name: str, baseline, defaults: dict, fds_path: Path, cutoff: str, mode: str,
                method_cfg: dict | None = None) -> None:
    print("=" * 72)
    display = (method_cfg or {}).get("display_name", method_name)
    short = (method_cfg or {}).get("short_name", method_name)
    desc = (method_cfg or {}).get("description", "")
    print(f"[baselines] method={method_name}  ({short} - \"{display}\")  mode={mode}")
    if desc:
        print(f"  description  = {desc}")
    print(f"  model        = {baseline.model}")
    print(f"  temperature  = {baseline.temperature}")
    print(f"  max_tokens   = {baseline.max_tokens}")
    print(f"  batch_api    = {defaults.get('batch_api', True)}")
    print(f"  max_articles = {getattr(baseline, 'max_articles', 'n/a')}")
    print(f"  cutoff       = {cutoff}")
    print(f"  fds_path     = {fds_path}")
    print("=" * 72)


def enforce_model_fairness(method_name: str, baseline, defaults: dict) -> None:
    allow = getattr(baseline, "allow_model_override", False)
    if not allow and baseline.model != defaults["model"]:
        raise SystemExit(
            f"[fairness] {method_name} uses model={baseline.model!r} but defaults.model="
            f"{defaults['model']!r}. Only B9 is allowed to deviate."
        )


# ---------------------------------------------------------------------------
# Run ID / manifest helpers (Rule 1)
# ---------------------------------------------------------------------------

def git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or "nogit"
    except Exception:
        return "nogit"


def make_run_id() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{git_sha_short()}"


def write_manifest(
    results_dir: Path,
    *,
    run_id: str,
    method_name: str,
    cutoff: str,
    baseline,
    defaults: dict,
    method_cfg: dict,
    fds: list[dict],
    articles: dict,
    mode: str,
    smoke_n: int | None,
    sync: bool,
) -> None:
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method_name,
        "cutoff": cutoff,
        "model": baseline.model,
        "temperature": baseline.temperature,
        "max_tokens": baseline.max_tokens,
        "git_sha": git_sha_short(),
        "mode": mode,
        "smoke_n": smoke_n,
        "sync": sync,
        "n_fds": len(fds),
        "n_articles": len(articles),
        "fd_ids_sample": [fd.get("id") for fd in fds[:3]],
        "config_snapshot": {
            "defaults": defaults,
            "method": method_cfg,
        },
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def update_latest_pointer(method_dir: Path, run_id: str) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "latest.txt").write_text(run_id + "\n")


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def dispatch_requests(
    client: BatchClient | None,
    requests: list[BatchRequest],
    job_name: str,
    method_name: str,
    results_dir: Path,
) -> dict[str, BatchResult]:
    """Submit via BatchClient. On quota/transient failure, persist pending requests."""
    if client is None:
        raise RuntimeError("BatchClient not initialized (dry-run path should not reach here).")
    try:
        return client.run(requests, job_name=job_name)
    except Exception as exc:
        pending_path = results_dir / f"pending_{method_name}.jsonl"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pending_path, "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps({
                    "custom_id": r.custom_id,
                    "model": r.model,
                    "messages": r.messages,
                    "max_tokens": r.max_tokens,
                    "temperature": r.temperature,
                    "response_format": r.response_format,
                }) + "\n")
        print(f"[quota] BatchClient failed: {exc}")
        print(f"[quota] Wrote {len(requests)} pending requests to {pending_path}")
        raise SystemExit(2)


def dispatch_sync(requests: list[BatchRequest]) -> dict[str, BatchResult]:
    """Synchronous (non-Batch) dispatch for --sync smoke mode.

    Uses the OpenAI Chat Completions endpoint directly (one call per request).
    Intended only for small smoke runs; no retries.
    """
    from openai import OpenAI  # local import so dry-run never requires openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[sync] OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    results: dict[str, BatchResult] = {}
    for i, r in enumerate(requests):
        t0 = time.time()
        kwargs: dict[str, Any] = dict(
            model=r.model,
            messages=r.messages,
            temperature=r.temperature,
            max_tokens=r.max_tokens,
        )
        if r.response_format:
            kwargs["response_format"] = r.response_format
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            latency = time.time() - t0
            usage = getattr(resp, "usage", None)
            usage_dict = usage.__dict__ if usage else {}
            results[r.custom_id] = BatchResult(
                custom_id=r.custom_id,
                content=content,
                raw={"latency_sec": latency, "usage": usage_dict},
            )
        except Exception as exc:
            print(f"[sync] request {i+1}/{len(requests)} failed: {exc}")
            results[r.custom_id] = BatchResult(custom_id=r.custom_id, content="", raw={"error": str(exc)})
    return results


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def infer_cutoff(fds_path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(fds_path))
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Smoke debug dumps (Rule 2)
# ---------------------------------------------------------------------------

def dump_smoke_debug(
    debug_dir: Path,
    fds: list[dict],
    requests: list[BatchRequest],
    results: dict[str, BatchResult],
    preds: list[dict],
) -> tuple[int, int]:
    """Write per-FD debug files; return (n_parse_ok, n_parse_fail).

    A parse is "ok" if the result produced a non-uniform distribution.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    req_by_fd: dict[str, list[BatchRequest]] = {}
    for r in requests:
        fd_id = r.custom_id.split("::", 1)[0]
        req_by_fd.setdefault(fd_id, []).append(r)

    pred_by_fd = {p["id"]: p for p in preds}

    n_ok = 0
    n_fail = 0
    for fd in fds:
        fd_id = fd["id"]
        fd_reqs = req_by_fd.get(fd_id, [])
        fd_results = []
        latencies = []
        for r in fd_reqs:
            res = results.get(r.custom_id)
            raw = res.raw if isinstance(res, BatchResult) else {}
            content = res.content if isinstance(res, BatchResult) else ""
            if isinstance(raw, dict) and "latency_sec" in raw:
                latencies.append(raw["latency_sec"])
            fd_results.append({
                "custom_id": r.custom_id,
                "rendered_prompt_system": r.messages[0]["content"] if r.messages else "",
                "rendered_prompt_user": r.messages[1]["content"] if len(r.messages) > 1 else "",
                "response_raw": content,
                "usage": raw.get("usage") if isinstance(raw, dict) else None,
            })
        pred = pred_by_fd.get(fd_id, {})
        hs = fd.get("hypothesis_set", [])
        predicted_class = pred.get("predicted_class")
        # Pick-only success: parsed prediction is one of hypothesis_set
        if predicted_class and predicted_class in hs:
            n_ok += 1
        else:
            n_fail += 1

        debug_payload = {
            "fd_id": fd_id,
            "hypothesis_set": hs,
            "ground_truth": fd.get("ground_truth"),
            "predicted_class": predicted_class,
            "latency_sec": round(sum(latencies), 3) if latencies else None,
            "requests": fd_results,
        }
        # default=str stringifies OpenAI SDK objects (CompletionTokensDetails etc.)
        # that aren't natively JSON-serializable.
        (debug_dir / f"{fd_id}.json").write_text(
            json.dumps(debug_payload, indent=2, default=str))
    return n_ok, n_fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Baselines battery runner")
    p.add_argument("--method", required=True, help="Baseline key (e.g. b4 or b4_self_consistency)")
    p.add_argument("--fds", required=True, help="Path to forecasts.jsonl")
    p.add_argument("--articles", required=True, help="Path to articles.jsonl")
    p.add_argument("--config", default=str(CONFIG_DEFAULT))
    p.add_argument("--dry-run", action="store_true",
                   help="Build requests, print first 3, no API call.")
    p.add_argument("--smoke", type=int, nargs="?", const=5, default=None,
                   help="Smoke mode: limit to first N FDs (default 5) and write per-FD debug dumps.")
    p.add_argument("--sync", action="store_true",
                   help="Use synchronous Chat Completions calls instead of the Batch API.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of FDs (overridden by --smoke).")
    p.add_argument("--results-dir", default=None,
                   help="Override base results dir (default: benchmark/results/).")
    args = p.parse_args()

    # --- Resolve config ---
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        candidate_bench = _BENCHMARK_ROOT / cfg_path
        candidate_root = _REPO_ROOT / cfg_path
        candidate_cwd = Path.cwd() / cfg_path
        for cand in (candidate_cwd, candidate_bench, candidate_root):
            if cand.exists():
                cfg_path = cand
                break
    with open(cfg_path) as f:
        cfg_yaml = yaml.safe_load(f)
    defaults = cfg_yaml["defaults"]
    baselines = cfg_yaml["baselines"]

    # Accept short aliases: "b4" matches "b4_self_consistency"
    method_name = args.method
    if method_name not in baselines:
        matches = [k for k in baselines if k.startswith(method_name + "_") or k == method_name]
        if len(matches) == 1:
            method_name = matches[0]
        else:
            raise SystemExit(f"Unknown method {args.method!r}. Options: {list(baselines)}")

    method_cfg = baselines[method_name]
    baseline_cls = load_baseline_class(method_cfg["class"])
    baseline = baseline_cls(method_cfg, defaults)

    # --- Resolve data paths ---
    fds_path = Path(args.fds)
    if not fds_path.is_absolute():
        for base in (Path.cwd(), _BENCHMARK_ROOT, _REPO_ROOT):
            cand = base / args.fds
            if cand.exists():
                fds_path = cand
                break
    articles_path = Path(args.articles)
    if not articles_path.is_absolute():
        for base in (Path.cwd(), _BENCHMARK_ROOT, _REPO_ROOT):
            cand = base / args.articles
            if cand.exists():
                articles_path = cand
                break

    cutoff = infer_cutoff(fds_path)

    # --- Determine mode (Rule 3) ---
    smoke_n = args.smoke
    if args.dry_run:
        mode = "dry-run"
    elif smoke_n is not None and args.sync:
        mode = f"smoke-sync(N={smoke_n})"
    elif smoke_n is not None:
        mode = f"smoke-batch(N={smoke_n})"
    elif args.sync:
        mode = "sync"
    else:
        mode = "batch"

    effective_limit = smoke_n if smoke_n is not None else args.limit

    # --- Load data ---
    fds = load_jsonl(fds_path, limit=effective_limit) if fds_path.exists() else []
    articles = load_articles_index(articles_path) if articles_path.exists() else {}

    # --- Run id + results dir (Rule 1) ---
    run_id = make_run_id()
    results_base = Path(args.results_dir) if args.results_dir else _BENCHMARK_ROOT / "results"
    method_dir = results_base / cutoff / method_name
    results_dir = method_dir / run_id

    # --- Banner + fairness ---
    emit_banner(method_name, baseline, defaults, fds_path, cutoff, mode, method_cfg=method_cfg)
    enforce_model_fairness(method_name, baseline, defaults)
    print(f"[baselines] run_id      = {run_id}")
    print(f"[baselines] results_dir = {results_dir}")
    print(f"[baselines] loaded {len(fds)} FDs, {len(articles)} articles")

    if not fds:
        print("[baselines] no FDs loaded; exiting.")
        return 1

    # --- Build round-0 requests ---
    requests = baseline.build_requests(fds, articles)
    print(f"[baselines] built {len(requests)} requests for round 0")

    # --- Dry-run: print first 3 and stop ---
    if args.dry_run:
        print("\n[dry-run] First 3 built BatchRequests:")
        for i, r in enumerate(requests[:3]):
            print(f"\n--- request {i+1} ---")
            print(f"custom_id   : {r.custom_id}")
            print(f"model       : {r.model}")
            print(f"temperature : {r.temperature}")
            print(f"max_tokens  : {r.max_tokens}")
            sys_msg = r.messages[0]["content"]
            usr_msg = r.messages[1]["content"]
            print(f"system      : {sys_msg[:120]}{'...' if len(sys_msg) > 120 else ''}")
            print(f"user (first 400 chars):\n{usr_msg[:400]}{'...' if len(usr_msg) > 400 else ''}")
        print(f"\n[dry-run] total requests that would be dispatched: {len(requests)}")
        print(f"[dry-run] mode={mode}; sync={args.sync}; smoke_n={smoke_n}")
        print("[dry-run] done (no results written).")
        return 0

    # --- Materialize results dir and write manifest ---
    results_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        results_dir,
        run_id=run_id,
        method_name=method_name,
        cutoff=cutoff,
        baseline=baseline,
        defaults=defaults,
        method_cfg=method_cfg,
        fds=fds,
        articles=articles,
        mode=mode,
        smoke_n=smoke_n,
        sync=args.sync,
    )
    update_latest_pointer(method_dir, run_id)

    # --- Dispatch ---
    all_results: dict[str, BatchResult] = {}
    job_prefix = f"{method_name}_{cutoff}"
    client = None
    if args.sync:
        round_results = dispatch_sync(requests)
    else:
        ach_cfg = get_config()
        client = BatchClient(mode="batch" if defaults.get("batch_api", True) else "direct", config=ach_cfg)
        round_results = dispatch_requests(client, requests, f"{job_prefix}_r0", method_name, results_dir)
    all_results.update(round_results)

    # Multi-round baselines
    if getattr(baseline, "multi_round", False):
        total_rounds = getattr(baseline, "n_rounds", None) or getattr(baseline, "depth", None) \
            or getattr(baseline, "n_iterations", None) or 1
        for r in range(1, total_rounds):
            reqs_r = baseline.build_requests_round(r, fds, articles, all_results)
            print(f"[baselines] built {len(reqs_r)} requests for round {r}")
            requests.extend(reqs_r)
            if args.sync:
                round_results = dispatch_sync(reqs_r)
            else:
                round_results = dispatch_requests(client, reqs_r, f"{job_prefix}_r{r}", method_name, results_dir)
            all_results.update(round_results)

    # --- Parse + save ---
    preds = baseline.parse_responses(all_results, fds)
    pred_path = results_dir / f"predictions_{method_name}.jsonl"
    met_path = results_dir / f"metrics_{method_name}.json"
    save_jsonl(preds, pred_path)

    # Dump parse failures (raw response + FD id) for prompt-tuning feedback.
    failed_ids = {p["id"] for p in preds if p.get("parse_failed")}
    if failed_ids:
        pf_path = results_dir / f"parse_failures_{method_name}.jsonl"
        with open(pf_path, "w", encoding="utf-8") as f:
            for fd in fds:
                if fd["id"] not in failed_ids:
                    continue
                # Find any BatchResult whose custom_id starts with this fd_id
                for cid, res in all_results.items():
                    if cid.startswith(fd["id"] + "::"):
                        content = res.content if hasattr(res, "content") else str(res)
                        f.write(json.dumps({
                            "fd_id": fd["id"],
                            "custom_id": cid,
                            "hypothesis_set": fd.get("hypothesis_set"),
                            "ground_truth": fd.get("ground_truth"),
                            "raw_content": content,
                        }, ensure_ascii=False) + "\n")
                        break
        print(f"[baselines] wrote {len(failed_ids)} parse failures -> {pf_path}")

    metrics = compute_metrics(preds)
    metrics.update({
        "method": method_name,
        "short_name": method_cfg.get("short_name", method_name),
        "display_name": method_cfg.get("display_name", method_name),
        "description": method_cfg.get("description", ""),
        "model": baseline.model,
        "temperature": baseline.temperature,
        "cutoff": cutoff,
        "run_id": run_id,
    })
    save_json(metrics, met_path)

    # --- Smoke debug dump + summary (Rule 2) ---
    if smoke_n is not None:
        debug_dir = results_dir / "debug"
        n_ok, n_fail = dump_smoke_debug(debug_dir, fds, requests, all_results, preds)
        correct = sum(1 for p in preds if p["predicted_class"] == p["ground_truth"])
        sample_prompt_len = len(requests[0].messages[1]["content"]) if requests else 0
        print("\n[smoke] summary")
        print(f"  FDs         : {len(fds)}")
        print(f"  parse_ok    : {n_ok}")
        print(f"  parse_fail  : {n_fail}")
        print(f"  accuracy    : {correct}/{len(preds)}")
        print(f"  sample prompt length: {sample_prompt_len} chars")
        print(f"  debug dir   : {debug_dir}")

    print(f"[baselines] saved {len(preds)} predictions -> {pred_path}")
    print(f"[baselines] metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
