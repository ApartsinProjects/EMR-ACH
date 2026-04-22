"""Reparse cached raw batch responses with the current parser and recompute
metrics. No new API calls.

Usage:
  python -m evaluation.baselines.reparse --method b3_rag --cutoff 2026-01-01
  python -m evaluation.baselines.reparse --method all --cutoff 2026-01-01

Writes results under a new run_id (<ts>_reparse) so original runs are not
overwritten. Useful for comparing parser versions side-by-side.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import yaml

from ._shim import BatchResult
from .runner import (
    _BENCHMARK_ROOT,
    CONFIG_DEFAULT,
    compute_metrics,
    save_json,
    save_jsonl,
)


def load_raw(raw_path: Path) -> dict[str, BatchResult]:
    """Load a cached raw batch output file as {custom_id: BatchResult}."""
    results: dict[str, BatchResult] = {}
    if not raw_path.exists():
        return results
    for line in raw_path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        cid = r.get("custom_id", "")
        # Three formats seen historically:
        #  (a) OpenAI Batch API: {custom_id, response:{body:{choices:[{message:{content}}]}}}
        #  (b) Sync mode dump:   {custom_id, content, raw?}
        #  (c) Nested:           {custom_id, response:{choices:[...]}}
        content = ""
        raw = {}
        if "content" in r:
            content = r.get("content") or ""
            raw = r.get("raw") or {}
        else:
            body = r.get("response", {}).get("body") or r.get("response", {})
            if isinstance(body, dict):
                choices = body.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {}) or {}
                    content = msg.get("content") or ""
                raw = body
        results[cid] = BatchResult(custom_id=cid, content=content, raw=raw)
    return results


def load_fds_and_articles(cutoff: str) -> tuple[list[dict], dict[str, dict]]:
    fc = _BENCHMARK_ROOT / "data" / cutoff / "forecasts.jsonl"
    art = _BENCHMARK_ROOT / "data" / cutoff / "articles.jsonl"
    fds = [json.loads(l) for l in fc.open(encoding="utf-8")]
    articles: dict[str, dict] = {}
    for l in art.open(encoding="utf-8"):
        try:
            a = json.loads(l); articles[a["id"]] = a
        except Exception:
            pass
    return fds, articles


def load_method_class(dotted: str):
    module_path, cls_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(f"evaluation.baselines.{module_path}")
    return getattr(mod, cls_name)


def reparse_one(method_name: str, cutoff: str, cfg: dict) -> dict:
    # Build the baseline instance (needed so multi-round baselines know their own schema)
    method_cfg = cfg["baselines"][method_name]
    defaults = cfg["defaults"]
    baseline_cls = load_method_class(method_cfg["class"])
    baseline = baseline_cls(method_cfg, defaults)

    # Collect raw results for all rounds this method may have emitted
    raw_dir = _BENCHMARK_ROOT.parent / "results" / "raw"
    all_results: dict[str, BatchResult] = {}
    for r in range(8):  # max rounds any baseline uses
        p = raw_dir / f"{method_name}_{cutoff}_r{r}.jsonl"
        all_results.update(load_raw(p))
    if not all_results:
        print(f"[reparse] {method_name}: no cached raw responses found")
        return {}

    fds, articles = load_fds_and_articles(cutoff)

    # For single-round baselines this is trivial; for multi-round the
    # baseline expects intermediate rounds to already be in all_results.
    preds = baseline.parse_responses(all_results, fds)

    # Write to new run dir tagged "_reparse"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_reparse"
    out_dir = _BENCHMARK_ROOT / "results" / cutoff / method_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(preds, out_dir / f"predictions_{method_name}.jsonl")
    metrics = compute_metrics(preds)
    metrics.update({
        "method": method_name,
        "model": baseline.model,
        "cutoff": cutoff,
        "run_id": run_id,
        "source": "reparse",
        "n_raw_responses": len(all_results),
    })
    save_json(metrics, out_dir / f"metrics_{method_name}.json")
    # parse-failure dump
    failed = [p["id"] for p in preds if p.get("parse_failed")]
    if failed:
        pf_path = out_dir / f"parse_failures_{method_name}.jsonl"
        with pf_path.open("w", encoding="utf-8") as f:
            fd_by_id = {fd["id"]: fd for fd in fds}
            for fd_id in failed:
                for cid, res in all_results.items():
                    if cid.startswith(fd_id + "::"):
                        fd = fd_by_id.get(fd_id, {})
                        f.write(json.dumps({
                            "fd_id": fd_id,
                            "custom_id": cid,
                            "hypothesis_set": fd.get("hypothesis_set"),
                            "ground_truth": fd.get("ground_truth"),
                            "raw_content": res.content,
                        }, ensure_ascii=False) + "\n")
                        break
        print(f"[reparse] {method_name}: {len(failed)} parse failures -> {pf_path.name}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    help="Method id (b1_direct, b2_cot, ...) or 'all'.")
    ap.add_argument("--cutoff", required=True,
                    help="Cutoff date, e.g. 2026-01-01.")
    ap.add_argument("--config", default=str(CONFIG_DEFAULT))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    methods = list(cfg["baselines"].keys()) if args.method == "all" else [args.method]

    for m in methods:
        met = reparse_one(m, args.cutoff, cfg)
        if not met:
            continue
        # Print short summary
        print(f"\n=== {m} (reparsed) ===")
        if "per_benchmark" in met:
            for bench, m2 in met["per_benchmark"].items():
                print(f"  {bench}: n={m2['n']}  parse_failed={m2.get('n_parse_failed',0)}  "
                      f"acc={m2['accuracy']}%  bal={m2['balanced_accuracy']}%  "
                      f"mcc={m2['mcc']}  acc_ci={m2.get('accuracy_ci95','-')}")
        else:
            print(f"  n={met['n']}  acc={met.get('accuracy','-')}  mcc={met.get('mcc','-')}")


if __name__ == "__main__":
    main()
