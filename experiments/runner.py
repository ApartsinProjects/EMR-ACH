"""
Shared experiment runner utilities.

Provides:
  - save_predictions / load_predictions: persist prediction dicts to JSONL
  - save_metrics / load_metrics: persist EvalResult dicts
  - ExperimentRunner: base class that handles smoke vs full mode, logging, cost reporting
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.config import get_config
from src.eval.metrics import EvalResult


ROOT = Path(__file__).parent.parent


def save_predictions(predictions: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"[runner] Saved {len(predictions)} predictions to {path}")


def load_predictions(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[runner] Saved metrics to {path}")


def load_metrics(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_all_metrics(results_dir: str | Path) -> dict[str, dict]:
    """Load all metrics_*.json from results/processed/."""
    results_dir = Path(results_dir)
    all_metrics = {}
    for p in sorted(results_dir.glob("**/metrics_*.json")):
        name = p.stem.replace("metrics_", "")
        all_metrics[name] = load_metrics(p)
    return all_metrics


def print_cost_estimate(client, requests: list) -> None:
    estimate = client.estimate_cost(requests)
    print(
        f"[cost] {estimate['n_requests']} requests | "
        f"~{estimate['estimated_input_tokens']:,} input tokens | "
        f"~{estimate['estimated_output_tokens']:,} output tokens | "
        f"est. cost: ${estimate['estimated_cost_usd']:.3f} (batch, 50% discount)"
    )


def parse_args(description: str = "EMR-ACH experiment") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
        help="smoke: 5 queries, direct API; full: all queries, batch API",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to override config YAML",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="Override number of queries (useful for quick partial runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cost estimate only, do not submit any batch jobs",
    )
    return parser.parse_args()


def get_client(mode: str, config=None):
    from src.batch_client import BatchClient
    cfg = config or get_config()
    batch_mode = "direct" if mode == "smoke" else "batch"
    return BatchClient(mode=batch_mode, config=cfg)


def get_n_queries(mode: str, args=None) -> int | None:
    if args and args.n_queries:
        return args.n_queries
    if mode == "smoke":
        return 5
    return None


def timing(label: str):
    """Context manager that prints elapsed time."""
    class _Timer:
        def __enter__(self):
            self._start = time.time()
            print(f"[{label}] starting...")
            return self
        def __exit__(self, *_):
            elapsed = time.time() - self._start
            print(f"[{label}] done in {elapsed:.1f}s")
    return _Timer()
