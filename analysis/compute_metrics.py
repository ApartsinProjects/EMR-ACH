"""
Compute and print all metrics from experiment results.

Reads all predictions_*.jsonl from results/processed/,
computes evaluation metrics against MIRAI ground truth,
and writes a consolidated metrics summary.

Usage:
  python analysis/compute_metrics.py
  python analysis/compute_metrics.py --bootstrap-n 1000
"""

import json
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.mirai import MiraiDataset, make_mock_queries
from src.eval.metrics import evaluate, per_class_f1, mcnemar_test
from experiments.runner import load_predictions, save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    results_dir = cfg.results_dir / "processed"

    # Load ground truth
    try:
        ds = MiraiDataset(cfg)
        queries = ds.queries()
        labels_by_id = {q.id: q.label for q in queries}
        print(f"Loaded {len(queries)} MIRAI queries")
    except FileNotFoundError:
        print("[WARN] MIRAI data not found. Using mock labels.")
        queries = make_mock_queries(100)
        labels_by_id = {q.id: q.label for q in queries}

    # Find all prediction files
    pred_files = sorted(results_dir.glob("predictions_*.jsonl"))
    if not pred_files:
        print(f"No prediction files found in {results_dir}")
        print("Run experiments first.")
        return

    print(f"\nFound {len(pred_files)} prediction files\n")
    print(f"{'System':<25} {'F1%':>6} {'Prec%':>6} {'Rec%':>6} {'Acc%':>5} {'ECE':>7} {'KL':>8}  CI95")
    print("-" * 90)

    all_results = {}
    all_predictions = {}

    for pred_file in pred_files:
        name = pred_file.stem.replace("predictions_", "")
        preds = load_predictions(pred_file)
        if not preds:
            continue

        # Align with ground truth
        labels = []
        aligned_preds = []
        for p in preds:
            qid = p.get("query_id", "")
            if qid in labels_by_id:
                labels.append(labels_by_id[qid])
                aligned_preds.append(p)

        if len(aligned_preds) < 5:
            print(f"  {name}: insufficient aligned predictions ({len(aligned_preds)})")
            continue

        result = evaluate(
            aligned_preds, labels,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=cfg.get("evaluation", "bootstrap_seed", default=42),
        )
        all_results[name] = result
        all_predictions[name] = (aligned_preds, labels)

        ci = f"[{result.f1_ci[0]*100:.1f}, {result.f1_ci[1]*100:.1f}]" if result.f1_ci else ""
        print(
            f"  {name:<23} {result.f1*100:>6.1f} {result.precision*100:>6.1f} "
            f"{result.recall*100:>6.1f} {result.accuracy*100:>5.1f} "
            f"{result.ece:>7.4f} {result.kl_div:>8.4f}  {ci}"
        )

        # Per-class F1
        pf1 = per_class_f1(aligned_preds, labels)
        print(f"    Per-class F1: {' '.join(f'{h}={pf1[h]*100:.1f}%' for h in ['VC','MC','VK','MK'])}")

        save_metrics(result.to_dict(), results_dir / f"metrics_{name}.json")

    # McNemar's test: full system vs. each baseline
    if "emrach_full" in all_predictions and len(all_predictions) > 1:
        print("\n--- McNemar's Test (vs. Full System) ---")
        full_preds, full_labels = all_predictions["emrach_full"]
        full_pred_strs = [p["prediction"] for p in full_preds]
        for name, (preds, labels) in all_predictions.items():
            if name == "emrach_full":
                continue
            other_pred_strs = [p["prediction"] for p in preds]
            if len(other_pred_strs) != len(full_pred_strs):
                continue
            test = mcnemar_test(full_pred_strs, other_pred_strs, full_labels)
            sig = "*" if test["significant"] else ""
            print(f"  full vs {name}: p={test['p_value']:.4f}{sig}  (b={test['b']}, c={test['c']})")

    # Save consolidated summary
    summary = {
        name: result.to_dict()
        for name, result in all_results.items()
    }
    summary_path = cfg.results_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nConsolidated metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
