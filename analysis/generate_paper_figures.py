"""
Generate updated paper figures from REAL experimental results.

Reads metrics_summary.json and predictions_*.jsonl,
produces updated versions of all 8 paper figures.
Outputs to results/figures/ (separate from paper/figures/ which has mock data).

Usage:
  python analysis/generate_paper_figures.py
  python analysis/generate_paper_figures.py --also-update-paper  # overwrite paper/figures/
"""

import json
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import get_config
from experiments.runner import load_predictions

COLORS = {
    "baseline": "#9ecae1",
    "base":     "#f4a582",
    "ours":     "#d6604d",
    "best":     "#b2182b",
    "gray":     "#aaaaaa",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TABLE1_KEYS = [
    "b1_direct", "b2_cot", "b3_ragonly",
    "emrach_base", "emrach_calib", "emrach_contrast",
    "emrach_diag", "emrach_absence", "emrach_multiagent", "emrach_full",
]
TABLE1_LABELS = [
    "Direct\nPrompt", "CoT", "RAG\nOnly",
    "EMR-ACH\n(base)", "+Calib", "+Contrast",
    "+Diag", "+Absence", "+Multi\nAgent", "Full\nSystem",
]
ABLATION_KEYS = [
    "emrach_full", "abl_no_multiagent", "abl_no_contrastive",
    "abl_no_diag", "abl_no_mmr", "abl_no_deepanalysis",
    "abl_no_decay", "abl_no_absence", "abl_no_calib",
]
ABLATION_LABELS = [
    "Full System", "w/o Multi-Agent", "w/o Contrastive Ind.",
    "w/o Diag. Weighting", "w/o MMR/RRF", "w/o Deep Analysis",
    "w/o Temporal Decay", "w/o Absence Evid.", "w/o Calibration",
]


def load_summary(cfg) -> dict:
    path = cfg.results_dir / "metrics_summary.json"
    if not path.exists():
        print(f"[WARN] metrics_summary.json not found at {path}")
        print("  Run: python analysis/compute_metrics.py")
        return {}
    with open(path) as f:
        return json.load(f)


def get_metric(summary, key, metric, default=0.0):
    return summary.get(key, {}).get(metric, default)


def fig_main_results(summary, out_dir):
    if not any(k in summary for k in TABLE1_KEYS):
        print("  [skip] No main results data")
        return

    f1s = [get_metric(summary, k, "f1") for k in TABLE1_KEYS]
    precs = [get_metric(summary, k, "precision") for k in TABLE1_KEYS]
    recs = [get_metric(summary, k, "recall") for k in TABLE1_KEYS]
    accs = [get_metric(summary, k, "accuracy") for k in TABLE1_KEYS]

    x = np.arange(len(TABLE1_KEYS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - 1.5*width, precs, width, label="Precision", color="#4393c3", alpha=0.88)
    ax.bar(x - 0.5*width, recs, width, label="Recall", color="#74c476", alpha=0.88)
    ax.bar(x + 0.5*width, f1s, width, label="F1 Score", color="#fd8d3c", alpha=0.88)
    ax.bar(x + 1.5*width, accs, width, label="Accuracy", color="#9e9ac8", alpha=0.88)

    ax.axvline(x=2.5, color="#aaa", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TABLE1_LABELS, fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max(max(f1s), 70) + 8)
    ax.set_title("Main Results — Quad-Class Relation Classification (MIRAI Test Set)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_main_results.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_main_results.png done")


def fig_ablation(summary, out_dir):
    f1s = [get_metric(summary, k, "f1") for k in ABLATION_KEYS]
    full_f1 = f1s[0]
    drops = [0.0] + [full_f1 - v for v in f1s[1:]]

    colors = [COLORS["best"]] + [
        COLORS["baseline"] if d < 2 else COLORS["ours"] if d < 4 else COLORS["base"]
        for d in drops[1:]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(ABLATION_KEYS))
    bars = ax.barh(y, f1s, color=colors, height=0.6, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(ABLATION_LABELS, fontsize=10)
    ax.set_xlabel("F1 Score (%)")
    min_f1 = min(f1s)
    ax.set_xlim(max(0, min_f1 - 3), full_f1 + 4)
    ax.set_title("Ablation Study — Component Contributions to F1", fontweight="bold")
    ax.axvline(x=full_f1, color=COLORS["best"], linestyle="--", linewidth=1.2, alpha=0.7)

    for bar, val, drop in zip(bars, f1s, drops):
        label = f"{val:.1f}%" if drop == 0 else f"{val:.1f}% (\u2193{drop:.1f})"
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    ax.xaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_ablation.png done")


def fig_multiagent(summary, out_dir):
    ns, f1s, costs = [], [], []
    for n in [1, 2, 3, 4, 5, 6]:
        k = f"ma_agents{n}"
        if k in summary:
            ns.append(n)
            f1s.append(summary[k].get("f1", 0))
            costs.append(summary[k].get("api_calls_per_query", n * 30))

    if not ns:
        print("  [skip] No multi-agent data")
        return

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    ax1.plot(ns, f1s, "o-", color=COLORS["best"], linewidth=2, markersize=6, label="F1 (%)")
    ax2.bar(ns, costs, color="#4393c3", alpha=0.35, width=0.4, label="API calls/query")
    ax1.set_xlabel("Number of Competing Agents")
    ax1.set_ylabel("F1 Score (%)", color=COLORS["best"])
    ax2.set_ylabel("API Calls per Query", color="#4393c3")
    ax1.tick_params(axis="y", labelcolor=COLORS["best"])
    ax2.tick_params(axis="y", labelcolor="#4393c3")
    ax1.set_xticks(ns)
    ax1.set_title("Multi-Agent ACH: Accuracy vs. Compute Cost", fontweight="bold")
    if len(ns) >= 4 and ns[3] == 4:
        ax1.annotate("Sweet spot", xy=(4, f1s[3]), xytext=(4.5, f1s[3] - 2),
                     arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_multiagent.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_multiagent.png done")


def fig_ece(summary, out_dir):
    ece_keys = [
        ("b1_direct", "Direct\nPrompt"), ("b2_cot", "CoT"), ("b3_ragonly", "RAG\nOnly"),
        ("emrach_base", "EMR-ACH\n(base)"), ("emrach_calib", "+Calib"),
        ("emrach_contrast", "+Contrast+\nCalib"), ("emrach_diag", "+Diag+\nAbsence"),
        ("emrach_full", "Full\nSystem"),
    ]
    vals, labels, colors = [], [], []
    for k, lbl in ece_keys:
        if k in summary:
            vals.append(summary[k].get("ece", 0))
            labels.append(lbl)
            if k.startswith("b"):
                colors.append("#9ecae1")
            elif k == "emrach_base":
                colors.append("#f4a582")
            elif k == "emrach_full":
                colors.append(COLORS["best"])
            else:
                colors.append(COLORS["ours"])

    if not vals:
        print("  [skip] No ECE data")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor="white")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title("Expected Calibration Error (ECE) Across Methods", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_ece.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_ece.png done")


def fig_kl(summary, out_dir):
    kl_keys = [
        ("b1_direct", "Direct\nPrompt"), ("b2_cot", "CoT"), ("b3_ragonly", "RAG\nOnly"),
        ("emrach_base", "EMR-ACH\n(base)"), ("emrach_calib", "+Calib"),
        ("emrach_contrast", "+Contrast"), ("emrach_diag", "+Diag"),
        ("emrach_absence", "+Absence"), ("emrach_multiagent", "+Multi\nAgent"),
        ("emrach_full", "Full\nSystem"),
    ]
    vals, labels, colors = [], [], []
    for k, lbl in kl_keys:
        if k in summary:
            vals.append(summary[k].get("kl_div", 0))
            labels.append(lbl)
            colors.append("#9ecae1" if k.startswith("b") else (COLORS["best"] if k == "emrach_full" else COLORS["ours"]))

    if not vals:
        print("  [skip] No KL data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("KL Divergence (log scale)")
    ax.set_title("KL Divergence — Predicted vs. Ground-Truth Distribution", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4, which="both")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_kl_divergence.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_kl_divergence.png done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--also-update-paper", action="store_true",
                        help="Also copy figures to paper/figures/")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    summary = load_summary(cfg)
    if not summary:
        return

    out_dir = cfg.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures from real data -> {out_dir}")

    fig_main_results(summary, out_dir)
    fig_ablation(summary, out_dir)
    fig_multiagent(summary, out_dir)
    fig_ece(summary, out_dir)
    fig_kl(summary, out_dir)

    print(f"\nFigures saved to {out_dir}")

    if args.also_update_paper:
        paper_figs = Path(__file__).parent.parent / "paper" / "figures"
        import shutil
        for fig in out_dir.glob("*.png"):
            dest = paper_figs / fig.name
            shutil.copy(fig, dest)
            print(f"  Updated {dest}")
        print("Paper figures updated.")


if __name__ == "__main__":
    main()
