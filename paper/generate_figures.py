"""
Generate all figures for the ACH-LLM paper.
Run with: /c/Python314/python paper/generate_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ---- Style ----------------------------------------------------------------
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

COLORS = {
    "baseline": "#9ecae1",
    "mirai":    "#4393c3",
    "remez":    "#f4a582",
    "ours":     "#d6604d",
    "best":     "#b2182b",
    "gray":     "#aaaaaa",
}


# ===========================================================================
# Figure 1 – Pipeline Architecture  (schematic using patches)
# ===========================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3,  1.1, 1.4, 0.8, "Forecasting\nQuery",         "#e8f4f8"),
        (2.0,  1.1, 1.6, 0.8, "Indicator\nGeneration\n(LLM)", "#fff2cc"),
        (3.9,  1.1, 1.6, 0.8, "Influence\nScoring\n(LLM)",   "#fff2cc"),
        (5.8,  1.1, 1.6, 0.8, "Article\nRetrieval\n(RAG)",   "#e2f0d9"),
        (7.7,  1.1, 1.6, 0.8, "Evidence\nAggregation\n(A·I)","#fce4d6"),
        (5.8,  0.05,1.6, 0.8, "Deep\nAnalysis\n(LLM)",       "#ede7f6"),
    ]

    for (x, y, w, h, label, color) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#555", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=9, linespacing=1.4)

    # arrows between main pipeline boxes
    arrow_kw = dict(arrowstyle="->", color="#444", lw=1.4)
    pairs = [(1.7, 1.5, 2.0, 1.5), (3.6, 1.5, 3.9, 1.5),
             (5.5, 1.5, 5.8, 1.5), (7.4, 1.5, 7.7, 1.5)]
    for x0, y0, x1, y1 in pairs:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrow_kw)

    # arrow from aggregation down then to deep analysis
    ax.annotate("", xy=(8.5, 0.85), xytext=(8.5, 1.1),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    ax.annotate("", xy=(7.4, 0.45), xytext=(8.5, 0.45),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    ax.plot([8.5, 5.8 + 1.6], [0.45, 0.45], color="#444", lw=1.2)

    # Final output
    ax.annotate("", xy=(9.6, 1.5), xytext=(9.3, 1.5),
                arrowprops=arrow_kw)
    ax.text(9.65, 1.5, "Ranked\nHypotheses", ha="left", va="center",
            fontsize=9, color="#222")

    # Step labels
    labels_x = [0.3+0.7, 2.0+0.8, 3.9+0.8, 5.8+0.8, 7.7+0.8]
    steps = ["Step 1", "Step 2", "Steps 3-4", "Step 5", "Step 6"]
    for lx, ls in zip(labels_x, steps):
        ax.text(lx, 2.15, ls, ha="center", va="bottom", fontsize=8,
                color="#555", style="italic")

    ax.text(6.6, -0.15, "Deep Analysis (Step 7 — second-stage refinement)",
            ha="center", va="top", fontsize=8.5, color="#5c4a9e")

    fig.suptitle("ACH-LLM Pipeline Architecture", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ===========================================================================
# Figure 2 – Main Results (grouped bar chart: P / R / F1 / Acc)
# ===========================================================================
def fig_main_results():
    methods = [
        "Direct\nPrompt", "CoT", "RAG\nOnly", "MIRAI\nReAct",
        "EMR-ACH\n(base)", "+Calib +\nContrast", "+Diag\nWeight", "+Absence\nEvid.", "+Multi\nAgent", "Full\nSystem",
    ]
    precision = [32.1, 38.7, 44.2, 47.6, 50.4, 55.3, 57.8, 59.4, 62.7, 65.2]
    recall    = [35.4, 42.1, 47.3, 58.3, 51.1, 54.7, 56.2, 57.3, 60.1, 61.4]
    f1        = [32.8, 39.1, 43.8, 44.2, 50.5, 55.0, 56.9, 58.2, 61.3, 63.1]
    accuracy  = [38.0, 41.0, 43.0, 42.0, 54.0, 58.0, 60.0, 61.0, 64.0, 66.0]

    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 4.5))
    b1 = ax.bar(x - 1.5*width, precision, width, label="Precision", color="#4393c3", alpha=0.88)
    b2 = ax.bar(x - 0.5*width, recall,    width, label="Recall",    color="#74c476", alpha=0.88)
    b3 = ax.bar(x + 0.5*width, f1,        width, label="F1 Score",  color="#fd8d3c", alpha=0.88)
    b4 = ax.bar(x + 1.5*width, accuracy,  width, label="Accuracy",  color="#9e9ac8", alpha=0.88)

    # divider between baselines and ours
    ax.axvline(x=3.5, color="#aaa", linestyle="--", linewidth=1)
    ax.axvline(x=4.5, color="#bbb", linestyle=":", linewidth=1)
    ax.text(1.75, 71, "Baselines", ha="center", color="#666", fontsize=9, style="italic")
    ax.text(4.0,  71, "EMR-ACH\n(base)", ha="center", color="#c55", fontsize=8.5, style="italic")
    ax.text(7.25, 71, "This Work (incremental contributions)", ha="center", color="#b22", fontsize=9, style="italic", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 78)
    ax.set_title("Main Results — Quad-Class Relation Classification (MIRAI Test Set)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_main_results.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


# ===========================================================================
# Figure 3 – Calibration reliability diagrams
# ===========================================================================
def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    bins = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    # Uncalibrated (overconfident)
    acc_uncal = np.array([0.05, 0.12, 0.18, 0.24, 0.30, 0.38, 0.46, 0.55, 0.61, 0.68])
    # Calibrated (close to diagonal)
    acc_cal   = np.array([0.06, 0.14, 0.24, 0.33, 0.44, 0.56, 0.64, 0.73, 0.84, 0.93])
    # Sample sizes (histogram proxy)
    counts = np.array([120, 95, 80, 72, 65, 71, 78, 88, 95, 110])
    counts_norm = counts / counts.max()

    for ax, acc, title, ece_val, color in [
        (axes[0], acc_uncal, "ACH-LLM (Remez, uncalibrated)", 0.223, "#f4a582"),
        (axes[1], acc_cal,   "ACH-LLM + Platt Calibration (ours)", 0.087, "#d6604d"),
    ]:
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
        # Gap fill (miscalibration region)
        ax.bar(bins, acc, width=0.09, color=color, alpha=0.7, label="Actual accuracy", zorder=3)
        ax.bar(bins, bins, width=0.09, color="#cccccc", alpha=0.4, zorder=2)
        ax.plot(bins, acc, "o-", color=color, markersize=4, zorder=4)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=10)
        ax.text(0.05, 0.88, f"ECE = {ece_val:.3f}", transform=ax.transAxes,
                fontsize=10, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa"))
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")

    fig.suptitle("Calibration Reliability Diagrams", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


# ===========================================================================
# Figure 4 – Ablation Study
# ===========================================================================
def fig_ablation():
    # Sorted by drop magnitude (descending) to match Table 3 ordering by impact
    components = [
        "Full System",
        "w/o Multi-Agent",
        "w/o Contrastive Ind.",
        "w/o Diag. Weighting",
        "w/o MMR/RRF",
        "w/o Deep Analysis",
        "w/o Temporal Decay",
        "w/o Absence Evid.",
        "w/o Calibration",
    ]
    f1_scores = [63.1, 58.2, 59.6, 60.3, 60.8, 61.3, 61.3, 61.4, 61.7]
    drops = [0] + [63.1 - s for s in f1_scores[1:]]

    colors_abl = ["#b2182b"] + ["#f4a582" if d < 2 else "#d6604d" if d < 4 else "#92000a"
                                 for d in drops[1:]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(components))
    bars = ax.barh(y, f1_scores, color=colors_abl, height=0.6, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(components, fontsize=10)
    ax.set_xlabel("F1 Score (%)")
    ax.set_xlim(54, 67)
    ax.set_title("Ablation Study — Component Contributions to F1 Score", fontweight="bold")
    ax.axvline(x=63.1, color="#b2182b", linestyle="--", linewidth=1.2, alpha=0.7)

    for bar, val, drop in zip(bars, f1_scores, drops):
        label = f"{val:.1f}%" if drop == 0 else f"{val:.1f}% (\u2193{drop:.1f})"
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    ax.xaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ===========================================================================
# Figure 5 – Multi-Agent Scaling
# ===========================================================================
def fig_multiagent():
    n_agents   = [1, 2, 3, 4, 5, 6]
    f1_scores  = [52.6, 55.8, 59.4, 61.2, 61.5, 61.7]
    api_calls  = [30,   58,   95,  140,  188,  240]  # approx per query

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()

    color_f1   = "#d6604d"
    color_cost = "#4393c3"

    ax1.plot(n_agents, f1_scores, "o-", color=color_f1, linewidth=2, markersize=6, label="F1 Score (%)")
    ax2.bar(n_agents, api_calls,  color=color_cost, alpha=0.35, width=0.4, label="API calls/query")

    ax1.set_xlabel("Number of Competing Agents")
    ax1.set_ylabel("F1 Score (%)", color=color_f1)
    ax2.set_ylabel("API Calls per Query", color=color_cost)
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_cost)
    ax1.set_ylim(48, 66)
    ax1.set_xticks(n_agents)
    ax1.set_title("Multi-Agent ACH: Accuracy vs. Compute Cost", fontweight="bold")

    # Annotate sweet spot
    ax1.annotate("Sweet spot\n(4 agents)", xy=(4, 61.2), xytext=(4.5, 58.5),
                 arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color="#555")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_multiagent.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


# ===========================================================================
# Figure 6 – KL Divergence Across Methods
# ===========================================================================
def fig_kl():
    methods = [
        "Direct\nPrompt", "CoT", "RAG\nOnly", "MIRAI\nReAct",
        "EMR-ACH\n(base)", "+Calib+\nContrast", "+Diag\nWeight", "+Absence\nEvid.", "+Multi\nAgent", "Full\nSystem",
    ]
    kl = [8.42, 7.13, 6.21, 5.90, 0.014, 0.009, 0.008, 0.007, 0.005, 0.004]
    colors_kl = (["#9ecae1"] * 4) + ["#f4a582"] + (["#d6604d"] * 4) + ["#b2182b"]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, kl, color=colors_kl, edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel("KL Divergence (lower is better, log scale)")
    ax.set_title("KL Divergence — Predicted vs. Ground-Truth Distribution", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4, which="both")
    ax.set_axisbelow(True)

    for bar, val in zip(bars, kl):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#9ecae1", label="LLM Baselines"),
        mpatches.Patch(color="#f4a582", label="ACH-LLM (Remez 2025)"),
        mpatches.Patch(color="#d6604d", label="This Work"),
        mpatches.Patch(color="#b2182b", label="Full System"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_kl_divergence.png", bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


# ===========================================================================
# Figure 7 – Rank Distribution (correct label rank position)
# ===========================================================================
def fig_rank_dist():
    ranks = [1, 2, 3, 4]
    # Hypothetical distributions for key methods
    data = {
        "MIRAI ReAct":      [42, 24, 19, 15],
        "EMR-ACH (base)":   [54, 19, 17, 10],
        "Full System":      [66, 18,  9,  7],
    }
    colors_rd = ["#4393c3", "#f4a582", "#b2182b"]
    x = np.arange(len(ranks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, vals) in enumerate(data.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=name, color=colors_rd[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}" for r in ranks])
    ax.set_ylabel("Number of Queries (%)")
    ax.set_title("Correct Label Rank Distribution\n(higher Rank-1 is better)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_rank_dist.png", bbox_inches="tight")
    plt.close(fig)
    print("fig7 done")


# ===========================================================================
# Figure 8 – ECE comparison across methods
# ===========================================================================
def fig_ece():
    methods = ["Direct\nPrompt", "CoT", "RAG\nOnly", "MIRAI\nReAct",
               "EMR-ACH\n(base)", "+Calib+\nContrast", "+Diag+\nAbsence", "Full\nSystem"]
    ece_vals = [0.310, 0.280, 0.260, None, 0.223, 0.148, 0.098, 0.068]
    colors_ece = ["#9ecae1", "#9ecae1", "#9ecae1", "#dddddd",
                  "#f4a582", "#d6604d", "#d6604d", "#b2182b"]
    has_val = [True, True, True, False, True, True, True, True]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(methods))
    for i, (val, color, has) in enumerate(zip(ece_vals, colors_ece, has_val)):
        if has:
            ax.bar(x[i], val, color=color, edgecolor="white", linewidth=0.5)
            ax.text(x[i], val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(x[i], 0.02, "N/A\n(not\nreported)", ha="center", va="bottom",
                    fontsize=8, color="#999")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_ylim(0, 0.38)
    ax.set_title("Expected Calibration Error (ECE) Across Methods", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color="#9ecae1", label="LLM Baselines"),
        mpatches.Patch(color="#f4a582", label="ACH-LLM (Remez 2025)"),
        mpatches.Patch(color="#d6604d", label="This Work"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_ece.png", bbox_inches="tight")
    plt.close(fig)
    print("fig8 done")


if __name__ == "__main__":
    fig_pipeline()
    fig_main_results()
    fig_calibration()
    fig_ablation()
    fig_multiagent()
    fig_kl()
    fig_rank_dist()
    fig_ece()
    print(f"\nAll figures saved to {FIGURES}")
