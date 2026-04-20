"""
Generate comparison tables (LaTeX + plain text) for the paper.

Reads metrics_summary.json and outputs:
  - Table 1: MIRAI build-up results
  - Table 3: Ablation study
  - Multi-agent scaling numbers

Usage:
  python analysis/compare_systems.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

# Mapping: internal name -> display name for tables
TABLE1_ORDER = [
    ("b1_direct",         "Direct Prompting"),
    ("b2_cot",            "CoT Prompting"),
    ("b3_ragonly",        "RAG-Only"),
    ("emrach_base",       "EMR-ACH (base)"),
    ("emrach_calib",      "+ Calibrated mapping"),
    ("emrach_contrast",   "+ Contrastive indicators"),
    ("emrach_diag",       "+ Diagnostic weighting"),
    ("emrach_absence",    "+ Absence-of-evidence"),
    ("emrach_multiagent", "+ Multi-agent ACH"),
    ("emrach_full",       "+ Deep Analysis (Full System)"),
]

TABLE3_ORDER = [
    ("emrach_full",       "Full System",                  None),
    ("abl_no_multiagent", "w/o Multi-agent (single)",     "emrach_full"),
    ("abl_no_deepanalysis","w/o Deep Analysis",           "emrach_full"),
    ("abl_no_contrastive","w/o Contrastive indicators",   "emrach_full"),
    ("abl_no_diag",       "w/o Diagnostic weighting",     "emrach_full"),
    ("abl_no_absence",    "w/o Absence-of-evidence",      "emrach_full"),
    ("abl_no_calib",      "w/o Calibrated mapping",       "emrach_full"),
    ("abl_no_mmr",        "w/o MMR/RRF (BM25 only)",      "emrach_full"),
    ("abl_no_decay",      "w/o Temporal decay",           "emrach_full"),
]


def load_summary(cfg) -> dict:
    path = cfg.results_dir / "metrics_summary.json"
    if not path.exists():
        print(f"[WARN] {path} not found. Run compute_metrics.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def print_table1(summary: dict) -> None:
    print("\n" + "=" * 90)
    print("TABLE 1: MIRAI Quad-Class Results")
    print("=" * 90)
    header = f"{'Method':<35} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} {'Acc%':>5} {'KL':>8} {'ECE':>7}  {'CI [F1]':>16}"
    print(header)
    print("-" * 90)

    for key, display_name in TABLE1_ORDER:
        if key not in summary:
            print(f"  {display_name:<33} {'N/A':>6}")
            continue
        m = summary[key]
        ci = f"[{m['f1_ci'][0]:.1f}, {m['f1_ci'][1]:.1f}]" if m.get("f1_ci") else ""
        print(
            f"  {display_name:<33} {m.get('precision', 0):>6.1f} {m.get('recall', 0):>6.1f} "
            f"{m.get('f1', 0):>6.1f} {m.get('accuracy', 0):>5.1f} "
            f"{m.get('kl_div', 0):>8.4f} {m.get('ece', 0):>7.4f}  {ci:>16}"
        )


def print_table3(summary: dict) -> None:
    print("\n" + "=" * 80)
    print("TABLE 3: Ablation Study")
    print("=" * 80)
    header = f"{'Configuration':<35} {'F1%':>6} {'ΔF1':>7} {'Acc%':>5} {'ECE':>7}"
    print(header)
    print("-" * 80)

    full_f1 = summary.get("emrach_full", {}).get("f1", 0.0)
    for key, display_name, ref_key in TABLE3_ORDER:
        if key not in summary:
            print(f"  {display_name:<33} {'N/A':>6}")
            continue
        m = summary[key]
        f1 = m.get("f1", 0.0)
        delta = ""
        if ref_key and ref_key in summary:
            df = f1 - summary[ref_key].get("f1", 0.0)
            delta = f"{df:+.1f}"
        print(
            f"  {display_name:<33} {f1:>6.1f} {delta:>7} "
            f"{m.get('accuracy', 0):>5.1f} {m.get('ece', 0):>7.4f}"
        )


def print_multiagent_table(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("MULTI-AGENT SCALING")
    print("=" * 70)
    print(f"{'N Agents':>8}  {'F1%':>6}  {'Acc%':>6}  {'API calls/q':>12}")
    print("-" * 40)
    for n_agents in [1, 2, 3, 4, 5, 6]:
        key = f"ma_agents{n_agents}"
        if key not in summary:
            continue
        m = summary[key]
        print(
            f"{n_agents:>8}  {m.get('f1', 0):>6.1f}  {m.get('accuracy', 0):>6.1f}  "
            f"{m.get('api_calls_per_query', 0):>12}"
        )


def generate_latex_table1(summary: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{MIRAI quad-class results. Bold = best value.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Prec & Rec & F1 & Acc & KL$\downarrow$ & ECE$\downarrow$ \\",
        r"\midrule",
    ]
    for key, display_name in TABLE1_ORDER:
        if key not in summary:
            lines.append(f"{display_name} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        m = summary[key]
        lines.append(
            f"{display_name} & {m.get('precision',0):.1f} & {m.get('recall',0):.1f} & "
            f"{m.get('f1',0):.1f} & {m.get('accuracy',0):.1f} & "
            f"{m.get('kl_div',0):.4f} & {m.get('ece',0):.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    cfg = get_config()
    summary = load_summary(cfg)
    if not summary:
        return

    print_table1(summary)
    print_table3(summary)
    print_multiagent_table(summary)

    # Save LaTeX
    latex = generate_latex_table1(summary)
    latex_path = cfg.results_dir / "table1_latex.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX Table 1 saved to {latex_path}")


if __name__ == "__main__":
    main()
