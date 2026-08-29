"""Regularization-robustness figure: depth-wise purpose AUC and cross-format transfer
gap swept over the logistic C penalty (0.01, 0.1, 1, 10 -- 1000x range) at the pinned
min-gap/final/spike depths, all three checkpoints.

Self-contained. Reads only results/full320_v3_c_robustness.json (no activation
extraction, no probe refitting here).

Message: neither the purpose-AUC magnitude nor the transfer gap is sensitive to the
pinned C=0.1 operating point -- both are stable across three orders of magnitude in
regularization strength, at every depth and every checkpoint tested. This rules out
"the effect is a regularization artifact" as an explanation for either quantity.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results" / "full320_v3_c_robustness.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.3-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.1-70B": "llama31_70b", "Llama-3.3-70B": "llama33_70b"}
DEPTH_ORDER = ["spike", "min_gap", "final"]
DEPTH_COLOR = {"spike": "#C9502F", "min_gap": "#1F7A5C", "final": "#2A5FA5"}
INK = "#202124"
GRID = "#D9DDE3"
C_GRID = [0.01, 0.1, 1.0, 10.0]


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    data = json.loads(DATA_PATH.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), constrained_layout=True, sharex=True)

    for col, label in enumerate(MODEL_ORDER):
        key = MODEL_KEYS[label]
        d = data[key]
        depth_role = d["depth_role"]  # {"14": "min_gap", ...}
        layer_by_role = {v: k for k, v in depth_role.items()}

        ax_auc, ax_gap = axes[0, col], axes[1, col]
        for depth in DEPTH_ORDER:
            layer = layer_by_role[depth]
            by_c = d["by_layer"][layer]
            aucs = [by_c[str(c)]["auc_decorr"] for c in C_GRID]
            gaps = [by_c[str(c)]["gap_point"] for c in C_GRID]
            color = DEPTH_COLOR[depth]
            ax_auc.plot(C_GRID, aucs, "o-", color=color, label=depth, markersize=4.5, linewidth=1.3)
            ax_gap.plot(C_GRID, gaps, "o-", color=color, label=depth, markersize=4.5, linewidth=1.3)

        for ax in (ax_auc, ax_gap):
            ax.set_xscale("log")
            ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
            ax.set_axisbelow(True)
        ax_auc.set_title(label, loc="left", fontsize=10, fontweight="bold")
        ax_gap.set_xlabel("logistic C (pinned value = 0.1)")

    axes[0, 0].set_ylabel("Purpose AUC\n(decorrelated regime)")
    axes[1, 0].set_ylabel("Cross-format transfer gap")
    axes[0, 0].legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    fig.suptitle(
        "Purpose AUC and transfer gap are stable across a 1000x range of C",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"c_robustness_sweep.{suffix}", **kwargs)
    plt.close(fig)

    for label in MODEL_ORDER:
        key = MODEL_KEYS[label]
        d = data[key]
        depth_role = d["depth_role"]
        layer_by_role = {v: k for k, v in depth_role.items()}
        for depth in DEPTH_ORDER:
            layer = layer_by_role[depth]
            by_c = d["by_layer"][layer]
            aucs = [by_c[str(c)]["auc_decorr"] for c in C_GRID]
            gaps = [by_c[str(c)]["gap_point"] for c in C_GRID]
            print(f"{label:<14} {depth:<8} AUC range=[{min(aucs):.4f},{max(aucs):.4f}] "
                  f"gap range=[{min(gaps):.4f},{max(gaps):.4f}]")
    print(f"wrote {OUT_DIR / 'c_robustness_sweep.pdf'}")


if __name__ == "__main__":
    main()
