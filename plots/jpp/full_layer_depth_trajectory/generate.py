"""Full-layer depth trajectory, replacing the "spike/min_gap/final" 3-point convention.

That 3-point labeling was found to be inconsistent across checkpoints: "min_gap" sits
at 23.8-26.2% relative depth for the 70Bs but 43.8% for the 8B -- not the "same depth
role" across models despite sharing a label. This figure instead plots the actual
continuous depth trajectory (relative depth 0->1, so the 32-layer 8B and 80-layer 70Bs
are directly comparable) for purpose AUC and cross-format transfer gap, both raw
(no erasure, pinned per-layer sweep) and after matched-format-PCA erasure at rank=64
(new full-layer sweep, computed on uahpc: scripts/full320_v3_erasure_full_layer_sweep.py).
The old spike/min_gap/final points are kept as small markers for continuity with
earlier figures/discussion, not as the primary story.

Self-contained. Reads only:
  results/layer_sweep_full320-v3_<model>.json           (raw baseline, pinned, all layers)
  results/full320_v3_erasure_full_layer_sweep.json      (rank=64 erasure, all layers)
No activation extraction or probe refitting in this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RAW_PATHS = {
    "llama31_8b": ROOT / "results" / "layer_sweep_full320-v3_llama31_8b.json",
    "llama31_70b": ROOT / "results" / "layer_sweep_full320-v3_llama31_70b.json",
    "llama33_70b": ROOT / "results" / "layer_sweep_full320-v3_llama33_70b.json",
}
ERASED_PATH = ROOT / "results" / "full320_v3_erasure_full_layer_sweep.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.3-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.1-70B": "llama31_70b", "Llama-3.3-70B": "llama33_70b"}
MODEL_COLOR = {"llama31_8b": "#2A5FA5", "llama31_70b": "#C9502F", "llama33_70b": "#1F7A5C"}
# Kept only as reference markers on the continuous curve, not as the primary x-axis.
OLD_DEPTH_LAYERS = {
    "llama31_8b": {"spike": 3, "min_gap": 14, "final": 31},
    "llama31_70b": {"spike": 4, "min_gap": 19, "final": 79},
    "llama33_70b": {"spike": 4, "min_gap": 21, "final": 79},
}
INK = "#202124"
GRID = "#D9DDE3"


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
    erased = json.loads(ERASED_PATH.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax_purpose, ax_gap = axes

    for label in MODEL_ORDER:
        key = MODEL_KEYS[label]
        raw = json.loads(RAW_PATHS[key].read_text())
        n_layers = len(raw)
        rel_depth = np.array([r["layer"] / (n_layers - 1) for r in raw])
        raw_purpose = np.array([r["decorrelated"]["auc"] for r in raw])
        raw_gap = np.array([r["decorrelated"]["auc"] - (r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in raw])

        erased_by_layer = erased["by_model"][key]["by_layer"]
        erased_purpose = np.array([erased_by_layer[str(L)]["purpose_auc_after_erasure"] for L in range(n_layers)])
        erased_gap = np.array([erased_by_layer[str(L)]["gap_after_erasure"] for L in range(n_layers)])

        color = MODEL_COLOR[key]
        ax_purpose.plot(rel_depth, raw_purpose, "-", color=color, linewidth=1.4, alpha=0.9, label=f"{label} (raw)")
        ax_purpose.plot(rel_depth, erased_purpose, "--", color=color, linewidth=1.1, alpha=0.7)
        ax_gap.plot(rel_depth, raw_gap, "-", color=color, linewidth=1.4, alpha=0.9, label=f"{label} (raw)")
        ax_gap.plot(rel_depth, erased_gap, "--", color=color, linewidth=1.1, alpha=0.7)

        for name, L in OLD_DEPTH_LAYERS[key].items():
            rd = L / (n_layers - 1)
            ax_purpose.plot(rd, raw_purpose[L], "o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
            ax_gap.plot(rd, raw_gap[L], "o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    for ax in (ax_purpose, ax_gap):
        ax.set_xlabel("relative depth (0 = first layer, 1 = last layer)")
        ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.02, 1.02)

    ax_purpose.set_ylabel("Purpose AUC\n(solid = raw, dashed = rank-64 erased)")
    ax_gap.set_ylabel("Cross-format transfer gap\n(solid = raw, dashed = rank-64 erased)")
    ax_purpose.set_title("Purpose decodability vs. relative depth", loc="left", fontsize=10, fontweight="bold")
    ax_gap.set_title("Transfer gap vs. relative depth", loc="left", fontsize=10, fontweight="bold")
    ax_purpose.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "Full-layer depth trajectory (dots: old spike/min_gap/final reference points)",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"full_layer_depth_trajectory.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'full_layer_depth_trajectory.pdf'}")


if __name__ == "__main__":
    main()
