"""Matched-format-PCA erasure dose-response: cross-format transfer gap collapses with
rank while purpose AUC stays flat, across all three checkpoints and all three pinned
depths.

Self-contained. Reads only results/full320_v3_format_erasure_retrain.json (no
activation extraction or probe refitting here). Rank grid differs by model
(llama31_8b: 1/4/16/64; both 70Bs: 1/16) because the 70B (8192-dim) runs used a
thinned grid for compute-cost reasons -- this is preserved as-is, not padded or
interpolated.

Message: as more of the matched-format PCA subspace is removed, the b2c/c2b transfer
gap falls sharply and close to monotonically, while purpose AUC is essentially flat
(within ~1 point) at every depth and every checkpoint. This is the raw erasure-retrain
result the direction-geometry / conditional-null analyses explain mechanistically.
Format itself is NOT driven to chance by this method (format_auc_after_erasure stays
well above 0.5 even at the largest tested rank) -- see
scripts/full320_v3_format_erasure_retrain.py's own module docstring; this figure
does not claim complete format erasure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results" / "full320_v3_format_erasure_retrain.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.3-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.1-70B": "llama31_70b", "Llama-3.3-70B": "llama33_70b"}
DEPTH_ORDER = ["spike", "min_gap", "final"]
DEPTH_COLOR = {"spike": "#C9502F", "min_gap": "#1F7A5C", "final": "#2A5FA5"}
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
    data = json.loads(DATA_PATH.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), constrained_layout=True, sharex=False)

    for col, label in enumerate(MODEL_ORDER):
        key = MODEL_KEYS[label]
        d = data[key]
        depth_role = d["depth_role"]
        layer_by_role = {v: k for k, v in depth_role.items()}
        ranks = sorted(int(r) for r in d["method_grid"]["matched_pca"])

        ax_purpose, ax_gap = axes[0, col], axes[1, col]
        for depth in DEPTH_ORDER:
            layer = layer_by_role[depth]
            by_layer = d["by_layer"][layer]
            raw_purpose = by_layer["raw_baseline"]["purpose_auc"]
            raw_gap = by_layer["raw_baseline"]["gap"]
            purpose = [raw_purpose] + [by_layer["matched_pca"][str(r)]["purpose_auc_after_erasure"] for r in ranks]
            gap = [raw_gap] + [by_layer["matched_pca"][str(r)]["gap_after_erasure"] for r in ranks]
            x = [0] + ranks
            color = DEPTH_COLOR[depth]
            ax_purpose.plot(x, purpose, "o-", color=color, label=depth, markersize=4.5, linewidth=1.3)
            ax_gap.plot(x, gap, "o-", color=color, label=depth, markersize=4.5, linewidth=1.3)

        for ax in (ax_purpose, ax_gap):
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([0] + ranks)
            ax.set_xticklabels(["raw"] + [str(r) for r in ranks])
            ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
            ax.set_axisbelow(True)
        ax_purpose.set_title(label, loc="left", fontsize=10, fontweight="bold")
        ax_gap.set_xlabel("matched-format PCA rank removed")

    axes[0, 0].set_ylabel("Purpose AUC\nafter erasure")
    axes[1, 0].set_ylabel("Cross-format transfer gap\nafter erasure")
    axes[0, 0].legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    fig.suptitle(
        "Transfer gap collapses with matched-format rank; purpose AUC stays flat",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"matched_format_erasure_dose_response.{suffix}", **kwargs)
    plt.close(fig)

    for label in MODEL_ORDER:
        key = MODEL_KEYS[label]
        d = data[key]
        depth_role = d["depth_role"]
        layer_by_role = {v: k for k, v in depth_role.items()}
        ranks = sorted(int(r) for r in d["method_grid"]["matched_pca"])
        for depth in DEPTH_ORDER:
            layer = layer_by_role[depth]
            by_layer = d["by_layer"][layer]
            raw_gap = by_layer["raw_baseline"]["gap"]
            final_gap = by_layer["matched_pca"][str(ranks[-1])]["gap_after_erasure"]
            print(f"{label:<14} {depth:<8} gap raw={raw_gap:.4f} -> rank{ranks[-1]}={final_gap:.4f} "
                  f"({100*(1-final_gap/raw_gap):.0f}% reduction)")
    print(f"wrote {OUT_DIR / 'matched_format_erasure_dose_response.pdf'}")


if __name__ == "__main__":
    main()
