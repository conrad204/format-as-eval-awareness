"""Matched-format-PCA erasure dose-response: cross-format transfer gap collapses with
rank while purpose AUC stays flat, across all three checkpoints and all three pinned
depths, with payload-block-bootstrapped 95% CIs (N_BOOT=1000).

Self-contained. Reads only results/full320_v3_erasure_dose_response_bootstrap.json
(computed on the uahpc cluster via scripts/full320_v3_erasure_dose_response_bootstrap.py
-- 5-fold cross-fitted OOF scoring at ranks {0(raw),1,4,16,64}, then payload-block
resampling of the cached scores; no refitting in the bootstrap stage itself). No
activation extraction or probe refitting in this plotting script.

Message: as more of the matched-format PCA subspace is removed, the b2c/c2b transfer
gap falls sharply, with confidence intervals that do not overlap the raw baseline at
higher ranks, while purpose AUC's CI stays essentially unchanged from its raw value at
every depth and every checkpoint. Format itself is NOT driven to chance by this method
(format_auc_after_erasure stays well above 0.5 at every rank tested) -- see
scripts/full320_v3_format_erasure_retrain.py's own module docstring; this figure does
not claim complete format erasure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results" / "full320_v3_erasure_dose_response_bootstrap.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.3-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.1-70B": "llama31_70b", "Llama-3.3-70B": "llama33_70b"}
DEPTH_ORDER = ["spike", "min_gap", "final"]
DEPTH_COLOR = {"spike": "#C9502F", "min_gap": "#1F7A5C", "final": "#2A5FA5"}
INK = "#202124"
GRID = "#D9DDE3"
RANKS = [0, 1, 4, 16, 64]


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
    n_boot = data["n_boot"]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), constrained_layout=True, sharex=True)

    for col, label in enumerate(MODEL_ORDER):
        key = MODEL_KEYS[label]
        d = data["by_model"][key]
        depth_role = d["depth_role"]
        layer_by_role = {v: k for k, v in depth_role.items()}

        ax_purpose, ax_gap = axes[0, col], axes[1, col]
        for depth in DEPTH_ORDER:
            by_rank = d["by_depth"][depth]
            purpose = np.array([by_rank[str(r)]["purpose_auc"] for r in RANKS])
            purpose_ci = np.array([by_rank[str(r)]["purpose_auc_ci95"] for r in RANKS])
            gap = np.array([by_rank[str(r)]["gap"] for r in RANKS])
            gap_ci = np.array([by_rank[str(r)]["gap_ci95"] for r in RANKS])
            color = DEPTH_COLOR[depth]

            purpose_err = np.abs(purpose_ci.T - purpose)
            gap_err = np.abs(gap_ci.T - gap)
            ax_purpose.errorbar(RANKS, purpose, yerr=purpose_err, fmt="o-", color=color,
                                 label=depth, markersize=4.5, linewidth=1.3, capsize=2.5, elinewidth=1.0)
            ax_gap.errorbar(RANKS, gap, yerr=gap_err, fmt="o-", color=color,
                             label=depth, markersize=4.5, linewidth=1.3, capsize=2.5, elinewidth=1.0)

        for ax in (ax_purpose, ax_gap):
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks(RANKS)
            ax.set_xticklabels(["raw"] + [str(r) for r in RANKS[1:]])
            ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
            ax.set_axisbelow(True)
        ax_purpose.set_title(label, loc="left", fontsize=10, fontweight="bold")
        ax_gap.set_xlabel("matched-format PCA rank removed")

    axes[0, 0].set_ylabel("Purpose AUC\nafter erasure")
    axes[1, 0].set_ylabel("Cross-format transfer gap\nafter erasure")
    axes[0, 0].legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    fig.suptitle(
        f"Transfer gap collapses with matched-format rank; purpose AUC stays flat\n"
        f"(error bars: payload-block-bootstrap 95% CI, N={n_boot})",
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
        d = data["by_model"][key]
        depth_role = d["depth_role"]
        layer_by_role = {v: k for k, v in depth_role.items()}
        for depth in DEPTH_ORDER:
            by_rank = d["by_depth"][depth]
            raw = by_rank["0"]
            final = by_rank["64"]
            overlap = not (final["gap_ci95"][1] < raw["gap_ci95"][0] or final["gap_ci95"][0] > raw["gap_ci95"][1])
            print(f"{label:<14} {depth:<8} gap raw={raw['gap']:.4f} CI={raw['gap_ci95']} -> "
                  f"rank64={final['gap']:.4f} CI={final['gap_ci95']} "
                  f"{'[CIs OVERLAP]' if overlap else '[CIs separate]'}")
    print(f"wrote {OUT_DIR / 'matched_format_erasure_dose_response.pdf'}")


if __name__ == "__main__":
    main()
