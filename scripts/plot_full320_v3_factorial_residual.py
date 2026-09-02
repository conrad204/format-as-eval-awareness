#!/usr/bin/env python3
"""Plot integrated gaps, cross-fitted C capture, and incremental specificity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GAP_CONDITIONS = (
    ("raw", "Raw", "#555555"),
    ("minus_b", r"$-Q_B$", "#377eb8"),
    ("minus_b_cperp", r"$-[Q_B,Q_{C\perp B}]$", "#e41a1c"),
    ("minus_b_rperp", r"$-[Q_B,Q_{R\perp B}]$", "#4daf4a"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    models = list(data["by_model"])
    fig, axes = plt.subplots(3, len(models), figsize=(5.1 * len(models), 10.0), squeeze=False)

    for col, model in enumerate(models):
        result = data["by_model"][model]
        ranks = result["ranks"]
        x = np.arange(len(ranks))
        gap_ax, capture_ax, specific_ax = axes[:, col]
        for condition, label, color in GAP_CONDITIONS:
            means, lows, highs = [], [], []
            for rank in ranks:
                cell = result["by_rank"][str(rank)]["by_condition"][condition]
                means.append(cell["integrated_gap"])
                lows.append(cell["integrated_gap_ci95"][0])
                highs.append(cell["integrated_gap_ci95"][1])
            means = np.asarray(means)
            gap_ax.errorbar(
                x,
                means,
                yerr=np.vstack([means - lows, np.asarray(highs) - means]),
                marker="o",
                linewidth=1.7,
                capsize=3,
                color=color,
                label=label,
            )
        for rank in ranks:
            capture = result["c_capture_analysis"]["by_rank"][str(rank)]["c_capture_by_layer"]
            capture_ax.plot(result["q"], capture, label=f"r={rank}", linewidth=1.6)
        specific, low, high = [], [], []
        for rank in ranks:
            cell = result["by_rank"][str(rank)]["paired"]["specific_C_increment"]
            specific.append(cell["specific_C"])
            low.append(cell["specific_C_ci95"][0])
            high.append(cell["specific_C_ci95"][1])
        specific = np.asarray(specific)
        specific_ax.bar(x, specific, color="#984ea3", width=0.65)
        specific_ax.errorbar(
            x,
            specific,
            yerr=np.vstack([specific - low, np.asarray(high) - specific]),
            fmt="none",
            color="black",
            capsize=3,
        )
        specific_ax.axhline(0, color="black", linewidth=0.8)
        gap_ax.set_title(model)
        gap_ax.set_ylabel("Integrated transfer gap I")
        capture_ax.set_ylabel(r"Cross-fitted $C$ capture")
        capture_ax.set_xlabel("Normalized depth q")
        specific_ax.set_ylabel(r"Specific $C$: extra$_C$ - extra$_{random}$")
        specific_ax.set_xlabel("Rank r")
        gap_ax.set_xticks(x, [str(r) for r in ranks])
        specific_ax.set_xticks(x, [str(r) for r in ranks])
        for axis in (gap_ax, capture_ax, specific_ax):
            axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("B-derived subspace: C capture and residualized-C specificity")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
