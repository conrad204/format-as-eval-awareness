"""Why does the cross-format transfer gap re-widen in late layers?

Self-contained. This is the explanation that survives scrutiny, after two
correlation-based geometric candidates did not:

  1. rho(l) = ||C(l)|| / ||A(l)|| (interaction-to-shared-effect norm ratio):
     raw r=0.74-0.94 with the gap, but ~entirely explained by depth (both
     norms track depth at r~0.97-1.00, almost certainly because residual-
     stream activation norms grow with depth for purely architectural
     reasons -- see scripts/format_interaction_geometry_full320.py).
  2. cos_align(dir_benchmark, dir_casual) (does the purpose direction itself
     rotate apart between formats?): same story -- raw r=-0.70 to -0.94,
     consistent sign across all 3 models, but only 1-of-3 significant after
     partialling out depth, and the layer-to-layer delta test is null in all
     three (see scripts/format_direction_alignment_full320.py).

Both of those are CORRELATIONS between two curves that both trend with
depth, which is inherently confound-prone. This figure instead makes a
direct, non-circular comparison: from each model's decorrelated-AUC PEAK to
its final layer, how much does the both-format-trained probe's AUC drop,
versus how much does the cross-format-TRANSFERRED probe's AUC drop? No
correlation, no depth-partialling needed -- just two magnitudes, read
directly off the pinned sweeps, replicated cleanly in all three models.

Result: the transfer-tested probe consistently loses ~1.7-1.8x more AUC than
the jointly-trained probe by the final layer. Reading: late transformer
layers are known to reorganize residual-stream content toward next-token /
output-production computation (diluting earlier abstract semantic content).
A probe trained on BOTH formats can track whatever that reorganization does,
even if it becomes more format-conditional. A probe trained on only ONE
format cannot follow a reorganization it never saw -- so it disproportionately
loses ground. That asymmetry, not a specific geometric magnitude or
direction, is what makes the GAP (dual minus cross) widen even as both
curves are declining. Still observational/descriptive, not causal -- but
immune to the depth-confound critique that sank the two geometry attempts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PINNED_DIR = ROOT / "artifacts" / "pinned_full320"

MODEL_FILES = {
    "Llama-3.1-8B": "layer_sweep_full320-v3_llama31_8b.json",
    "Llama-3.3-70B": "layer_sweep_full320-v3_llama33_70b.json",
    "Llama-3.1-70B": "layer_sweep_full320-v3_llama31_70b.json",
}
MODEL_COLORS = {
    "Llama-3.1-8B": "#0072B2",
    "Llama-3.3-70B": "#D55E00",
    "Llama-3.1-70B": "#009E73",
}
INK = "#202124"
MUTED = "#666A70"
GRID = "#D9DDE3"


def curves(rows: list[dict]) -> dict:
    n = len(rows)
    depth = np.array([r["layer"] / (n - 1) for r in rows])
    dual = np.array([r["decorrelated"]["auc"] for r in rows])
    cross = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in rows])
    return {"depth": depth, "dual": dual, "cross": cross}


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.2,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), constrained_layout=True, sharey=True)

    for ax, (label, fname) in zip(axes, MODEL_FILES.items()):
        rows = json.loads((PINNED_DIR / fname).read_text())
        c = curves(rows)
        depth_pct = 100 * c["depth"]
        color = MODEL_COLORS[label]
        peak_i = int(np.argmax(c["dual"]))

        ax.plot(depth_pct, c["dual"], color=color, linewidth=2.0, label="both-format trained", zorder=3)
        ax.plot(depth_pct, c["cross"], color=color, linewidth=2.0, linestyle=(0, (4, 2)), label="cross-format transfer", zorder=3)
        ax.axvline(depth_pct[peak_i], color=MUTED, linewidth=0.7, linestyle=":", zorder=1)

        dual_drop = c["dual"][peak_i] - c["dual"][-1]
        cross_drop = c["cross"][peak_i] - c["cross"][-1]
        ratio = cross_drop / dual_drop

        ax.annotate("", xy=(100, c["dual"][-1]), xytext=(100, c["dual"][peak_i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.55, linewidth=1.1))
        ax.annotate("", xy=(100, c["cross"][-1]), xytext=(depth_pct[peak_i], c["cross"][peak_i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.3, linewidth=1.1, linestyle="dashed"))

        ax.set_title(f"{label}\n\u0394dual={dual_drop:.3f}  \u0394cross={cross_drop:.3f}  ({ratio:.1f}\u00d7)",
                     loc="left", fontsize=9.0, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(0.5, 1.02)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Relative depth (%)")
        ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        print(f"{label}: peak at {depth_pct[peak_i]:.0f}% depth | dual drop={dual_drop:.3f} | "
              f"cross drop={cross_drop:.3f} | cross drops {ratio:.2f}x more than dual")

    axes[0].set_ylabel("Held-out purpose AUC")
    axes[0].legend(loc="lower left", fontsize=7.6, frameon=False)
    fig.suptitle("Cross-format-transferred probes lose disproportionately more AUC in late layers",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.text(
        0.0, -0.09,
        "From each model's decorrelated-AUC peak (dotted vertical line) to the final layer: the cross-format-transferred\n"
        "probe (dashed) loses ~1.7\u20131.8\u00d7 more AUC than the jointly-trained probe (solid), consistently in all 3 models. Reading:\n"
        "late transformer layers reorganize toward output/next-token computation; a probe trained on both formats can track\n"
        "that reorganization even as it becomes more format-conditional, but a probe trained on one format alone cannot follow\n"
        "a shift it never saw \u2014 so transfer degrades faster than joint decodability, and the gap between them widens.\n"
        "Descriptive, not causal; unlike the two geometry checks in this project, not a correlation between depth-trending curves.",
        ha="left", va="top", fontsize=7.2, color=MUTED,
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"why_the_gap_widens.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'why_the_gap_widens.pdf'}")


if __name__ == "__main__":
    main()
