"""The core finding, drawn honestly: cross-format purpose-transfer gap vs depth.

Self-contained (no shared plots/_shared import — that module kept disappearing
mid-session from causes outside this script's control, so this one script owns
everything it needs: data loading, style, and rendering).

Data: artifacts/pinned_full320/layer_sweep_full320-v3_*.json (three Llama
checkpoints). gap(layer) = decorrelated-both-format AUC minus the mean of the
two one-format-trained transfer directions (b2c, c2b). No fabricated
confidence intervals here: the earlier "final minus argmin(gap)" bootstrap was
statistically biased (the comparison point was selected from the same curve
being tested) and is deliberately NOT reproduced. What's plotted is the
observed curve, annotated with its own minimum and final value, replicated
across three independently trained checkpoints -- that cross-model agreement,
not a within-dataset p-value, is the actual evidence here.
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


def gap_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    depth = np.array([r["layer"] / (n - 1) for r in rows])
    dual = np.array([r["decorrelated"]["auc"] for r in rows])
    cross_mean = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in rows])
    return depth, dual - cross_mean


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax.axvspan(0, 12, color=mpl.colors.to_rgba(MUTED, 0.05), zorder=0)
    ax.text(0.6, 0.225, "early spike", fontsize=7.5, color=MUTED, va="top", style="italic")

    for label, fname in MODEL_FILES.items():
        rows = json.loads((PINNED_DIR / fname).read_text())
        depth, gap = gap_curve(rows)
        depth_pct = 100 * depth
        color = MODEL_COLORS[label]
        ax.plot(depth_pct, gap, color=color, linewidth=2.0, label=label, zorder=3, solid_capstyle="round")

        min_idx = int(np.argmin(gap))
        ax.scatter([depth_pct[min_idx]], [gap[min_idx]], color=color, s=28, zorder=4, edgecolor="white", linewidth=0.7)
        ax.scatter([depth_pct[-1]], [gap[-1]], color=color, s=28, zorder=4, marker="D", edgecolor="white", linewidth=0.7)

        print(f"{label}: min at {depth_pct[min_idx]:.0f}% depth (gap={gap[min_idx]:.3f}), "
              f"final gap={gap[-1]:.3f} ({gap[-1]/gap[min_idx]:.1f}x the minimum)")

    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.006, 0.235)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Relative depth through the network (%)")
    ax.set_ylabel("Transfer gap: both-format AUC \u2212 cross-format AUC")
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("The cross-format transfer gap narrows mid-network, then re-widens",
                 loc="left", fontsize=11, fontweight="bold", pad=10)
    handles, labels_ = ax.get_legend_handles_labels()
    marker_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", color=MUTED, markersize=5, label="layer of minimum gap"),
        mpl.lines.Line2D([], [], marker="D", linestyle="none", color=MUTED, markersize=5, label="final layer"),
    ]
    ax.legend(handles=handles + marker_handles, loc="upper right", frameon=False, fontsize=8.2)

    fig.text(0.0, -0.06,
              "Held cue-vocabulary-family OOF probes \u00b7 L2 logistic \u00b7 7,680 stated purpose rows / 320 payload blocks\n"
              "(3,840 no-cue control rows excluded \u2014 they carry no purpose label). Curve is descriptive: replicated\n"
              "across 3 independently trained checkpoints, no within-dataset significance claim attached.",
              ha="left", va="top", fontsize=7.3, color=MUTED)

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"cross_format_gap_trajectory.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'cross_format_gap_trajectory.pdf'}")


if __name__ == "__main__":
    main()
