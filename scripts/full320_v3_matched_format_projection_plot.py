"""Publication figure for the matched-format-projection transfer diagnostic.

Reads results/full320_v3_matched_format_projection_diagnostic.json (produced
by scripts/full320_v3_matched_format_projection_diagnostic.py) and renders a
3 (model rows) x 3 (metric columns) small-multiples grid:

  Column A. raw gap and rank-64 matched-format-projected gap vs relative
            depth.
  Column B. gap rescue delta_G = raw_gap - proj_gap vs relative depth, zero
            reference line.
  Column C. directional transfer rescue delta_AUC_b2c / delta_AUC_c2b vs
            relative depth, with joint-purpose delta_AUC shown as a faint
            reference line.

One row per model keeps each model's own shape legible (no 9-line overlay)
and lets a shared per-row y-scale make cross-model amplitude comparisons
honest. The shaded band in every row marks that model's post-purpose-
formation region (raw joint AUC >= 0.95), the same region the model-level
summary table is computed over.

Usage: python scripts/full320_v3_matched_format_projection_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
IN_JSON = ROOT / "results" / "full320_v3_matched_format_projection_diagnostic.json"
OUT_DIR = ROOT / "plots" / "matched format projection diagnostic 8-28-2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#3a3a37"; MUTED = "#898781"
GRID = "#e4e3dc"; BASE = "#c3c2b7"
S8B = "#2a78d6"; S71 = "#8a5cd6"; S70 = "#eb6834"
BAND = "#e9f3ec"; BAND_EDGE = "#bfe0cc"
COLORS = {"llama31_8b": S8B, "llama31_70b": S71, "llama33_70b": S70}
ORDER = ["llama31_8b", "llama31_70b", "llama33_70b"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": BASE, "font.size": 12,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.axisbelow": True,
})

COL_TITLES = [
    "A. raw vs. rank-64 projected gap",
    "B. gap rescue  " + r"$\Delta G = G_{raw}-G_{proj}$",
    "C. directional rescue vs. joint-purpose " + r"$\Delta$AUC",
]


def axchrome(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)


def by_model(rows):
    out = {m: [] for m in ORDER}
    for r in rows:
        out[r["model"]].append(r)
    for m in out:
        out[m].sort(key=lambda r: r["layer"])
    return out


def main():
    payload = json.loads(IN_JSON.read_text())
    rows = payload["per_layer"]
    data = by_model(rows)
    summaries = payload["model_summaries"]

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 12.6), sharex="col")

    for i, m in enumerate(ORDER):
        d = data[m]
        c = COLORS[m]
        x = [r["relative_depth"] * 100 for r in d]
        lo, hi = summaries[m]["eligible_relative_depth_range"]
        axA, axB, axC = axes[i]

        for ax in (axA, axB, axC):
            ax.axvspan(lo * 100, hi * 100, color=BAND, zorder=0, ec=BAND_EDGE, lw=0.8)
            axchrome(ax)

        # A: raw vs projected gap
        axA.plot(x, [r["raw_gap"] for r in d], color=c, lw=2.6, zorder=3)
        axA.plot(x, [r["proj_gap"] for r in d], color=c, lw=2.6, ls=(0, (4, 2.4)), alpha=0.75, zorder=2)
        axA.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)

        # B: gap rescue
        axB.plot(x, [r["delta_gap"] for r in d], color=c, lw=2.6, zorder=3)
        axB.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)

        # C: directional rescue + faint joint-purpose delta
        axC.plot(x, [r["delta_joint_auc"] for r in d], color=INK2, lw=1.4, alpha=0.45, zorder=2)
        axC.plot(x, [r["delta_b2c_auc"] for r in d], color=c, lw=2.6, ls="-", zorder=3)
        axC.plot(x, [r["delta_c2b_auc"] for r in d], color=c, lw=2.6, ls=(0, (1, 1.3)), zorder=3)
        axC.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)

        axA.set_ylabel(f"{summaries[m]['model_label']}\n\npurpose gap (AUC)", fontsize=11.5, color=INK, linespacing=1.5)
        axB.set_ylabel("gap rescue " + r"$\Delta G$" + " (AUC)", fontsize=11.5, color=INK)
        axC.set_ylabel(r"$\Delta$AUC after projection", fontsize=11.5, color=INK)
        if i == 2:
            for ax in (axA, axB, axC):
                ax.set_xlabel("relative depth through the network (%)", fontsize=11.5)

    # shared per-column y-limits so amplitude differences across models are honest
    for col in range(3):
        vals = []
        for row_axes in axes:
            for ln in row_axes[col].get_lines():
                vals.extend(ln.get_ydata())
        lo_y, hi_y = min(vals), max(vals)
        pad = 0.06 * (hi_y - lo_y)
        for row_axes in axes:
            row_axes[col].set_ylim(lo_y - pad, hi_y + pad)

    for ax, title in zip(axes[0], COL_TITLES):
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)

    fig.text(0.006, 0.99, "Matched-format projection rescues cross-format transfer, not joint-purpose decodability",
              fontsize=16, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.006, 0.965,
              "Rank-64 residualization of the matched-format subspace (cross-fitted per held-out cue-vocabulary "
              "fold, probe retrained on the residual). One row per model; shaded band = that model's "
              "post-purpose-formation region (raw joint AUC \u2265 0.95), the region the summary table is computed over.",
              fontsize=10.5, color=INK2, va="top", ha="left")

    style_handles = [
        Line2D([], [], color=INK2, lw=2.6, label="raw"),
        Line2D([], [], color=INK2, lw=2.6, ls=(0, (4, 2.4)), alpha=0.75, label="rank-64 matched-format projected"),
        Line2D([], [], color=INK2, lw=2.6, ls="-", label=r"$\Delta$AUC benchmark$\to$casual"),
        Line2D([], [], color=INK2, lw=2.6, ls=(0, (1, 1.3)), label=r"$\Delta$AUC casual$\to$benchmark"),
        Line2D([], [], color=INK2, lw=1.4, alpha=0.45, label=r"$\Delta$AUC joint purpose (faint reference)"),
        Line2D([], [], marker="s", ls="", ms=11, mfc=BAND, mec=BAND_EDGE, label="post-purpose-formation region"),
    ]
    fig.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=6, frameon=False, fontsize=10.2, labelcolor=INK2, handlelength=2.1, columnspacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.905])
    fig.subplots_adjust(hspace=0.30, wspace=0.24)
    out_png = OUT_DIR / "matched_format_projection_diagnostic.png"
    out_pdf = OUT_DIR / "matched_format_projection_diagnostic.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
