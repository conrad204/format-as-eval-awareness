"""Main-text summary figure for the matched-format projection diagnostic.

Reads results/full320_v3_matched_format_projection_diagnostic.json only (no
new probes or activations). Produces a 2-panel figure meant to communicate
the headline result at a glance; the full-depth 3x3 small-multiples figure
(scripts/full320_v3_matched_format_projection_plot.py) remains the
appendix/supporting-material figure and is not touched by this script.

  Panel A: raw vs. matched-format-projected cross-format transfer gap,
           one faint point per post-purpose-formation layer (raw both-format
           purpose AUC >= 0.95), all 3 models, plus one bold per-model
           median marker. y=x reference line: points below it are layers
           where projection reduced the transfer penalty.
  Panel B: per-model dumbbell of median raw gap -> median projected gap,
           annotated with the recomputed median fraction of the gap removed,
           plus a small side panel contrasting the mean cross-format
           transfer gain (benchmark->casual, casual->benchmark) against the
           mean absolute change in both-format purpose AUC.

Usage: python scripts/full320_v3_matched_format_projection_summary_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
IN_JSON = ROOT / "results" / "full320_v3_matched_format_projection_diagnostic.json"
OUT_DIR = ROOT / "plots" / "matched format projection summary 8-28-2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#3a3a37"; MUTED = "#898781"
GRID = "#e4e3dc"; BASE = "#c3c2b7"
S8B = "#2a78d6"; S71 = "#8a5cd6"; S70 = "#eb6834"
COLORS = {"llama31_8b": S8B, "llama31_70b": S71, "llama33_70b": S70}
ORDER = ["llama31_8b", "llama31_70b", "llama33_70b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": BASE, "font.size": 12,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.axisbelow": True,
})


def axchrome(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)


def main():
    payload = json.loads(IN_JSON.read_text())
    rows = payload["per_layer"]
    summaries = payload["model_summaries"]

    eligible = {m: [r for r in rows if r["model"] == m and r["raw_joint_auc"] >= payload["purpose_formed_auc_cutoff"]]
                for m in ORDER}

    fig = plt.figure(figsize=(16.6, 7.6))
    axA = fig.add_axes([0.055, 0.12, 0.335, 0.63])
    axB = fig.add_axes([0.470, 0.12, 0.335, 0.63])
    axIns = fig.add_axes([0.865, 0.12, 0.115, 0.63])

    # ---------------- Panel A: raw vs. projected gap scatter ----------------
    lims_hi = 0.0
    for m in ORDER:
        d = eligible[m]
        c = COLORS[m]
        xr = np.array([r["raw_gap"] for r in d])
        yp = np.array([r["proj_gap"] for r in d])
        lims_hi = max(lims_hi, xr.max(), yp.max())
        axA.scatter(xr, yp, s=26, color=c, alpha=0.35, linewidths=0, zorder=2)
        axA.scatter([np.median(xr)], [np.median(yp)], s=220, color=c, edgecolors=SURF,
                    linewidths=1.8, zorder=4, marker="o")
    lo, hi = -0.005, lims_hi * 1.08
    axA.plot([lo, hi], [lo, hi], color=BASE, lw=1.6, ls=(0, (4, 3)), zorder=1)
    axA.text(hi, hi * 0.985, "no change (y = x)", color=MUTED, fontsize=9.3, ha="right", va="bottom")
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.set_aspect("equal")
    axA.set_xlabel("raw cross-format transfer gap (AUC)", fontsize=11.5)
    axA.set_ylabel("matched-format-projected transfer gap (AUC)", fontsize=11.5)
    axA.set_title("A. Projection pulls the transfer gap toward zero", loc="left",
                   fontsize=12.5, fontweight="bold", color=INK, pad=10)
    axA.text(0.04, 0.95, "points below the line:\nprojection reduced the gap\n\nbold dot: per-model median",
              transform=axA.transAxes, fontsize=9.0, color=INK2, ha="left", va="top", style="italic")
    axchrome(axA)

    # ---------------- Panel B: per-model dumbbell ----------------
    y_pos = {m: i for i, m in enumerate(ORDER)}
    xmax = max(summaries[m]["raw_gap_median"] for m in ORDER) * 1.42
    for m in ORDER:
        c = COLORS[m]
        s = summaries[m]
        yi = y_pos[m]
        raw_med, proj_med = s["raw_gap_median"], s["proj_gap_median"]
        axB.plot([proj_med, raw_med], [yi, yi], color=c, lw=3.2, zorder=2, solid_capstyle="round")
        axB.scatter([raw_med], [yi], s=170, color=SURF, edgecolors=c, linewidths=2.4, zorder=4)
        axB.scatter([proj_med], [yi], s=170, color=c, edgecolors=SURF, linewidths=1.6, zorder=4)
        rescue_pct = 100 * s["rescue_fraction_median_excl_near_zero"]
        axB.text((raw_med + proj_med) / 2, yi + 0.30, f"\u2212{rescue_pct:.0f}%",
                  color=c, fontsize=12.5, fontweight="bold", va="bottom", ha="center")

    axB.set_yticks([y_pos[m] for m in ORDER])
    axB.set_yticklabels([summaries[m]["model_label"] for m in ORDER], fontsize=11.5, color=INK)
    axB.set_ylim(-0.6, len(ORDER) - 1 + 0.62)
    axB.set_xlim(0, xmax)
    axB.set_xlabel("median cross-format transfer gap (AUC)", fontsize=11.5)
    axB.set_title("B. Median gap shrinks by roughly half to three-quarters", loc="left",
                   fontsize=12.5, fontweight="bold", color=INK, pad=10)
    axchrome(axB)
    axB.grid(axis="y", visible=False)
    axB.text(0.98, 0.03, "label: % of median gap removed", transform=axB.transAxes,
              fontsize=8.6, color=MUTED, ha="right", va="bottom", style="italic")

    legend_handles = [
        Line2D([], [], marker="o", ls="", ms=11, mfc=SURF, mec=INK2, mew=2, label="raw"),
        Line2D([], [], marker="o", ls="", ms=11, mfc=INK2, mec=SURF, mew=1.6, label="matched-format projected"),
    ]
    axB.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.13),
               frameon=False, fontsize=9.3, labelcolor=INK2, handletextpad=0.6, ncol=2, columnspacing=1.2)

    # ---- side panel: mean AUC change comparison (transfer gain vs. purpose-decoding change) ----
    bar_x = np.arange(3)
    width = 0.24
    for k, m in enumerate(ORDER):
        s = summaries[m]
        vals = [s["mean_delta_b2c_auc"], s["mean_delta_c2b_auc"], s["mean_abs_delta_joint_auc"]]
        axIns.bar(bar_x + (k - 1) * width, vals, width, color=COLORS[m], zorder=3)
    axIns.axhline(0, color=BASE, lw=1.0, zorder=1)
    axIns.set_xticks(bar_x)
    axIns.set_xticklabels(["benchmark\n\u2192casual\ngain", "casual\n\u2192benchmark\ngain",
                            "both-format\npurpose |\u0394|"], fontsize=8.0, color=INK2)
    axIns.set_ylabel("mean \u0394AUC after projection", fontsize=9.3)
    axIns.tick_params(axis="y", labelsize=8.5)
    axchrome(axIns)
    axIns.grid(axis="x", visible=False)
    axIns.set_title("C. Transfer moves far\nmore than purpose\ndecoding does", fontsize=11, fontweight="bold",
                     color=INK, loc="left", pad=10)

    fig.text(0.006, 0.975,
              "Matched-format projection removes most of the cross-format transfer penalty, "
              "while preserving purpose decodability",
              fontsize=15.5, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.006, 0.925,
              "Post-purpose-formation layers (raw both-format purpose AUC \u2265 0.95), all 3 models.",
              fontsize=10.4, color=INK2, va="top", ha="left")
    model_handles = [Line2D([], [], color=COLORS[m], lw=3, label=summaries[m]["model_label"]) for m in ORDER]
    fig.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.006, 0.875),
               ncol=3, frameon=False, fontsize=10.8, labelcolor=INK2, handlelength=1.8, columnspacing=1.4)

    out_png = OUT_DIR / "matched_format_projection_summary.png"
    out_pdf = OUT_DIR / "matched_format_projection_summary.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    return summaries


if __name__ == "__main__":
    main()
