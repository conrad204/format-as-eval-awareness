"""Main-text summary figure for the matched-format projection diagnostic.

Reads results/full320_v3_matched_format_projection_diagnostic.json only (no
new probes or activations). Produces a single wide bar-chart panel meant to
communicate the headline result at a glance; the full-depth 3x3
small-multiples figure (scripts/full320_v3_matched_format_projection_plot.py)
remains the appendix/supporting-material figure and is not touched by this
script.

  Mean cross-format transfer gain (benchmark->casual, casual->benchmark)
  after rank-64 matched-format projection, contrasted with the mean
  absolute change in both-format purpose AUC after the same projection,
  averaged over each model's post-purpose-formation region (raw both-format
  purpose AUC >= 0.95).

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
    cutoff = payload["purpose_formed_auc_cutoff"]

    eligible = {m: [r for r in rows if r["model"] == m and r["raw_joint_auc"] >= cutoff] for m in ORDER}

    fig = plt.figure(figsize=(13.2, 8.0))
    ax = fig.add_axes([0.09, 0.10, 0.86, 0.56])

    bar_x = np.arange(3)
    width = 0.24
    for k, m in enumerate(ORDER):
        s = summaries[m]
        d = eligible[m]
        vals = [s["mean_delta_b2c_auc"], s["mean_delta_c2b_auc"], s["mean_abs_delta_joint_auc"]]
        errs = [
            np.std([r["delta_b2c_auc"] for r in d]),
            np.std([r["delta_c2b_auc"] for r in d]),
            np.std([abs(r["delta_joint_auc"]) for r in d]),
        ]
        bars = ax.bar(bar_x + (k - 1) * width, vals, width, yerr=errs, color=COLORS[m], zorder=3,
                       capsize=4, error_kw=dict(ecolor=INK2, elinewidth=1.3, capthick=1.3, zorder=4))
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.0012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.2, color=INK2)
    ax.axhline(0, color=BASE, lw=1.0, zorder=1)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["benchmark \u2192 casual\ntransfer gain", "casual \u2192 benchmark\ntransfer gain",
                         "both-format purpose\ndecoding, |change|"], fontsize=11.5, color=INK2)
    ax.set_ylabel("mean \u0394AUC after matched-format projection", fontsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    axchrome(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title("Transfer moves far more than purpose decoding does", fontsize=14.5, fontweight="bold",
                  color=INK, loc="left", pad=12)
    ax.text(0.99, 0.97, "error bars: \u00b1 1 SD across post-purpose-formation layers\n(descriptive spread, not an independent-sample CI)",
             transform=ax.transAxes, fontsize=8.6, color=MUTED, ha="right", va="top", style="italic")

    fig.text(0.006, 0.975,
              "Matched-format projection removes most of the cross-format transfer penalty,\n"
              "while preserving purpose decodability",
              fontsize=16.5, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.006, 0.865,
              "Post-purpose-formation layers (raw both-format purpose AUC \u2265 0.95), all 3 models.",
              fontsize=11, color=INK2, va="top", ha="left")
    model_handles = [Line2D([], [], color=COLORS[m], lw=3, label=summaries[m]["model_label"]) for m in ORDER]
    fig.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.006, 0.805),
               ncol=3, frameon=False, fontsize=11.5, labelcolor=INK2, handlelength=1.8, columnspacing=1.4)

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
