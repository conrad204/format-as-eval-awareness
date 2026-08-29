"""Publication figure for the matched-format-projection transfer diagnostic.

Reads results/full320_v3_matched_format_projection_diagnostic.json (produced
by scripts/full320_v3_matched_format_projection_diagnostic.py) and renders a
3-panel figure:

  A. raw gap and rank-64 matched-format-projected gap vs relative depth,
     all 3 models.
  B. gap rescue delta_G = raw_gap - proj_gap vs relative depth, zero line.
  C. directional transfer rescue delta_AUC_b2c / delta_AUC_c2b vs relative
     depth, with joint-purpose delta_AUC shown as a faint reference line, so
     the plot answers whether transfer improves substantially more than
     ordinary purpose decodability moves.

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

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
S8B = "#2a78d6"; S71 = "#8a5cd6"; S70 = "#eb6834"
COLORS = {"llama31_8b": S8B, "llama31_70b": S71, "llama33_70b": S70}
ORDER = ["llama31_8b", "llama31_70b", "llama33_70b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": BASE, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def axchrome(ax, xlab, ylab):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
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

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4))
    axA, axB, axC = axes

    # Panel A: raw vs projected gap
    for m in ORDER:
        d = data[m]
        x = [r["relative_depth"] * 100 for r in d]
        c = COLORS[m]
        axA.plot(x, [r["raw_gap"] for r in d], color=c, lw=2, zorder=3)
        axA.plot(x, [r["proj_gap"] for r in d], color=c, lw=2, ls=(0, (4, 3)), alpha=0.85, zorder=2)
    axA.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)
    axchrome(axA, "relative depth (%)", "purpose gap  (joint AUC − mean cross-format AUC)")
    axA.set_title("A. raw vs. projected gap", loc="left", fontsize=11, fontweight="bold", color=INK)

    # Panel B: gap rescue delta_G
    for m in ORDER:
        d = data[m]
        x = [r["relative_depth"] * 100 for r in d]
        axB.plot(x, [r["delta_gap"] for r in d], color=COLORS[m], lw=2, zorder=3)
    axB.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)
    axchrome(axB, "relative depth (%)", r"gap rescue  $\Delta G = G_{raw} - G_{proj}$")
    axB.set_title("B. gap rescue by depth", loc="left", fontsize=11, fontweight="bold", color=INK)

    # Panel C: directional rescue + faint joint-purpose delta
    for m in ORDER:
        d = data[m]
        x = [r["relative_depth"] * 100 for r in d]
        c = COLORS[m]
        axC.plot(x, [r["delta_joint_auc"] for r in d], color=c, lw=1.1, alpha=0.30, zorder=1)
        axC.plot(x, [r["delta_b2c_auc"] for r in d], color=c, lw=2, ls="-", zorder=3)
        axC.plot(x, [r["delta_c2b_auc"] for r in d], color=c, lw=2, ls=(0, (1, 1.4)), zorder=3)
    axC.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)
    axchrome(axC, "relative depth (%)", r"$\Delta$AUC after projection")
    axC.set_title("C. directional transfer rescue vs. joint-purpose change", loc="left",
                   fontsize=11, fontweight="bold", color=INK)

    model_handles = [Line2D([], [], color=COLORS[m], lw=2, label=summaries[m]["model_label"])
                      for m in ORDER]
    style_handles_A = [
        Line2D([], [], color=INK2, lw=2, label="raw"),
        Line2D([], [], color=INK2, lw=2, ls=(0, (4, 3)), label="rank-64 matched-format projected"),
    ]
    style_handles_C = [
        Line2D([], [], color=INK2, lw=2, ls="-", label=r"$\Delta$AUC benchmark$\to$casual"),
        Line2D([], [], color=INK2, lw=2, ls=(0, (1, 1.4)), label=r"$\Delta$AUC casual$\to$benchmark"),
        Line2D([], [], color=INK2, lw=1.1, alpha=0.5, label=r"$\Delta$AUC joint purpose (faint)"),
    ]

    fig.text(0.006, 0.975, "Matched-format projection rescues cross-format transfer, not joint-purpose decodability",
              fontsize=13, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.006, 0.930,
              "Rank-64 residualization of the matched-format subspace (cross-fitted per held-out cue-vocabulary "
              "fold, probe retrained on the residual), all layers, 3 models.",
              fontsize=9.5, color=INK2, va="top", ha="left")

    fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.5, 0.895),
               ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.8, columnspacing=1.3)
    fig.legend(handles=style_handles_A, loc="upper left", bbox_to_anchor=(0.005, 0.845),
               ncol=1, frameon=False, fontsize=8.6, labelcolor=INK2, handlelength=2.0)
    fig.legend(handles=style_handles_C, loc="upper right", bbox_to_anchor=(0.995, 0.845),
               ncol=1, frameon=False, fontsize=8.6, labelcolor=INK2, handlelength=2.2)

    fig.tight_layout(rect=[0, 0, 1, 0.80])

    out_png = OUT_DIR / "matched_format_projection_diagnostic.png"
    out_pdf = OUT_DIR / "matched_format_projection_diagnostic.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
