"""Supplementary figure: does format still exist after the rank-64
matched-format projection?

Reads results/full320_v3_matched_format_projection_diagnostic.json only (no
new probes or activations). For every layer of all 3 models, plots the
format-decodability sanity probe's AUC (trained to predict benchmark vs.
casual format from the activations) before vs. after the same rank-64
matched-format projection used for the purpose-transfer diagnostic.

This answers a distinct question from the purpose-transfer figures: the
projection is built to remove the matched-format subspace that specifically
hurts cross-format purpose transfer, not to anonymize format in general.
This figure checks how much format signal is left over.

Usage: python scripts/full320_v3_format_persistence_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    summaries = payload["model_summaries"]
    data = by_model(rows)

    fig = plt.figure(figsize=(13.2, 7.6))
    ax = fig.add_axes([0.08, 0.15, 0.87, 0.58])

    for m in ORDER:
        d = data[m]
        c = COLORS[m]
        x = [r["relative_depth"] * 100 for r in d]
        ax.plot(x, [r["raw_format_auc"] for r in d], color=c, lw=2.2, ls=(0, (4, 2.2)), alpha=0.55, zorder=2)
        ax.plot(x, [r["proj_format_auc"] for r in d], color=c, lw=2.6, zorder=3)

    ax.axhline(0.5, color=BASE, lw=1.2, ls=":", zorder=1)
    ax.text(100, 0.51, "chance", color=MUTED, fontsize=9.3, ha="right", va="bottom")
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("relative depth through the network (%)", fontsize=12)
    ax.set_ylabel("format decodability (AUC)", fontsize=12)
    ax.tick_params(labelsize=10.5)
    axchrome(ax)
    ax.set_title("Format is still substantially decodable after the projection", loc="left",
                  fontsize=14.5, fontweight="bold", color=INK, pad=12)
    ax.text(0.02, 0.30, "faint dashed: raw activations (\u2248 1.0 AUC, every layer, every model)\n"
                          "solid: after the rank-64 matched-format projection",
             transform=ax.transAxes, fontsize=9.6, color=INK2, ha="left", va="top", style="italic")

    fig.text(0.006, 0.975,
              "Removing the rank-64 matched-format subspace does not anonymize format",
              fontsize=16.5, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.006, 0.915,
              "All layers, all 3 models. Same projection used for the transfer-gap results.",
              fontsize=11, color=INK2, va="top", ha="left")
    model_handles = [Line2D([], [], color=COLORS[m], lw=3, label=summaries[m]["model_label"]) for m in ORDER]
    fig.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.006, 0.865),
               ncol=3, frameon=False, fontsize=11.5, labelcolor=INK2, handlelength=1.8, columnspacing=1.4)

    out_png = OUT_DIR / "format_persistence_after_projection.png"
    out_pdf = OUT_DIR / "format_persistence_after_projection.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
