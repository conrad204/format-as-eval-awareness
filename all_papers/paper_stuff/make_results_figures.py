"""Results-section figures for paper_stuff/current.tex.

Reads only checked-in result JSONs (never activations):
  results/layer_sweep_full320-v3_{llama31_8b,llama31_70b,llama33_70b}.json
  results/nocue_sweep_full320-v3_{...}.json
  results/full_layer_gap_erasure_summary.json
Writes fig_results_01_replication, fig_results_02_nocue, fig_results_03_transfer
as PDF (for LaTeX) and PNG (preview) into paper_stuff/.

Palette: three categorical slots validated all-pairs for CVD (blue / aqua / orange);
the confounded probe is drawn in the same model hue with a dashed line, or in a
de-emphasis gray for bars. Layers are drawn as one trajectory per checkpoint.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
MODELS = [  # (key, label, color)
    ("llama31_8b",  "Llama-3.1-8B",  "#2a78d6"),
    ("llama31_70b", "Llama-3.1-70B", "#1baf7a"),
    ("llama33_70b", "Llama-3.3-70B", "#eb6834"),
]
DASH = (0, (3.2, 2.0))
LW = 1.6

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "pdf.fonttype": 42,
})

SWEEP = {k: json.load(open(R / f"layer_sweep_full320-v3_{k}.json")) for k, _, _ in MODELS}
NOCUE = {k: json.load(open(R / f"nocue_sweep_full320-v3_{k}.json")) for k, _, _ in MODELS}
ERASE = json.load(open(R / "full_layer_gap_erasure_summary.json"))["by_model"]
PEAK = {k: max(SWEEP[k], key=lambda r: r["decorrelated"]["auc"])["layer"] for k, _, _ in MODELS}


def depth(rows):
    n = len(rows)
    return np.array([r["layer"] / (n - 1) * 100 for r in rows])


def chrome(ax, xlab, ylab, title=None):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.5, color=BASE)
    if title:
        ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=5)


def panel_tag(ax, tag):
    ax.text(-0.14, 1.06, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def model_handles():
    return [Line2D([], [], color=c, lw=LW, label=lab) for _, lab, c in MODELS]


def style_handles(solid, dashed):
    return [Line2D([], [], color=INK2, lw=LW, label=solid),
            Line2D([], [], color=INK2, lw=LW, ls=DASH, label=dashed)]


def mark_peak(ax, x, y, color):
    ax.plot([x], [y], "o", ms=4.8, color=color, mec=SURF, mew=1.1, zorder=6)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)
    print("wrote", stem)


# ----------------------------------------------------------------- figure 1
fig = plt.figure(figsize=(5.5, 2.55))
gsA = fig.add_gridspec(1, 1, left=0.085, right=0.50, bottom=0.2, top=0.84)
gsB = fig.add_gridspec(1, 3, wspace=0.14, left=0.60, right=0.99, bottom=0.2, top=0.84)
a = fig.add_subplot(gsA[0, 0])
bs = [fig.add_subplot(gsB[0, i]) for i in range(3)]
for k, lab, c in MODELS:
    rows = SWEEP[k]; x = depth(rows)
    a.plot(x, [r["decorrelated"]["auc"] for r in rows], color=c, lw=LW, solid_capstyle="round", zorder=3)
    a.plot(x, [r["confounded"]["auc"] for r in rows], color=c, lw=LW, ls=DASH, zorder=2)
    p = PEAK[k]; mark_peak(a, x[p], rows[p]["decorrelated"]["auc"], c)
a.axhline(0.5, color=BASE, lw=0.7, zorder=1)
a.text(99, 0.507, "chance", ha="right", va="bottom", color=MUTED, fontsize=6.3)
a.set_xlim(0, 100); a.set_ylim(0.42, 1.02)
a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
chrome(a, "relative depth (%)", "held-family-out purpose AUC", "Purpose AUC by depth")
a.legend(handles=model_handles() + style_handles("decorrelated", "confounded"),
         loc="center right", bbox_to_anchor=(1.0, 0.63), ncol=2, columnspacing=0.8,
         frameon=False, handlelength=1.4, handletextpad=0.5, labelcolor=INK2, borderaxespad=0.1, fontsize=5.8)
panel_tag(a, "A")

cells = ["evaluation::benchmark", "evaluation::casual", "deployment::benchmark", "deployment::casual"]
cell_lbl = ["EB", "EC", "DB", "DC"]
w = 0.36; gap = 0.05
for bi, (ax, (k, lab, c)) in enumerate(zip(bs, MODELS)):
    p = SWEEP[k][PEAK[k]]
    for ci, cell in enumerate(cells):
        cv = p["confounded"]["cells"][cell]; dv = p["decorrelated"]["cells"][cell]
        ax.bar(ci - w / 2 - gap / 2, cv, w, color=MUTED, zorder=3, linewidth=0)
        ax.bar(ci + w / 2 + gap / 2, dv, w, color=c, zorder=3, linewidth=0)
        if cell in ("evaluation::casual", "deployment::benchmark"):
            ax.text(ci - w / 2 - gap / 2, cv + 2.5, f"{cv:.0f}", ha="center", va="bottom", color=INK2, fontsize=5.6)
            ax.text(ci + w / 2 + gap / 2, dv + 2.5, f"{dv:.0f}", ha="center", va="bottom", color=INK2, fontsize=5.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(cell_lbl, fontsize=6.2, color=INK2)
    ax.set_xlim(-0.7, 3.7); ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    chrome(ax, "", "% called “evaluation”" if bi == 0 else "")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{lab.replace('Llama-', '')}  ·  L{PEAK[k]}", loc="center", color=INK, fontsize=6.8, pad=4)
    if bi > 0:
        ax.tick_params(labelleft=False)
bs[0].text(0.0, 1.16, "Decisions by cell, selected layer", transform=bs[0].transAxes,
           color=INK, fontsize=8, fontweight="bold", va="bottom", ha="left")
fig.legend(handles=[Patch(color=MUTED, label="confounded probe"), Patch(color=INK2, label="decorrelated probe (model color)")],
           loc="lower right", bbox_to_anchor=(0.995, 0.005), ncol=2, frameon=False, labelcolor=INK2,
           handlelength=1.0, handleheight=0.9, borderaxespad=0.0, fontsize=6.0, columnspacing=1.0)
panel_tag(bs[0], "B")
save(fig, "fig_results_01_replication")

# ----------------------------------------------------------------- figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.55), gridspec_kw={"width_ratios": [1.3, 1]})
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.2, top=0.86, wspace=0.34)
CEIL = 1 / np.sqrt((2080 / 3840) * (1760 / 3840))
for k, lab, c in MODELS:
    rows = NOCUE[k]; x = depth(rows)
    d = np.array([r["nocue"]["decorrelated"]["shift_sd"] for r in rows])
    lo = np.array([r["nocue"]["decorrelated"]["shift_sd_ci95"][0] for r in rows])
    hi = np.array([r["nocue"]["decorrelated"]["shift_sd_ci95"][1] for r in rows])
    cf = np.array([r["nocue"]["confounded"]["shift_sd"] for r in rows])
    a.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
    a.plot(x, d, color=c, lw=LW, zorder=3)
    a.plot(x, cf, color=c, lw=LW, ls=DASH, zorder=2)
    p = PEAK[k]; mark_peak(a, x[p], d[p], c)
    b.plot(x, [r["nocue"]["decorrelated"]["format_auc"] for r in rows], color=c, lw=LW, zorder=3)
    b.plot(x, [r["nocue"]["confounded"]["format_auc"] for r in rows], color=c, lw=LW, ls=DASH, zorder=2)
    mark_peak(b, x[p], rows[p]["nocue"]["decorrelated"]["format_auc"], c)
a.axhline(0, color=BASE, lw=0.7, zorder=1)
a.axhline(CEIL, color=BASE, lw=0.7, zorder=1)
a.text(1, CEIL + 0.03, f"two-point ceiling ({CEIL:.2f})", ha="left", va="bottom", color=MUTED, fontsize=6.3)
a.set_xlim(0, 100); a.set_ylim(-1.4, 2.4)
a.set_yticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
chrome(a, "relative depth (%)", "no-cue shift, benchmark − casual (SD units)", "Format shift on the purpose score")
a.legend(handles=model_handles() + style_handles("decorrelated (95% CI)", "confounded"),
         loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2, columnspacing=0.8,
         frameon=False, handlelength=1.4, handletextpad=0.5, labelcolor=INK2, borderaxespad=0.1, fontsize=5.8)
panel_tag(a, "A")
b.axhline(0.5, color=BASE, lw=0.7, zorder=1)
b.text(99, 0.507, "chance", ha="right", va="bottom", color=MUTED, fontsize=6.3)
b.set_xlim(0, 100); b.set_ylim(0.3, 1.04)
b.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
chrome(b, "relative depth (%)", "AUC, bench vs casual (purpose score)", "Ranking version")
panel_tag(b, "B")
save(fig, "fig_results_02_nocue")

# ----------------------------------------------------------------- figure 3
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.55))
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.2, top=0.86, wspace=0.34)
for k, lab, c in MODELS:
    rows = SWEEP[k]; x = depth(rows)
    dec = np.array([r["decorrelated"]["auc"] for r in rows])
    tr = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in rows])
    a.plot(x, dec, color=c, lw=LW, zorder=3)
    a.plot(x, tr, color=c, lw=LW, ls=DASH, zorder=2)
    p = PEAK[k]; mark_peak(a, x[p], dec[p], c)
    e = ERASE[k]; q = np.array(e["q"]) * 100
    g0 = np.array(e["by_rank"]["0"]["gap_by_layer"]); g64 = np.array(e["by_rank"]["64"]["gap_by_layer"])
    assert len(q) == len(rows) and np.allclose(g0, dec - tr, atol=1e-9)
    b.plot(q, g0, color=c, lw=LW, zorder=3)
    b.plot(q, g64, color=c, lw=LW, ls=DASH, zorder=2)
    mark_peak(b, q[p], g0[p], c)
a.axhline(0.5, color=BASE, lw=0.7, zorder=1)
a.text(99, 0.507, "chance", ha="right", va="bottom", color=MUTED, fontsize=6.3)
a.set_xlim(0, 100); a.set_ylim(0.42, 1.02); a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
chrome(a, "relative depth (%)", "held-family-out purpose AUC", "Single-format transfer by depth")
a.legend(handles=model_handles() + style_handles("decorrelated", "mean B→C / C→B"),
         loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=2, columnspacing=0.8,
         frameon=False, handlelength=1.4, handletextpad=0.5, labelcolor=INK2, borderaxespad=0.1, fontsize=5.8)
panel_tag(a, "A")
b.axhline(0, color=BASE, lw=0.7, zorder=1)
b.set_xlim(0, 100); b.set_ylim(-0.01, 0.225); b.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
chrome(b, "relative depth (%)", "transfer gap (AUC)", "Gap, raw vs. rank-64 projected")
b.legend(handles=style_handles("raw activations", "format subspace removed"),
         loc="center right", bbox_to_anchor=(1.0, 0.72), frameon=False, handlelength=1.8, labelcolor=INK2, borderaxespad=0.2)
panel_tag(b, "B")
save(fig, "fig_results_03_transfer")
