#!/usr/bin/env python3
"""Paper figures 1 and 2 — claim-first versions.

Replaces the exhaustive analysis figures (which move to the appendix) with two
two-panel figures whose claim is readable in a few seconds.

Fig 1 — where the nuisance lives:
  A  per-layer raw gap vs rank-64 projected gap, one dot per layer, y=x diagonal.
     Points below the diagonal = projection reduced the gap at that layer.
  B  fraction of the depth-integrated gap removed vs projection rank, for the
     cue-bearing basis and the cue-free basis.

Fig 2 — the network uses it:
  A  interchange effect above the magnitude-matched control vs normalized depth,
     one curve per checkpoint, with a depth-integrated strip at the right.
  B  per-layer tradeoff: format moved toward donor (x) against purpose
     preservation relative to control (y), both relative to the same control.

Fig 2B's y-axis is expressed in units of the baseline purpose-score SD. The
underlying quantity is a difference of mean absolute shifts in the frozen
purpose probe's decision score, which is scale-dependent; the baseline SD is
constant within a checkpoint (7.75 / 7.73 / 7.18), so this is a per-model
rescale that makes the three checkpoints comparable without altering shape.
Same score-SD convention as the no-cue displacement metric.

Layers are repeated observations along one depth trajectory per model, not
independent replicates; no per-layer significance count is derived from these
figures.

Reads only committed result JSONs. No activations, no probe fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

MODELS = ["llama31_8b", "llama31_70b", "llama33_70b"]
LABEL = {"llama31_8b": "Llama-3.1-8B", "llama31_70b": "Llama-3.1-70B", "llama33_70b": "Llama-3.3-70B"}
COLOR = {"llama31_8b": "#2A5FA5", "llama31_70b": "#C9502F", "llama33_70b": "#1F7A5C"}
MARKER = {"llama31_8b": "o", "llama31_70b": "s", "llama33_70b": "^"}
NLAYERS = {"llama31_8b": 32, "llama31_70b": 80, "llama33_70b": 80}
RANKS = [0, 1, 4, 16, 64]
INK, GRID = "#202124", "#D9DDE3"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        kw = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == "png":
            kw["dpi"] = 400
        fig.savefig(OUT_DIR / f"{name}.{suffix}", **kw)
    plt.close(fig)
    print(f"wrote {OUT_DIR / (name + '.pdf')}")


def figure1() -> None:
    erasure = json.loads((ROOT / "results" / "full_layer_gap_erasure_summary.json").read_text())["by_model"]

    nocue = {}
    for m in MODELS:
        nocue[m] = {0: (0.0, (0.0, 0.0))}
        for r in RANKS[1:]:
            d = json.loads((ROOT / "results" / f"nocue_format_subspace_{m}_r{r}.json").read_text())
            nocue[m][r] = (d["fraction_integrated_gap_removed"],
                           tuple(d["fraction_integrated_gap_removed_ci95"]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)

    hi = 0.0
    for m in MODELS:
        raw = np.asarray(erasure[m]["by_rank"]["0"]["gap_by_layer"], dtype=float)
        prj = np.asarray(erasure[m]["by_rank"]["64"]["gap_by_layer"], dtype=float)
        hi = max(hi, float(np.nanmax(raw)), float(np.nanmax(prj)))
        below = int(np.sum(prj < raw))
        axA.scatter(raw, prj, s=17, marker=MARKER[m], facecolor=COLOR[m], edgecolor="white",
                    linewidth=0.4, alpha=0.85, zorder=3,
                    label=f"{LABEL[m]}  ({below}/{len(raw)} layers reduced)")
    lim = (0.0, hi * 1.06)
    axA.plot(lim, lim, color=INK, linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.55, zorder=2)
    axA.annotate("$y=x$", xy=(lim[1] * 0.87, lim[1] * 0.87), xytext=(-30, 6),
                 textcoords="offset points", fontsize=7.5, color=INK, alpha=0.7)
    axA.annotate("below the line:\nprojection reduced the gap",
                 xy=(lim[1] * 0.60, lim[1] * 0.18), fontsize=7.5, color=INK, alpha=0.75)
    axA.set_xlim(*lim)
    axA.set_ylim(*lim)
    axA.set_aspect("equal")
    axA.set_xlabel("raw transfer gap (per layer)")
    axA.set_ylabel("gap after rank-64 projection")
    axA.set_title("A. Every layer, before vs after", loc="left", fontsize=10, fontweight="bold")
    axA.legend(loc="upper left", fontsize=6.8, framealpha=0.9)

    x = np.arange(len(RANKS))
    for m in MODELS:
        cue = np.array([erasure[m]["by_rank"][str(r)]["fraction_gap_removed"] * 100 for r in RANKS])
        cue_ci = np.array([erasure[m]["by_rank"][str(r)]["fraction_gap_removed_ci95"] for r in RANKS]) * 100
        nc = np.array([nocue[m][r][0] * 100 for r in RANKS])
        nc_ci = np.array([nocue[m][r][1] for r in RANKS]) * 100
        axB.errorbar(x, cue, yerr=np.abs(cue_ci.T - cue), color=COLOR[m], marker=MARKER[m],
                     markersize=4.5, linewidth=1.7, capsize=2, zorder=3)
        axB.errorbar(x, nc, yerr=np.abs(nc_ci.T - nc), color=COLOR[m], marker=MARKER[m],
                     markersize=4.0, linewidth=1.4, capsize=2, linestyle=(0, (2, 1.6)),
                     alpha=0.75, zorder=3)
    axB.axhline(0, color=INK, linewidth=0.9, linestyle=(0, (1, 1)), alpha=0.45)
    axB.set_xticks(x)
    axB.set_xticklabels([str(r) for r in RANKS])
    axB.set_xlabel("projection rank")
    axB.set_ylabel("% of depth-integrated gap removed")
    axB.set_title("B. Structure is distributed, and about half is\n     recoverable without purpose cues",
                  loc="left", fontsize=10, fontweight="bold")

    model_keys = [Line2D([], [], color=COLOR[m], marker=MARKER[m], linewidth=1.7,
                         markersize=4.5, label=LABEL[m]) for m in MODELS]
    basis_keys = [Line2D([], [], color=INK, linewidth=1.6, label="cue-bearing basis"),
                  Line2D([], [], color=INK, linewidth=1.4, linestyle=(0, (2, 1.6)),
                         alpha=0.75, label="cue-free basis")]
    leg1 = axB.legend(handles=model_keys, loc="upper left", fontsize=6.8, framealpha=0.9)
    axB.add_artist(leg1)
    axB.legend(handles=basis_keys, loc="upper left", bbox_to_anchor=(0.0, 0.76),
               fontsize=6.8, framealpha=0.9)
    axB.annotate("rank-64 cue-bearing: 58.5 / 62.8 / 65.3%\n"
                 "rank-64 cue-free:      29.8 / 25.8 / 32.7%\n"
                 r"$|\Delta$ joint-format purpose AUC$|<.002$; format AUC $\approx$ .99",
                 xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=6.6, color=INK, alpha=0.85)

    for ax in (axA, axB):
        ax.grid(color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
    save(fig, "fig1_residual_low_rank")


def figure2() -> None:
    boot = {
        "llama31_8b": "tae_ixpatch_full_llama31_8b_bootstrap.json",
        "llama31_70b": "tae_ixpatch_full_llama31_70b_evenlayers_bootstrap.json",
        "llama33_70b": "tae_ixpatch_full_llama33_70b_bootstrap.json",
    }
    integ = {m: json.loads((ROOT / "results" / f"tae_ixpatch_integrated_{m}.json").read_text())
             for m in MODELS}
    scale = {m: json.loads((ROOT / "results" / f"tae_ixpatch_purposescale_{m}.json").read_text())["by_layer"]
             for m in MODELS}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    strip = axA.inset_axes([1.04, 0.0, 0.15, 1.0])

    for m in MODELS:
        d = json.loads((ROOT / "results" / boot[m]).read_text())["by_control"]["matched_control"]["by_layer"]
        layers = sorted(int(k) for k in d)
        q = np.array([n / (NLAYERS[m] - 1) for n in layers])
        gap = np.array([d[str(n)]["format_iia_gap"] for n in layers])
        pos = sum(1 for n in layers if d[str(n)]["format_iia_gap_ci95"][0] > 0)
        axA.plot(q, gap, color=COLOR[m], linewidth=1.8, zorder=3,
                 label=f"{LABEL[m]}  ({pos}/{len(layers)} layers $>$ 0)")
        prot = np.array([scale[m][str(n)]["purpose_gap_sd_units"] for n in layers])
        axB.scatter(gap, prot, s=18, marker=MARKER[m], facecolor=COLOR[m], edgecolor="white",
                    linewidth=0.4, alpha=0.85, zorder=3, label=LABEL[m])

    axA.axhline(0, color=INK, linewidth=1.0, zorder=2)
    axA.set_xlim(-0.02, 1.02)
    axA.set_xlabel("relative depth (0 = first layer, 1 = last)")
    axA.set_ylabel("format shift toward donor, above control")
    axA.set_title("A. The swap moves the format readout, at nearly every layer",
                  loc="left", fontsize=10, fontweight="bold")
    axA.legend(loc="upper left", fontsize=6.8, framealpha=0.9)

    for i, m in enumerate(MODELS):
        v = integ[m]["integrated_format_iia_gap"]
        lo, hi = integ[m]["integrated_format_iia_gap_ci95"]
        strip.errorbar([i], [v], yerr=[[v - lo], [hi - v]], color=COLOR[m], marker=MARKER[m],
                       markersize=6, capsize=3, linewidth=1.6, zorder=3)
    strip.set_ylim(axA.get_ylim())
    strip.set_xlim(-0.7, len(MODELS) - 0.3)
    strip.axhline(0, color=INK, linewidth=1.0, zorder=2)
    strip.set_xticks([])
    strip.set_yticks([])
    strip.grid(color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    strip.set_axisbelow(True)
    strip.set_title("depth-\nintegrated", fontsize=6.6, color=INK, pad=3)

    axB.axhline(0, color=INK, linewidth=1.0, zorder=2)
    axB.axvline(0, color=INK, linewidth=1.0, zorder=2)
    axB.set_xlabel("format moved toward donor, above control")
    axB.set_ylabel("purpose preservation vs control\n(baseline purpose-score SD units)")
    axB.set_title("B. Format is carried, purpose is not spared",
                  loc="left", fontsize=10, fontweight="bold")
    axB.legend(loc="upper left", fontsize=6.8, framealpha=0.9)
    axB.annotate("purpose protected\n(late depth)", xy=(0.72, 0.90), xycoords="axes fraction",
                 ha="left", va="top", fontsize=6.6, color=INK, alpha=0.8)
    axB.annotate("purpose disrupted more than control",
                 xy=(0.34, 0.13), xycoords="axes fraction", xytext=(0.44, 0.05),
                 textcoords="axes fraction", fontsize=6.6, color=INK, alpha=0.8,
                 arrowprops={"arrowstyle": "->", "color": INK, "alpha": 0.55, "linewidth": 0.7})

    for ax in (axA, axB):
        ax.grid(color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
    save(fig, "fig2_interchange")


if __name__ == "__main__":
    style()
    figure1()
    figure2()
