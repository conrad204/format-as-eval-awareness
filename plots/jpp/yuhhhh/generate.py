"""Quick 2-model (Llama-3.1-8B + Llama-3.3-70B) preview of the full-layer,
normalized-depth matched-format-PCA erasure gap analysis, while the
Llama-3.1-70B sweep is still running on the cluster. Same data/methodology as
plots/jpp/full_layer_gap_erasure/ (reads results/full_layer_gap_erasure_summary.json;
no activation access, no model inference here). Not the final 3-model figure --
see plots/jpp/full_layer_gap_erasure/ for that once all 3 checkpoints land.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results" / "full_layer_gap_erasure_summary.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.3-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.3-70B": "llama33_70b"}
MODEL_COLOR = {"llama31_8b": "#2A5FA5", "llama33_70b": "#1F7A5C"}
MODEL_MARKER = {"llama31_8b": "o", "llama33_70b": "^"}
RANKS = [0, 1, 4, 16, 64]
RANK_CMAP = plt.get_cmap("viridis")
RANK_COLOR = {r: RANK_CMAP(0.92 - 0.85 * i / (len(RANKS) - 1)) for i, r in enumerate(RANKS)}
INK = "#202124"
GRID = "#D9DDE3"


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    data = json.loads(DATA_PATH.read_text())
    n_boot = data["n_boot"]
    available = [lbl for lbl in MODEL_ORDER if MODEL_KEYS[lbl] in data["by_model"]]

    fig = plt.figure(figsize=(9.4, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    traj_axes = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(len(available))]
    ax_G = fig.add_subplot(gs[1, 0:2])
    ax_frac = fig.add_subplot(gs[1, 2:4])

    for ax, label in zip(traj_axes, available):
        key = MODEL_KEYS[label]
        m = data["by_model"][key]
        q = np.array(m["q"])
        for r in RANKS:
            gap = np.array(m["by_rank"][str(r)]["gap_by_layer"])
            lw = 2.0 if r == 0 else 1.3
            ax.plot(q, gap, color=RANK_COLOR[r], linewidth=lw, alpha=0.95,
                     label=("raw (r=0)" if r == 0 else f"r={r}"))
        ax.axhline(0.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0)
        ax.set_title(f"{label}  (N={m['n_layers']} layers)", loc="left", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("normalized depth  q = layer / (N-1)")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
    traj_axes[0].set_ylabel("cross-format transfer gap  $g(q,r)$")
    traj_axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9, title="erasure rank r", title_fontsize=7)

    x_pos = np.arange(len(RANKS))
    for label in available:
        key = MODEL_KEYS[label]
        m = data["by_model"][key]
        color = MODEL_COLOR[key]
        G = np.array([m["by_rank"][str(r)]["G"] for r in RANKS])
        lo = np.array([m["by_rank"][str(r)]["G_ci95"][0] for r in RANKS])
        hi = np.array([m["by_rank"][str(r)]["G_ci95"][1] for r in RANKS])
        ax_G.plot(x_pos, G, "-", color=color, linewidth=1.6, marker=MODEL_MARKER[key], markersize=5,
                   markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=5)
        ax_G.fill_between(x_pos, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)
    ax_G.axhline(0.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0)
    ax_G.set_xticks(x_pos)
    ax_G.set_xticklabels([str(r) for r in RANKS])
    ax_G.set_xlabel("matched-format PCA rank removed, r")
    ax_G.set_ylabel("integrated transfer gap\n$G(r) = \\int_0^1 g(q,r)\\,dq$")
    ax_G.set_title("Integrated gap vs. rank", loc="left", fontsize=9.5, fontweight="bold")
    ax_G.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_G.set_axisbelow(True)
    ax_G.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    for label in available:
        key = MODEL_KEYS[label]
        m = data["by_model"][key]
        color = MODEL_COLOR[key]
        frac = np.array([m["by_rank"][str(r)]["fraction_gap_removed"] for r in RANKS])
        lo = np.array([m["by_rank"][str(r)]["fraction_gap_removed_ci95"][0] for r in RANKS])
        hi = np.array([m["by_rank"][str(r)]["fraction_gap_removed_ci95"][1] for r in RANKS])
        ax_frac.plot(x_pos, frac, "-", color=color, linewidth=1.6, marker=MODEL_MARKER[key], markersize=5,
                      markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=5)
        ax_frac.fill_between(x_pos, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)
    ax_frac.axhline(1.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0, linestyle="--")
    ax_frac.axhline(0.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0)
    ax_frac.set_xticks(x_pos)
    ax_frac.set_xticklabels([str(r) for r in RANKS])
    ax_frac.set_xlabel("matched-format PCA rank removed, r")
    ax_frac.set_ylabel("fraction of integrated gap removed\n$1 - G(r)/G(0)$")
    ax_frac.set_title("Fraction of gap removed vs. rank", loc="left", fontsize=9.5, fontweight="bold")
    ax_frac.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_frac.set_axisbelow(True)

    fig.suptitle(
        "Full-layer, normalized-depth matched-format-PCA erasure gap -- 2/3 checkpoints "
        f"(payload-block bootstrap, N={n_boot}); Llama-3.1-70B pending",
        fontsize=11, fontweight="bold", x=0.01, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"yuhhhh.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'yuhhhh.pdf'}")


if __name__ == "__main__":
    main()
