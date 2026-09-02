"""Quick single-model (Llama-3.1-8B) preview of the full-layer, normalized-depth
matched-format-PCA erasure gap analysis, while the 70B sweeps are still running
on the cluster. Same data/methodology as plots/jpp/full_layer_gap_erasure/
(reads results/full_layer_gap_erasure_summary.json; no activation access, no
model inference here). Not the final 3-model figure -- see
plots/jpp/full_layer_gap_erasure/ for that once all 3 checkpoints land.
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
MODEL_KEY = "llama31_8b"
MODEL_LABEL = "Llama-3.1-8B"
RANKS = [0, 1, 4, 16, 64]
RANK_CMAP = plt.get_cmap("viridis")
RANK_COLOR = {r: RANK_CMAP(0.92 - 0.85 * i / (len(RANKS) - 1)) for i, r in enumerate(RANKS)}
INK = "#202124"
GRID = "#D9DDE3"
COLOR = "#2A5FA5"


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
    m = data["by_model"][MODEL_KEY]
    q = np.array(m["q"])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    ax_traj, ax_G, ax_frac = axes

    for r in RANKS:
        gap = np.array(m["by_rank"][str(r)]["gap_by_layer"])
        lw = 2.0 if r == 0 else 1.3
        ax_traj.plot(q, gap, color=RANK_COLOR[r], linewidth=lw, alpha=0.95,
                      label=("raw (r=0)" if r == 0 else f"r={r}"))
    ax_traj.axhline(0.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0)
    ax_traj.set_xlabel("normalized depth  q = layer / (N-1)")
    ax_traj.set_ylabel("cross-format transfer gap  $g(q,r)$")
    ax_traj.set_title(f"{MODEL_LABEL} (N={m['n_layers']} layers)", loc="left", fontsize=10, fontweight="bold")
    ax_traj.set_xlim(-0.02, 1.02)
    ax_traj.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_traj.set_axisbelow(True)
    ax_traj.legend(loc="upper right", fontsize=7.5, framealpha=0.9, title="erasure rank r", title_fontsize=7.5)

    x_pos = np.arange(len(RANKS))
    G = np.array([m["by_rank"][str(r)]["G"] for r in RANKS])
    lo = np.array([m["by_rank"][str(r)]["G_ci95"][0] for r in RANKS])
    hi = np.array([m["by_rank"][str(r)]["G_ci95"][1] for r in RANKS])
    ax_G.plot(x_pos, G, "-o", color=COLOR, linewidth=1.6, markersize=5,
               markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax_G.fill_between(x_pos, lo, hi, color=COLOR, alpha=0.2, linewidth=0, zorder=1)
    ax_G.axhline(0.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0)
    ax_G.set_xticks(x_pos)
    ax_G.set_xticklabels([str(r) for r in RANKS])
    ax_G.set_xlabel("matched-format PCA rank removed, r")
    ax_G.set_ylabel("integrated gap  $G(r)=\\int_0^1 g(q,r)\\,dq$")
    ax_G.set_title("Integrated gap vs. rank", loc="left", fontsize=10, fontweight="bold")
    ax_G.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_G.set_axisbelow(True)

    frac = np.array([m["by_rank"][str(r)]["fraction_gap_removed"] for r in RANKS])
    flo = np.array([m["by_rank"][str(r)]["fraction_gap_removed_ci95"][0] for r in RANKS])
    fhi = np.array([m["by_rank"][str(r)]["fraction_gap_removed_ci95"][1] for r in RANKS])
    ax_frac.plot(x_pos, frac, "-o", color=COLOR, linewidth=1.6, markersize=5,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax_frac.fill_between(x_pos, flo, fhi, color=COLOR, alpha=0.2, linewidth=0, zorder=1)
    ax_frac.axhline(1.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0, linestyle="--")
    ax_frac.axhline(0.0, color=INK, linewidth=0.6, alpha=0.4, zorder=0)
    ax_frac.set_xticks(x_pos)
    ax_frac.set_xticklabels([str(r) for r in RANKS])
    ax_frac.set_xlabel("matched-format PCA rank removed, r")
    ax_frac.set_ylabel("fraction of gap removed  $1-G(r)/G(0)$")
    ax_frac.set_title("Fraction removed vs. rank", loc="left", fontsize=10, fontweight="bold")
    ax_frac.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_frac.set_axisbelow(True)

    fig.suptitle(
        f"{MODEL_LABEL} preview -- full-layer normalized-depth matched-format-PCA erasure "
        f"(payload-block bootstrap, N={n_boot}); 70B checkpoints pending",
        fontsize=11.5, fontweight="bold", x=0.01, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"8b_lol.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / '8b_lol.pdf'}")


if __name__ == "__main__":
    main()
