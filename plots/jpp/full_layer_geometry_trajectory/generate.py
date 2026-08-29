"""Full-layer-resolution matched-format subspace geometry, rank 16, all 3 checkpoints.

Extends plots/jpp/matched_format_subspace_geometry (3 pinned depths x 3 models x
4 ranks, Holm/Bonferroni-corrected) to every transformer layer, using the SAME
apples-to-apples method (matched-format PCA basis Q_format(16) projected into and
re-orthonormalized within the empirical top-64 PC subspace, compared against a
conditional null of random rank-16 subspaces drawn from within that same top-64
subspace, using the real purpose-direction coordinate vector). Source:
scripts/full320_v3_geometry_full_layer_sweep.py, computed on uahpc.

Explicitly descriptive, not a new inferential claim: adjacent layers share most of
their variance (residual-stream continuity), so per-layer p-values here are
repeated measurements of one underlying depth trajectory, not independent tests.
This figure is NOT Holm/Bonferroni-corrected across layers and no per-layer
significance count is reported anywhere in this script or its caption -- doing so
would be exactly the layers-as-independent-replicates pseudoreplication RULES.md
prohibits. The only corrected, confirmatory claim is the pre-registered 3-depth
27-cell family in results/full320_v3_conditional_null_multiplicity.json; this plot
exists to show that the corrected 3-point result is not an artifact of which 3
layers happened to be picked, by showing the whole trajectory around them.

Self-contained. Reads only:
  results/full320_v3_geometry_full_layer_sweep.json
No activation extraction or probe refitting here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
SWEEP_PATH = ROOT / "results" / "full320_v3_geometry_full_layer_sweep.json"

DIMS = {"llama31_8b": 4096, "llama31_70b": 8192, "llama33_70b": 8192}
MODEL_ORDER = ["llama31_8b", "llama31_70b", "llama33_70b"]
MODEL_LABEL = {"llama31_8b": "Llama-3.1-8B", "llama31_70b": "Llama-3.1-70B", "llama33_70b": "Llama-3.3-70B"}
MODEL_COLOR = {"llama31_8b": "#2A5FA5", "llama31_70b": "#C9502F", "llama33_70b": "#1F7A5C"}
# Pre-registered pinned depths, kept only as reference markers for continuity with
# the corrected 3-depth test (matched_format_subspace_geometry), not re-tested here.
PINNED_DEPTHS = {
    "llama31_8b": {"spike": 3, "min_gap": 14, "final": 31},
    "llama31_70b": {"spike": 4, "min_gap": 19, "final": 79},
    "llama33_70b": {"spike": 4, "min_gap": 21, "final": 79},
}
RANK = 16
N_DRAWS = 10_000
P_FLOOR = 1 / (N_DRAWS + 1)
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
    sweep = json.loads(SWEEP_PATH.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax_energy, ax_p = axes

    for key in MODEL_ORDER:
        label = MODEL_LABEL[key]
        color = MODEL_COLOR[key]
        d = DIMS[key]
        by_layer = sweep["by_model"][key]["by_layer"]
        n_layers = sweep["by_model"][key]["n_layers"]
        layers = list(range(n_layers))
        rel_depth = np.array([L / (n_layers - 1) for L in layers])

        energy = np.array([by_layer[str(L)]["by_rank"][str(RANK)]["E_purpose_in_format_subspace"] for L in layers])
        p_cond = np.array([by_layer[str(L)]["by_rank"][str(RANK)]["p_conditional_apples_to_apples"] for L in layers])

        ax_energy.plot(rel_depth, energy, "-", color=color, linewidth=1.4, alpha=0.9, label=label)
        ax_energy.axhline(RANK / d, color=color, linewidth=0.9, linestyle=":", alpha=0.6)
        ax_p.plot(rel_depth, np.clip(p_cond, P_FLOOR, None), "-", color=color, linewidth=1.4, alpha=0.9, label=label)

        for L in PINNED_DEPTHS[key].values():
            rd = L / (n_layers - 1)
            ax_energy.plot(rd, energy[layers.index(L)], "o", color=color, markersize=5,
                            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
            ax_p.plot(rd, max(p_cond[layers.index(L)], P_FLOOR), "o", color=color, markersize=5,
                       markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    ax_energy.set_yscale("log")
    ax_p.set_yscale("log")
    ax_p.axhline(0.05, color=INK, linewidth=0.9, linestyle="--", alpha=0.5)
    ax_p.axhline(P_FLOOR, color=INK, linewidth=0.7, linestyle=":", alpha=0.35)

    for ax in (ax_energy, ax_p):
        ax.set_xlabel("relative depth (0 = first layer, 1 = last layer)")
        ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.02, 1.02)

    ax_energy.set_ylabel(f"E_purpose→format({RANK})\n(dotted: exact chance {RANK}/d)")
    ax_p.set_ylabel(f"conditional-null p (rank {RANK})\n(dashed: p=0.05, dotted: {N_DRAWS}-draw floor)")
    ax_energy.set_title("Purpose energy in matched-format subspace vs. depth", loc="left", fontsize=10, fontweight="bold")
    ax_p.set_title("Apples-to-apples conditional-null p vs. depth", loc="left", fontsize=10, fontweight="bold")
    ax_energy.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "Full-layer geometry trajectory, rank 16 (dots: pre-registered pinned depths)",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"full_layer_geometry_trajectory.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'full_layer_geometry_trajectory.pdf'}")


if __name__ == "__main__":
    main()
