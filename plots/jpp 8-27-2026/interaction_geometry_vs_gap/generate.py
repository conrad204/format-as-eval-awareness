"""Does interaction-vs-shared geometry track the transfer gap? Checked, and mostly no.

Self-contained (see cross_format_gap_trajectory/generate.py for why).

The raw post-spike Spearman correlation between rho(l) = ||C(l)||/||A(l)|| and
the transfer gap G(l) is strong (r = 0.74 / 0.94 / 0.85 across the three
models) -- but rho(l) correlates with relative depth alone at r ~ 0.97-0.98 in
every model, and so does G(l) (r ~ 0.76-0.96). That is close to a confound by
construction: two quantities that both increase almost monotonically with
depth will correlate with each other even with no direct relationship.

Two harsher tests, run to check this directly:
  1. Partial Spearman r(rho, G | depth) -- residualize both on depth-rank,
     correlate the residuals.
  2. Spearman r(delta_rho, delta_G) on layer-to-layer first differences --
     does an interaction increase from one layer to the next coincide with a
     gap increase, independent of the overall depth trend?

Result: the partial correlation is non-significant in 2 of 3 models (8B
r=0.15 p=0.45; 3.3-70B r=0.06 p=0.65) and only partially survives in the
third (3.1-70B r=0.46 p<0.001, versus a raw r of 0.85). The delta-delta test
is non-significant in all three models (p in [0.09, 0.90]). Conclusion: most
of the raw correlation is a shared-depth-trend artifact, not evidence that
growing interaction geometry specifically drives the transfer-gap regrowth.
This figure reports that null result plainly rather than the flattering raw
scatter -- do not upgrade this to a mechanistic claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
GEOM_PATH = ROOT / "artifacts" / "format_interaction_geometry_full320.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.3-70B", "Llama-3.1-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.3-70B": "llama33_70b", "Llama-3.1-70B": "llama31_70b"}
INK = "#202124"
MUTED = "#666A70"
GRID = "#D9DDE3"
FAIL = "#B23A48"


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m, c = np.polyfit(b, a, 1)
        return a - (m * b + c)

    return spearmanr(resid(rx, rz), resid(ry, rz))


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
    geom = json.loads(GEOM_PATH.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), constrained_layout=True, sharey=True)
    cmap = plt.get_cmap("viridis")

    for ax, label in zip(axes, MODEL_ORDER):
        d = geom[MODEL_KEYS[label]]
        depth = np.array(d["depth"])
        rho = np.array(d["rho_C_over_A"])
        gap = np.array(d["gap"])
        post_spike = depth >= 0.15
        depth_p, rho_p, gap_p = depth[post_spike], rho[post_spike], gap[post_spike]

        ax.scatter(rho[~post_spike], gap[~post_spike], s=20, color=MUTED, alpha=0.4,
                   edgecolor="none", label="early spike (depth<15%)", zorder=2)
        sc = ax.scatter(rho_p, gap_p, s=28, c=depth_p, cmap=cmap, vmin=0.15, vmax=1.0,
                        edgecolor="white", linewidth=0.4, zorder=3, label="depth \u2265 15%")

        m, b = np.polyfit(rho_p, gap_p, 1)
        xs = np.linspace(rho_p.min(), rho_p.max(), 20)
        ax.plot(xs, m * xs + b, color=INK, linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)

        r_raw, _ = spearmanr(rho_p, gap_p)
        r_partial, p_partial = partial_spearman(rho_p, gap_p, depth_p)
        d_rho, d_gap = np.diff(rho_p), np.diff(gap_p)
        r_delta, p_delta = spearmanr(d_rho, d_gap)


        ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        stats_lines = "\n".join([
            f"raw r={r_raw:.2f}",
            f"controlling for depth: r={r_partial:.2f} (p={p_partial:.2f})",
            f"$\\Delta\\rho$ vs $\\Delta G$: r={r_delta:.2f} (p={p_delta:.2f})",
        ])
        ax.text(0.02, 0.97, stats_lines, transform=ax.transAxes, fontsize=6.6,
                va="top", ha="left", color=INK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID, linewidth=0.6))
        sig_partial = "sig." if p_partial < 0.05 else "not sig."
        sig_delta = "sig." if p_delta < 0.05 else "not sig."
        ax.text(0.98, 0.03, f"partial: {sig_partial}   \u0394-test: {sig_delta}", transform=ax.transAxes,
                fontsize=6.6, color=(INK if (p_partial < 0.05 and p_delta < 0.05) else FAIL),
                va="bottom", ha="right", fontstyle="italic")

        ax.set_xlabel(r"$\rho(\ell) = \|C(\ell)\| / \|A(\ell)\|$")
        ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        print(f"{label}: raw r={r_raw:.3f} | partial(depth) r={r_partial:.3f} p={p_partial:.3f} | "
              f"delta r={r_delta:.3f} p={p_delta:.3f}")

    axes[0].set_ylabel("Transfer gap  G($\\ell$)")
    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.75, pad=0.015)
    cbar.set_label("relative depth", fontsize=8)
    fig.suptitle(
        "Interaction geometry correlates with the transfer gap \u2014 almost entirely through depth",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )
    fig.text(
        0.0, -0.1,
        "\u03c1(\u2113) tracks relative depth at r\u2248 0.97\u20130.98 in every model, and so does the gap \u2014 two depth-driven curves\n"
        "correlate with each other even absent a direct link. Controlling for depth (partial Spearman), the association is NOT\n"
        "significant in 2 of 3 models and survives only partially in the third (3.1-70B, r=0.46 vs. raw 0.85). The harsher test \u2014\n"
        "does a layer-to-layer increase in \u03c1 coincide with a layer-to-layer increase in the gap (\u0394\u03c1 vs. \u0394G) \u2014 is non-significant in\n"
        "all three models. Read: most of the raw correlation is a shared-depth-trend artifact, not evidence that interaction\n"
        "geometry specifically drives the transfer-gap regrowth. Not promoted to a mechanistic claim.",
        ha="left", va="top", fontsize=7.1, color=MUTED,
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"interaction_geometry_vs_gap.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'interaction_geometry_vs_gap.pdf'}")


if __name__ == "__main__":
    main()
