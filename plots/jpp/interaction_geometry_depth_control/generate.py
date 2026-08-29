"""Final publication figure: visual depth-confound demonstration for the interaction-
geometry / transfer-gap result.

Self-contained. Reads only pinned artifacts (no activation extraction):
  - artifacts/format_interaction_geometry_full320.json       (per-layer rho, gap, depth)
  - artifacts/format_interaction_geometry_partial_corr.json  (pinned raw/partial/delta stats)

Three panels, same post-spike layers (relative depth >= 0.15) used by the pinned stats:
  1. Raw scatter: format-dependence of the purpose representation vs. cross-format
     transfer gap, all three models overlaid, colored/marker-coded by model.
  2. The same points after residualizing BOTH quantities on relative depth, using the
     exact rank-residualization method that produced the pinned partial-Spearman
     statistics (rank-transform x, y, z; linearly regress ranked x and ranked y on
     ranked z; residuals are what gets plotted and Spearman-correlated).
  3. Layer-to-layer first differences (diff along depth-sorted layers) of the same two
     quantities -- the stricter test used in the pinned artifact.

Every plotted statistic is asserted against the pinned artifact (to 1e-6) before the
figure is drawn; a stale/rederived analysis would fail loudly rather than silently
mismatch what's on screen.
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
STATS_PATH = ROOT / "artifacts" / "format_interaction_geometry_partial_corr.json"

MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.3-70B", "Llama-3.1-70B"]
MODEL_KEYS = {"Llama-3.1-8B": "llama31_8b", "Llama-3.3-70B": "llama33_70b", "Llama-3.1-70B": "llama31_70b"}
MODEL_STYLE = {
    "Llama-3.1-8B": {"color": "#2E75B6", "marker": "o"},
    "Llama-3.3-70B": {"color": "#D06B29", "marker": "^"},
    "Llama-3.1-70B": {"color": "#5B9B4B", "marker": "s"},
}
INK = "#202124"
GRID = "#E3E6EA"
ZERO = "#B7BBC1"


def fmt_p(p: float) -> str:
    return "p<.001" if p < 1e-3 else f"p={p:.2f}"


def center_on_zero(ax, x: bool = True, y: bool = True) -> None:
    """Make (0, 0) the visual center of the axes by symmetrizing the current limits."""
    if x:
        x0, x1 = ax.get_xlim()
        m = max(abs(x0), abs(x1))
        ax.set_xlim(-m, m)
    if y:
        y0, y1 = ax.get_ylim()
        m = max(abs(y0), abs(y1))
        ax.set_ylim(-m, m)


def partial_spearman_resid(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Exact method used to produce the pinned r_partial/p_partial: rank-transform,
    then linearly residualize the ranked x and ranked y on ranked z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m, c = np.polyfit(b, a, 1)
        return a - (m * b + c)

    rx_resid, ry_resid = resid(rx, rz), resid(ry, rz)
    r, p = spearmanr(rx_resid, ry_resid)
    return rx_resid, ry_resid, r, p


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    geom = json.loads(GEOM_PATH.read_text())
    stats = json.loads(STATS_PATH.read_text())

    per_model = {}
    for label in MODEL_ORDER:
        key = MODEL_KEYS[label]
        d = geom[key]
        s = stats[key]
        depth = np.array(d["depth"])
        rho = np.array(d["rho_C_over_A"])
        gap = np.array(d["gap"])
        post = depth >= 0.15  # pinned stats (n=27/68) exclude the early-layer spike
        depth_p, rho_p, gap_p = depth[post], rho[post], gap[post]

        r_raw, _ = spearmanr(rho_p, gap_p)
        assert abs(r_raw - s["r_raw"]) < 1e-6, f"{label}: raw r mismatch"

        rx_resid, ry_resid, r_partial, p_partial = partial_spearman_resid(rho_p, gap_p, depth_p)
        assert abs(r_partial - s["r_partial"]) < 1e-6, f"{label}: partial r mismatch"
        assert abs(p_partial - s["p_partial"]) < 1e-6, f"{label}: partial p mismatch"

        d_rho, d_gap = np.diff(rho_p), np.diff(gap_p)
        r_delta, p_delta = spearmanr(d_rho, d_gap)
        assert abs(r_delta - s["r_delta"]) < 1e-6, f"{label}: delta r mismatch"
        assert abs(p_delta - s["p_delta"]) < 1e-6, f"{label}: delta p mismatch"

        per_model[label] = dict(
            depth=depth_p, rho=rho_p, gap=gap_p, rx_resid=rx_resid, ry_resid=ry_resid,
            d_rho=d_rho, d_gap=d_gap, s=s,
        )
    # Within-model standardization for Panel 2: z-score each model's rank-residuals
    # using that model's own mean/std (not pooled across models). This is a per-model,
    # per-axis affine rescale -- it does not change point order, relative spacing, or
    # any within-model correlation.
    for label in MODEL_ORDER:
        m = per_model[label]
        m["rx_z"] = (m["rx_resid"] - m["rx_resid"].mean()) / m["rx_resid"].std()
        m["ry_z"] = (m["ry_resid"] - m["ry_resid"].mean()) / m["ry_resid"].std()

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.35), constrained_layout=False)

    # --- Panel 1: raw relationship -----------------------------------------------
    ax = axes[0]
    for label in MODEL_ORDER:
        st = MODEL_STYLE[label]
        m = per_model[label]
        ax.scatter(m["rho"], m["gap"], s=24, color=st["color"], marker=st["marker"],
                   alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3, label=label)
        a, b = np.polyfit(m["rho"], m["gap"], 1)
        xs = np.linspace(m["rho"].min(), m["rho"].max(), 20)
        ax.plot(xs, a * xs + b, color=st["color"], linewidth=1.0, alpha=0.4, zorder=2)
    ax.set_title("Raw relationship", fontsize=11, fontweight="bold")
    ax.set_xlabel("Format-dependence of\npurpose representation")
    ax.set_ylabel("Cross-format transfer gap")
    ax.grid(color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    # --- Panel 2: after removing depth --------------------------------------------
    ax = axes[1]
    for label in MODEL_ORDER:
        st = MODEL_STYLE[label]
        m = per_model[label]
        ax.scatter(m["rx_z"], m["ry_z"], s=24, color=st["color"], marker=st["marker"],
                   alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(0, color=ZERO, linewidth=0.9, zorder=1)
    ax.axvline(0, color=ZERO, linewidth=0.9, zorder=1)
    ax.margins(x=0.08, y=0.08)
    center_on_zero(ax)
    ax.set_title("After removing the effect of depth", fontsize=11, fontweight="bold")
    ax.set_xlabel("Depth-adjusted format-dependence")
    ax.set_ylabel("Depth-adjusted transfer gap")
    ax.grid(color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    stat_lines = "\n".join(
        f"{label}: r={per_model[label]['s']['r_partial']:.2f}, {fmt_p(per_model[label]['s']['p_partial'])}"
        for label in MODEL_ORDER
    )
    ax.text(0.03, 0.97, stat_lines, transform=ax.transAxes, fontsize=7.1,
            va="top", ha="left", color=INK, linespacing=1.55)

    # --- Panel 3: layer-to-layer change test --------------------------------------
    ax = axes[2]
    for label in MODEL_ORDER:
        st = MODEL_STYLE[label]
        m = per_model[label]
        ax.scatter(m["d_rho"], m["d_gap"], s=24, color=st["color"], marker=st["marker"],
                   alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(0, color=ZERO, linewidth=0.9, zorder=1)
    ax.axvline(0, color=ZERO, linewidth=0.9, zorder=1)
    ax.margins(x=0.08, y=0.08)
    center_on_zero(ax)
    ax.set_title("Layer-to-layer changes", fontsize=11, fontweight="bold")
    ax.set_xlabel("Layer-to-layer change in\nformat-dependence")
    ax.set_ylabel("Layer-to-layer change\nin transfer gap")
    ax.grid(color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    stat_lines = "\n".join(
        f"{label}: r={per_model[label]['s']['r_delta']:.2f}, {fmt_p(per_model[label]['s']['p_delta'])}"
        for label in MODEL_ORDER
    )
    ax.text(0.03, 0.97, stat_lines, transform=ax.transAxes, fontsize=7.1,
            va="top", ha="left", color=INK, linespacing=1.55)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.24, wspace=0.32)

    # Shared legend identifying the three models (marker/color only).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06),
               ncol=3, frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.4)

    fig.suptitle(
        "Interaction geometry appears related to transfer failure,\nbut the relationship is largely explained by depth",
        fontsize=12.5, fontweight="bold", x=0.5, y=0.97, ha="center",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.15}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"interaction_geometry_depth_control_residualized.{suffix}", **kwargs)
    plt.close(fig)

    for label in MODEL_ORDER:
        s = per_model[label]["s"]
        print(f"{label}: raw r={s['r_raw']:.3f} | partial(depth) r={s['r_partial']:.3f} p={s['p_partial']:.4f} "
              f"| delta r={s['r_delta']:.3f} p={s['p_delta']:.4f}")
    print(f"wrote {OUT_DIR / 'interaction_geometry_depth_control_residualized.pdf'}")


if __name__ == "__main__":
    main()
