"""Final validation pass: preregistered selectivity prediction, bootstrapped.

Self-contained. Source: artifacts/format_transport_intervention.json
(scripts/format_transport_intervention.py).

This is the FROZEN endpoint of the transport-intervention branch (rotation /
covariance-alignment hypotheses for the two failure modes identified in
format_probe_signal_noise_decomposition.py). No further transforms, no
tuning past this point -- see the module docstring of
scripts/format_transport_intervention.py for the two real bugs found and
fixed during development (full-rank O(d^3) Procrustes SVD on a rank-deficient
matrix; under-regularized CORAL covariance inverse amplifying sampling
noise), both verified against synthetic ground truth before trusting any
real-data result.

The test is NOT "does the transform beat baseline" (a weaker, less
informative comparison already reported as null: Procrustes 2/18, CORAL
0/18). It is the actual preregistered SELECTIVITY prediction:

  interaction_procrustes = (AUC_procrustes - AUC_baseline)_{c2b}
                          - (AUC_procrustes - AUC_baseline)_{b2c}
  interaction_coral      = (AUC_coral - AUC_baseline)_{b2c}
                          - (AUC_coral - AUC_baseline)_{c2b}

Both predicted > 0 (and > a predeclared practical threshold, +0.02 AUC, for
a "meaningful rescue" claim) if rotation/covariance mismatch selectively
explain their respective failure mode. 95% CIs are a 1000-rep block bootstrap
over the 80 held-out payload blocks (paired: same resample used for both
legs of each interaction).

Result, 9 model x depth cells per hypothesis (18 total):
  - Procrustes: mean interaction +0.009 (far below +0.02). 5/9 positive sign,
    5/9 significant in the predicted direction, but 2/9 SIGNIFICANT IN THE
    WRONG DIRECTION. Per-model averages disagree in sign (Llama-3.1-70B
    averages negative). Only 2/9 cells clear +0.02.
  - CORAL: mean interaction +0.002 (~zero). 4/9 positive sign, only 3/9
    significant in the predicted direction versus 4/9 SIGNIFICANT IN THE
    WRONG DIRECTION -- more cells contradict the prediction than support it.
    Only 2/9 cells clear +0.02.

Neither aggregate estimate clears the predeclared +0.02 practical threshold,
and the per-cell sign/significance pattern does not even consistently favor
the predicted direction. This is stronger than "CIs cross zero broadly": it
is an internally inconsistent pattern across cells, which is evidence AGAINST
a stable underlying selectivity effect, not merely underpowered null evidence
for one. Per pre-registration: the simple transport hypotheses are not
supported. The exact signal/noise decomposition remains the defensible
mechanistic-level endpoint. Branch closed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "artifacts" / "format_transport_intervention.json"

MODEL_ORDER = ["llama31_8b", "llama33_70b", "llama31_70b"]
LABELS = {"llama31_8b": "Llama-3.1-8B", "llama33_70b": "Llama-3.3-70B", "llama31_70b": "Llama-3.1-70B"}
MODEL_COLORS = {"llama31_8b": "#0072B2", "llama33_70b": "#D55E00", "llama31_70b": "#009E73"}
INK = "#202124"
MUTED = "#666A70"
GRID = "#D9DDE3"
THRESH = 0.02


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    data = json.loads(DATA_PATH.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True, sharex=True)
    specs = [
        ("interaction_procrustes", "Procrustes selectivity\n(predicted: c2b helped more than b2c)"),
        ("interaction_coral", "CORAL selectivity\n(predicted: b2c helped more than c2b)"),
    ]

    all_points = {"interaction_procrustes": [], "interaction_coral": []}
    for ax, (key, title) in zip(axes, specs):
        labels, pts, los, his, colors = [], [], [], [], []
        for model in MODEL_ORDER:
            d = data[model]
            for depth_label, L in [("27%", d["anchor_layer"]), ("50%", d["mid_layer"]), ("100%", d["final_layer"])]:
                boot = d["layers"][str(L)]["bootstrap"]
                p = boot[f"{key}_point"]
                lo, hi = boot[f"{key}_ci95"]
                labels.append(f"{LABELS[model]} {depth_label}")
                pts.append(p); los.append(lo); his.append(hi)
                colors.append(MODEL_COLORS[model])
                all_points[key].append(p)

        y = np.arange(len(labels))[::-1]
        for yi, p, lo, hi, c in zip(y, pts, los, his, colors):
            sig_wrong = hi < 0
            ax.plot([lo, hi], [yi, yi], color=("#B23A48" if sig_wrong else c), linewidth=1.6, zorder=2,
                    alpha=1.0 if sig_wrong else 0.85)
            ax.scatter([p], [yi], color=("#B23A48" if sig_wrong else c), s=32, zorder=3, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.axvline(THRESH, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(THRESH, len(labels) - 0.3, f" +{THRESH} threshold", fontsize=6.6, color=MUTED, va="top")
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.set_xlabel("Interaction (AUC), predicted > 0")
        ax.set_title(title, fontsize=9.3, fontweight="bold", loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.13, 0.10)

    fig.suptitle("Preregistered selectivity prediction: not supported, and internally inconsistent",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    proc_mean = np.mean(all_points["interaction_procrustes"])
    coral_mean = np.mean(all_points["interaction_coral"])
    fig.text(
        0.0, -0.13,
        f"95% CI, 1,000-rep bootstrap over the 80 held-out payload blocks (paired per cell). Red segments = CI entirely below\n"
        f"zero (significant in the WRONG direction). Mean interaction: Procrustes {proc_mean:+.3f}, CORAL {coral_mean:+.3f} \u2014 both\n"
        f"far below the predeclared +{THRESH} practical-rescue threshold. Procrustes: 5/9 cells significant in the predicted\n"
        f"direction but 2/9 significant in the WRONG direction; per-model averages disagree in sign. CORAL: only 3/9 cells\n"
        f"significant in the predicted direction versus 4/9 significant in the WRONG direction \u2014 more cells contradict the\n"
        f"prediction than support it. Neither hypothesis is supported. Signal/noise decomposition remains the endpoint.",
        ha="left", va="top", fontsize=7.0, color=MUTED,
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"transport_intervention_results.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'transport_intervention_results.pdf'}")
    print(f"procrustes mean interaction: {proc_mean:+.4f}")
    print(f"coral mean interaction: {coral_mean:+.4f}")


if __name__ == "__main__":
    main()
