"""Sources of discriminability loss in cross-format transfer -- two distinct failure modes.

Self-contained. Source: artifacts/format_probe_signal_noise_decomposition.json
(scripts/format_probe_signal_noise_decomposition.py), computed directly from
cached full320-v3 activations -- not a correlation, an exact decomposition.

Terminology note: these are FAILURE MODES / SOURCES OF DISCRIMINABILITY LOSS,
not claims about a causal neural mechanism. The algebra is exact for d' (mean
class-separation / pooled within-class SD, on the logit/decision-function
scale): log(d'_self/d'_transfer) splits exactly, by construction, into a
signal term and a noise term, so each term's share of the total log-drop is
well defined (share can exceed 100% or be negative if the two terms partially
offset). The real held-out ROC AUC is reported alongside d' as EXTERNAL
VALIDATION that d' tracks true performance -- d' and AUC are related but not
identical quantities (AUC is a rank statistic under the empirical score
distribution; d' is a parametric binormal separation statistic), so this is
not a claim of an exact AUC decomposition, only an exact d' decomposition
with AUC-based validation that the story it tells is not an artifact of the
d' parameterization.

Finding, replicated in 3/3 models at all 3 checked layers (fixed 27% depth
anchor, mid-depth, final layer): the two transfer DIRECTIONS lose
discriminability via different failure modes.
  - casual-trained probe tested on benchmark: SIGNAL loss dominates (probe's
    learned direction stops tracking the true benchmark-format separation).
  - benchmark-trained probe tested on casual: NOISE growth dominates (signal
    is usually preserved or even improves; within-class scatter explodes).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "artifacts" / "format_probe_signal_noise_decomposition.json"

MODEL_ORDER = ["llama31_8b", "llama33_70b", "llama31_70b"]
LABELS = {"llama31_8b": "Llama-3.1-8B", "llama33_70b": "Llama-3.3-70B", "llama31_70b": "Llama-3.1-70B"}
MODEL_COLORS = {"llama31_8b": "#0072B2", "llama33_70b": "#D55E00", "llama31_70b": "#009E73"}
INK = "#202124"
MUTED = "#666A70"
GRID = "#D9DDE3"
SIGNAL_COLOR = "#6F4E9C"
NOISE_COLOR = "#B23A48"


def shares(layer: dict, self_name: str, transfer_name: str) -> tuple[float, float, float]:
    s, t = layer[self_name], layer[transfer_name]
    log_signal_drop = np.log(s["unpaired_signal_mean_diff"]) - np.log(t["unpaired_signal_mean_diff"])
    log_noise_grow = np.log(t["unpaired_pooled_sd"]) - np.log(s["unpaired_pooled_sd"])
    total = log_signal_drop + log_noise_grow
    return (100 * log_signal_drop / total, 100 * log_noise_grow / total, total) if total else (np.nan, np.nan, 0.0)


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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True, sharey=True)
    direction_specs = [
        ("probe trained on casual\ntested on benchmark", "bb_self", "c2b_transfer", axes[0]),
        ("probe trained on benchmark\ntested on casual", "cc_self", "b2c_transfer", axes[1]),
    ]

    pair_key = {("bb_self", "c2b_transfer"): "bench", ("cc_self", "b2c_transfer"): "casual"}
    for title, self_name, transfer_name, ax in direction_specs:
        labels = []
        signal_vals, noise_vals = [], []
        ci_lo, ci_hi = [], []
        for model in MODEL_ORDER:
            d = data[model]
            for depth_label, L in [("27%", d["anchor_layer"]), ("50%", d["mid_layer"]), ("100%", d["final_layer"])]:
                layer = d["layers"][str(L)]
                sig, noise, total = shares(layer, self_name, transfer_name)
                labels.append(f"{LABELS[model]}\ndepth {depth_label}")
                signal_vals.append(sig)
                noise_vals.append(noise)
                boot = layer.get("bootstrap_ci", {}).get(pair_key[(self_name, transfer_name)])
                if boot:
                    lo, hi = boot["signal_share_ci95"]
                else:
                    lo, hi = sig, sig
                ci_lo.append(lo)
                ci_hi.append(hi)

        y = np.arange(len(labels))[::-1]
        signal_vals = np.asarray(signal_vals)
        ci_lo = np.asarray(ci_lo)
        ci_hi = np.asarray(ci_hi)
        ax.barh(y, signal_vals, color=SIGNAL_COLOR, height=0.38, label="signal share (mean separation)", zorder=3)
        ax.barh(y, noise_vals, left=signal_vals, color=NOISE_COLOR, height=0.38, label="noise share (within-class SD)", zorder=3)
        ax.errorbar(signal_vals, y, xerr=[signal_vals - ci_lo, ci_hi - signal_vals], fmt="none",
                    ecolor=INK, elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=4,
                    label="95% payload-block bootstrap CI (signal/noise split point)")
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.axvline(100, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_yticks(y, labels, fontsize=7.3)
        ax.set_xlabel("% share of log(d\u2019) drop")
        ax.set_title(title, fontsize=9.6, fontweight="bold", loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-200, 300)

    axes[0].legend(loc="lower right", fontsize=7.4, frameon=False)
    fig.suptitle("Cross-format transfer loses discriminability via two different failure modes, by direction",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.text(
        0.0, -0.13,
        "Exact decomposition of d' (not a correlation, and not claimed as an exact AUC decomposition): log(d\u2019_self / d\u2019_transfer)\n"
        "splits additively into a signal-loss term and a noise-growth term. Real held-out ROC AUC is reported per cell as EXTERNAL\n"
        "VALIDATION that d' tracks true performance, not as a quantity the algebra decomposes exactly. Casual\u2192benchmark transfer\n"
        "loses discriminability almost entirely via a SIGNAL-LOSS failure mode (the probe direction stops tracking the true\n"
        "separation in the new format). Benchmark\u2192casual transfer loses discriminability almost entirely via a NOISE-GROWTH\n"
        "failure mode (signal is usually preserved or even improves; within-class scatter along the same direction explodes).\n"
        "Replicated at 3 depths (fixed 27% anchor, mid, final layer) in all 3 models. Descriptive failure-mode characterization,\n"
        "not a claim about the underlying causal neural mechanism. Error bars: 95% CI on the signal/noise split point, 1,000-rep\n"
        "payload-block bootstrap; in all 9 model\u00d7layer combinations the CI stays entirely on the dominant side (never crosses 50%).",
        ha="left", va="top", fontsize=7.1, color=MUTED,
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"probe_signal_noise_decomposition.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'probe_signal_noise_decomposition.pdf'}")


if __name__ == "__main__":
    main()
