"""Full-layer interchange-patching (real forward-pass causal test), all 3
checkpoints, real_swap vs. magnitude-matched random control.

Same method as plots/jpp/interchange_patch_llama31_8b (see that caption for
the full methodology). This figure is the cross-model version: one line per
model per condition, x-axis in relative depth (%) so the 32-layer 8B and
80-layer 70Bs are directly comparable, matching the convention in
plots/jpp/full_layer_geometry_trajectory.

Only the magnitude-matched comparison (real_swap vs matched_control) is
plotted here -- that is the fair, magnitude-controlled test. The unmatched
random_control is deliberately omitted from this figure (it understates both
effects, as shown in the single-model 8B figure) to keep the cross-model
comparison uncluttered; it remains available in the underlying per-model
figure and raw records for anyone who wants it.

Gracefully skips any model whose matched_control sweep hasn't landed yet
(prints a warning, plots only the models with data present) so this script
can be run as soon as the first model finishes rather than waiting on all 3.

Self-contained. Reads only:
  results/tae_ixpatch_full_<model>.jsonl      (real_swap)
  results/tae_ixpatch_matched_<model>.jsonl   (matched_control)
No activation extraction, probe refitting, or model inference in this script.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
N_LAYERS = {"llama31_8b": 32, "llama31_70b": 80, "llama33_70b": 80}
MODEL_LABEL = {"llama31_8b": "Llama-3.1-8B", "llama31_70b": "Llama-3.1-70B", "llama33_70b": "Llama-3.3-70B"}
MODEL_COLOR = {"llama31_8b": "#2A5FA5", "llama31_70b": "#C9502F", "llama33_70b": "#1F7A5C"}
COND_STYLE = {"real_swap": "-", "matched_control": ":"}
INK = "#202124"
GRID = "#D9DDE3"


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def per_layer_stats(records: list[dict]) -> dict:
    by_layer_cond = defaultdict(list)
    for r in records:
        target_gap = r["format_score_baseline_tgt"] - r["format_score_baseline_src"]
        actual_shift = r["format_score_patched"] - r["format_score_baseline_src"]
        iia = float(np.sign(actual_shift) == np.sign(target_gap)) if target_gap != 0 else np.nan
        purpose_shift = abs(r["purpose_score_patched"] - r["purpose_score_baseline"])
        by_layer_cond[(r["layer"], r["condition"])].append((iia, purpose_shift))
    out = {}
    for (layer, cond), vals in by_layer_cond.items():
        iia_vals = np.array([v[0] for v in vals])
        iia_vals = iia_vals[~np.isnan(iia_vals)]
        pshift_vals = np.array([v[1] for v in vals])
        out[(layer, cond)] = {
            "format_iia": float(np.mean(iia_vals)) if len(iia_vals) else float("nan"),
            "mean_abs_purpose_shift": float(np.mean(pshift_vals)),
        }
    return out


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

    available = {}
    for model, n_layers in N_LAYERS.items():
        real_path = ROOT / "results" / f"tae_ixpatch_full_{model}.jsonl"
        matched_path = ROOT / "results" / f"tae_ixpatch_matched_{model}.jsonl"
        if not (real_path.exists() and matched_path.exists()):
            print(f"skipping {model}: missing {real_path.name if not real_path.exists() else matched_path.name}",
                  file=sys.stderr)
            continue
        stats = per_layer_stats(load_records(real_path))
        stats.update(per_layer_stats(load_records(matched_path)))
        available[model] = (n_layers, stats)

    if not available:
        raise SystemExit("no models with both real_swap and matched_control data present")

    fig, (ax_iia, ax_purpose) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    for model, (n_layers, stats) in available.items():
        color = MODEL_COLOR[model]
        for cond in ("real_swap", "matched_control"):
            layers = sorted(L for L, c in stats if c == cond)
            rel_depth_pct = [100.0 * L / (n_layers - 1) for L in layers]
            iia = [stats[(L, cond)]["format_iia"] for L in layers]
            pshift = [stats[(L, cond)]["mean_abs_purpose_shift"] for L in layers]
            label = f"{MODEL_LABEL[model]} ({cond})"
            ax_iia.plot(rel_depth_pct, iia, COND_STYLE[cond], color=color, linewidth=1.6,
                        alpha=0.95 if cond == "real_swap" else 0.6, label=label)
            ax_purpose.plot(rel_depth_pct, pshift, COND_STYLE[cond], color=color, linewidth=1.6,
                             alpha=0.95 if cond == "real_swap" else 0.6, label=label)

    ax_iia.axhline(0.5, color=INK, linewidth=0.9, linestyle=(0, (1, 1)), alpha=0.4)

    for ax in (ax_iia, ax_purpose):
        ax.set_xlabel("relative depth (% of layers, 0 = first, 100 = last)")
        ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-2, 102)

    ax_iia.set_ylim(0.4, 1.02)
    ax_iia.set_ylabel("format interchange-intervention accuracy")
    ax_purpose.set_ylabel("mean |purpose-readout shift| (patched - baseline)")
    ax_iia.set_title("Format readout follows real_swap, not matched_control", loc="left", fontsize=10, fontweight="bold")
    ax_purpose.set_title("Purpose-readout disruption: real_swap vs. matched_control", loc="left", fontsize=10, fontweight="bold")
    ax_iia.legend(loc="lower right", fontsize=6.5, framealpha=0.9, ncol=1)

    n_models = len(available)
    fig.suptitle(
        f"Interchange-patching causal test, rank 16, magnitude-matched control "
        f"({n_models}/3 checkpoints)",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"interchange_patch_all_models.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'interchange_patch_all_models.pdf'} ({n_models}/3 models)")


if __name__ == "__main__":
    main()
