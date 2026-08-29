"""Full-layer interchange-patching (real forward-pass causal test), llama31_8b.

Swaps ONLY the projection onto Q_format(16) between matched real benchmark/
casual counterparts (real content, not a synthetic steering vector) at every
layer, via register_forward_hook on real bf16 forward passes, and reads out
purpose/format probes at the final layer. Three conditions:

  real_swap        -- swap through the identified Q_format(16) subspace.
  random_control    -- same swap operation, random rank-16 subspace (same
                        real counterpart content, UNMATCHED perturbation size:
                        a random rank-16 projection captures only ~16/4096 =
                        0.4% of a vector's energy on average, so this
                        understates both effects below).
  matched_control    -- fresh random ADDITIVE direction, magnitude-matched
                        per-row to that row's own real_swap delta norm
                        (computed offline from static activations). Isolates
                        whether real_swap's effects are specific to
                        Q_format(16) or just a function of perturbation size.

Left panel: format interchange-intervention accuracy (does the format readout
move, in sign, toward the counterpart's own baseline format score).
Right panel: mean |purpose-readout shift| (patched vs. unpatched baseline).

Descriptive full-layer trajectory, not a per-layer significance test (layers
are repeated measurements of one depth profile, not independent replicates --
see plots/jpp/full_layer_geometry_trajectory/important_note.md for the same
convention applied to the geometry sweep).

Self-contained. Reads only:
  results/tae_ixpatch_full_llama31_8b.jsonl      (real_swap, random_control)
  results/tae_ixpatch_matched_llama31_8b.jsonl   (matched_control)
No activation extraction, probe refitting, or model inference in this script.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
REAL_PATH = ROOT / "results" / "tae_ixpatch_full_llama31_8b.jsonl"
MATCHED_PATH = ROOT / "results" / "tae_ixpatch_matched_llama31_8b.jsonl"
N_LAYERS = 32

COND_LABEL = {"real_swap": "real_swap (Q_format)", "random_control": "random_control (unmatched)",
              "matched_control": "matched_control (norm-matched random)"}
COND_COLOR = {"real_swap": "#C9502F", "random_control": "#8A8F98", "matched_control": "#2A5FA5"}
COND_STYLE = {"real_swap": "-", "random_control": "--", "matched_control": ":"}
INK = "#202124"
GRID = "#D9DDE3"


def load_records(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def per_layer_stats(records):
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
    stats = per_layer_stats(load_records(REAL_PATH))
    stats.update(per_layer_stats(load_records(MATCHED_PATH)))

    layers = list(range(N_LAYERS))
    rel_depth_pct = [100.0 * L / (N_LAYERS - 1) for L in layers]

    fig, (ax_iia, ax_purpose) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    for cond in ("real_swap", "random_control", "matched_control"):
        iia = [stats[(L, cond)]["format_iia"] for L in layers]
        pshift = [stats[(L, cond)]["mean_abs_purpose_shift"] for L in layers]
        color, style, label = COND_COLOR[cond], COND_STYLE[cond], COND_LABEL[cond]
        ax_iia.plot(rel_depth_pct, iia, style, color=color, linewidth=1.6, label=label)
        ax_purpose.plot(rel_depth_pct, pshift, style, color=color, linewidth=1.6, label=label)

    ax_iia.axhline(0.5, color=INK, linewidth=0.9, linestyle=(0, (1, 1)), alpha=0.4)
    ax_iia.text(101.5, 0.5, "chance", fontsize=7, color=INK, alpha=0.6, va="center")

    for ax in (ax_iia, ax_purpose):
        ax.set_xlabel("relative depth (% of layers, 0 = first, 100 = last)")
        ax.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-2, 102)

    ax_iia.set_ylim(0.4, 1.02)
    ax_iia.set_ylabel("format interchange-intervention accuracy")
    ax_purpose.set_ylabel("mean |purpose-readout shift| (patched - baseline)")
    ax_iia.set_title("Does the format readout follow the swap?", loc="left", fontsize=10, fontweight="bold")
    ax_purpose.set_title("How much does the purpose readout move?", loc="left", fontsize=10, fontweight="bold")
    ax_iia.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "Interchange-patching causal test, Llama-3.1-8B, rank 16 (real forward passes)",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"interchange_patch_llama31_8b.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'interchange_patch_llama31_8b.pdf'}")


if __name__ == "__main__":
    main()
