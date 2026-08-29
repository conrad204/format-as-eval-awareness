"""Matched-format subspace geometry: purpose-direction energy inside the matched-format
PCA subspace, vs. the exact high-dimensional chance level, across all 3 checkpoints x
3 depths x rank 1/4/16/64 -- plus the multiple-testing-corrected significance summary
for both the full-space isotropic null and the high-variance-subspace conditional
null.

Self-contained. Reads only:
  results/full320_v3_direction_geometry.csv         (observed E_purpose_in_format(r))
  results/full320_v3_direction_geometry_significance.json   (full-space Beta-null test)
  results/full320_v3_conditional_hv_null.json               (conditional HV-subspace null)
  results/full320_v3_conditional_null_multiplicity.json     (Holm/Bonferroni correction)
No activation extraction or probe refitting here.

Message: purpose energy inside the matched-format subspace is far below the exact
high-dimensional chance level r/d at every model, depth, and rank tested (left panel).
This survives Holm and Bonferroni correction across the full 27-test family at ranks
4 and 16 (all 9/9 cells each), but does NOT at rank 1 (underpowered at that tiny
magnitude) -- shown honestly in the right panel, not smoothed over.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "full320_v3_direction_geometry.csv"
SIG_PATH = ROOT / "results" / "full320_v3_direction_geometry_significance.json"
COND_PATH = ROOT / "results" / "full320_v3_conditional_hv_null.json"
MULT_PATH = ROOT / "results" / "full320_v3_conditional_null_multiplicity.json"

DIMS = {"llama31_8b": 4096, "llama31_70b": 8192, "llama33_70b": 8192}
MODEL_ORDER = ["llama31_8b", "llama31_70b", "llama33_70b"]
MODEL_LABEL = {"llama31_8b": "Llama-3.1-8B", "llama31_70b": "Llama-3.1-70B", "llama33_70b": "Llama-3.3-70B"}
MODEL_COLOR = {"llama31_8b": "#2A5FA5", "llama31_70b": "#C9502F", "llama33_70b": "#1F7A5C"}
DEPTH_MARKER = {"spike": "o", "min_gap": "s", "final": "^"}
RANKS = [1, 4, 16, 64]
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
    rows = list(csv.DictReader(open(CSV_PATH)))
    sig = json.loads(SIG_PATH.read_text())
    cond = json.loads(COND_PATH.read_text())
    mult = json.loads(MULT_PATH.read_text())

    fig, (ax_geom, ax_sig) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    # --- left: observed energy vs full-space chance null, per model/depth ---
    for model in MODEL_ORDER:
        d = DIMS[model]
        for depth in ("spike", "min_gap", "final"):
            cell_rows = [r for r in rows if r["model"] == model and r["depth"] == depth]
            cell_rows.sort(key=lambda r: int(r["rank"]))
            xs = [int(r["rank"]) for r in cell_rows]
            ys = [float(r["E_purpose_in_format_subspace"]) for r in cell_rows]
            ax_geom.plot(xs, ys, marker=DEPTH_MARKER[depth], color=MODEL_COLOR[model],
                         linewidth=1.1, markersize=5, alpha=0.9)
        chance = [r / d for r in RANKS]
        ax_geom.plot(RANKS, chance, "--", color=MODEL_COLOR[model], linewidth=1.0, alpha=0.5)

    ax_geom.set_xscale("log", base=2)
    ax_geom.set_xticks(RANKS)
    ax_geom.set_xticklabels(RANKS)
    ax_geom.set_xlabel("matched-format PCA rank r")
    ax_geom.set_ylabel(r"$E_{purpose \to format}(r)$")
    ax_geom.set_title("Observed (solid) vs. exact chance r/d (dashed)", loc="left", fontsize=10, fontweight="bold")
    ax_geom.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_geom.set_axisbelow(True)

    model_handles = [plt.Line2D([0], [0], color=MODEL_COLOR[m], lw=2, label=MODEL_LABEL[m]) for m in MODEL_ORDER]
    depth_handles = [plt.Line2D([0], [0], color="gray", marker=mk, linestyle="", label=dep)
                      for dep, mk in DEPTH_MARKER.items()]
    ax_geom.legend(handles=model_handles + depth_handles, loc="upper left", fontsize=7, ncol=2)

    # --- right: significance counts (of 9 cells) per rank, both null models, both corrections ---
    full_space_by_rank = sig["summary_by_rank"]
    cond_cells = cond["cells"]
    cond_by_rank = defaultdict(list)
    for c in cond_cells:
        cond_by_rank[c["rank"]].append(c)
    holm_bonf_by_cell = {(c["model"], c["depth"], c["rank"]): c for c in
                          json.loads(COND_PATH.read_text())["cells"]}  # noqa: F841 (kept for clarity)

    ranks_for_cond = [1, 4, 16]
    bonf_flags_emp = None  # recomputed below purely for per-rank breakdown display
    from statsmodels.stats.multitest import multipletests
    p_emp_all = [c["p_below_conditional_null_empirical"] for c in cond_cells]
    rej_bonf, _, _, _ = multipletests(p_emp_all, alpha=0.05, method="bonferroni")
    rej_holm, _, _, _ = multipletests(p_emp_all, alpha=0.05, method="holm")

    bar_w = 0.13
    group_x = {1: 0, 4: 1, 16: 2, 64: 3}
    series = [
        ("full-space\nnominal", lambda r: full_space_by_rank[str(r)]["n_significant_0.05"] if r in RANKS else None, "#9AA0A6"),
        ("full-space\nBonferroni", lambda r: full_space_by_rank[str(r)]["n_significant_bonferroni"] if r in RANKS else None, "#5F6368"),
        ("cond-HV\nnominal", lambda r: sum(1 for c in cond_by_rank[r] if c["p_below_conditional_null_empirical"] < 0.05) if r in ranks_for_cond else None, "#8AB4F8"),
        ("cond-HV\nHolm", lambda r: sum(1 for i, c in enumerate(cond_cells) if c["rank"] == r and rej_holm[i]) if r in ranks_for_cond else None, "#1A73E8"),
        ("cond-HV\nBonferroni", lambda r: sum(1 for i, c in enumerate(cond_cells) if c["rank"] == r and rej_bonf[i]) if r in ranks_for_cond else None, "#174EA6"),
    ]
    for i, (name, fn, color) in enumerate(series):
        xs, ys = [], []
        for r in RANKS:
            v = fn(r)
            if v is None:
                continue
            xs.append(group_x[r] + (i - 2) * bar_w)
            ys.append(v)
        ax_sig.bar(xs, ys, width=bar_w, color=color, label=name)

    ax_sig.set_xticks(list(group_x.values()))
    ax_sig.set_xticklabels([f"r={r}" for r in RANKS])
    ax_sig.set_ylabel("cells significant / 9")
    ax_sig.set_ylim(0, 9.8)
    ax_sig.set_title("Significant cells, with correction (of 9 per rank)", loc="left", fontsize=10, fontweight="bold")
    ax_sig.legend(loc="lower left", fontsize=6.5, ncol=2, framealpha=0.9)
    ax_sig.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax_sig.set_axisbelow(True)

    fig.suptitle(
        "Purpose energy in the matched-format subspace: below chance, with a rank-1 blind spot",
        fontsize=12, fontweight="bold", x=0.02, ha="left",
    )

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_DIR / f"matched_format_subspace_geometry.{suffix}", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'matched_format_subspace_geometry.pdf'}")


if __name__ == "__main__":
    main()
