"""Emit the binding_v1 primary results table and publication figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MODELS = (("llama31_8b", "Llama-3.1-8B", "#2A5FA5"),
          ("llama31_70b", "Llama-3.1-70B", "#C9502F"),
          ("llama33_70b", "Llama-3.3-70B", "#1F7A5C"))
INK = "#202124"
GRID = "#D9DDE3"
FIELDS = ("model_key", "n_layers_model", "layer", "relative_depth",
          "fixed_joint_auc_family_mean", "fixed_joint_auc_ci_lo", "fixed_joint_auc_ci_hi",
          "fixed_families_above_chance", "fixed_interaction_I_family_mean",
          "fixed_interaction_I_families_positive", "fixed_interaction_I_benchmark",
          "fixed_interaction_I_casual", "binding_joint_auc_family_mean",
          "binding_benchmark_to_casual_auc", "binding_casual_to_benchmark_auc",
          "binding_transfer_gap_family_mean", "binding_transfer_gap_ci_lo",
          "binding_transfer_gap_ci_hi", "binding_families_gap_positive",
          "norm_P", "norm_M", "norm_S", "norm_F", "cos_P_explicit_joint_probe",
          "cos_P_benchmark_vs_casual", "cos_P_across_families_mean")
N_LAYERS = {"llama31_8b": 32, "llama31_70b": 80, "llama33_70b": 80}


def rows_for(model_key: str, path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for rec in d["layers"]:
        fx, bb, g = rec["fixed_explicit_to_binding"], rec["binding_to_binding"], rec["factorial_geometry"]
        s = fx["family_unit_summary"]["joint_format_auc"]
        i = fx["interaction_family_unit_summary"]
        sb = bb["family_unit_summary"]
        n_layers = N_LAYERS[model_key]
        out.append({
            "model_key": model_key, "n_layers_model": n_layers, "layer": rec["layer"],
            "relative_depth": round(100.0 * rec["layer"] / (n_layers - 1), 2),
            "fixed_joint_auc_family_mean": round(s["family_mean"], 5),
            "fixed_joint_auc_ci_lo": round(s["family_bootstrap_ci95"][0], 5),
            "fixed_joint_auc_ci_hi": round(s["family_bootstrap_ci95"][1], 5),
            "fixed_families_above_chance": s["families_above_reference"],
            "fixed_interaction_I_family_mean": round(i["mean_interaction_I"]["family_mean"], 5),
            "fixed_interaction_I_families_positive": i["mean_interaction_I"]["families_above_reference"],
            "fixed_interaction_I_benchmark": round(i["mean_interaction_I_benchmark"]["family_mean"], 5),
            "fixed_interaction_I_casual": round(i["mean_interaction_I_casual"]["family_mean"], 5),
            "binding_joint_auc_family_mean": round(sb["joint_format_auc"]["family_mean"], 5),
            "binding_benchmark_to_casual_auc": round(sb["benchmark_to_casual_auc"]["family_mean"], 5),
            "binding_casual_to_benchmark_auc": round(sb["casual_to_benchmark_auc"]["family_mean"], 5),
            "binding_transfer_gap_family_mean": round(sb["transfer_gap"]["family_mean"], 5),
            "binding_transfer_gap_ci_lo": round(sb["transfer_gap"]["family_bootstrap_ci95"][0], 5),
            "binding_transfer_gap_ci_hi": round(sb["transfer_gap"]["family_bootstrap_ci95"][1], 5),
            "binding_families_gap_positive": sb["transfer_gap"]["families_above_reference"],
            "norm_P": round(g["norms"]["P"], 6), "norm_M": round(g["norms"].get("M", float("nan")), 6),
            "norm_S": round(g["norms"].get("S", float("nan")), 6),
            "norm_F": round(g["norms"].get("F", float("nan")), 6),
            "cos_P_explicit_joint_probe": round(g["cos_P_explicit_joint_probe"], 5),
            "cos_P_benchmark_vs_casual": round(g["cos_P_benchmark_vs_casual"], 5),
            "cos_P_across_families_mean": round(g["cos_P_across_families_mean"], 5),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--csv-out", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()
    table: list[dict] = []
    present = {}
    for key, label, colour in MODELS:
        full = args.results_dir / f"binding_v1_{key}_all_layers_analysis.json"
        single = args.results_dir / f"binding_v1_{key}_analysis.json"
        path = full if full.exists() else single
        if not path.exists():
            print(f"skipping {key}: missing {single.name}")
            continue
        print(f"{key}: using {path.name}")
        rows = rows_for(key, path)
        table.extend(rows)
        present[key] = (label, colour, rows)
    if not table:
        raise SystemExit("no binding_v1 analysis files found")
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(table)

    mpl.rcParams.update({"font.family": "sans-serif",
                        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                        "font.size": 9.0, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=True)
    for key, (label, colour, rows) in present.items():
        depth = [r["relative_depth"] for r in rows]
        fixed = [r["fixed_joint_auc_family_mean"] for r in rows]
        lo = [r["fixed_joint_auc_ci_lo"] for r in rows]
        hi = [r["fixed_joint_auc_ci_hi"] for r in rows]
        binding = [r["binding_joint_auc_family_mean"] for r in rows]
        cross = [0.5 * (r["binding_benchmark_to_casual_auc"] + r["binding_casual_to_benchmark_auc"])
                 for r in rows]
        if len(rows) > 1:
            ax_a.plot(depth, fixed, "-", color=colour, lw=1.7, label=label)
            ax_a.fill_between(depth, lo, hi, color=colour, alpha=0.16, lw=0)
            ax_b.plot(depth, binding, "-", color=colour, lw=1.7, label=f"{label} joint")
            ax_b.plot(depth, cross, ":", color=colour, lw=1.7, alpha=0.75,
                      label=f"{label} cross-format")
            ax_c.plot(depth, [r["norm_P"] for r in rows], "-", color=colour, lw=1.7, label="|P|")
            ax_c.plot(depth, [r["norm_F"] for r in rows], "--", color=colour, lw=1.4, alpha=0.7,
                      label="|F| (format)")
            ax_c.plot(depth, [r["norm_M"] for r in rows], ":", color=colour, lw=1.4, alpha=0.7,
                      label="|M|, |S| (main effects)")
            ax_c.plot(depth, [r["norm_S"] for r in rows], ":", color=colour, lw=1.4, alpha=0.7)
        else:
            r = rows[0]
            ax_a.errorbar([r["relative_depth"]], [r["fixed_joint_auc_family_mean"]],
                          yerr=[[r["fixed_joint_auc_family_mean"] - r["fixed_joint_auc_ci_lo"]],
                                [r["fixed_joint_auc_ci_hi"] - r["fixed_joint_auc_family_mean"]]],
                          fmt="o", color=colour, ms=6, capsize=3, label=f"{label} (L18)")
            ax_b.plot([r["relative_depth"]], [r["binding_joint_auc_family_mean"]], "o",
                      color=colour, ms=6, label=f"{label} joint (L18)")
            ax_b.plot([r["relative_depth"]], [0.5 * (r["binding_benchmark_to_casual_auc"]
                                                     + r["binding_casual_to_benchmark_auc"])],
                      "s", mfc="none", mec=colour, ms=6, label=f"{label} cross-format (L18)")
    ax_a.axhline(0.5, color=INK, lw=0.9, ls=(0, (1, 1)), alpha=0.45)
    ax_b.axhline(0.5, color=INK, lw=0.9, ls=(0, (1, 1)), alpha=0.45)
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlabel("relative depth (% of layers)")
        ax.grid(axis="both", color=GRID, lw=0.55, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-2, 102)
    ax_a.set_ylim(0.40, 0.75)
    ax_a.set_ylabel("frozen explicit probe: AUC for bound purpose")
    ax_a.set_title("A. Frozen explicit purpose probe transfers only in the\ndeeper half, and only at 70B scale",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_b.set_ylabel("family-held-out AUC")
    ax_b.set_title("B. Bound purpose IS decodable, and cross-format\ntransfer stays worse",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_c.set_yscale("log")
    ax_c.set_ylabel("contrast norm")
    ax_c.set_title("C. Interaction P grows with depth while\nadditive main effects stay near zero",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_b.legend(loc="lower right", fontsize=6.3, framealpha=0.9, ncol=1)
    ax_c.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("binding_v1: response purpose as a factorial relational binding "
                 f"({len(present)}/3 checkpoints, inference unit = relational family, n=8)",
                 fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(args.fig_dir / f"binding_v1.{suffix}", **kwargs)
    plt.close(fig)
    print(json.dumps({"csv": str(args.csv_out), "figure": str(args.fig_dir / "binding_v1.png"),
                      "models": sorted(present), "rows": len(table)}, indent=2))


if __name__ == "__main__":
    main()
