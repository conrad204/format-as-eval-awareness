"""Assemble the binding_v2 + provenance figure and machine-readable summary.

Reads only committed result JSON. Performs no fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")
LABELS = {"llama31_8b": "Llama-3.1-8B", "llama31_70b": "Llama-3.1-70B",
          "llama33_70b": "Llama-3.3-70B"}
COLORS = {"llama31_8b": "#1f77b4", "llama31_70b": "#d62728", "llama33_70b": "#2ca02c"}


def binding_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        data = json.loads((RESULTS / f"binding_v2_{model}_analysis.json").read_text())
        for rec in data["layers"]:
            for section, keys in (
                ("binding_v2_within_condition",
                 [f"within:{a}:h{h}" for a in ("consequence", "neutral") for h in (0, 1, 2)]),
                ("hop_transfer",
                 [f"hop:{a}:0to{h}" for a in ("consequence", "neutral") for h in (1, 2)]),
            ):
                for key in keys:
                    s = rec[section][key]["family_unit_summary"]["joint_format_auc"]
                    rows.append({
                        "model": model, "layer": rec["layer"], "section": section, "split": key,
                        "family_mean_auc": s["family_mean"],
                        "ci95_lo": s["family_bootstrap_ci95"][0],
                        "ci95_hi": s["family_bootstrap_ci95"][1],
                        "families_above_chance": s["families_above_reference"],
                        "n_families": s["n_families"],
                        "sign_test_p": s["exact_family_sign_test_p_one_sided"],
                    })
            fixed = rec.get("fixed_explicit_to_binding_v2")
            if fixed:
                for key, value in fixed["by_arm_hop"].items():
                    s = value["family_unit_summary"]["joint_format_auc"]
                    rows.append({
                        "model": model, "layer": rec["layer"],
                        "section": "frozen_stated_purpose_probe", "split": key,
                        "family_mean_auc": s["family_mean"],
                        "ci95_lo": s["family_bootstrap_ci95"][0],
                        "ci95_hi": s["family_bootstrap_ci95"][1],
                        "families_above_chance": s["families_above_reference"],
                        "n_families": s["n_families"],
                        "sign_test_p": s["exact_family_sign_test_p_one_sided"],
                    })
    return rows


def subspace_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        path = RESULTS / f"binding_v2_{model}_template_subspace.json"
        data = json.loads(path.read_text())
        for rec in data["layers"]:
            block = rec["by_rank"]["64"]
            raw = block["raw_template_summary"]
            proj = block["projected_template_summary"]
            fmt = block["format_auc_template_summary"]
            g_raw = raw["transfer_gap"]["template_mean"]
            g_proj = proj["transfer_gap"]["template_mean"]
            rows.append({
                "model": model, "layer": rec["layer"], "rank": 64,
                "n_held_templates": len(block["per_template"]),
                "raw_joint_auc": raw["joint_format_auc"]["template_mean"],
                "raw_cross_auc": raw["cross_format_mean_auc"]["template_mean"],
                "raw_transfer_gap": g_raw,
                "projected_joint_auc": proj["joint_format_auc"]["template_mean"],
                "projected_cross_auc": proj["cross_format_mean_auc"]["template_mean"],
                "projected_transfer_gap": g_proj,
                "fraction_gap_removed": (g_raw - g_proj) / g_raw,
                "raw_format_auc": fmt["raw_format_auc"]["template_mean"],
                "projected_format_auc": fmt["projected_format_auc"]["template_mean"],
                "templates_with_gap_reduced": sum(
                    1 for v in block["per_template"].values()
                    if v["projected_minus_raw"]["delta_transfer_gap"] < 0
                ),
            })
    return rows


def provenance_rows() -> list[dict]:
    rows = []
    for bank in ("provenance_v1", "provenance_v2"):
        for model in MODELS:
            path = RESULTS / f"{bank}_{model}_frozen_probe.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for rec in data["layers"]:
                r = rec["regimes"]["joint"]
                eb, ec = r["mean_score_eval_benchmark"], r["mean_score_eval_casual"]
                db, dc = r["mean_score_deploy_benchmark"], r["mean_score_deploy_casual"]
                rows.append({
                    "bank": bank, "model": model, "layer": rec["layer"],
                    "normalized_depth": rec["normalized_depth"],
                    "pooled_auc": r["pooled_auc"],
                    "pooled_auc_ci_lo": r["block_bootstrap_ci95"]["pooled_auc"][0],
                    "pooled_auc_ci_hi": r["block_bootstrap_ci95"]["pooled_auc"][1],
                    "benchmark_auc": r["benchmark_auc"],
                    "casual_auc": r["casual_auc"],
                    "source_pair_auc_mean": r["source_pair_auc_mean"],
                    "source_pair_auc_min": r["source_pair_auc_min"],
                    "format_effect_score": ((eb - ec) + (db - dc)) / 2,
                    "provenance_effect_score": ((eb - db) + (ec - dc)) / 2,
                    "provenance_x_format_interaction": r["provenance_x_format_score_interaction"],
                    "interaction_ci_lo": r["block_bootstrap_ci95"]["provenance_x_format_score_interaction"][0],
                    "interaction_ci_hi": r["block_bootstrap_ci95"]["provenance_x_format_score_interaction"][1],
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.write_text("", encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(binding: list[dict], subspace: list[dict], provenance: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    ax = axes[0]
    conditions = [f"h{h}" for h in (0, 1, 2)]
    width = 0.26
    for k, model in enumerate(MODELS):
        best = {}
        for arm in ("consequence", "neutral"):
            vals = []
            for h in (0, 1, 2):
                cand = [r for r in binding if r["model"] == model
                        and r["split"] == f"within:{arm}:h{h}"]
                layer = max(
                    [r for r in binding if r["model"] == model
                     and r["split"] == "within:consequence:h0"],
                    key=lambda r: r["family_mean_auc"],
                )["layer"]
                vals.append(next(r["family_mean_auc"] for r in cand if r["layer"] == layer))
            best[arm] = vals
        x = np.arange(3) + (k - 1) * width
        ax.bar(x, best["consequence"], width * 0.45, color=COLORS[model],
               label=f"{LABELS[model]} consequence")
        ax.bar(x + width * 0.45, best["neutral"], width * 0.45, color=COLORS[model],
               alpha=0.4, hatch="//", label=f"{LABELS[model]} neutral twin")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(conditions)
    ax.set_xlabel("indirection depth")
    ax.set_ylabel("family-mean AUC")
    ax.set_title("Semantics-ablated twin control\n(family-held-out probes)")
    ax.legend(fontsize=6, ncol=1)

    ax = axes[1]
    for model in MODELS:
        rows = sorted([r for r in provenance if r["bank"] == "provenance_v2"
                       and r["model"] == model], key=lambda r: r["layer"])
        ax.plot([r["normalized_depth"] for r in rows], [r["pooled_auc"] for r in rows],
                color=COLORS[model], lw=1.4, label=LABELS[model])
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.axhline(0.9615, color="grey", lw=1.0, ls="--")
    ax.text(0.02, 0.975, "surface-text ceiling 0.96", fontsize=6, color="grey")
    ax.set_xlabel("normalized depth")
    ax.set_ylabel("provenance AUC")
    ax.set_title("Frozen stated-purpose probe on\nreal corpus provenance (no retraining)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for model in MODELS:
        rows = sorted([r for r in provenance if r["bank"] == "provenance_v2"
                       and r["model"] == model], key=lambda r: r["layer"])
        ax.plot([r["normalized_depth"] for r in rows],
                [abs(r["format_effect_score"]) / max(abs(r["provenance_effect_score"]), 1e-9)
                 for r in rows], color=COLORS[model], lw=1.4, label=LABELS[model])
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("normalized depth")
    ax.set_ylabel("|format effect| / |provenance effect|")
    ax.set_title("Scaffold vs real provenance on the\nsame eval-ness score scale")
    ax.legend(fontsize=7)

    ax = axes[3]
    labels, raw_vals, proj_vals = [], [], []
    for r in subspace:
        labels.append(f"{LABELS[r['model']]}\nL{r['layer']}")
        raw_vals.append(r["raw_transfer_gap"])
        proj_vals.append(r["projected_transfer_gap"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, raw_vals, 0.4, color="#555555", label="raw")
    ax.bar(x + 0.2, proj_vals, 0.4, color="#ff7f0e", label="rank-64 projected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("cross-format transfer gap")
    ax.set_title("Template-HELD-OUT format subspace\n(basis excludes held template)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="binding_v2_provenance")
    args = ap.parse_args()
    binding, subspace, provenance = binding_rows(), subspace_rows(), provenance_rows()
    write_csv(RESULTS / f"{args.out_prefix}_binding.csv", binding)
    write_csv(RESULTS / f"{args.out_prefix}_subspace.csv", subspace)
    write_csv(RESULTS / f"{args.out_prefix}_provenance.csv", provenance)
    summary = {
        "schema_version": 1,
        "binding_v2_rows": len(binding),
        "subspace_rows": len(subspace),
        "provenance_rows": len(provenance),
        "template_subspace": subspace,
    }
    (RESULTS / f"{args.out_prefix}_summary.json").write_bytes(
        (json.dumps(summary, indent=2) + "\n").encode("utf-8")
    )
    figure(binding, subspace, provenance,
           ROOT / "results" / f"{args.out_prefix}_figure.png")
    print(f"wrote {len(binding)} binding, {len(subspace)} subspace, {len(provenance)} provenance rows")


if __name__ == "__main__":
    main()
