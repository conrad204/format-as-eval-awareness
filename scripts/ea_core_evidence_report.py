"""Create the local-only core EA evidence table and figure."""
from __future__ import annotations

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
FIXED = {"llama31_8b": 8, "llama31_70b": 18, "llama33_70b": 18}


def main() -> None:
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for model in MODELS:
        d = json.loads((RESULTS / f"layer_sweep_full320-v3_{model}.json").read_text())
        q = np.array([r["layer"] / (len(d) - 1) for r in d])
        joint = np.array([r["decorrelated"]["auc"] for r in d])
        transfer = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in d])
        ax.plot(q, joint, color=COLORS[model], lw=2, label=LABELS[model])
        ax.plot(q, transfer, color=COLORS[model], lw=1.2, ls="--")
        best = max(d, key=lambda r: r["decorrelated"]["auc"])
        rows.append({"model": model, "controlled_layer": best["layer"],
                     "controlled_joint_auc": best["decorrelated"]["auc"],
                     "controlled_b2c_auc": best["b2c"]["auc"],
                     "controlled_c2b_auc": best["c2b"]["auc"]})
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("normalized depth")
    ax.set_ylabel("held-family-out ROC-AUC")
    ax.set_title("Controlled factorial identification\nsolid: joint; dashed: cross-format mean")
    ax.legend(fontsize=7, loc="lower right")

    directions = ("purpose", "purpose_orthogonal_to_format", "format", "nocue_format")
    short = ("purpose", "purpose minus format", "format", "no-cue format")
    x = np.arange(len(directions))
    width = 0.24
    ax = axes[1]
    for mi, model in enumerate(MODELS):
        d = json.loads((RESULTS / f"ea_provenance_specificity_{model}.json").read_text())
        by = {r["name"]: r for r in d["directions"]}
        vals = [by[k]["source_pair_auc_min"] for k in directions]
        cis = [by[k]["block_bootstrap_ci95"]["source_pair_auc_min"] for k in directions]
        err = np.array([[v-c[0] for v,c in zip(vals,cis)], [c[1]-v for v,c in zip(vals,cis)]])
        ax.bar(x + (mi-1)*width, vals, width, color=COLORS[model], label=LABELS[model],
               yerr=err, capsize=2)
        rows[mi].update({f"external_{k}_min_pair_auc": by[k]["source_pair_auc_min"]
                         for k in directions})
        rows[mi]["external_purpose_null_p"] = d["null_summary"]["purpose_min_pair_empirical_p_greater_equal"]
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("minimum of four source-pair AUCs")
    ax.set_title("External provenance at controlled layer\npurpose fails; format transfers")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    for mi, model in enumerate(MODELS):
        d = json.loads((RESULTS / f"ea_provenance_specificity_{model}.json").read_text())
        null = [r["source_pair_auc_min"] for r in d["directions"]
                if r["name"].startswith("purpose_block_signflip_")]
        purpose = next(r["source_pair_auc_min"] for r in d["directions"] if r["name"] == "purpose")
        parts = ax.violinplot(null, positions=[mi], widths=.7, showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[model]); body.set_alpha(.35)
        ax.scatter([mi], [purpose], color=COLORS[model], edgecolor="black", zorder=3, s=45,
                   marker="D")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("minimum source-pair AUC")
    ax.set_title("100 block-sign control directions\ndiamond: actual purpose direction")

    fig.suptitle("Representational EA is strong under factorial control; real-provenance transfer is nonspecific",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(RESULTS / "ea_core_evidence_figure.png", dpi=180)

    with (RESULTS / "ea_core_evidence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"figure": str(RESULTS / "ea_core_evidence_figure.png"),
                      "csv": str(RESULTS / "ea_core_evidence_summary.csv"), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
