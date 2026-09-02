"""Summarize frozen explicit-purpose probes on the implicit-v2 bank."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = (
    ("Llama-3.1-8B", RESULTS / "implicit_v2_8b_analysis.json", 8),
    ("Llama-3.1-70B", RESULTS / "implicit_v2_llama31_70b_l18_analysis.json", 18),
    ("Llama-3.3-70B", RESULTS / "implicit_v2_llama33_70b_l18_analysis.json", 18),
)
METRICS = ("joint_format_auc", "benchmark_to_casual_auc", "casual_to_benchmark_auc")


def summary(path: Path, layer: int) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    fixed = result["fixed_explicit_to_implicit"]
    if "layers" in fixed:
        record = next(row for row in fixed["layers"] if row["layer"] == layer)
        return record["family_unit_summary"]
    if result["pre_specified_layer"] != layer:
        raise RuntimeError("unexpected pre-specified layer")
    return fixed["family_unit_summary"]


def main() -> None:
    rows = []
    for model, path, layer in MODELS:
        values = summary(path, layer)
        row = {"model": model, "layer": layer}
        for metric in METRICS:
            item = values[metric]
            row[metric] = item["mean"]
            row[f"{metric}_ci_low"] = item["family_bootstrap_ci95"][0]
            row[f"{metric}_ci_high"] = item["family_bootstrap_ci95"][1]
        rows.append(row)

    csv_path = RESULTS / "implicit_ea_core_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = ("joint", "benchmark→casual", "casual→benchmark")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True)
    for ax, row in zip(axes, rows):
        means = [row[m] for m in METRICS]
        low = [row[m] - row[f"{m}_ci_low"] for m in METRICS]
        high = [row[f"{m}_ci_high"] - row[m] for m in METRICS]
        ax.errorbar(range(3), means, yerr=[low, high], fmt="o", capsize=3, color="#2457a6")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(3), labels, rotation=28, ha="right")
        ax.set_title(f"{row['model']} · L{row['layer']}")
        ax.set_ylim(0.45, 0.75)
    axes[0].set_ylabel("held-family mean ROC AUC")
    fig.suptitle("Frozen explicit-purpose probes on implicit evaluative situations", fontsize=11)
    fig.tight_layout()
    figure_path = RESULTS / "implicit_ea_core_figure.png"
    fig.savefig(figure_path, dpi=180)
    print(json.dumps({"csv": str(csv_path), "figure": str(figure_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
