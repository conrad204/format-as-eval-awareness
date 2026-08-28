"""Generate submission figures only from pinned full320 results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "artifacts" / "pinned_full320"
OUT = ROOT / "paper" / "figures"
MODELS = {
    "Llama-3.1-8B": "layer_sweep_full320-v3_llama31_8b.json",
    "Llama-3.1-70B": "layer_sweep_full320-v3_llama31_70b.json",
    "Llama-3.3-70B": "layer_sweep_full320-v3_llama33_70b.json",
}
COLORS = {"Llama-3.1-8B": "#315f8c", "Llama-3.1-70B": "#b24c32", "Llama-3.3-70B": "#4c8c61"}


def load():
    data = {}
    for model, filename in MODELS.items():
        data[model] = json.loads((PINNED / filename).read_text(encoding="utf-8"))
    return data


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for model, rows in data.items():
        x = [r["layer"] for r in rows]
        ax.plot(x, [r["decorrelated"]["auc"] for r in rows], label=model, color=COLORS[model], lw=2)
    ax.axhline(.5, color="0.6", ls="--", lw=1)
    ax.set(xlabel="Transformer block (zero-based)", ylabel="Held-out purpose AUC", title="Purpose decodability across depth")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "main_purpose_by_depth.png")

    peaks = {model: max(rows, key=lambda r: r["decorrelated"]["auc"]) for model, rows in data.items()}
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    x = np.arange(len(peaks)); width = .34
    auc = [r["decorrelated"]["auc"] for r in peaks.values()]
    ba = [r["decorrelated"]["bal_acc"] for r in peaks.values()]
    ax.bar(x - width / 2, auc, width, label="AUC", color="#315f8c")
    ax.bar(x + width / 2, ba, width, label="Balanced accuracy", color="#d08b43")
    ax.axhline(.5, color="0.6", ls="--", lw=1)
    ax.set_xticks(x, ["8B", "3.1-70B", "3.3-70B"])
    ax.set_ylim(.45, 1.03)
    ax.set(ylabel="Score", title="AUC and threshold accuracy at each model's peak")
    ax.legend(frameon=False)
    save(fig, "main_auc_balanced_accuracy.png")

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    x = np.arange(len(peaks)); width = .34
    b2c = [r["b2c"]["auc"] for r in peaks.values()]
    c2b = [r["c2b"]["auc"] for r in peaks.values()]
    ax.bar(x - width / 2, b2c, width, label="Benchmark → casual", color="#7f9fc4")
    ax.bar(x + width / 2, c2b, width, label="Casual → benchmark", color="#c77d67")
    ax.axhline(.5, color="0.6", ls="--", lw=1)
    ax.set_xticks(x, ["8B", "3.1-70B", "3.3-70B"])
    ax.set_ylim(.45, 1.03)
    ax.set(ylabel="AUC", title="Cross-format transfer at selected observational peaks")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "main_cross_format_transfer.png")

    eq = json.loads((ROOT / "artifacts" / "full320_equal_n_llama31_8b_l12.json").read_text(encoding="utf-8"))
    grouped = {name: [] for name in ("benchmark_only", "casual_only", "equal_n_mixed")}
    for result in eq["results"]:
        for name in grouped:
            grouped[name].append(result[name]["auc"])
    labels = ["Benchmark-only", "Casual-only", "Equal-N mixed"]
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    vals = [grouped[k] for k in grouped]
    ax.boxplot(vals, tick_labels=labels, showmeans=True, meanline=True)
    ax.axhline(.5, color="0.6", ls="--", lw=1)
    ax.set_ylim(.5, 1.01)
    ax.set_ylabel("Held-out purpose AUC")
    ax.set_title("8B equal-N control at layer 12")
    ax.tick_params(axis="x", labelrotation=15)
    save(fig, "main_equal_n_8b.png")

    assay = json.loads((ROOT / "runs" / "tae_2026" / "jlens_v3" / "20260818_full320_8b_retry03" / "outputs" / "assay.json").read_text(encoding="utf-8"))
    layers = [4, 9, 12, 15, 20, 24]
    purpose = [assay["summary"]["purpose"][str(layer)]["oriented_auc"] for layer in layers]
    fmt = [assay["summary"]["format"][str(layer)]["oriented_auc"] for layer in layers]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(layers, purpose, marker="o", label="Purpose", color="#315f8c")
    ax.plot(layers, fmt, marker="o", label="No-cue format", color="#b24c32")
    ax.axhline(.5, color="0.6", ls="--", lw=1)
    ax.set(xticks=layers, ylim=(.45, .72), xlabel="Source layer", ylabel="Oriented AUC", title="8B jLens direct diagnostic")
    ax.legend(frameon=False)
    save(fig, "appendix_jlens_direct_diagnostic.png")

    print(json.dumps({"figures": sorted(p.name for p in OUT.glob("*.png"))}))


if __name__ == "__main__":
    main()
