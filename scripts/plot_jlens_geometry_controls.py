"""Plot development J-lens energy trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    d = json.loads(args.report.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for metric, label, name in (("top_auc", "top-J purpose AUC", "purpose_auc_by_energy"), ("low_auc", "lower-gain complement purpose AUC", "complement_auc_by_energy")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for layer in d["layers"]:
            xs = sorted(float(x) for x in layer["thresholds"])
            ys = []
            for x in xs:
                vals = [r["thresholds"][str(x)][metric] for r in layer["folds"].values()]
                ys.append(sum(vals) / len(vals))
            ax.plot(xs, ys, marker="o", label=f"L{layer['layer']}")
        ax.set_xlabel("Jacobian Frobenius energy retained")
        ax.set_ylabel(label)
        ax.set_ylim(0.45, 1.0)
        ax.legend(ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{name}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for threshold in (0.8, 0.9, 0.95, 0.99):
        xs, ys = [], []
        for layer in d["layers"]:
            xs.append(layer["layer"])
            vals = [r["thresholds"][str(threshold)]["top_norm_fraction"] for r in layer["folds"].values()]
            ys.append(sum(vals) / len(vals))
        ax.plot(xs, ys, marker="o", label=f"{threshold:.3g}")
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Purpose norm fraction in dominant-J subspace")
    ax.set_ylim(0, 1)
    ax.legend(title="Energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "purpose_norm_fraction_by_layer.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
