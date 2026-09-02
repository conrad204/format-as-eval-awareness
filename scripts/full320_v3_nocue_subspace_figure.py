"""Two-panel figure for the no-cue-derived format-associated subspace result.

Left: per-layer cross-format transfer gap over normalized depth, raw vs
no-cue-projected (rank 64). Right: fraction of the integrated gap removed as a
function of rank, no-cue-derived vs stated-purpose-derived basis, with paired
payload-block bootstrap CIs.

Usage: python scripts/full320_v3_nocue_subspace_figure.py --model llama31_8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RANKS = (1, 4, 16, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args()

    main_json = json.loads((RESULTS / f"nocue_format_subspace_{args.model}_r{args.rank}.json").read_text())
    layers = main_json["layers"]
    q = [x["normalized_depth"] for x in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax1.plot(q, [x["raw_gap"] for x in layers], label="Raw", lw=2, color="#33415c")
    ax1.plot(q, [x["projected_gap"] for x in layers], label=f"No-cue projected (r={args.rank})",
             lw=2, color="#c1440e")
    ax1.axhline(0, color="black", lw=.7)
    ax1.set_xlabel("Normalized depth"); ax1.set_ylabel("Cross-format transfer gap $G_l$ (AUC)")
    ax1.set_title(f"{args.model}: gap over depth"); ax1.legend(frameon=False)

    for label, k, color in (("Stated-purpose-derived $Q_B$", "stated_qb", "#33415c"),
                            ("No-cue-derived $Q_{nocue}$", "nocue", "#c1440e")):
        xs, ys, lo, hi = [], [], [], []
        for r in RANKS:
            c = json.loads((RESULTS / f"nocue_vs_stated_subspace_compare_{args.model}_r{r}.json").read_text())
            key = "stated_qb_fraction_removed" if k == "stated_qb" else "nocue_fraction_removed"
            xs.append(r); ys.append(c[key])
            lo.append(c[key] - c[key + "_ci95"][0]); hi.append(c[key + "_ci95"][1] - c[key])
        ax2.errorbar(xs, ys, yerr=[lo, hi], marker="o", lw=2, capsize=3, label=label, color=color)
    ax2.axhline(0, color="black", lw=.7)
    ax2.set_xscale("log", base=2); ax2.set_xticks(RANKS); ax2.set_xticklabels([str(r) for r in RANKS])
    ax2.set_xlabel("Projection rank $r$"); ax2.set_ylabel("Fraction of integrated gap removed")
    ax2.set_title("Gap removed vs rank"); ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out = RESULTS / f"nocue_format_subspace_figure_{args.model}.png"
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
