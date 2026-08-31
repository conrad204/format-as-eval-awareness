"""Figure and CSV for provenance_v3 (two independent deployment-log corpora)."""
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
CEILING = 0.9586  # word 1-2 gram grouped five-fold, from the v3 audit


def rows() -> tuple[list[dict], list[dict]]:
    prov, contrast = [], []
    for model in MODELS:
        d = json.loads((RESULTS / f"provenance_v3_{model}_frozen_probe.json").read_text())
        for rec in d["layers"]:
            r = rec["regimes"]["joint"]
            prov.append({
                "model": model, "layer": rec["layer"],
                "normalized_depth": rec["normalized_depth"],
                "pooled_auc": r["pooled_auc"],
                "benchmark_auc": r["benchmark_auc"], "casual_auc": r["casual_auc"],
                "source_pair_auc_mean": r["source_pair_auc_mean"],
                "source_pair_auc_min": r["source_pair_auc_min"],
                **{f"srcpair::{k}": v for k, v in r["source_pair_auc"].items()},
            })
        p = RESULTS / f"provenance_v3_{model}_log_corpus_contrast.json"
        if p.exists():
            c = json.loads(p.read_text())
            for rec in c["layers"]:
                r = rec["regimes"]["joint"]
                contrast.append({
                    "model": model, "layer": rec["layer"],
                    "auc_lmsys_over_wildchat": r["point"]["all__auc_lmsys_over_wildchat"],
                    "ci_lo": r["block_bootstrap_ci95"]["all__auc_lmsys_over_wildchat"][0],
                    "ci_hi": r["block_bootstrap_ci95"]["all__auc_lmsys_over_wildchat"][1],
                    "auc_eval_vs_wildchat": r["point"]["all__auc_eval_vs_wildchat"],
                    "auc_eval_vs_lmsys": r["point"]["all__auc_eval_vs_lmsys"],
                })
    return prov, contrast


def write_csv(path: Path, data: list[dict]) -> None:
    keys = sorted({k for d in data for k in d})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(data)


def main() -> None:
    prov, contrast = rows()
    write_csv(RESULTS / "provenance_v3_provenance.csv", prov)
    if contrast:
        write_csv(RESULTS / "provenance_v3_log_corpus_contrast.csv", contrast)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for m in MODELS:
        r = sorted([x for x in prov if x["model"] == m], key=lambda x: x["layer"])
        ax.plot([x["normalized_depth"] for x in r], [x["pooled_auc"] for x in r],
                color=COLORS[m], lw=1.4, label=LABELS[m])
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.axhline(CEILING, color="grey", lw=1.0, ls="--")
    ax.text(0.02, CEILING + 0.008, f"surface-text ceiling {CEILING:.2f}", fontsize=6, color="grey")
    ax.set_xlabel("normalized depth"); ax.set_ylabel("provenance AUC")
    ax.set_title("Frozen stated-purpose probe on\nprovenance (2 eval + 2 log corpora)")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels, mins = [], []
    for m in MODELS:
        r = sorted([x for x in prov if x["model"] == m], key=lambda x: -x["pooled_auc"])[0]
        pairs = {k.split("::", 1)[1]: v for k, v in r.items() if k.startswith("srcpair::")}
        for name, v in sorted(pairs.items()):
            labels.append(f"{LABELS[m][-4:]}\n{name.split('|')[0].split('/')[-1][:8]}/"
                          f"{name.split('|')[1].split('/')[-1][:8]}")
            mins.append(v)
    ax.bar(np.arange(len(mins)), mins, color=[COLORS[m] for m in MODELS for _ in range(4)])
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=4.5, rotation=90)
    ax.set_ylabel("provenance AUC at best layer")
    ax.set_title("Every eval-source x log-corpus pair\n(2x2 source robustness)")

    ax = axes[2]
    for m in MODELS:
        r = sorted([x for x in contrast if x["model"] == m], key=lambda x: x["layer"])
        if not r:
            continue
        ax.plot([x["layer"] for x in r], [x["auc_lmsys_over_wildchat"] for x in r],
                color=COLORS[m], lw=1.4, marker="o", ms=2.5, label=LABELS[m])
        ax.fill_between([x["layer"] for x in r], [x["ci_lo"] for x in r],
                        [x["ci_hi"] for x in r], color=COLORS[m], alpha=0.15)
    ax.axhline(0.5, color="k", lw=1.0, ls="--")
    ax.set_xlabel("layer"); ax.set_ylabel("AUC(arena ranked above general use)")
    ax.set_title("Within the deployment class:\narena traffic vs general ChatGPT use")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = RESULTS / "provenance_v3_figure.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}  ({len(prov)} provenance rows, {len(contrast)} contrast rows)")


if __name__ == "__main__":
    main()
