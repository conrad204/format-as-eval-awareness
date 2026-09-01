"""Figure: latent-status inference versus the explicit-statement ceiling."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LABELS = {"llama31_8b": "Llama-3.1-8B", "llama31_70b": "Llama-3.1-70B", "llama33_70b": "Llama-3.3-70B"}


def main() -> None:
    data = json.loads((RESULTS / "behavioral_ea_latent_vs_explicit.json").read_text(encoding="utf-8"))
    models = [k for k in ("llama31_8b", "llama31_70b", "llama33_70b") if k in data["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    ax = axes[0]
    x = np.arange(len(models))
    for offset, key, colour, name in ((-0.18, "latent", "#b3452c", "status must be inferred"),
                                      (+0.18, "explicit", "#2457a6", "status stated once")):
        point = [data["models"][m][f"{key}_accuracy"] for m in models]
        lo = [point[i] - data["models"][m][f"{key}_ci95"][0] for i, m in enumerate(models)]
        hi = [data["models"][m][f"{key}_ci95"][1] - point[i] for i, m in enumerate(models)]
        ax.bar(x + offset, point, width=0.34, color=colour, label=name)
        ax.errorbar(x + offset, point, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, [LABELS[m] for m in models], rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recognition accuracy")
    ax.set_title("Which channel is the closed one?", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    conditions = [("latent_action_rates", "map0_K", "closed, inferred"),
                  ("latent_action_rates", "map0_M", "live, inferred"),
                  ("explicit_action_rates", "explicit_closed", "closed, rule stated"),
                  ("explicit_action_rates", "explicit_active", "live, rule stated")]
    width = 0.2
    for i, (block, cond, name) in enumerate(conditions):
        vals = []
        for m in models:
            rates = data["models"][m].get(block, {})
            vals.append(rates.get(cond, {}).get("legacy", np.nan))
        ax.bar(np.arange(len(models)) + (i - 1.5) * width, vals, width=width, label=name)
    ax.set_xticks(np.arange(len(models)), [LABELS[m] for m in models], rotation=12)
    ax.set_ylabel("rate of emitting the flawed legacy answer")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does the model game the retention metric?", fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle("Hidden-channel test: inferring evaluation status versus being told it", fontsize=12)
    fig.tight_layout()
    out = RESULTS / "behavioral_ea_hidden_channel_figure.png"
    fig.savefig(out, dpi=180)
    print(json.dumps({"figure": str(out), "models": models}, indent=2))


if __name__ == "__main__":
    main()
