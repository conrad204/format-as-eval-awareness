"""Recompute the earlier behavioral-EA experiments with correct crossed inference.

The original analyses resampled the 128 history x decision cells independently and
identically, which ignores dependence along both crossed axes and understates
generalization uncertainty. This script recomputes the same point estimates with a
two-way crossed bootstrap that independently resamples history families and
decision families, using one shared set of replicate indices for every condition,
endpoint, and model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONDS = ("evaluation_coherent", "deployment_coherent", "evaluation_scrambled", "deployment_scrambled")
EXPERIMENTS = {
    "initial": "behavioral_ea",
    "validation": "behavioral_ea_validation",
}
MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_matrix(rows: list[dict], field: str) -> tuple[np.ndarray, list[str], list[str]]:
    table: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        table.setdefault((row["history_family"], row["decision_family"]), {})[row["condition"]] = row[field]
    histories = sorted({key[0] for key in table})
    decisions = sorted({key[1] for key in table})
    if any(set(cell) != set(CONDS) for cell in table.values()):
        raise RuntimeError("incomplete condition crossing")
    grid = np.array([[[table[(h, d)][c] for c in CONDS] for d in decisions] for h in histories])
    return grid, histories, decisions


def crossed_ci(values: np.ndarray, ih: np.ndarray, idd: np.ndarray) -> list[float]:
    draws = np.array([values[np.ix_(a, b)].mean() for a, b in zip(ih, idd)])
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {
        "schema_version": 1,
        "purpose": "correct two-way crossed re-inference of superseded iid-cell analyses",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "experiments": {},
    }
    for label, prefix in EXPERIMENTS.items():
        per_model = {}
        indices = None
        for model in MODELS:
            path = args.results_dir / f"{prefix}_{model}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            grid, histories, decisions = cell_matrix(data["rows"], "logit_evaluator_minus_user")
            if indices is None:
                indices = (
                    rng.integers(0, len(histories), (args.n_bootstrap, len(histories))),
                    rng.integers(0, len(decisions), (args.n_bootstrap, len(decisions))),
                )
            ih, idd = indices
            coherent = grid[:, :, 0] - grid[:, :, 1]
            interaction = coherent - (grid[:, :, 2] - grid[:, :, 3])
            per_model[model] = {
                "source": str(path),
                "source_sha256": sha256(path),
                "load_in_8bit": data.get("load_in_8bit", False),
                "n_history_families": len(histories),
                "n_decision_families": len(decisions),
                "coherent_logodds": float(coherent.mean()),
                "coherent_logodds_crossed_ci95": crossed_ci(coherent, ih, idd),
                "specificity_interaction_logodds": float(interaction.mean()),
                "specificity_interaction_crossed_ci95": crossed_ci(interaction, ih, idd),
                "positive_history_families": int((coherent.mean(axis=1) > 0).sum()),
                "positive_decision_families": int((coherent.mean(axis=0) > 0).sum()),
            }
            per_model[model]["coherent_crossed_ci_excludes_zero"] = per_model[model]["coherent_logodds_crossed_ci95"][0] > 0
            per_model[model]["interaction_crossed_ci_excludes_zero"] = per_model[model]["specificity_interaction_crossed_ci95"][0] > 0
        output["experiments"][label] = {
            "models": per_model,
            "models_with_coherent_ci_excluding_zero": sum(v["coherent_crossed_ci_excludes_zero"] for v in per_model.values()),
            "models_with_interaction_ci_excluding_zero": sum(v["interaction_crossed_ci_excludes_zero"] for v in per_model.values()),
        }
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
