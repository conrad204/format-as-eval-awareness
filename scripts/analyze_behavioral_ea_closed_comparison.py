"""Apply the frozen two-way crossed-bootstrap closed-comparison EA gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONDS = ("evaluation_coherent", "deployment_coherent", "evaluation_scrambled", "deployment_scrambled")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    n_boot = protocol["design"]["n_bootstrap"]
    rng = np.random.default_rng(protocol["design"]["seed"])
    ih = rng.integers(0, 32, (n_boot, 32))
    id_ = rng.integers(0, 16, (n_boot, 16))
    models, contrasts = {}, []
    histories = decisions = None
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        by = {}
        for row in data["rows"]:
            by.setdefault((row["history_family"], row["decision_family"]), {})[row["condition"]] = row
        histories = sorted({key[0] for key in by})
        decisions = sorted({key[1] for key in by})
        if len(histories) != 32 or len(decisions) != 16 or any(set(v) != set(CONDS) for v in by.values()):
            raise RuntimeError("focused bank is not a complete 32 x 16 x 4 crossing")
        x = np.array([[[by[(h, d)][c]["logit_evaluator_minus_user"] for c in CONDS] for d in decisions] for h in histories])
        p = np.array([[[by[(h, d)][c]["p_evaluator_favored"] for c in CONDS] for d in decisions] for h in histories])
        coherent = x[:, :, 0] - x[:, :, 1]
        interaction = coherent - (x[:, :, 2] - x[:, :, 3])
        contrasts.append(coherent)
        boot_c = np.empty(n_boot)
        boot_i = np.empty(n_boot)
        for k in range(n_boot):
            sel = np.ix_(ih[k], id_[k])
            boot_c[k] = coherent[sel].mean()
            boot_i[k] = interaction[sel].mean()
        record = {
            "source": str(path), "source_sha256": sha256(path), "load_in_8bit": data.get("load_in_8bit", False),
            "coherent_logodds": float(coherent.mean()),
            "coherent_logodds_two_way_ci95": [float(v) for v in np.quantile(boot_c, [.025, .975])],
            "specificity_interaction_logodds": float(interaction.mean()),
            "specificity_interaction_logodds_two_way_ci95": [float(v) for v in np.quantile(boot_i, [.025, .975])],
            "coherent_probability_difference": float((p[:, :, 0] - p[:, :, 1]).mean()),
            "direction_consistency": float((coherent > 0).mean()),
        }
        record["model_gate_pass"] = record["coherent_logodds_two_way_ci95"][0] > 0 and record["specificity_interaction_logodds_two_way_ci95"][0] > 0
        models[data["model_key"]] = record
    pooled = np.mean(np.stack(contrasts), axis=0)
    history_means = {h: float(pooled[i].mean()) for i, h in enumerate(histories)}
    decision_means = {d: float(pooled[:, j].mean()) for j, d in enumerate(decisions)}
    gate = protocol["gate"]
    result = {
        "schema_version": 1, "protocol_sha256": sha256(args.protocol), "n_bootstrap": n_boot,
        "inference": "two-way crossed history-family x decision-family bootstrap", "models": models,
        "pooled_history_family_means": history_means, "pooled_decision_family_means": decision_means,
        "pooled_positive_history_families": sum(v > 0 for v in history_means.values()),
        "pooled_positive_decision_families": sum(v > 0 for v in decision_means.values()),
    }
    result["confirmation_gate_pass"] = bool(
        all(k in models and models[k]["model_gate_pass"] for k in gate["required_models"])
        and result["pooled_positive_history_families"] >= gate["pooled_positive_domain_families_min"]
        and result["pooled_positive_decision_families"] >= gate["pooled_positive_decision_families_min"]
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
