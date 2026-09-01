"""Analyze the frozen behavioral-EA intervention with complete-block bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("evaluation_coherent", "deployment_coherent", "evaluation_scrambled", "deployment_scrambled")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    n_boot = int(protocol["design"]["n_bootstrap"])
    seed = int(protocol["design"]["seed"])

    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    common_blocks = sorted(set.intersection(*[{row["block_id"] for row in data["rows"]} for data in loaded]))
    if len(common_blocks) != 128:
        raise RuntimeError(f"expected 128 complete shared blocks, found {len(common_blocks)}")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(common_blocks), size=(n_boot, len(common_blocks)))
    result_models = {}
    for path, data in zip(args.inputs, loaded):
        by_block = {}
        for row in data["rows"]:
            by_block.setdefault(row["block_id"], {})[row["condition"]] = row
        if any(set(by_block[block]) != set(CONDITIONS) for block in common_blocks):
            raise RuntimeError("incomplete condition block")
        logit = np.array([[by_block[block][condition]["logit_evaluator_minus_user"] for condition in CONDITIONS] for block in common_blocks])
        prob = np.array([[by_block[block][condition]["p_evaluator_favored"] for condition in CONDITIONS] for block in common_blocks])
        coherent_logit = logit[:, 0] - logit[:, 1]
        scrambled_logit = logit[:, 2] - logit[:, 3]
        interaction_logit = coherent_logit - scrambled_logit
        coherent_prob = prob[:, 0] - prob[:, 1]
        interaction_prob = coherent_prob - (prob[:, 2] - prob[:, 3])
        boot = {
            "coherent_logit": coherent_logit[samples].mean(axis=1),
            "specificity_interaction_logit": interaction_logit[samples].mean(axis=1),
            "coherent_probability": coherent_prob[samples].mean(axis=1),
            "specificity_interaction_probability": interaction_prob[samples].mean(axis=1),
        }
        record = {
            "source": str(path),
            "source_sha256": sha256(path),
            "model_key": data["model_key"],
            "n_blocks": len(common_blocks),
            "mean_p_evaluator_by_condition": {condition: float(prob[:, i].mean()) for i, condition in enumerate(CONDITIONS)},
            "coherent_logit_contrast": float(coherent_logit.mean()),
            "coherent_logit_contrast_ci95": interval(boot["coherent_logit"]),
            "coherent_probability_difference": float(coherent_prob.mean()),
            "coherent_probability_difference_ci95": interval(boot["coherent_probability"]),
            "specificity_interaction_logit": float(interaction_logit.mean()),
            "specificity_interaction_logit_ci95": interval(boot["specificity_interaction_logit"]),
            "specificity_interaction_probability": float(interaction_prob.mean()),
            "specificity_interaction_probability_ci95": interval(boot["specificity_interaction_probability"]),
            "direction_consistency": float((coherent_logit > 0).mean()),
            "history_family_means": {},
            "decision_family_means": {},
        }
        item_lookup = {row["block_id"]: row for row in data["rows"] if row["condition"] == "evaluation_coherent"}
        for field, output_key in (("history_family", "history_family_means"), ("decision_family", "decision_family_means")):
            labels = sorted({item_lookup[block][field] for block in common_blocks})
            for label in labels:
                mask = np.array([item_lookup[block][field] == label for block in common_blocks])
                record[output_key][label] = float(coherent_logit[mask].mean())
        gate = protocol["primary_gate"]
        record["model_gate_pass"] = bool(
            record["coherent_logit_contrast_ci95"][0] > gate["coherent_contrast_ci_low_gt"]
            and record["specificity_interaction_logit_ci95"][0] > gate["specificity_interaction_ci_low_gt"]
            and record["coherent_probability_difference"] >= gate["minimum_probability_point_difference"]
            and record["direction_consistency"] >= gate["direction_consistency_min_fraction"]
        )
        result_models[data["model_key"]] = record

    required = protocol["primary_gate"]["models_required"]
    result = {
        "schema_version": 1,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "n_bootstrap": n_boot,
        "same_block_samples_across_models": True,
        "models": result_models,
        "primary_gate_pass": all(key in result_models and result_models[key]["model_gate_pass"] for key in required),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
