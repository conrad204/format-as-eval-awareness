"""Analyze the frozen implicit-context behavioral recognition confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
    idom = rng.integers(0, 32, (n_boot, 32))
    iq = rng.integers(0, 4, (n_boot, 4))
    models, pooled_domain = {}, []
    domains = questions = None
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        by = {}
        for row in data["rows"]:
            by[(row["history_family"], row["question_family"], row["condition"])] = row
        domains = sorted({key[0] for key in by})
        questions = sorted({key[1] for key in by})
        conditions = ("implicit_closed_comparison", "implicit_active_use")
        if len(domains) != 32 or len(questions) != 4 or len(by) != 256:
            raise RuntimeError("recognition output is not a complete 32 x 4 x 2 crossing")
        margins = np.array([[[by[(d, q, c)]["target_logit_margin"] for c in conditions] for q in questions] for d in domains])
        correct = margins > 0
        accuracy = correct.mean(axis=2)
        paired_correct = correct.all(axis=2)
        mean_margin = margins.mean(axis=2)
        pooled_domain.append(mean_margin.mean(axis=1))
        boot_acc = np.empty(n_boot)
        for k in range(n_boot):
            boot_acc[k] = accuracy[np.ix_(idom[k], iq[k])].mean()
        q_acc = {q: float(accuracy[:, j].mean()) for j, q in enumerate(questions)}
        record = {
            "source": str(path), "source_sha256": sha256(path), "load_in_8bit": data.get("load_in_8bit", False),
            "accuracy": float(accuracy.mean()),
            "accuracy_two_way_ci95": [float(v) for v in np.quantile(boot_acc, [.025, .975])],
            "both_contexts_correct_rate": float(paired_correct.mean()),
            "mean_correct_choice_logit_margin": float(margins.mean()),
            "question_family_accuracy": q_acc,
            "positive_domain_families": int((mean_margin.mean(axis=1) > 0).sum()),
        }
        gate = protocol["gate"]
        record["model_gate_pass"] = bool(
            record["accuracy"] >= gate["each_model_accuracy_min"]
            and record["accuracy_two_way_ci95"][0] >= gate["each_model_accuracy_two_way_ci_low_min"]
            and min(q_acc.values()) >= gate["each_question_family_accuracy_min"]
        )
        models[data["model_key"]] = record
    pooled = np.mean(np.stack(pooled_domain), axis=0)
    pooled_positive = int((pooled > 0).sum())
    gate = protocol["gate"]
    result = {
        "schema_version": 1, "protocol_sha256": sha256(args.protocol), "n_bootstrap": n_boot,
        "inference": "two-way crossed domain x question-family bootstrap", "models": models,
        "pooled_domains_positive": pooled_positive,
        "recognition_gate_pass": bool(
            all(models[k]["model_gate_pass"] for k in gate["required_models"])
            and pooled_positive >= gate["pooled_domains_positive_min"]
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
