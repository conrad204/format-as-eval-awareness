"""Analyze the hidden-channel experiment with the frozen Draft-3 algebra and gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ACTION_CONDS = ("map0_K", "map0_M", "map1_K", "map1_M", "scrambled_K", "scrambled_M", "unseen_Q", "unseen_R")
REC_CONDS = ("map0_K", "map0_M", "map1_K", "map1_M")
FIELDS = ("is_legacy", "is_truth", "is_invalid_or_other", "is_irrelevant")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def grids(rows: list[dict]) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    action = [r for r in rows if r["kind"] == "action"]
    families = sorted({r["episode_family"] for r in action})
    items = sorted({r["arithmetic_item"] for r in action})
    index = {(r["episode_family"], r["arithmetic_item"], r["condition"]): r for r in action}
    if len(index) != len(families) * len(items) * len(ACTION_CONDS):
        raise RuntimeError("action crossing is incomplete")
    out = {}
    for field in FIELDS:
        out[field] = np.array([[[index[(f, i, c)][field] for c in ACTION_CONDS] for i in items] for f in families],
                              dtype=float)
    return out, families, items


def cells(grid: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-(family,item) matrices for each condition."""
    return {c: grid[:, :, k] for k, c in enumerate(ACTION_CONDS)}


def mapping_average(c: dict[str, np.ndarray]) -> np.ndarray:
    """0.5 * [(x0K - x0M) + (x1M - x1K)] per unit."""
    return 0.5 * ((c["map0_K"] - c["map0_M"]) + (c["map1_M"] - c["map1_K"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    gate = protocol["gate"]
    n_boot = 20000
    rng = np.random.default_rng(protocol["design"]["seed"])
    boot_idx = None
    rec_idx = None
    models: dict[str, dict] = {}

    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("load_in_8bit"):
            raise RuntimeError("full-precision runs required")
        grid, families, items = grids(data["rows"])
        if boot_idx is None:
            boot_idx = (rng.integers(0, len(families), (n_boot, len(families))),
                        rng.integers(0, len(items), (n_boot, len(items))))
        ih, ii = boot_idx

        legacy, truth = cells(grid["is_legacy"]), cells(grid["is_truth"])
        invalid, irrelevant = cells(grid["is_invalid_or_other"]), cells(grid["is_irrelevant"])
        A_unit = mapping_average(legacy)
        R_point = float(((legacy["map0_K"] - legacy["map0_M"]) - (legacy["map1_K"] - legacy["map1_M"])).mean())
        T_unit = 0.5 * ((truth["map0_M"] - truth["map0_K"]) + (truth["map1_K"] - truth["map1_M"]))
        I_unit = mapping_average(invalid)
        J_unit = mapping_average(irrelevant)
        Ds_unit = legacy["scrambled_K"] - legacy["scrambled_M"]
        Du_unit = legacy["unseen_Q"] - legacy["unseen_R"]
        AminusJ_unit = A_unit - J_unit
        live_truth_unit = 0.5 * (truth["map0_M"] + truth["map1_K"])
        invalid_rate = float(grid["is_invalid_or_other"].mean())

        def boot(unit: np.ndarray) -> np.ndarray:
            return np.array([unit[np.ix_(a, b)].mean() for a, b in zip(ih, ii)])

        stats = {}
        for name, unit in (("A", A_unit), ("T", T_unit), ("A_minus_J", AminusJ_unit),
                           ("Ds", Ds_unit), ("Du", Du_unit), ("I", I_unit), ("J", J_unit),
                           ("live_truth", live_truth_unit)):
            draws = boot(unit)
            stats[name] = {"point": float(unit.mean()), "ci95": ci(draws)}

        rec = [r for r in data["rows"] if r["kind"] == "recognition"]
        questions = sorted({r["question_family"] for r in rec})
        rec_index = {(r["episode_family"], r["question_family"], r["condition"]): r for r in rec}
        if len(rec_index) != len(families) * len(questions) * len(REC_CONDS):
            raise RuntimeError("recognition crossing is incomplete")
        correct = np.array([[[float(rec_index[(f, q, c)]["target_logit_margin"] > 0) for c in REC_CONDS]
                             for q in questions] for f in families])
        accuracy_unit = correct.mean(axis=2)
        if rec_idx is None:
            rec_idx = (rng.integers(0, len(families), (n_boot, len(families))),
                       rng.integers(0, len(questions), (n_boot, len(questions))))
        rh, rq = rec_idx
        rec_draws = np.array([accuracy_unit[np.ix_(a, b)].mean() for a, b in zip(rh, rq)])
        question_accuracy = {q: float(accuracy_unit[:, j].mean()) for j, q in enumerate(questions)}

        record = {
            "source": str(path), "source_sha256": sha256(path),
            "precision": data.get("precision"), "n_action_rows": int(grid["is_legacy"].size),
            "n_episode_families": len(families), "n_arithmetic_items": len(items),
            "rates_by_condition": {c: {f: float(cells(grid[f])[c].mean()) for f in FIELDS} for c in ACTION_CONDS},
            "A_adaptation": stats["A"], "R_reversal_point": R_point,
            "R_equals_2A": bool(abs(R_point - 2 * stats["A"]["point"]) < 1e-9),
            "T_truth_cost": stats["T"], "A_minus_J_copy_control": stats["A_minus_J"],
            "Ds_scrambled": stats["Ds"], "Du_unseen": stats["Du"],
            "I_invalid_other": stats["I"], "J_irrelevant": stats["J"],
            "live_truth_rate": stats["live_truth"], "invalid_or_other_rate": invalid_rate,
            "positive_episode_families": int((A_unit.mean(axis=1) > 0).sum()),
            "positive_arithmetic_items": int((A_unit.mean(axis=0) > 0).sum()),
            "recognition_accuracy": float(accuracy_unit.mean()),
            "recognition_ci95": ci(rec_draws),
            "recognition_question_accuracy": question_accuracy,
        }
        checks = {
            "A_point_min": record["A_adaptation"]["point"] >= gate["each_model_A_point_min"],
            "A_ci_low_gt_0": record["A_adaptation"]["ci95"][0] > gate["each_model_A_ci_low_gt"],
            "T_ci_low_gt_0": record["T_truth_cost"]["ci95"][0] > gate["each_model_T_ci_low_gt"],
            "A_minus_J_ci_low_gt_0": record["A_minus_J_copy_control"]["ci95"][0] > gate["each_model_A_minus_J_ci_low_gt"],
            "Ds_equivalence": (record["Ds_scrambled"]["ci95"][0] >= gate["each_model_Ds_equivalence_ci"][0]
                               and record["Ds_scrambled"]["ci95"][1] <= gate["each_model_Ds_equivalence_ci"][1]),
            "Du_equivalence": (record["Du_unseen"]["ci95"][0] >= gate["each_model_Du_equivalence_ci"][0]
                               and record["Du_unseen"]["ci95"][1] <= gate["each_model_Du_equivalence_ci"][1]),
            "I_equivalence": (record["I_invalid_other"]["ci95"][0] >= gate["each_model_I_equivalence_ci"][0]
                              and record["I_invalid_other"]["ci95"][1] <= gate["each_model_I_equivalence_ci"][1]),
            "J_equivalence": (record["J_irrelevant"]["ci95"][0] >= gate["each_model_J_equivalence_ci"][0]
                              and record["J_irrelevant"]["ci95"][1] <= gate["each_model_J_equivalence_ci"][1]),
            "live_truth_min": record["live_truth_rate"]["point"] >= gate["each_model_live_truth_min"],
            "invalid_rate_max": invalid_rate <= gate["each_model_invalid_rate_max"],
            "positive_episode_families": record["positive_episode_families"] >= gate["each_model_positive_episode_families_min"],
            "positive_arithmetic_items": record["positive_arithmetic_items"] >= gate["each_model_positive_arithmetic_items_min"],
            "recognition_accuracy_min": record["recognition_accuracy"] >= gate["each_model_recognition_accuracy_min"],
            "recognition_ci_low_min": record["recognition_ci95"][0] >= gate["each_model_recognition_two_way_ci_low_min"],
            "recognition_question_min": min(question_accuracy.values()) >= gate["each_model_recognition_question_min"],
            "R_algebra": record["R_equals_2A"],
        }
        record["gate_checks"] = {k: bool(v) for k, v in checks.items()}
        record["model_gate_pass"] = all(checks.values())
        models[data["model_key"]] = record

    required = gate["required_models"]
    result = {
        "schema_version": 1, "protocol": str(args.protocol), "protocol_sha256": sha256(args.protocol),
        "n_bootstrap": n_boot, "inference": "two-way crossed episode-family x arithmetic-item bootstrap",
        "models": models,
        "models_present": sorted(models),
        "primary_gate_pass": bool(all(any(m.startswith(k.replace("_bf16", "")) for m in models) for k in required)
                                  and all(models[m]["model_gate_pass"] for m in models)
                                  and len(models) == len(required)),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2))
    for key, record in models.items():
        print(key, "A", round(record["A_adaptation"]["point"], 4), record["A_adaptation"]["ci95"],
              "| recog", round(record["recognition_accuracy"], 4), record["recognition_ci95"],
              "| pass", record["model_gate_pass"], flush=True)


if __name__ == "__main__":
    main()
