#!/usr/bin/env python3
"""Analyze complete frozen factorial A/B/C causal response surfaces."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    from scripts.run_pilot80_factorial_causal import (
        condition_specs, expected_record_keys, load_rows, record_key, sha256_file,
        validate_plan_arrays, validate_plan_contract, validate_resumed_raw,
    )
except ImportError:
    from run_pilot80_factorial_causal import (
        condition_specs, expected_record_keys, load_rows, record_key, sha256_file,
        validate_plan_arrays, validate_plan_contract, validate_resumed_raw,
    )

ENDPOINTS = ("shared_purpose", "purpose_interaction", "format")
SUPPORTED_PROTOCOL_IDS = {
    "pilot80_factorial_causal_mediation_v1",
    "pilot80_factorial_causal_mediation_v2_reextracted",
    "pilot80_factorial_causal_mediation_v3_determinism_patched",
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def strict_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_raw(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = record_key(row)
        if key in index:
            raise ValueError(f"duplicate raw record {path}:{line_number}: {key}")
        scores = row.get("readout_scores", {})
        if set(scores) != {"purpose_benchmark", "purpose_casual", "format"}:
            raise ValueError(f"invalid readout score fields {key}")
        numeric = list(scores.values()) + [
            row.get("source_baseline_max_abs"), row.get("source_baseline_cosine"),
            row.get("target_baseline_max_abs"), row.get("target_baseline_cosine"),
        ]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise ValueError(f"nonfinite raw record {key}")
        rows.append(row)
        index[key] = row
    return rows, index


def response_values(plus: dict[str, Any], minus: dict[str, Any]) -> dict[str, float]:
    if plus["format"] != minus["format"]:
        raise ValueError("paired records disagree on format")
    purpose_name = "purpose_benchmark" if plus["format"] == "benchmark" else "purpose_casual"
    purpose = (plus["readout_scores"][purpose_name] - minus["readout_scores"][purpose_name]) / 2
    format_value = (plus["readout_scores"]["format"] - minus["readout_scores"]["format"]) / 2
    format_sign = 1.0 if plus["format"] == "benchmark" else -1.0
    return {
        "shared_purpose": float(purpose),
        "purpose_interaction": float(format_sign * purpose),
        "format": float(format_value),
    }


def ci(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.025, 0.5, 0.975])
    return {"ci_2.5": float(q[0]), "ci_50": float(q[1]), "ci_97.5": float(q[2])}


def repo_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError("unexpected protocol")
    if not (len(args.plan_json) == len(args.plan_npz) == len(args.raw) == len(protocol["models"])):
        raise ValueError("supply exactly one aligned plan-json/plan-npz/raw bundle per configured model")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "causal_results.json"
    metrics_path = args.out_dir / "causal_metrics.tsv"
    figure_path = args.out_dir / "causal_response_matrix.png"
    manifest_path = args.out_dir / "run_manifest.json"
    existing_outputs = [path for path in (results_path, metrics_path, figure_path, manifest_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite immutable analysis outputs: {existing_outputs}")
    expected_specs = condition_specs(protocol)
    n_random = int(protocol["interventions"]["random_directions"])
    if n_random != len(protocol["interventions"]["random_seeds"]) or n_random < 20:
        raise ValueError("frozen random-control bank mismatch")
    model_data: dict[str, dict[str, Any]] = {}
    all_raw_hashes: dict[str, str] = {}
    config_sha256 = sha256_file(config_path)
    runner_path = Path(__file__).with_name("run_pilot80_factorial_causal.py")
    builder_path = Path(__file__).with_name("build_pilot80_factorial_causal_plan.py")
    runner_script_sha256 = sha256_file(runner_path)
    builder_script_sha256 = sha256_file(builder_path)
    canonical_items_path = Path(protocol["dataset"]["items_path"])
    if not canonical_items_path.exists() or sha256_file(canonical_items_path) != protocol["dataset"]["items_sha256"]:
        raise ValueError("canonical item bank missing or hash-mismatched")
    canonical_rows = load_rows(canonical_items_path)
    for plan_json_arg, plan_npz_arg, raw_arg in zip(args.plan_json, args.plan_npz, args.raw):
        plan_json_path, plan_npz_path, raw_path = map(Path.resolve, (plan_json_arg, plan_npz_arg, raw_arg))
        plan = json.loads(plan_json_path.read_text(encoding="utf-8"))
        model_id = plan["model"]["model_id"]
        if model_id in model_data:
            raise ValueError(f"duplicate model bundle {model_id}")
        configured = next((model for model in protocol["models"] if model["model_id"] == model_id), None)
        if configured is None or configured != plan["model"]:
            raise ValueError(f"model config mismatch {model_id}")
        if plan.get("config_sha256") != config_sha256 or plan.get("plan_npz_sha256") != sha256_file(plan_npz_path):
            raise ValueError(f"plan provenance mismatch {model_id}")
        if plan.get("script_sha256") != builder_script_sha256:
            raise ValueError(f"plan builder provenance mismatch {model_id}")
        validate_plan_contract(protocol, plan, canonical_rows)
        with np.load(plan_npz_path, allow_pickle=False) as plan_archive:
            validate_plan_arrays(plan, plan_archive)
        invalid_path = raw_path.with_name("INVALID.json")
        if invalid_path.exists():
            raise ValueError(f"invalid run marker present {model_id}: {invalid_path}")
        runner_manifest_path = raw_path.with_name("runner_manifest.json")
        if not runner_manifest_path.exists():
            raise ValueError(f"missing runner manifest {model_id}")
        runner_manifest = json.loads(runner_manifest_path.read_text(encoding="utf-8"))
        plan_json_sha256 = sha256_file(plan_json_path)
        plan_npz_sha256 = sha256_file(plan_npz_path)
        raw_sha256 = sha256_file(raw_path)
        expected_keys = expected_record_keys(plan, model_id, expected_specs)
        required_manifest = {
            "schema_version": 1,
            "status": "completed_uninspected_factorial_causal_raw",
            "protocol_id": protocol["protocol_id"],
            "model_id": model_id,
            "revision": configured["revision"],
            "config_sha256": config_sha256,
            "plan_json_sha256": plan_json_sha256,
            "plan_npz_sha256": plan_npz_sha256,
            "script_sha256": runner_script_sha256,
            "activation_sha256": configured["activation_sha256"],
            "items_sha256": plan["items_sha256"],
            "raw_path": raw_path.name,
            "raw_sha256": raw_sha256,
            "records": len(expected_keys),
            "expected_records": len(expected_keys),
        }
        for field, value in required_manifest.items():
            if runner_manifest.get(field) != value:
                raise ValueError(f"runner manifest mismatch {model_id}: {field}")
        model_file_hashes = runner_manifest.get("model_file_hashes")
        if (
            not isinstance(model_file_hashes, dict)
            or not model_file_hashes
            or not all(
                isinstance(name, str)
                and isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for name, value in model_file_hashes.items()
            )
        ):
            raise ValueError(f"runner model file hashes invalid {model_id}")
        strict_keys, _ = validate_resumed_raw(
            raw_path, protocol, plan, canonical_rows, expected_specs, config_sha256,
            plan_json_sha256, plan_npz_sha256, runner_script_sha256,
        )
        if strict_keys != expected_keys:
            raise ValueError(f"strict raw completeness mismatch {model_id}")
        rows, raw_index = load_raw(raw_path)
        if set(raw_index) != expected_keys:
            raise ValueError(f"raw index mismatch {model_id}")
        for row in rows:
            if row["hook_count"] != 1 or row["source_hook_count"] != 1 or row["target_hook_count"] != 1:
                raise ValueError(f"raw hook gate failure {record_key(row)}")
        model_data[model_id] = {
            "plan": plan, "plan_json_path": plan_json_path, "plan_npz_path": plan_npz_path,
            "raw_path": raw_path, "runner_manifest_path": runner_manifest_path,
            "runner_manifest": runner_manifest, "rows": rows, "raw_index": raw_index,
        }
        all_raw_hashes[model_id] = raw_sha256
    if set(model_data) != {model["model_id"] for model in protocol["models"]}:
        raise ValueError("model bundles do not match frozen model set")

    # Baseline endpoint and hook/reproduction gates.
    endpoint_metrics: dict[str, dict[str, Any]] = {}
    endpoint_valid = True
    hook_and_reproduction = True
    max_reproduction = {"source_max_abs": 0.0, "target_max_abs": 0.0, "source_min_cosine": 1.0, "target_min_cosine": 1.0}
    for model_id, data in model_data.items():
        labels_and_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for fold in data["plan"]["folds"]:
            for layer_text in fold["source_layers"]:
                # Baseline readouts are layer invariant; use low source layer only.
                if int(layer_text) != min(map(int, fold["source_layers"])):
                    continue
                for row_index in fold["test_indices"]:
                    zero = data["raw_index"][(model_id, int(layer_text), fold["fold_id"], row_index, "zero")]
                    purpose_label = int(zero["intended_purpose"] == "evaluation")
                    format_label = int(zero["format"] == "benchmark")
                    labels_and_scores["format"].append((format_label, zero["readout_scores"]["format"]))
                    purpose_name = "purpose_benchmark" if zero["format"] == "benchmark" else "purpose_casual"
                    labels_and_scores[purpose_name].append((purpose_label, zero["readout_scores"][purpose_name]))
        model_metrics: dict[str, Any] = {}
        for name in ("purpose_benchmark", "purpose_casual", "format"):
            pairs = labels_and_scores[name]
            labels = np.asarray([pair[0] for pair in pairs], dtype=np.int8)
            scores = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            metrics = {
                "n": len(pairs), "roc_auc": float(roc_auc_score(labels, scores)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0)),
            }
            metrics["pass"] = metrics["roc_auc"] >= 0.80 and metrics["balanced_accuracy"] >= 0.60
            endpoint_valid &= metrics["pass"]
            model_metrics[name] = metrics
        endpoint_metrics[model_id] = model_metrics
        for row in data["rows"]:
            max_reproduction["source_max_abs"] = max(max_reproduction["source_max_abs"], row["source_baseline_max_abs"])
            max_reproduction["target_max_abs"] = max(max_reproduction["target_max_abs"], row["target_baseline_max_abs"])
            max_reproduction["source_min_cosine"] = min(max_reproduction["source_min_cosine"], row["source_baseline_cosine"])
            max_reproduction["target_min_cosine"] = min(max_reproduction["target_min_cosine"], row["target_baseline_cosine"])
    hook_and_reproduction = (
        max_reproduction["source_max_abs"] <= 0.125 and max_reproduction["target_max_abs"] <= 0.125
        and max_reproduction["source_min_cosine"] >= 0.999 and max_reproduction["target_min_cosine"] >= 0.999
    )

    # Paired component and random responses, indexed at row level.
    paired: dict[tuple[str, str, int, str, float, str, int], dict[str, float]] = {}
    block_values: dict[tuple[str, int, str, float, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    random_block_values: dict[tuple[str, int, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model_id, data in model_data.items():
        for fold in data["plan"]["folds"]:
            fold_id = fold["fold_id"]
            for layer_text in fold["source_layers"]:
                layer = int(layer_text)
                for row_index in fold["test_indices"]:
                    zero = data["raw_index"][(model_id, layer, fold_id, row_index, "zero")]
                    block = zero["payload_block_id"]
                    for component in ("A", "B", "C"):
                        for alpha in (0.5, 1.0):
                            plus_id = f"{component}_p_alpha{alpha:g}"
                            minus_id = f"{component}_m_alpha{alpha:g}"
                            values = response_values(
                                data["raw_index"][(model_id, layer, fold_id, row_index, plus_id)],
                                data["raw_index"][(model_id, layer, fold_id, row_index, minus_id)],
                            )
                            paired[(model_id, fold_id, layer, component, alpha, block, row_index)] = values
                            for endpoint, value in values.items():
                                block_values[(model_id, layer, component, alpha, endpoint)][block].append(value)
                    for random_index, seed in enumerate(protocol["interventions"]["random_seeds"]):
                        plus_id = f"random_{seed}_p_alpha1"
                        minus_id = f"random_{seed}_m_alpha1"
                        values = response_values(
                            data["raw_index"][(model_id, layer, fold_id, row_index, plus_id)],
                            data["raw_index"][(model_id, layer, fold_id, row_index, minus_id)],
                        )
                        for endpoint, value in values.items():
                            random_block_values[(model_id, layer, random_index, endpoint)][block].append(value)

    all_blocks = sorted({block for values in block_values.values() for block in values})
    n_boot = int(protocol["uncertainty"]["bootstrap_resamples"])
    rng = np.random.default_rng(int(protocol["uncertainty"]["bootstrap_seed"]))
    resamples = rng.integers(0, len(all_blocks), size=(n_boot, len(all_blocks)))
    metrics_rows: list[dict[str, Any]] = []
    point_lookup: dict[tuple[str, int, str, float, str], dict[str, Any]] = {}
    for key, values_by_block in sorted(block_values.items()):
        model_id, layer, component, alpha, endpoint = key
        missing_blocks = set(all_blocks) - values_by_block.keys()
        if missing_blocks:
            raise ValueError(f"component response missing payload blocks {key}: {sorted(missing_blocks)[:5]}")
        means = np.asarray([np.mean(values_by_block[block]) for block in all_blocks], dtype=np.float64)
        boot = means[resamples].mean(axis=1)
        random_points = []
        if alpha == 1.0:
            for random_index in range(n_random):
                random_values = random_block_values[(model_id, layer, random_index, endpoint)]
                if set(random_values) != set(all_blocks):
                    raise ValueError(f"random response missing blocks {(model_id, layer, random_index, endpoint)}")
                random_points.append(float(np.mean([np.mean(random_values[block]) for block in all_blocks])))
        point = float(means.mean())
        random_p = None if alpha != 1.0 else float((1 + sum(abs(value) >= abs(point) for value in random_points)) / (1 + n_random))
        row = {
            "model_id": model_id, "source_layer": layer, "component": component,
            "alpha": alpha, "endpoint": endpoint, "point": point, **ci(boot),
            "random_p": random_p, "n_blocks": len(all_blocks),
        }
        metrics_rows.append(row)
        point_lookup[key] = row

    source_by_model = {model["model_id"]: model["source_layers"] for model in protocol["models"]}
    A_shared = True
    B_format = True
    C_interaction = True
    dose_response = True
    for model_id, sources in source_by_model.items():
        for layer in sources.values():
            a = point_lookup[(model_id, layer, "A", 1.0, "shared_purpose")]
            b = point_lookup[(model_id, layer, "B", 1.0, "format")]
            A_shared &= a["ci_2.5"] > 0 and a["random_p"] <= 0.05
            B_format &= b["ci_2.5"] > 0 and b["random_p"] <= 0.05
            for component, endpoint in (("A", "shared_purpose"), ("B", "format")):
                full = point_lookup[(model_id, layer, component, 1.0, endpoint)]["point"]
                half = point_lookup[(model_id, layer, component, 0.5, endpoint)]["point"]
                dose_response &= half * full > 0 and abs(half) <= abs(full)
        high = sources["high_interaction"]
        c = point_lookup[(model_id, high, "C", 1.0, "purpose_interaction")]
        C_interaction &= c["ci_2.5"] > 0 and c["random_p"] <= 0.05
        full = c["point"]
        half = point_lookup[(model_id, high, "C", 0.5, "purpose_interaction")]["point"]
        dose_response &= half * full > 0 and abs(half) <= abs(full)

    # Cross-model high-minus-low C response, paired on the common payload blocks.
    model_differences = []
    for model_id, sources in source_by_model.items():
        high_values = block_values[(model_id, sources["high_interaction"], "C", 1.0, "purpose_interaction")]
        low_values = block_values[(model_id, sources["low_interaction"], "C", 1.0, "purpose_interaction")]
        model_differences.append(np.asarray([np.mean(high_values[block]) - np.mean(low_values[block]) for block in all_blocks]))
    aggregate_difference = np.mean(model_differences, axis=0)
    difference_boot = aggregate_difference[resamples].mean(axis=1)
    geometry_to_causality_summary = {"point": float(aggregate_difference.mean()), **ci(difference_boot)}
    geometry_to_causality = geometry_to_causality_summary["ci_2.5"] > 0

    # Full cube response-surface checks and explicit intervention-side interactions.
    cube_summary: dict[str, Any] = {}
    factorial_cube_agreement = True
    for model_id, data in model_data.items():
        observed: list[float] = []
        predicted: list[float] = []
        interactions: dict[str, list[float]] = defaultdict(list)
        for fold in data["plan"]["folds"]:
            fold_id = fold["fold_id"]
            for layer_text in fold["source_layers"]:
                layer = int(layer_text)
                for row_index in fold["test_indices"]:
                    zero = data["raw_index"][(model_id, layer, fold_id, row_index, "zero")]
                    for readout in ("purpose_benchmark", "purpose_casual", "format"):
                        base = zero["readout_scores"][readout]
                        single = {}
                        for component in ("A", "B", "C"):
                            plus = data["raw_index"][(model_id, layer, fold_id, row_index, f"{component}_p_alpha1")]["readout_scores"][readout]
                            minus = data["raw_index"][(model_id, layer, fold_id, row_index, f"{component}_m_alpha1")]["readout_scores"][readout]
                            single[component] = (plus - minus) / 2
                        corners: list[tuple[int, int, int, float]] = []
                        for sa in (-1, 1):
                            for sb in (-1, 1):
                                for sc in (-1, 1):
                                    token = "_".join(f"{name}{'p' if sign > 0 else 'm'}" for name, sign in (("A", sa), ("B", sb), ("C", sc)))
                                    value = data["raw_index"][(model_id, layer, fold_id, row_index, "cube_" + token)]["readout_scores"][readout]
                                    corners.append((sa, sb, sc, value))
                                    observed.append(value - base)
                                    predicted.append(sa * single["A"] + sb * single["B"] + sc * single["C"])
                        for name, weights in (
                            ("AB", lambda a, b, c: a * b), ("AC", lambda a, b, c: a * c),
                            ("BC", lambda a, b, c: b * c), ("ABC", lambda a, b, c: a * b * c),
                        ):
                            interactions[name].append(float(np.mean([weights(a, b, c) * value for a, b, c, value in corners])))
        observed_array = np.asarray(observed)
        predicted_array = np.asarray(predicted)
        denominator = float(np.sum(observed_array ** 2))
        r2 = float(1 - np.sum((observed_array - predicted_array) ** 2) / denominator) if denominator else float("-inf")
        factorial_cube_agreement &= r2 > 0.50
        cube_summary[model_id] = {
            "n_corner_scores": len(observed), "additive_r2": r2,
            "rmse": float(np.sqrt(np.mean((observed_array - predicted_array) ** 2))),
            "intervention_side_interactions": {
                name: {"point": float(np.mean(values)), "mean_abs": float(np.mean(np.abs(values))), "max_abs": float(np.max(np.abs(values)))}
                for name, values in interactions.items()
            },
        }

    criteria = {
        "endpoint_valid": bool(endpoint_valid), "hook_and_reproduction": bool(hook_and_reproduction),
        "A_shared": bool(A_shared), "C_interaction": bool(C_interaction), "B_format": bool(B_format),
        "geometry_to_causality": bool(geometry_to_causality), "dose_response": bool(dose_response),
        "factorial_cube_agreement": bool(factorial_cube_agreement),
    }
    if all(criteria.values()):
        decision = "positive"
        status = "completed_targeted_factorial_internal_causal_mediation_positive"
        claim = protocol["claim_scope"]["allowed_if_positive"]
    elif criteria["endpoint_valid"] and criteria["hook_and_reproduction"] and criteria["A_shared"] and (criteria["C_interaction"] or criteria["B_format"]):
        decision = "partial"
        status = "completed_targeted_factorial_internal_causal_mediation_partial"
        claim = "On this fixed development item bank, at least two predeclared factorial directions shift validated held-out internal readouts under sign-symmetric interventions; the full factorial-basis claim did not pass every frozen gate, and no behavioral or evaluation-awareness claim follows."
    else:
        decision = "negative"
        status = "completed_targeted_factorial_internal_causal_mediation_negative"
        claim = "The frozen factorial causal-mediation hypothesis did not pass its required held-out and matched-random gates."

    result = {
        "schema_version": 1, "status": status, "protocol_id": protocol["protocol_id"],
        "decision": decision, "criteria": criteria, "claim": claim,
        "prohibited_claims": protocol["claim_scope"]["prohibited"],
        "endpoint_metrics": endpoint_metrics, "reproduction": max_reproduction,
        "response_metrics": metrics_rows, "geometry_to_causality": geometry_to_causality_summary,
        "factorial_cube": cube_summary, "bootstrap": {
            "unit": protocol["uncertainty"]["unit"], "resamples": n_boot,
            "seed": protocol["uncertainty"]["bootstrap_seed"], "n_blocks": len(all_blocks),
        },
        "inputs": {
            model_id: {
                "plan_json": data["plan_json_path"].as_posix(), "plan_json_sha256": sha256_file(data["plan_json_path"]),
                "plan_npz": data["plan_npz_path"].as_posix(), "plan_npz_sha256": sha256_file(data["plan_npz_path"]),
                "raw": data["raw_path"].as_posix(), "raw_sha256": sha256_file(data["raw_path"]),
                "runner_manifest": data["runner_manifest_path"].as_posix(),
                "runner_manifest_sha256": sha256_file(data["runner_manifest_path"]),
                "items_sha256": data["plan"]["items_sha256"],
                "activation_sha256": data["plan"]["activation_sha256"],
            } for model_id, data in model_data.items()
        },
    }
    atomic_text(results_path, strict_json(result))
    fieldnames = ["model_id", "source_layer", "component", "alpha", "endpoint", "point", "ci_2.5", "ci_50", "ci_97.5", "random_p", "n_blocks"]
    tmp_metrics = metrics_path.with_name(metrics_path.name + ".tmp")
    with tmp_metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_rows)
    os.replace(tmp_metrics, metrics_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(model_data)
    figure, axes = plt.subplots(len(models), 2, figsize=(10, 4.2 * len(models)), squeeze=False)
    for row_number, model_id in enumerate(models):
        sources = source_by_model[model_id]
        for column, (depth_label, layer) in enumerate(sources.items()):
            matrix = np.asarray([
                [point_lookup[(model_id, layer, component, 1.0, endpoint)]["point"] for endpoint in ENDPOINTS]
                for component in ("A", "B", "C")
            ])
            axis = axes[row_number, column]
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-max(1e-9, np.abs(matrix).max()), vmax=max(1e-9, np.abs(matrix).max()))
            axis.set_xticks(range(3), ["Shared purpose", "Purpose × format", "Format"], rotation=20, ha="right")
            axis.set_yticks(range(3), ["A", "B", "C"])
            axis.set_title(f"{model_id.split('/')[-1]} · {depth_label} L{layer}")
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=8)
            figure.colorbar(image, ax=axis, shrink=0.75)
    figure.tight_layout()
    tmp_figure = figure_path.with_name(figure_path.stem + ".tmp" + figure_path.suffix)
    figure.savefig(tmp_figure, dpi=180)
    plt.close(figure)
    os.replace(tmp_figure, figure_path)

    manifest = {
        "schema_version": 1, "status": status, "protocol_id": protocol["protocol_id"],
        "decision": decision, "repo_commit": repo_commit(), "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path), "script_sha256": sha256_file(Path(__file__)),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (results_path, metrics_path, figure_path)
        },
        "inputs": result["inputs"], "raw_sha256_by_model": all_raw_hashes,
        "builder_script_sha256": builder_script_sha256, "runner_script_sha256": runner_script_sha256,
        "elapsed_seconds": time.time() - started,
    }
    atomic_text(manifest_path, strict_json(manifest))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, action="append", required=True)
    parser.add_argument("--plan-npz", type=Path, action="append", required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = analyze(parse_args(argv))
    print(json.dumps({"decision": result["decision"], "criteria": result["criteria"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
