#!/usr/bin/env python3
"""Run frozen sign-symmetric factorial residual-stream interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Must be set before any CUDA context is created (first torch.cuda.* call
# anywhere in the process); required for torch.use_deterministic_algorithms
# to make cuBLAS reduction order reproducible across runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SUPPORTED_PROTOCOL_IDS = {
    "pilot80_factorial_causal_mediation_v1",
    "pilot80_factorial_causal_mediation_v2_reextracted",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)




def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else float("nan")


def condition_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{
        "condition_type": "zero", "condition_id": "zero", "component": None,
        "random_seed": None, "sign": None, "alpha": None, "cube_signs": None,
    }]
    for component in protocol["interventions"]["components"]:
        for alpha in protocol["interventions"]["alphas"]:
            for sign in protocol["interventions"]["signs"]:
                sign_name = "p" if sign > 0 else "m"
                specs.append({
                    "condition_type": "component", "condition_id": f"{component}_{sign_name}_alpha{alpha:g}",
                    "component": component, "random_seed": None, "sign": int(sign),
                    "alpha": float(alpha), "cube_signs": None,
                })
    for seed in protocol["interventions"]["random_seeds"]:
        for sign in protocol["interventions"]["signs"]:
            sign_name = "p" if sign > 0 else "m"
            specs.append({
                "condition_type": "random", "condition_id": f"random_{seed}_{sign_name}_alpha1",
                "component": None, "random_seed": int(seed), "sign": int(sign),
                "alpha": 1.0, "cube_signs": None,
            })
    for sa in (-1, 1):
        for sb in (-1, 1):
            for sc in (-1, 1):
                labels = {"A": sa, "B": sb, "C": sc}
                token = "_".join(f"{name}{'p' if value > 0 else 'm'}" for name, value in labels.items())
                specs.append({
                    "condition_type": "cube", "condition_id": "cube_" + token,
                    "component": None, "random_seed": None, "sign": None,
                    "alpha": 1.0, "cube_signs": labels,
                })
    if len(specs) != 85 or len({spec["condition_id"] for spec in specs}) != len(specs):
        raise AssertionError("frozen condition enumeration must contain 85 unique conditions")
    return specs


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["model_id"], int(record["source_layer"]), record["fold_id"],
        int(record["row_index"]), record["condition_id"],
    )


def load_resume(path: Path) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    if not path.exists():
        return keys
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record_key(record)
        if key in keys:
            raise ValueError(f"duplicate existing raw record at line {line_number}: {key}")
        keys.add(key)
    return keys


def load_baseline_metrics(path: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    metrics: dict[tuple[str, int, int], dict[str, float]] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["condition_id"] != "zero":
            continue
        key = (record["fold_id"], int(record["source_layer"]), int(record["row_index"]))
        value = {
            "source_baseline_max_abs": float(record["source_baseline_max_abs"]),
            "source_baseline_cosine": float(record["source_baseline_cosine"]),
            "target_baseline_max_abs": float(record["target_baseline_max_abs"]),
            "target_baseline_cosine": float(record["target_baseline_cosine"]),
        }
        if key in metrics and metrics[key] != value:
            raise ValueError(f"conflicting resumed baseline metrics: {key}")
        metrics[key] = value
    return metrics


def resolve_layers(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise TypeError("unable to locate transformer block list")


def output_hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def readout_scores(target: np.ndarray, archive: Any, fold_id: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ("purpose_benchmark", "purpose_casual", "format"):
        prefix = f"fold_{fold_id}_readout_{name}_"
        mean = archive[prefix + "mean"]
        scale = archive[prefix + "scale"]
        coef = archive[prefix + "coef"]
        intercept = float(archive[prefix + "intercept"])
        score_sd = float(archive[prefix + "score_sd"])
        score = float(np.dot((target - mean) / scale, coef) + intercept)
        result[name] = score / score_sd
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite readout score")
    return result


def delta_for_spec(spec: dict[str, Any], archive: Any, fold_id: str, layer: int, common_scale: float) -> np.ndarray:
    prefix = f"fold_{fold_id}_layer_{layer}_"
    if spec["condition_type"] == "zero":
        return np.zeros_like(archive[prefix + "A"], dtype=np.float32)
    if spec["condition_type"] == "component":
        vector = archive[prefix + spec["component"]].astype(np.float64)
        vector /= np.linalg.norm(vector)
        return (spec["sign"] * spec["alpha"] * common_scale * vector).astype(np.float32)
    if spec["condition_type"] == "random":
        seeds = list(range(32))
        random_index = spec["random_seed"] - 20261201
        if random_index not in seeds:
            raise ValueError("random seed does not match frozen contiguous seed bank")
        vector = archive[prefix + "random"][random_index].astype(np.float64)
        vector /= np.linalg.norm(vector)
        return (spec["sign"] * common_scale * vector).astype(np.float32)
    if spec["condition_type"] == "cube":
        total = np.zeros_like(archive[prefix + "A"], dtype=np.float64)
        for component, sign in spec["cube_signs"].items():
            vector = archive[prefix + component].astype(np.float64)
            total += sign * vector / np.linalg.norm(vector)
        return (common_scale * total).astype(np.float32)
    raise ValueError(f"unknown condition type {spec['condition_type']}")


def pad_left(sequences: Sequence[Sequence[int]], pad_id: int, torch: Any, device: Any) -> tuple[Any, Any]:
    width = max(map(len, sequences))
    ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(sequences), width), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        ids[row, width - len(sequence):] = sequence
        mask[row, width - len(sequence):] = 1
    return torch.as_tensor(ids, device=device), torch.as_tensor(mask, device=device)


def encode_rows(tokenizer: Any, rows: Sequence[dict[str, Any]], indices: Iterable[int]) -> dict[int, list[int]]:
    encoded: dict[int, list[int]] = {}
    for index in indices:
        row = rows[index]
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}], tokenize=True,
            add_generation_prompt=True, add_special_tokens=False,
        )
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        encoded[index] = [int(token) for token in tokens]
    return encoded


def append_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(strict_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

def validate_plan_contract(protocol: dict[str, Any], plan: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    if plan.get("schema_version") != 1 or plan.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("plan schema/protocol mismatch")
    if plan.get("items_sha256") != protocol["dataset"]["items_sha256"]:
        raise ValueError("plan item provenance mismatch")
    if plan.get("canonical_item_ids") != [row["item_id"] for row in rows]:
        raise ValueError("plan canonical item order mismatch")
    blocks = {row["payload_block_id"] for row in rows}
    block_fold = plan.get("payload_fold_by_block", {})
    if set(block_fold) != blocks or any(value not in range(5) for value in block_fold.values()):
        raise ValueError("plan payload-fold map mismatch")
    expected_ids = {f"{family}_pf{payload_fold}" for family in protocol["dataset"]["purpose_families"] for payload_fold in range(5)}
    folds = plan.get("folds", [])
    if len(folds) != 25 or {fold.get("fold_id") for fold in folds} != expected_ids:
        raise ValueError("plan outer-fold inventory mismatch")
    model = plan["model"]
    expected_source = {str(layer): label for label, layer in model["source_layers"].items()}
    stated_indices = {index for index, row in enumerate(rows) if row["render_slot"] == "stated"}
    test_union: list[int] = []
    for fold in folds:
        fold_id = fold["fold_id"]
        family = fold["held_family"]
        payload_fold = int(fold["held_payload_fold"])
        if fold_id != f"{family}_pf{payload_fold}" or family not in protocol["dataset"]["purpose_families"] or payload_fold not in range(5):
            raise ValueError(f"invalid fold identity {fold_id}")
        source_layers = fold.get("source_layers", {})
        if set(source_layers) != set(expected_source):
            raise ValueError(f"source-layer inventory mismatch {fold_id}")
        if any(source_layers[layer].get("depth_label") != expected_source[layer] for layer in expected_source):
            raise ValueError(f"source-depth label mismatch {fold_id}")
        if set(fold.get("readouts", {})) != {"purpose_benchmark", "purpose_casual", "format"}:
            raise ValueError(f"readout inventory mismatch {fold_id}")
        groups = {}
        for name in ("test_indices", "direction_indices", "readout_indices"):
            indices = fold.get(name)
            if not isinstance(indices, list) or len(indices) != len(set(indices)):
                raise ValueError(f"invalid {name} {fold_id}")
            if any(not isinstance(index, int) or index < 0 or index >= len(rows) for index in indices):
                raise ValueError(f"out-of-range {name} {fold_id}")
            groups[name] = set(indices)
        if groups["test_indices"] & groups["direction_indices"] or groups["test_indices"] & groups["readout_indices"] or groups["direction_indices"] & groups["readout_indices"]:
            raise ValueError(f"row split overlap {fold_id}")
        block_sets = {name: {rows[index]["payload_block_id"] for index in indices} for name, indices in groups.items()}
        if block_sets["test_indices"] & block_sets["direction_indices"] or block_sets["test_indices"] & block_sets["readout_indices"] or block_sets["direction_indices"] & block_sets["readout_indices"]:
            raise ValueError(f"block split overlap {fold_id}")
        for index in groups["test_indices"]:
            row = rows[index]
            if row["render_slot"] != "stated" or row["purpose_family_id"] != family or block_fold[row["payload_block_id"]] != payload_fold:
                raise ValueError(f"invalid test row {fold_id}:{index}")
        for name in ("direction_indices", "readout_indices"):
            for index in groups[name]:
                row = rows[index]
                if row["render_slot"] != "stated" or row["purpose_family_id"] == family or block_fold[row["payload_block_id"]] == payload_fold:
                    raise ValueError(f"invalid training row {fold_id}:{name}:{index}")
        test_union.extend(fold["test_indices"])
    if len(test_union) != len(set(test_union)) or set(test_union) != stated_indices or len(test_union) != protocol["dataset"]["n_stated_rows"]:
        raise ValueError("plan test folds do not partition the frozen stated rows")


def required_plan_npz_keys(plan: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for fold in plan["folds"]:
        fold_id = fold["fold_id"]
        for layer in fold["source_layers"]:
            prefix = f"fold_{fold_id}_layer_{layer}_"
            keys.update(prefix + name for name in ("A", "B", "C", "random"))
        for readout in ("purpose_benchmark", "purpose_casual", "format"):
            prefix = f"fold_{fold_id}_readout_{readout}_"
            keys.update(prefix + name for name in ("mean", "scale", "coef", "intercept", "score_sd"))
    return keys



def expected_record_keys(plan: dict[str, Any], model_id: str, specs: Sequence[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {
        (model_id, int(layer), fold["fold_id"], int(row_index), spec["condition_id"])
        for fold in plan["folds"]
        for layer in fold["source_layers"]
        for row_index in fold["test_indices"]
        for spec in specs
    }


def validate_resumed_raw(
    path: Path,
    protocol: dict[str, Any],
    plan: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    specs: Sequence[dict[str, Any]],
    config_sha256: str,
    plan_json_sha256: str,
    plan_npz_sha256: str,
    script_sha256: str,
) -> tuple[set[tuple[Any, ...]], dict[tuple[str, int, int], dict[str, float]]]:
    if not path.exists():
        return set(), {}
    expected = expected_record_keys(plan, plan["model"]["model_id"], specs)
    spec_by_id = {spec["condition_id"]: spec for spec in specs}
    fold_by_id = {fold["fold_id"]: fold for fold in plan["folds"]}
    keys: set[tuple[Any, ...]] = set()
    baselines: dict[tuple[str, int, int], dict[str, float]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record_key(record)
        if key in keys or key not in expected:
            raise ValueError(f"invalid or duplicate resumed raw key at line {line_number}: {key}")
        keys.add(key)
        model_id, layer, fold_id, row_index, condition_id = key
        row = rows[row_index]
        fold = fold_by_id[fold_id]
        source_meta = fold["source_layers"][str(layer)]
        fixed = {
            "schema_version": 1, "protocol_id": protocol["protocol_id"],
            "model_id": plan["model"]["model_id"], "revision": plan["model"]["revision"],
            "fold_id": fold_id, "row_index": row_index, "item_id": row["item_id"],
            "payload_block_id": row["payload_block_id"], "purpose_family_id": row["purpose_family_id"],
            "intended_purpose": row["intended_purpose"], "format": row["format"],
            "source_layer": layer, "source_depth_label": source_meta["depth_label"],
            "target_layer": plan["model"]["target_layer"],
            "token_position": -1, "inference_dtype": "bfloat16",
            "common_residual_scale": float(source_meta["common_scale"]),
            "config_sha256": config_sha256, "plan_json_sha256": plan_json_sha256,
            "plan_npz_sha256": plan_npz_sha256, "script_sha256": script_sha256,
        }
        for field, value in fixed.items():
            if record.get(field) != value:
                raise ValueError(f"resumed raw metadata mismatch line {line_number}: {field}")
        spec = spec_by_id[condition_id]
        for field in ("condition_type", "condition_id", "component", "random_seed", "sign", "alpha", "cube_signs"):
            if record.get(field) != spec[field]:
                raise ValueError(f"resumed condition metadata mismatch line {line_number}: {field}")
        if record.get("source_hook_count") != 1 or record.get("target_hook_count") != 1 or record.get("hook_count") != 1:
            raise ValueError(f"resumed hook count mismatch line {line_number}")
        if record.get("source_finite") is not True or record.get("target_finite") is not True:
            raise ValueError(f"resumed finite gate mismatch line {line_number}")
        scores = record.get("readout_scores", {})
        if set(scores) != {"purpose_benchmark", "purpose_casual", "format"} or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in scores.values()
        ):
            raise ValueError(f"resumed readout schema mismatch line {line_number}")
        if (
            not isinstance(record.get("token_length"), int)
            or isinstance(record.get("token_length"), bool)
            or record["token_length"] < 1
            or not isinstance(record.get("last_prompt_token_id"), int)
            or isinstance(record.get("last_prompt_token_id"), bool)
            or not isinstance(record.get("intervention_norm"), (int, float))
            or isinstance(record.get("intervention_norm"), bool)
            or not math.isfinite(record["intervention_norm"])
            or record["intervention_norm"] < 0
        ):
            raise ValueError(f"resumed token or intervention metadata mismatch line {line_number}")
        metric_names = (
            "source_baseline_max_abs", "source_baseline_cosine",
            "target_baseline_max_abs", "target_baseline_cosine",
        )
        metric_values = {name: record.get(name) for name in metric_names}
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in metric_values.values()
        ):
            raise ValueError(f"resumed baseline metric mismatch line {line_number}")
        if condition_id == "zero":
            if (
                metric_values["source_baseline_max_abs"] > 0.125
                or metric_values["source_baseline_cosine"] < 0.999
                or metric_values["target_baseline_max_abs"] > 0.125
                or metric_values["target_baseline_cosine"] < 0.999
            ):
                raise ValueError(f"resumed baseline reproduction gate failure line {line_number}")
            baseline_key = (fold_id, layer, row_index)
            baselines[baseline_key] = {name: float(metric_values[name]) for name in metric_names}
    if any(key[-1] != "zero" and (key[2], key[1], key[3]) not in baselines for key in keys):
        raise ValueError("resumed interventions exist without their zero baseline")
    return keys, baselines


def validate_plan_arrays(plan: dict[str, Any], archive: Any) -> None:
    model = plan["model"]
    hidden = int(model["hidden_size"])
    n_random = 32
    if set(archive.files) != required_plan_npz_keys(plan):
        raise ValueError("plan NPZ key inventory mismatch")
    for fold in plan["folds"]:
        fold_id = fold["fold_id"]
        for layer, source_meta in fold["source_layers"].items():
            prefix = f"fold_{fold_id}_layer_{layer}_"
            for component in ("A", "B", "C"):
                value = archive[prefix + component]
                if value.shape != (hidden,) or value.dtype != np.float32 or not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
                    raise ValueError(f"invalid plan component {prefix + component}")
            random = archive[prefix + "random"]
            if random.shape != (n_random, hidden) or random.dtype != np.float32 or not np.isfinite(random).all():
                raise ValueError(f"invalid plan random bank {prefix}")
            if not math.isfinite(float(source_meta["common_scale"])) or float(source_meta["common_scale"]) <= 0:
                raise ValueError(f"invalid common scale {prefix}")
        for readout in ("purpose_benchmark", "purpose_casual", "format"):
            prefix = f"fold_{fold_id}_readout_{readout}_"
            for name in ("mean", "scale", "coef"):
                value = archive[prefix + name]
                if value.shape != (hidden,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid plan readout vector {prefix + name}")
            if np.any(archive[prefix + "scale"] <= 0):
                raise ValueError(f"invalid readout scale {prefix}")
            for name in ("intercept", "score_sd"):
                value = archive[prefix + name]
                if value.shape != () or not np.isfinite(value):
                    raise ValueError(f"invalid plan readout scalar {prefix + name}")
            if float(archive[prefix + "score_sd"]) <= 0:
                raise ValueError(f"invalid readout score SD {prefix}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pin deterministic GPU execution before any CUDA op runs below:
    # otherwise non-deterministic cuBLAS/cuDNN reduction order under BF16
    # accumulates into late-layer activation drift across runs, which is
    # exactly what the source_activation_reproduction gate detects.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    config_path = args.config.resolve()
    plan_json_path = args.plan_json.resolve()
    plan_npz_path = args.plan_npz.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_json_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in SUPPORTED_PROTOCOL_IDS or plan.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("protocol mismatch")
    if plan["config_sha256"] != sha256_file(config_path):
        raise ValueError("config hash mismatch")
    if plan["plan_npz_sha256"] != sha256_file(plan_npz_path):
        raise ValueError("plan NPZ hash mismatch")
    builder_path = Path(__file__).with_name("build_pilot80_factorial_causal_plan.py")
    if not builder_path.exists() or plan.get("script_sha256") != sha256_file(builder_path):
        raise ValueError("plan builder script provenance mismatch")
    configured_models = [model for model in protocol["models"] if model["model_id"] == args.model_id]
    if len(configured_models) != 1:
        raise ValueError("configured model is missing or duplicated")
    model_config = plan["model"]
    if model_config != configured_models[0]:
        raise ValueError("plan model metadata differs from the frozen config")
    activation_path = Path(model_config["activation_path_cluster"])
    if not activation_path.exists():
        activation_path = Path(model_config["activation_path"])
    if sha256_file(activation_path) != model_config["activation_sha256"]:
        raise ValueError("activation hash mismatch")
    items_path = Path(plan["items_path"])
    if not items_path.exists():
        items_path = Path(protocol["dataset"]["items_path"])
    if not items_path.exists():
        items_path = activation_path.with_name("meta.jsonl")
    if sha256_file(items_path) != plan["items_sha256"]:
        raise ValueError("item hash mismatch")
    rows = load_rows(items_path)
    if [row["item_id"] for row in rows] != plan["canonical_item_ids"]:
        raise ValueError("item order mismatch")
    validate_plan_contract(protocol, plan, rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "causal_raw.jsonl"
    complete_path = args.out_dir / "runner_manifest.json"
    invalid_path = args.out_dir / "INVALID.json"
    if complete_path.exists():
        raise FileExistsError(f"completed runner manifest already exists: {complete_path}")
    if invalid_path.exists():
        raise FileExistsError(f"invalid run marker already exists; refusing resume: {invalid_path}")
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"raw output exists without --resume: {raw_path}")
    config_sha256 = sha256_file(config_path)
    plan_json_sha256 = sha256_file(plan_json_path)
    plan_npz_sha256 = sha256_file(plan_npz_path)
    script_sha256 = sha256_file(Path(__file__))
    specs = condition_specs(protocol)
    expected_keys = expected_record_keys(plan, args.model_id, specs)
    if args.resume:
        done_keys, baseline_metrics = validate_resumed_raw(
            raw_path, protocol, plan, rows, specs, config_sha256,
            plan_json_sha256, plan_npz_sha256, script_sha256,
        )
    else:
        done_keys, baseline_metrics = set(), {}
    plan_archive = np.load(plan_npz_path, allow_pickle=False)
    validate_plan_arrays(plan, plan_archive)
    with np.load(activation_path, allow_pickle=False) as activation_archive:
        required_arrays = {"X", "item_ids", "layers", "token_lengths"}
        if required_arrays - set(activation_archive.files):
            raise ValueError("activation archive missing required arrays")
        activation_X = activation_archive["X"]
        activation_ids = activation_archive["item_ids"].astype(str).tolist()
        activation_layers = activation_archive["layers"].astype(int).tolist()
        token_lengths = activation_archive["token_lengths"]
        expected_shape = (len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"]))
        if activation_ids != plan["canonical_item_ids"] or activation_X.shape != expected_shape:
            raise ValueError("activation item order or shape mismatch")
        if activation_layers != list(range(model_config["n_layers"])) or activation_X.dtype != np.float32:
            raise ValueError("activation layer/dtype mismatch")
        if token_lengths.shape != (len(rows),) or not np.issubdtype(token_lengths.dtype, np.integer) or np.any(token_lengths <= 0):
            raise ValueError("activation token lengths are invalid")
        wanted_layers = sorted(set(model_config["source_layers"].values()) | {model_config["target_layer"]})
        stored = {layer: np.asarray(activation_X[:, layer, :], dtype=np.float32) for layer in wanted_layers}
    checkpoint = Path(model_config["checkpoint_path_cluster"])
    tokenizer_path = Path(model_config["tokenizer_path_cluster"])
    if not checkpoint.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"configured checkpoint/tokenizer missing: {checkpoint} {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer lacks pad and eos token")
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen BF16 causal run")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, revision=model_config["revision"], local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        # Pinned rather than left on transformers' auto-select, which can
        # choose SDPA or flash-attention kernels with different, run-to-run
        # non-deterministic reduction order.
        attn_implementation="eager",
    )
    model.eval()
    layers = resolve_layers(model)
    if len(layers) != model_config["n_layers"]:
        raise ValueError(f"model layer count mismatch: {len(layers)}")
    layer_device_types = {next(layer.parameters()).device.type for layer in layers}
    if layer_device_types != {"cuda"}:
        raise RuntimeError(f"all transformer blocks must be resident on CUDA, got {layer_device_types}")
    input_device = model.get_input_embeddings().weight.device
    all_test_indices = sorted({index for fold in plan["folds"] for index in fold["test_indices"]})
    encoded = encode_rows(tokenizer, rows, all_test_indices)
    for index, tokens in encoded.items():
        if len(tokens) != int(token_lengths[index]):
            raise ValueError(f"chat-template token length mismatch row {index}: {len(tokens)} != {token_lengths[index]}")
    records_written = 0
    started = time.time()
    try:
        for fold in plan["folds"]:
            fold_id = fold["fold_id"]
            for source_layer_text, source_meta in fold["source_layers"].items():
                source_layer = int(source_layer_text)
                depth_label = source_meta["depth_label"]
                common_scale = float(source_meta["common_scale"])
                pairs: list[tuple[int, dict[str, Any]]] = []
                for spec in specs:
                    for row_index in fold["test_indices"]:
                        candidate = {
                            "model_id": args.model_id, "source_layer": source_layer,
                            "fold_id": fold_id, "row_index": row_index,
                            "condition_id": spec["condition_id"],
                        }
                        if record_key(candidate) not in done_keys:
                            pairs.append((row_index, spec))
                # Baselines always precede interventions so their reproduction metrics can be copied.
                pairs.sort(key=lambda pair: (pair[1]["condition_type"] != "zero", pair[1]["condition_id"], pair[0]))
                n_zero = sum(pair[1]["condition_type"] == "zero" for pair in pairs)
                offset = 0
                while offset < len(pairs):
                    width = args.baseline_batch_size if offset < n_zero else args.batch_size
                    end = min(n_zero, offset + width) if offset < n_zero else min(len(pairs), offset + width)
                    batch = pairs[offset:end]
                    offset = end
                    sequences = [encoded[row_index] for row_index, _ in batch]
                    input_ids, attention_mask = pad_left(sequences, tokenizer.pad_token_id, torch, input_device)
                    deltas = np.stack([
                        delta_for_spec(spec, plan_archive, fold_id, source_layer, common_scale)
                        for _, spec in batch
                    ])
                    delta_tensor = torch.as_tensor(deltas, dtype=torch.float32)
                    capture: dict[str, Any] = {"source": None, "target": None, "source_count": 0, "target_count": 0}

                    def source_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                        hidden = output_hidden(output)
                        capture["source_count"] += 1
                        capture["source"] = hidden[:, -1, :].detach().float().cpu().numpy()
                        modified = hidden.clone()
                        modified[:, -1, :] = modified[:, -1, :] + delta_tensor.to(hidden.device, dtype=hidden.dtype)
                        return replace_hidden(output, modified)

                    def target_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                        hidden = output_hidden(output)
                        capture["target_count"] += 1
                        capture["target"] = hidden[:, -1, :].detach().float().cpu().numpy()
                        return output

                    source_handle = layers[source_layer].register_forward_hook(source_hook)
                    target_handle = layers[model_config["target_layer"]].register_forward_hook(target_hook)
                    try:
                        with torch.inference_mode():
                            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                    finally:
                        source_handle.remove()
                        target_handle.remove()
                    if capture["source_count"] != 1 or capture["target_count"] != 1:
                        raise RuntimeError(f"hook count violation: {capture['source_count']} {capture['target_count']}")
                    if capture["source"] is None or capture["target"] is None:
                        raise RuntimeError("hook failed to capture activations")
                    output_records = []
                    for batch_index, (row_index, spec) in enumerate(batch):
                        source_actual = capture["source"][batch_index]
                        source_expected = stored[source_layer][row_index]
                        source_max_abs = float(np.max(np.abs(source_actual - source_expected)))
                        source_cosine = cosine(source_actual, source_expected)
                        baseline_key = (fold_id, source_layer, row_index)
                        if spec["condition_type"] == "zero":
                            target_actual = capture["target"][batch_index]
                            target_expected = stored[model_config["target_layer"]][row_index]
                            target_max_abs = float(np.max(np.abs(target_actual - target_expected)))
                            target_cosine = cosine(target_actual, target_expected)
                            baseline_metrics[baseline_key] = {
                                "source_baseline_max_abs": source_max_abs,
                                "source_baseline_cosine": source_cosine,
                                "target_baseline_max_abs": target_max_abs,
                                "target_baseline_cosine": target_cosine,
                            }
                        if baseline_key not in baseline_metrics:
                            raise RuntimeError(f"missing baseline metrics before intervention {baseline_key}")
                        metrics = baseline_metrics[baseline_key]
                        if source_max_abs > 0.125 or source_cosine < 0.999:
                            raise RuntimeError(f"source activation reproduction gate failed {baseline_key}: {source_max_abs} {source_cosine}")
                        if metrics["target_baseline_max_abs"] > 0.125 or metrics["target_baseline_cosine"] < 0.999:
                            raise RuntimeError(f"target activation reproduction gate failed {baseline_key}: {metrics}")
                        row = rows[row_index]
                        record = {
                            "schema_version": 1, "protocol_id": protocol["protocol_id"],
                            "model_id": args.model_id, "revision": model_config["revision"],
                            "config_sha256": config_sha256, "plan_json_sha256": plan_json_sha256,
                            "plan_npz_sha256": plan_npz_sha256, "script_sha256": script_sha256,
                            "fold_id": fold_id, "row_index": int(row_index), "item_id": row["item_id"],
                            "payload_block_id": row["payload_block_id"], "purpose_family_id": row["purpose_family_id"],
                            "intended_purpose": row["intended_purpose"], "format": row["format"],
                            "source_layer": source_layer, "source_depth_label": depth_label,
                            "target_layer": int(model_config["target_layer"]), "token_position": -1,
                            "token_length": len(encoded[row_index]), "last_prompt_token_id": encoded[row_index][-1],
                            "inference_dtype": "bfloat16", "common_residual_scale": common_scale,
                            "intervention_norm": float(np.linalg.norm(deltas[batch_index])),
                            **spec, "hook_count": 1, "source_hook_count": 1, "target_hook_count": 1,
                            "source_baseline_max_abs": metrics["source_baseline_max_abs"],
                            "source_baseline_cosine": metrics["source_baseline_cosine"],
                            "target_baseline_max_abs": metrics["target_baseline_max_abs"],
                            "target_baseline_cosine": metrics["target_baseline_cosine"],
                            "readout_scores": readout_scores(capture["target"][batch_index], plan_archive, fold_id),
                            "source_finite": bool(np.isfinite(source_actual).all()),
                            "target_finite": bool(np.isfinite(capture["target"][batch_index]).all()),
                        }
                        if not record["source_finite"] or not record["target_finite"]:
                            raise ValueError("nonfinite activation")
                        output_records.append(record)
                        done_keys.add(record_key(record))
                    append_records(raw_path, output_records)
                    records_written += len(output_records)
    except Exception as exc:
        invalid = {
            "schema_version": 1,
            "status": "invalid_incomplete_causal_run",
            "use_for_inference": False,
            "protocol_id": protocol["protocol_id"],
            "model_id": args.model_id,
            "config_sha256": config_sha256,
            "plan_json_sha256": plan_json_sha256,
            "plan_npz_sha256": plan_npz_sha256,
            "script_sha256": script_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_path": raw_path.name,
            "raw_exists": raw_path.is_file(),
            "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
            "partial_record_count": (
                sum(1 for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip())
                if raw_path.is_file() else 0
            ),
        }
        atomic_json(invalid_path, invalid)
        raise
    finally:
        plan_archive.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final_keys, _ = validate_resumed_raw(
        raw_path, protocol, plan, rows, specs, config_sha256,
        plan_json_sha256, plan_npz_sha256, script_sha256,
    )
    if final_keys != expected_keys:
        missing = list(expected_keys - final_keys)[:5]
        extra = list(final_keys - expected_keys)[:5]
        raise RuntimeError(f"incomplete raw output: {len(final_keys)} != {len(expected_keys)} missing={missing} extra={extra}")
    model_file_hashes = {}
    for root_name, root in (("checkpoint", checkpoint), ("tokenizer", tokenizer_path)):
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            path = root / name
            if path.is_file():
                model_file_hashes[f"{root_name}/{name}"] = sha256_file(path)
    manifest = {
        "schema_version": 1, "status": "completed_uninspected_factorial_causal_raw",
        "protocol_id": protocol["protocol_id"], "model_id": args.model_id,
        "revision": model_config["revision"], "config_sha256": config_sha256,
        "plan_json_sha256": plan_json_sha256, "plan_npz_sha256": plan_npz_sha256,
        "script_sha256": script_sha256,
        "activation_sha256": model_config["activation_sha256"],
        "items_sha256": plan["items_sha256"],
        "raw_path": raw_path.name, "raw_sha256": sha256_file(raw_path), "records": len(final_keys),
        "expected_records": len(expected_keys), "records_written_this_invocation": records_written,
        "model_file_hashes": model_file_hashes,
        "elapsed_seconds": time.time() - started,
    }
    tmp = complete_path.with_name(complete_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, complete_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-npz", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--baseline-batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.baseline_batch_size < 1:
        parser.error("--batch-size and --baseline-batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        invalid_path = args.out_dir / "INVALID.json"
        if not invalid_path.exists():
            raw_path = args.out_dir / "causal_raw.jsonl"
            atomic_json(invalid_path, {
                "schema_version": 1,
                "status": "invalid_incomplete_causal_run",
                "use_for_inference": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_path": raw_path.name,
                "raw_exists": raw_path.is_file(),
                "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
                "partial_record_count": (
                    sum(1 for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip())
                    if raw_path.is_file() else 0
                ),
            })
        raise
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
