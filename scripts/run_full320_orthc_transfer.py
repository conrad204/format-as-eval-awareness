#!/usr/bin/env python3
"""Run the frozen pilot-trained orthogonalized-C transfer intervention."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts import run_pilot80_factorial_causal as base

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PROTOCOL_ID = "full320_orthogonalized_c_transfer_v1"


def condition_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    settings = protocol["interventions"]
    specs: list[dict[str, Any]] = [{
        "condition_type": "zero", "condition_id": "zero", "component": None,
        "random_seed": None, "sign": None, "alpha": None, "cube_signs": None,
    }]
    for alpha in settings["alphas"]:
        for sign in settings["signs"]:
            token = "p" if sign > 0 else "m"
            specs.append({
                "condition_type": "component", "condition_id": f"C_{token}_alpha{alpha:g}",
                "component": "C", "random_seed": None, "sign": int(sign),
                "alpha": float(alpha), "cube_signs": None,
            })
    for seed in settings["random_seeds"]:
        for sign in settings["signs"]:
            token = "p" if sign > 0 else "m"
            specs.append({
                "condition_type": "random", "condition_id": f"random_{seed}_{token}_alpha1",
                "component": None, "random_seed": int(seed), "sign": int(sign),
                "alpha": 1.0, "cube_signs": None,
            })
    expected = 1 + 2 * len(settings["alphas"]) + 2 * len(settings["random_seeds"])
    if len(specs) != expected or len({spec["condition_id"] for spec in specs}) != expected:
        raise AssertionError("transfer condition enumeration mismatch")
    return specs


def delta_for_spec(
    spec: dict[str, Any],
    archive: Any,
    fold_id: str,
    layer: int,
    common_scale: float,
    random_seeds: Sequence[int],
) -> np.ndarray:
    prefix = f"fold_{fold_id}_layer_{layer}_"
    if spec["condition_type"] == "zero":
        return np.zeros_like(archive[prefix + "C"], dtype=np.float32)
    if spec["condition_type"] == "component":
        vector = archive[prefix + "C"].astype(np.float64)
        vector /= np.linalg.norm(vector)
        return (spec["sign"] * spec["alpha"] * common_scale * vector).astype(np.float32)
    if spec["condition_type"] == "random":
        index = list(map(int, random_seeds)).index(int(spec["random_seed"]))
        vector = archive[prefix + "random"][index].astype(np.float64)
        vector /= np.linalg.norm(vector)
        return (spec["sign"] * common_scale * vector).astype(np.float32)
    raise ValueError(f"unknown transfer condition: {spec['condition_type']}")


def validate_arrays(plan: dict[str, Any], archive: Any) -> None:
    expected: set[str] = set()
    hidden = int(plan["model"]["hidden_size"])
    for fold in plan["folds"]:
        fold_id = fold["fold_id"]
        for layer in fold["source_layers"]:
            prefix = f"fold_{fold_id}_layer_{layer}_"
            expected.update({prefix + "C", prefix + "random"})
            c = archive[prefix + "C"]
            random = archive[prefix + "random"]
            if c.shape != (hidden,) or c.dtype != np.float32 or not np.isfinite(c).all() or np.linalg.norm(c) <= 0:
                raise ValueError(f"invalid C array: {prefix}")
            if random.shape != (32, hidden) or random.dtype != np.float32 or not np.isfinite(random).all():
                raise ValueError(f"invalid random array: {prefix}")
        for readout in ("purpose_benchmark", "purpose_casual", "format"):
            prefix = f"fold_{fold_id}_readout_{readout}_"
            for name in ("mean", "scale", "coef", "intercept", "score_sd"):
                expected.add(prefix + name)
            for name in ("mean", "scale", "coef"):
                value = archive[prefix + name]
                if value.shape != (hidden,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid readout vector: {prefix + name}")
            if np.any(archive[prefix + "scale"] <= 0) or float(archive[prefix + "score_sd"]) <= 0:
                raise ValueError(f"invalid readout scale: {prefix}")
    if set(archive.files) != expected:
        raise ValueError("transfer plan NPZ key inventory mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    config_path = args.config.resolve()
    plan_json_path = args.plan_json.resolve()
    plan_npz_path = args.plan_npz.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_json_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID or plan.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("transfer protocol mismatch")
    if plan["config_sha256"] != base.sha256_file(config_path) or plan["plan_npz_sha256"] != base.sha256_file(plan_npz_path):
        raise ValueError("transfer config or plan hash mismatch")
    builder_path = Path(__file__).with_name("build_full320_orthc_transfer_plan.py")
    if plan.get("script_sha256") != base.sha256_file(builder_path):
        raise ValueError("transfer plan builder provenance mismatch")
    configured = [model for model in protocol["models"] if model["model_id"] == args.model_id]
    if len(configured) != 1:
        raise ValueError("configured model is missing or duplicated")
    configured_model = {key: value for key, value in configured[0].items() if key != "source_plan"}
    model_config = plan["model"]
    if model_config != configured_model:
        raise ValueError("transfer plan model metadata differs from config")

    activation_path = Path(model_config["activation_path_cluster"])
    if not activation_path.exists():
        activation_path = Path(model_config["activation_path"])
    items_path = Path(plan["items_path"])
    if not items_path.exists():
        items_path = Path(protocol["dataset"]["items_path"])
    if base.sha256_file(activation_path) != model_config["activation_sha256"] or base.sha256_file(items_path) != plan["items_sha256"]:
        raise ValueError("transfer activation or item hash mismatch")
    rows = base.load_rows(items_path)
    if [row["item_id"] for row in rows] != plan["canonical_item_ids"]:
        raise ValueError("transfer item order mismatch")
    base.validate_plan_contract(protocol, plan, rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "causal_raw.jsonl"
    complete_path = args.out_dir / "runner_manifest.json"
    invalid_path = args.out_dir / "INVALID.json"
    if complete_path.exists() or invalid_path.exists():
        raise FileExistsError("completed or invalid transfer marker already exists")
    if raw_path.exists() and not args.resume:
        raise FileExistsError("partial transfer raw output requires --resume")

    config_sha256 = base.sha256_file(config_path)
    plan_json_sha256 = base.sha256_file(plan_json_path)
    plan_npz_sha256 = base.sha256_file(plan_npz_path)
    script_sha256 = base.sha256_file(Path(__file__))
    engine_sha256 = base.sha256_file(Path(base.__file__))
    specs = condition_specs(protocol)
    expected_keys = base.expected_record_keys(plan, args.model_id, specs)
    if args.resume:
        done_keys, baseline_metrics = base.validate_resumed_raw(
            raw_path, protocol, plan, rows, specs, config_sha256,
            plan_json_sha256, plan_npz_sha256, script_sha256,
        )
    else:
        done_keys, baseline_metrics = set(), {}

    plan_archive = np.load(plan_npz_path, allow_pickle=False)
    validate_arrays(plan, plan_archive)
    with np.load(activation_path, allow_pickle=False) as activation_archive:
        activation_x = activation_archive["X"]
        token_lengths = activation_archive["token_lengths"]
        if (
            activation_archive["item_ids"].astype(str).tolist() != plan["canonical_item_ids"]
            or activation_x.shape != (len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"]))
            or activation_archive["layers"].astype(int).tolist() != list(range(int(model_config["n_layers"])))
            or activation_x.dtype != np.float32
            or token_lengths.shape != (len(rows),)
        ):
            raise ValueError("transfer activation archive contract mismatch")
        wanted = sorted(set(model_config["source_layers"].values()) | {model_config["target_layer"]})
        stored = {layer: np.asarray(activation_x[:, layer, :], dtype=np.float32) for layer in wanted}

    checkpoint = Path(model_config["checkpoint_path_cluster"])
    tokenizer_path = Path(model_config["tokenizer_path_cluster"])
    if not checkpoint.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("configured checkpoint or tokenizer missing")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer lacks pad and eos tokens")
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transfer intervention")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        revision=model_config["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    layers = base.resolve_layers(model)
    if len(layers) != model_config["n_layers"]:
        raise ValueError("model layer count mismatch")
    if {next(layer.parameters()).device.type for layer in layers} != {"cuda"}:
        raise RuntimeError("all transformer blocks must be on CUDA")
    input_device = model.get_input_embeddings().weight.device
    all_test = sorted({index for fold in plan["folds"] for index in fold["test_indices"]})
    encoded = base.encode_rows(tokenizer, rows, all_test)
    for index, tokens in encoded.items():
        if len(tokens) != int(token_lengths[index]):
            raise ValueError(f"token length mismatch row {index}")

    records_written = 0
    started = time.time()
    try:
        for fold in plan["folds"]:
            fold_id = fold["fold_id"]
            for source_layer_text, source_meta in fold["source_layers"].items():
                source_layer = int(source_layer_text)
                common_scale = float(source_meta["common_scale"])
                pairs = [
                    (row_index, spec)
                    for spec in specs
                    for row_index in fold["test_indices"]
                    if base.record_key({"model_id": args.model_id, "source_layer": source_layer, "fold_id": fold_id, "row_index": row_index, "condition_id": spec["condition_id"]}) not in done_keys
                ]
                pairs.sort(key=lambda pair: (pair[1]["condition_type"] != "zero", pair[1]["condition_id"], pair[0]))
                n_zero = sum(spec["condition_type"] == "zero" for _, spec in pairs)
                offset = 0
                while offset < len(pairs):
                    width = args.baseline_batch_size if offset < n_zero else args.batch_size
                    end = min(n_zero, offset + width) if offset < n_zero else min(len(pairs), offset + width)
                    batch = pairs[offset:end]
                    offset = end
                    input_ids, attention_mask = base.pad_left([encoded[index] for index, _ in batch], tokenizer.pad_token_id, torch, input_device)
                    deltas = np.stack([
                        delta_for_spec(
                            spec,
                            plan_archive,
                            fold_id,
                            source_layer,
                            common_scale,
                            protocol["interventions"]["random_seeds"],
                        )
                        for _, spec in batch
                    ])
                    delta_tensor = torch.as_tensor(deltas, dtype=torch.float32)
                    capture: dict[str, Any] = {"source": None, "target": None, "source_count": 0, "target_count": 0}

                    def source_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                        hidden = base.output_hidden(output)
                        capture["source_count"] += 1
                        capture["source"] = hidden[:, -1, :].detach().float().cpu().numpy()
                        modified = hidden.clone()
                        modified[:, -1, :] = modified[:, -1, :] + delta_tensor.to(hidden.device, dtype=hidden.dtype)
                        return base.replace_hidden(output, modified)

                    def target_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                        hidden = base.output_hidden(output)
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
                        raise RuntimeError("hook count violation")
                    output_records = []
                    for batch_index, (row_index, spec) in enumerate(batch):
                        source_actual = capture["source"][batch_index]
                        source_expected = stored[source_layer][row_index]
                        source_max_abs = float(np.max(np.abs(source_actual - source_expected)))
                        source_cosine = base.cosine(source_actual, source_expected)
                        baseline_key = (fold_id, source_layer, row_index)
                        if spec["condition_type"] == "zero":
                            target_actual = capture["target"][batch_index]
                            target_expected = stored[model_config["target_layer"]][row_index]
                            baseline_metrics[baseline_key] = {
                                "source_baseline_max_abs": source_max_abs,
                                "source_baseline_cosine": source_cosine,
                                "target_baseline_max_abs": float(np.max(np.abs(target_actual - target_expected))),
                                "target_baseline_cosine": base.cosine(target_actual, target_expected),
                            }
                        if baseline_key not in baseline_metrics:
                            raise RuntimeError("missing zero baseline before intervention")
                        metrics = baseline_metrics[baseline_key]
                        if (
                            source_max_abs > 0.125
                            or source_cosine < 0.999
                            or metrics["target_baseline_max_abs"] > 0.125
                            or metrics["target_baseline_cosine"] < 0.999
                        ):
                            raise RuntimeError(f"activation reproduction gate failed {baseline_key}: {source_max_abs} {source_cosine} {metrics}")
                        row = rows[row_index]
                        record = {
                            "schema_version": 1,
                            "protocol_id": PROTOCOL_ID,
                            "model_id": args.model_id,
                            "revision": model_config["revision"],
                            "config_sha256": config_sha256,
                            "plan_json_sha256": plan_json_sha256,
                            "plan_npz_sha256": plan_npz_sha256,
                            "script_sha256": script_sha256,
                            "engine_sha256": engine_sha256,
                            "fold_id": fold_id,
                            "row_index": int(row_index),
                            "item_id": row["item_id"],
                            "payload_block_id": row["payload_block_id"],
                            "purpose_family_id": row["purpose_family_id"],
                            "intended_purpose": row["intended_purpose"],
                            "format": row["format"],
                            "source_layer": source_layer,
                            "source_depth_label": source_meta["depth_label"],
                            "target_layer": int(model_config["target_layer"]),
                            "token_position": -1,
                            "token_length": len(encoded[row_index]),
                            "last_prompt_token_id": encoded[row_index][-1],
                            "inference_dtype": "bfloat16",
                            "common_residual_scale": common_scale,
                            "intervention_norm": float(np.linalg.norm(deltas[batch_index])),
                            **spec,
                            "hook_count": 1,
                            "source_hook_count": 1,
                            "target_hook_count": 1,
                            **metrics,
                            "readout_scores": base.readout_scores(capture["target"][batch_index], plan_archive, fold_id),
                            "source_finite": bool(np.isfinite(source_actual).all()),
                            "target_finite": bool(np.isfinite(capture["target"][batch_index]).all()),
                        }
                        if not record["source_finite"] or not record["target_finite"]:
                            raise ValueError("nonfinite transfer activation")
                        output_records.append(record)
                        done_keys.add(base.record_key(record))
                    base.append_records(raw_path, output_records)
                    records_written += len(output_records)
    except Exception as exc:
        base.atomic_json(invalid_path, {
            "schema_version": 1,
            "status": "invalid_incomplete_full320_orthc_transfer",
            "use_for_inference": False,
            "protocol_id": PROTOCOL_ID,
            "model_id": args.model_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_path": raw_path.name,
            "raw_exists": raw_path.is_file(),
            "raw_sha256": base.sha256_file(raw_path) if raw_path.is_file() else None,
            "partial_record_count": sum(1 for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()) if raw_path.is_file() else 0,
        })
        raise
    finally:
        plan_archive.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_keys, _ = base.validate_resumed_raw(
        raw_path, protocol, plan, rows, specs, config_sha256,
        plan_json_sha256, plan_npz_sha256, script_sha256,
    )
    if final_keys != expected_keys:
        raise RuntimeError(f"incomplete transfer raw output: {len(final_keys)} != {len(expected_keys)}")
    model_file_hashes = {}
    for root_name, root in (("checkpoint", checkpoint), ("tokenizer", tokenizer_path)):
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            path = root / name
            if path.is_file():
                model_file_hashes[f"{root_name}/{name}"] = base.sha256_file(path)
    manifest = {
        "schema_version": 1,
        "status": "completed_uninspected_full320_orthc_transfer_raw",
        "protocol_id": PROTOCOL_ID,
        "model_id": args.model_id,
        "revision": model_config["revision"],
        "config_sha256": config_sha256,
        "plan_json_sha256": plan_json_sha256,
        "plan_npz_sha256": plan_npz_sha256,
        "script_sha256": script_sha256,
        "engine_sha256": engine_sha256,
        "activation_sha256": model_config["activation_sha256"],
        "items_sha256": plan["items_sha256"],
        "raw_path": raw_path.name,
        "raw_sha256": base.sha256_file(raw_path),
        "records": len(final_keys),
        "expected_records": len(expected_keys),
        "records_written_this_invocation": records_written,
        "model_file_hashes": model_file_hashes,
        "elapsed_seconds": time.time() - started,
    }
    base.atomic_json(complete_path, manifest)
    return manifest




def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-npz", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--baseline-batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size != 1 or args.baseline_batch_size != 1:
        parser.error("transfer protocol requires batch-size=1 and baseline-batch-size=1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        invalid = args.out_dir / "INVALID.json"
        if not invalid.exists():
            raw = args.out_dir / "causal_raw.jsonl"
            base.atomic_json(invalid, {
                "schema_version": 1,
                "status": "invalid_incomplete_full320_orthc_transfer",
                "use_for_inference": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_exists": raw.is_file(),
                "raw_sha256": base.sha256_file(raw) if raw.is_file() else None,
            })
        raise
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
