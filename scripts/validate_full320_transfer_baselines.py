#!/usr/bin/env python3
"""Check full320 source/target reproduction without computing intervention outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts import run_pilot80_factorial_causal as base

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PROTOCOL_ID = "full320_orthogonalized_c_transfer_v1"
SELECTION_DOMAIN = "full320_orthc_transfer_baseline_shakedown_v1"


def stable_hash(item_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_DOMAIN}\x1f{item_id}".encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID or plan.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("shakedown protocol mismatch")
    if plan["config_sha256"] != base.sha256_file(args.config) or plan["plan_npz_sha256"] != base.sha256_file(args.plan_npz):
        raise ValueError("shakedown config or plan hash mismatch")
    model_config = plan["model"]
    if model_config["model_id"] != args.model_id:
        raise ValueError("shakedown model mismatch")
    items_path = Path(plan["items_path"])
    activation_path = Path(model_config["activation_path_cluster"])
    rows = base.load_rows(items_path)
    if [row["item_id"] for row in rows] != plan["canonical_item_ids"]:
        raise ValueError("shakedown item order mismatch")
    test_indices = sorted(
        {index for fold in plan["folds"] for index in fold["test_indices"]},
        key=lambda index: stable_hash(rows[index]["item_id"]),
    )[: args.rows]
    if len(test_indices) != args.rows:
        raise ValueError("insufficient shakedown rows")
    source_layers = sorted(set(model_config["source_layers"].values()))
    wanted_layers = source_layers + [int(model_config["target_layer"])]
    with np.load(activation_path, allow_pickle=False) as archive:
        if archive["item_ids"].astype(str).tolist() != plan["canonical_item_ids"]:
            raise ValueError("shakedown activation order mismatch")
        token_lengths = archive["token_lengths"].astype(int)
        activation_x = archive["X"]
        stored = {
            layer: np.asarray(activation_x[test_indices, layer, :], dtype=np.float32)
            for layer in wanted_layers
        }

    checkpoint = Path(model_config["checkpoint_path_cluster"])
    tokenizer_path = Path(model_config["tokenizer_path_cluster"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    encoded = base.encode_rows(tokenizer, rows, test_indices)
    input_device = model.get_input_embeddings().weight.device
    comparisons = []
    for sample_index, row_index in enumerate(test_indices):
        if len(encoded[row_index]) != int(token_lengths[row_index]):
            raise ValueError(f"shakedown token mismatch row {row_index}")
        for source_layer in source_layers:
            capture: dict[str, Any] = {"source": None, "target": None, "source_count": 0, "target_count": 0}

            def source_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = base.output_hidden(output)
                capture["source_count"] += 1
                capture["source"] = hidden[:, -1, :].detach().float().cpu().numpy()[0]
                return output

            def target_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = base.output_hidden(output)
                capture["target_count"] += 1
                capture["target"] = hidden[:, -1, :].detach().float().cpu().numpy()[0]
                return output

            source_handle = layers[source_layer].register_forward_hook(source_hook)
            target_handle = layers[model_config["target_layer"]].register_forward_hook(target_hook)
            try:
                input_ids = torch.tensor([encoded[row_index]], dtype=torch.long, device=input_device)
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            finally:
                source_handle.remove()
                target_handle.remove()
            if capture["source_count"] != 1 or capture["target_count"] != 1:
                raise RuntimeError("shakedown hook count violation")
            source_max_abs = float(np.max(np.abs(capture["source"] - stored[source_layer][sample_index])))
            target_max_abs = float(np.max(np.abs(capture["target"] - stored[model_config["target_layer"]][sample_index])))
            source_cosine = base.cosine(capture["source"], stored[source_layer][sample_index])
            target_cosine = base.cosine(capture["target"], stored[model_config["target_layer"]][sample_index])
            comparisons.append({
                "row_index": int(row_index),
                "item_id": rows[row_index]["item_id"],
                "source_layer": int(source_layer),
                "target_layer": int(model_config["target_layer"]),
                "source_max_abs": source_max_abs,
                "source_cosine": source_cosine,
                "target_max_abs": target_max_abs,
                "target_cosine": target_cosine,
            })
    passed = all(
        record["source_max_abs"] <= 0.125
        and record["target_max_abs"] <= 0.125
        and record["source_cosine"] >= 0.999
        and record["target_cosine"] >= 0.999
        for record in comparisons
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scientific_outcomes_computed": False,
        "protocol_id": PROTOCOL_ID,
        "selection_domain": SELECTION_DOMAIN,
        "model_id": args.model_id,
        "config_sha256": base.sha256_file(args.config),
        "plan_json_sha256": base.sha256_file(args.plan_json),
        "plan_npz_sha256": base.sha256_file(args.plan_npz),
        "script_sha256": base.sha256_file(Path(__file__)),
        "n_rows": len(test_indices),
        "n_comparisons": len(comparisons),
        "gate": {"max_abs_lte": 0.125, "cosine_gte": 0.999},
        "maximum_source_max_abs": max(record["source_max_abs"] for record in comparisons),
        "maximum_target_max_abs": max(record["target_max_abs"] for record in comparisons),
        "minimum_source_cosine": min(record["source_cosine"] for record in comparisons),
        "minimum_target_cosine": min(record["target_cosine"] for record in comparisons),
        "comparisons": comparisons,
    }
    atomic_json(args.out, result)
    if not passed:
        raise RuntimeError("transfer baseline shakedown gate failed")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-npz", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("config", "plan_json", "plan_npz"):
        setattr(args, name, getattr(args, name).resolve())
    args.out = args.out.resolve()
    if args.rows != 8:
        parser.error("frozen shakedown requires exactly eight rows")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps({key: result[key] for key in ("status", "model_id", "n_comparisons", "maximum_source_max_abs", "maximum_target_max_abs", "minimum_source_cosine", "minimum_target_cosine")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
