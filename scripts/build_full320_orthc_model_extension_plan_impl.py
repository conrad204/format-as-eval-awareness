#!/usr/bin/env python3
"""Bind pilot80-trained orthogonalized-C plans to a frozen full320 transfer bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROTOCOL_ID = "full320_orthogonalized_c_transfer_model_extension_v1"
FOLD_DOMAIN = "full320_orthc_transfer_payload_fold_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def payload_folds(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    topic_by_block: dict[str, str] = {}
    for row in rows:
        block = row["payload_block_id"]
        topic = row["topic_id"]
        if block in topic_by_block and topic_by_block[block] != topic:
            raise ValueError(f"payload block spans topics: {block}")
        topic_by_block[block] = topic
    by_topic: dict[str, list[str]] = defaultdict(list)
    for block, topic in topic_by_block.items():
        by_topic[topic].append(block)
    if len(by_topic) != 16 or any(len(blocks) != 5 for blocks in by_topic.values()):
        raise ValueError("transfer bank must contain exactly five blocks in each of 16 topics")
    result: dict[str, int] = {}
    for topic, blocks in sorted(by_topic.items()):
        for fold, block in enumerate(sorted(blocks, key=lambda value: stable_hash(FOLD_DOMAIN, topic, value))):
            result[block] = fold
    return result


def required_keys(folds: Sequence[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for fold in folds:
        fold_id = fold["fold_id"]
        for layer in fold["source_layers"]:
            prefix = f"fold_{fold_id}_layer_{layer}_"
            keys.update({prefix + "C", prefix + "random"})
        for readout in ("purpose_benchmark", "purpose_casual", "format"):
            prefix = f"fold_{fold_id}_readout_{readout}_"
            keys.update(prefix + name for name in ("mean", "scale", "coef", "intercept", "score_sd"))
    return keys


def build(
    config_path: Path,
    source_plan_json: Path,
    source_plan_npz: Path,
    model_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol")
    models = [model for model in protocol["models"] if model["model_id"] == model_id]
    if len(models) != 1:
        raise ValueError("model is missing or duplicated in protocol")
    model = models[0]
    source_binding = model["source_plan"]
    if sha256_file(source_plan_json) != source_binding["json_sha256"]:
        raise ValueError("source plan JSON hash mismatch")
    if sha256_file(source_plan_npz) != source_binding["npz_sha256"]:
        raise ValueError("source plan NPZ hash mismatch")
    source = json.loads(source_plan_json.read_text(encoding="utf-8"))
    if source["model"]["model_id"] != model_id or source.get("experiment_tag") != "prospective_model_extension_C_residualized_against_A_B":
        raise ValueError("source plan is not the declared model-specific orthogonalized-C plan")

    items_path = Path(protocol["dataset"]["items_path_cluster"])
    if not items_path.exists():
        items_path = Path(protocol["dataset"]["items_path"])
    activation_path = Path(model["activation_path_cluster"])
    if not activation_path.exists():
        activation_path = Path(model["activation_path"])
    if sha256_file(items_path) != protocol["dataset"]["items_sha256"]:
        raise ValueError("transfer items hash mismatch")
    if sha256_file(activation_path) != model["activation_sha256"]:
        raise ValueError("transfer activation hash mismatch")
    rows = load_jsonl(items_path)
    if len(rows) != 2880 or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("transfer row inventory mismatch")
    stated = [row["render_slot"] == "stated" for row in rows]
    if sum(stated) != 1920:
        raise ValueError("transfer stated-row inventory mismatch")
    block_fold = payload_folds(rows)

    source_folds = {fold["fold_id"]: fold for fold in source["folds"]}
    families = protocol["dataset"]["purpose_families"]
    folds: list[dict[str, Any]] = []
    for family in families:
        for held_payload_fold in range(5):
            fold_id = f"{family}_pf{held_payload_fold}"
            source_fold = source_folds[fold_id]
            test_indices = [
                index for index, row in enumerate(rows)
                if stated[index]
                and row["purpose_family_id"] == family
                and block_fold[row["payload_block_id"]] == held_payload_fold
            ]
            if not test_indices:
                raise ValueError(f"empty transfer test fold: {fold_id}")
            folds.append({
                "fold_id": fold_id,
                "held_family": family,
                "held_payload_fold": held_payload_fold,
                "test_indices": test_indices,
                "direction_indices": [],
                "readout_indices": [],
                "source_layers": source_fold["source_layers"],
                "readouts": source_fold["readouts"],
                "source_plan_fold_id": fold_id,
            })
    test_union = [index for fold in folds for index in fold["test_indices"]]
    stated_indices = [index for index, value in enumerate(stated) if value]
    if sorted(test_union) != stated_indices or len(test_union) != len(set(test_union)):
        raise ValueError("transfer folds do not partition stated rows")

    arrays: dict[str, np.ndarray] = {}
    with np.load(source_plan_npz, allow_pickle=False) as archive:
        for key in sorted(required_keys(folds)):
            if key not in archive.files:
                raise ValueError(f"source plan missing required array: {key}")
            arrays[key] = np.asarray(archive[key])
    hidden = int(model["hidden_size"])
    for fold in folds:
        fold_id = fold["fold_id"]
        for layer in fold["source_layers"]:
            prefix = f"fold_{fold_id}_layer_{layer}_"
            if arrays[prefix + "C"].shape != (hidden,) or arrays[prefix + "C"].dtype != np.float32:
                raise ValueError(f"invalid C array: {prefix}")
            if arrays[prefix + "random"].shape != (32, hidden) or arrays[prefix + "random"].dtype != np.float32:
                raise ValueError(f"invalid random array: {prefix}")

    out_dir.mkdir(parents=True, exist_ok=False)
    plan_npz = out_dir / "causal_plan.npz"
    plan_json = out_dir / "causal_plan.json"
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".npz", delete=False) as handle:
        temp_npz = Path(handle.name)
    try:
        np.savez(temp_npz, **arrays)
        os.replace(temp_npz, plan_npz)
    finally:
        temp_npz.unlink(missing_ok=True)
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "items_path": items_path.as_posix(),
        "items_sha256": sha256_file(items_path),
        "activation_path": activation_path.as_posix(),
        "activation_sha256": sha256_file(activation_path),
        "canonical_item_ids": [row["item_id"] for row in rows],
        "payload_fold_by_block": block_fold,
        "folds": folds,
        "model": {key: value for key, value in model.items() if key != "source_plan"},
        "source_plan": {
            "json_path": source_plan_json.as_posix(),
            "json_sha256": sha256_file(source_plan_json),
            "npz_path": source_plan_npz.as_posix(),
            "npz_sha256": sha256_file(source_plan_npz),
            "scientific_role": "pilot80-trained orthogonalized-C directions, random controls, scales, and readouts copied without refitting",
        },
        "plan_npz": plan_npz.name,
        "plan_npz_sha256": sha256_file(plan_npz),
    }
    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-plan-json", type=Path, required=True)
    parser.add_argument("--source-plan-npz", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build(args.config, args.source_plan_json, args.source_plan_npz, args.model_id, args.out_dir)
    print(json.dumps({"model_id": args.model_id, "folds": len(plan["folds"]), "test_rows": sum(len(fold["test_indices"]) for fold in plan["folds"]), "plan_npz_sha256": plan["plan_npz_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
