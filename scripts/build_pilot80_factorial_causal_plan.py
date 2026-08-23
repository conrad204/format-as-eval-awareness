#!/usr/bin/env python3
"""Build the immutable cross-fitted plan for factorial A/B/C interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def strict_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(strict_json(value), encoding="utf-8")
    os.replace(tmp, path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {
        "item_id", "payload_block_id", "purpose_pair_id", "purpose_family_id", "topic_id",
        "render_slot", "intended_purpose", "format", "text",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("item_id values are not unique")
    return rows


def validate_groups(rows: Sequence[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_blocks = {row["payload_block_id"] for row in rows}
    stated_blocks: set[str] = set()
    for row in rows:
        if row["render_slot"] == "stated":
            if row["intended_purpose"] not in {"evaluation", "deployment"}:
                raise ValueError(f"invalid stated purpose: {row['intended_purpose']}")
            if row["format"] not in {"benchmark", "casual"}:
                raise ValueError(f"invalid stated format: {row['format']}")
            stated_blocks.add(row["payload_block_id"])
            groups.setdefault((row["payload_block_id"], row["purpose_pair_id"]), []).append(row)
    if stated_blocks != all_blocks:
        raise ValueError("unstated-only payload blocks would alter the payload-fold assignment")
    expected = {
        ("evaluation", "benchmark"), ("evaluation", "casual"),
        ("deployment", "benchmark"), ("deployment", "casual"),
    }
    for key, group in groups.items():
        cells = [(row["intended_purpose"], row["format"]) for row in group]
        if len(group) != 4 or set(cells) != expected or len(set(cells)) != 4:
            raise ValueError(f"incomplete or duplicate stated 2x2 group {key}: {cells}")
        for field in ("purpose_family_id", "topic_id"):
            if len({row[field] for row in group}) != 1:
                raise ValueError(f"stated 2x2 group {key} spans {field}")


def payload_folds(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    topic_by_block: dict[str, str] = {}
    for row in rows:
        block, topic = row["payload_block_id"], row["topic_id"]
        if block in topic_by_block and topic_by_block[block] != topic:
            raise ValueError(f"payload block {block} spans topics")
        topic_by_block[block] = topic
    by_topic: dict[str, list[str]] = {}
    for block, topic in topic_by_block.items():
        by_topic.setdefault(topic, []).append(block)
    result: dict[str, int] = {}
    for topic, blocks in by_topic.items():
        ordered = sorted(blocks, key=lambda block: stable_hash("factorial_causal_payload_fold_v1", topic, block))
        for rank, block in enumerate(ordered):
            result[block] = rank % 5
    return result


def role_blocks(
    rows: Sequence[dict[str, Any]], fold_id: str, held_payload_fold: int, block_fold: dict[str, int]
) -> tuple[set[str], set[str]]:
    topic_by_block = {row["payload_block_id"]: row["topic_id"] for row in rows}
    by_topic: dict[str, list[str]] = {}
    for block, topic in topic_by_block.items():
        if block_fold[block] != held_payload_fold:
            by_topic.setdefault(topic, []).append(block)
    direction: set[str] = set()
    readout: set[str] = set()
    for topic, blocks in by_topic.items():
        ordered = sorted(blocks, key=lambda block: stable_hash("factorial_causal_role_v1", fold_id, topic, block))
        for rank, block in enumerate(ordered):
            (direction if rank % 2 == 0 else readout).add(block)
    if not direction or not readout or direction & readout:
        raise ValueError(f"invalid role split for {fold_id}")
    return direction, readout


def component_arrays(X: np.ndarray, rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    grouped: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for index in indices:
        row = rows[index]
        key = (row["payload_block_id"], row["purpose_pair_id"])
        cell = (row["intended_purpose"], row["format"])
        if cell in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate training cell {key} {cell}")
        grouped[key][cell] = index
    expected = {
        ("evaluation", "benchmark"), ("evaluation", "casual"),
        ("deployment", "benchmark"), ("deployment", "casual"),
    }
    effects: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for key, cells in grouped.items():
        if set(cells) != expected:
            raise ValueError(f"direction split broke group {key}")
        eb = X[cells[("evaluation", "benchmark")]].astype(np.float64)
        ec = X[cells[("evaluation", "casual")]].astype(np.float64)
        db = X[cells[("deployment", "benchmark")]].astype(np.float64)
        dc = X[cells[("deployment", "casual")]].astype(np.float64)
        effects.append(((eb + ec - db - dc) / 4, (eb + db - ec - dc) / 4, (eb - ec - db + dc) / 4))
    if not effects:
        raise ValueError("no complete direction groups")
    A = np.mean([effect[0] for effect in effects], axis=0)
    B = np.mean([effect[1] for effect in effects], axis=0)
    C = np.mean([effect[2] for effect in effects], axis=0)
    common_scale = float(np.median([np.linalg.norm(effect[0]) for effect in effects]))
    if not np.isfinite(common_scale) or common_scale <= 0:
        raise ValueError("invalid common residual scale")
    for name, vector in (("A", A), ("B", B), ("C", C)):
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"invalid {name} direction")
    return A.astype(np.float32), B.astype(np.float32), C.astype(np.float32), common_scale, len(effects)


def random_directions(A: np.ndarray, B: np.ndarray, C: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
    basis = np.stack([A, B, C]).astype(np.float64).T
    q, r = np.linalg.qr(basis)
    rank = int(np.sum(np.abs(np.diag(r)) > np.finfo(np.float64).eps * max(basis.shape) * abs(r).max()))
    q = q[:, :rank]
    outputs = []
    for seed in seeds:
        vector = np.random.default_rng(seed).standard_normal(A.shape[0])
        if rank:
            vector -= q @ (q.T @ vector)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"degenerate random direction seed {seed}")
        vector /= norm
        if rank and float(np.max(np.abs(q.T @ vector))) > 1e-10:
            raise ValueError(f"random direction seed {seed} is not orthogonal")
        outputs.append(vector.astype(np.float32))
    return np.stack(outputs)


def fit_readout(
    X: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if len(np.unique(labels[train_indices])) != 2 or len(np.unique(labels[test_indices])) != 2:
        raise ValueError("readout train/test split lacks both classes")
    settings = protocol["readouts"]
    scaler = StandardScaler().fit(X[train_indices])
    model = LogisticRegression(
        C=float(settings["C"]), penalty=settings["penalty"], solver=settings["solver"],
        max_iter=int(settings["max_iter"]), tol=float(settings["tol"]),
        random_state=int(settings["random_state"]),
    ).fit(scaler.transform(X[train_indices]), labels[train_indices])
    train_scores = model.decision_function(scaler.transform(X[train_indices]))
    score_sd = float(np.std(train_scores))
    if not np.isfinite(score_sd) or score_sd <= 0:
        raise ValueError("readout score standard deviation is invalid")
    test_scores = model.decision_function(scaler.transform(X[test_indices]))
    metrics = {
        "roc_auc": float(roc_auc_score(labels[test_indices], test_scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test_indices], test_scores >= 0)),
        "n_train": int(len(train_indices)), "n_test": int(len(test_indices)),
        "train_score_sd": score_sd, "n_iter": int(np.max(model.n_iter_)),
    }
    arrays = {
        "mean": scaler.mean_.astype(np.float32), "scale": scaler.scale_.astype(np.float32),
        "coef": model.coef_[0].astype(np.float32),
        "intercept": np.asarray(float(model.intercept_[0]), dtype=np.float64),
        "score_sd": np.asarray(score_sd, dtype=np.float64),
    }
    return arrays, metrics


def build(config_path: Path, model_id: str, out_dir: Path) -> dict[str, Any]:
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError("unexpected protocol")
    models = [model for model in protocol["models"] if model["model_id"] == model_id]
    if len(models) != 1:
        raise ValueError(f"configured model not unique: {model_id}")
    model_config = models[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_npz = out_dir / "causal_plan.npz"
    plan_json = out_dir / "causal_plan.json"
    if plan_npz.exists() or plan_json.exists():
        raise FileExistsError(f"refusing to overwrite immutable causal plan in {out_dir}")
    activation_path = Path(model_config["activation_path_cluster"])
    if not activation_path.exists():
        activation_path = Path(model_config["activation_path"])
    items_path = Path(protocol["dataset"]["items_path"])
    if not items_path.exists():
        items_path = activation_path.with_name("meta.jsonl")
    if sha256_file(items_path) != protocol["dataset"]["items_sha256"]:
        raise ValueError("item hash mismatch")
    if sha256_file(activation_path) != model_config["activation_sha256"]:
        raise ValueError("activation hash mismatch")
    rows = load_rows(items_path)
    validate_groups(rows)
    dataset = protocol["dataset"]
    stated_rows = [row for row in rows if row["render_slot"] == "stated"]
    stated_groups = {(row["payload_block_id"], row["purpose_pair_id"]) for row in stated_rows}
    all_blocks = {row["payload_block_id"] for row in rows}
    stated_families = {row["purpose_family_id"] for row in stated_rows}
    if len(rows) != dataset["n_rows"]:
        raise ValueError("row count mismatch")
    if len(stated_rows) != dataset["n_stated_rows"]:
        raise ValueError("stated row count mismatch")
    if len(stated_groups) != dataset["n_complete_2x2_groups"]:
        raise ValueError("complete stated group count mismatch")
    if len(all_blocks) != dataset["n_payload_blocks"]:
        raise ValueError("payload block count mismatch")
    if stated_families != set(dataset["purpose_families"]):
        raise ValueError("stated purpose-family set mismatch")
    with np.load(activation_path, allow_pickle=False) as archive:
        required_arrays = {"X", "item_ids", "layers", "token_lengths"}
        if required_arrays - set(archive.files):
            raise ValueError("activation archive missing required arrays")
        X_all = archive["X"]
        item_ids = archive["item_ids"].astype(str).tolist()
        layers = archive["layers"].astype(int).tolist()
        token_lengths = archive["token_lengths"]
        if item_ids != [row["item_id"] for row in rows]:
            raise ValueError("activation item order mismatch")
        expected_shape = (len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"]))
        if X_all.shape != expected_shape or layers != list(range(model_config["n_layers"])):
            raise ValueError(f"activation shape/layer mismatch: {X_all.shape}")
        if X_all.dtype != np.float32 or not np.isfinite(X_all).all():
            raise ValueError("activation X must be finite float32")
        if token_lengths.shape != (len(rows),) or not np.issubdtype(token_lengths.dtype, np.integer) or np.any(token_lengths <= 0):
            raise ValueError("activation token_lengths are invalid")
        wanted_layers = sorted(set(model_config["source_layers"].values()) | {model_config["target_layer"]})
        X = {layer: np.asarray(X_all[:, layer, :], dtype=np.float32) for layer in wanted_layers}
    stated = np.asarray([row["render_slot"] == "stated" for row in rows])
    block_fold = payload_folds(rows)
    arrays: dict[str, np.ndarray] = {}
    folds: list[dict[str, Any]] = []
    families = protocol["dataset"]["purpose_families"]
    seeds = protocol["interventions"]["random_seeds"]
    for family in families:
        for held_payload_fold in range(5):
            fold_id = f"{family}_pf{held_payload_fold}"
            direction_blocks, readout_blocks = role_blocks(rows, fold_id, held_payload_fold, block_fold)
            test_indices = np.flatnonzero(stated & np.asarray([row["purpose_family_id"] == family and block_fold[row["payload_block_id"]] == held_payload_fold for row in rows]))
            eligible = stated & np.asarray([row["purpose_family_id"] != family and block_fold[row["payload_block_id"]] != held_payload_fold for row in rows])
            direction_indices = np.flatnonzero(eligible & np.asarray([row["payload_block_id"] in direction_blocks for row in rows]))
            readout_indices = np.flatnonzero(eligible & np.asarray([row["payload_block_id"] in readout_blocks for row in rows]))
            test_blocks = {rows[index]["payload_block_id"] for index in test_indices}
            direction_used_blocks = {rows[index]["payload_block_id"] for index in direction_indices}
            readout_used_blocks = {rows[index]["payload_block_id"] for index in readout_indices}
            if test_blocks & direction_used_blocks or test_blocks & readout_used_blocks or direction_used_blocks & readout_used_blocks:
                raise ValueError(f"fold block disjointness failure {fold_id}")
            if not len(test_indices) or set(test_indices) & set(direction_indices) or set(test_indices) & set(readout_indices) or set(direction_indices) & set(readout_indices):
                raise ValueError(f"fold disjointness/coverage failure {fold_id}")
            fold: dict[str, Any] = {
                "fold_id": fold_id, "held_family": family, "held_payload_fold": held_payload_fold,
                "test_indices": test_indices.tolist(), "direction_indices": direction_indices.tolist(),
                "readout_indices": readout_indices.tolist(), "source_layers": {}, "readouts": {},
            }
            for depth_label, layer in model_config["source_layers"].items():
                A, B, C, common_scale, n_groups = component_arrays(X[layer], rows, direction_indices)
                prefix = f"fold_{fold_id}_layer_{layer}_"
                arrays[prefix + "A"] = A
                arrays[prefix + "B"] = B
                arrays[prefix + "C"] = C
                arrays[prefix + "random"] = random_directions(A, B, C, seeds)
                fold["source_layers"][str(layer)] = {
                    "depth_label": depth_label, "n_direction_groups": n_groups, "common_scale": common_scale,
                    "norm_A": float(np.linalg.norm(A)), "norm_B": float(np.linalg.norm(B)), "norm_C": float(np.linalg.norm(C)),
                    "max_random_component_cosine": float(max(abs(arrays[prefix + "random"] @ (v / np.linalg.norm(v))).max() for v in (A, B, C))),
                }
            target = X[model_config["target_layer"]]
            purpose_labels = np.asarray([row["intended_purpose"] == "evaluation" for row in rows], dtype=np.int8)
            format_labels = np.asarray([row["format"] == "benchmark" for row in rows], dtype=np.int8)
            for name in ("purpose_benchmark", "purpose_casual", "format"):
                if name == "purpose_benchmark":
                    select = np.asarray([row["format"] == "benchmark" for row in rows])
                    labels = purpose_labels
                elif name == "purpose_casual":
                    select = np.asarray([row["format"] == "casual" for row in rows])
                    labels = purpose_labels
                else:
                    select = np.ones(len(rows), dtype=bool)
                    labels = format_labels
                train = readout_indices[select[readout_indices]]
                test = test_indices[select[test_indices]]
                readout_arrays, metrics = fit_readout(target, labels, train, test, protocol)
                prefix = f"fold_{fold_id}_readout_{name}_"
                for suffix, value in readout_arrays.items():
                    arrays[prefix + suffix] = value
                fold["readouts"][name] = metrics
            folds.append(fold)
    test_union = [index for fold in folds for index in fold["test_indices"]]
    stated_indices = np.flatnonzero(stated).tolist()
    if sorted(test_union) != stated_indices or len(test_union) != len(set(test_union)):
        raise ValueError("outer test folds do not partition stated rows exactly")
    # Output paths were reserved before expensive fitting; publish them atomically only after all checks pass.
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".npz", delete=False) as handle:
        tmp_npz = Path(handle.name)
    try:
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, plan_npz)
    finally:
        tmp_npz.unlink(missing_ok=True)
    plan = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "config_path": config_path.as_posix(), "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)), "items_path": items_path.as_posix(),
        "items_sha256": sha256_file(items_path), "activation_path": activation_path.as_posix(),
        "activation_sha256": sha256_file(activation_path), "model": model_config,
        "canonical_item_ids": [row["item_id"] for row in rows],
        "payload_fold_by_block": block_fold, "folds": folds,
        "plan_npz": plan_npz.name, "plan_npz_sha256": sha256_file(plan_npz),
    }
    atomic_json(plan_json, plan)
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build(args.config.resolve(), args.model_id, args.out_dir.resolve())
    print(strict_json({"model_id": args.model_id, "folds": len(plan["folds"]), "out": str(args.out_dir)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
