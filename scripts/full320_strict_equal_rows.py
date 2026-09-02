#!/usr/bin/env python3
"""Preregistered strict equal-row probe-training coverage analysis.

The analysis treats benchmark/casual as stored format/template environments,
not a pure format intervention. It excludes legacy rows, holds out purpose
families, exactly balances paired training rows per payload block, and uses
common payload-block resamples across checkpoints and conditions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ENVIRONMENTS = ("benchmark", "casual")
REGIME_CONTROLS = ("benchmark_only", "casual_only")
PRIMARY_VARIANT = "raw_ridge_0.1"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_digest(*parts: Any) -> bytes:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).digest()


def strict_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def repo_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def rounded_layer(q: float, n_layers: int) -> int:
    return int(math.floor(q * (n_layers - 1) + 0.5))


def parse_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"item hash mismatch: expected {expected_sha256}, got {actual}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def npz_small_array(npz_path: Path, name: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as archive:
        return np.asarray(archive[name])


def extract_x_memmap(npz_path: Path, work_root: Path, expected_shape: Sequence[int]) -> Path:
    """Extract X.npy atomically because compressed NPZ members cannot be mmap'ed."""
    work_root.mkdir(parents=True, exist_ok=True)
    target = work_root / f"{npz_path.parent.name}_{hashlib.sha256(str(npz_path).encode()).hexdigest()[:12]}_X.npy"
    if target.exists():
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        if list(array.shape) == list(expected_shape) and array.dtype == np.float32:
            return target
        target.unlink()
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo("X.npy")
        free = shutil.disk_usage(work_root).free
        if free < info.file_size + 2 * 1024**3:
            raise OSError(f"insufficient disk for {info.file_size} byte X.npy in {work_root}; free={free}")
        incomplete = target.with_suffix(".npy.incomplete")
        if incomplete.exists():
            incomplete.unlink()
        with archive.open(info) as source, incomplete.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        os.replace(incomplete, target)
    array = np.load(target, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != list(expected_shape) or array.dtype != np.float32:
        target.unlink(missing_ok=True)
        raise ValueError(f"unexpected extracted X contract: {array.shape} {array.dtype}")
    return target


def pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["payload_block_id"]),
        str(row["purpose_pair_id"]),
        str(row["format"]),
        str(row["template_id"]),
    )


def canonical_pair_units(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str, str], tuple[int, int]]:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["render_slot"] == "stated" and not bool(row["legacy"]):
            grouped[pair_key(row)].append(index)
    units: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    for key, indices in grouped.items():
        by_label = {str(rows[index]["intended_purpose"]): index for index in indices}
        if len(indices) != 2 or set(by_label) != {"evaluation", "deployment"}:
            raise ValueError(f"pair unit is not one evaluation/deployment pair: {key} -> {indices}")
        families = {str(rows[index]["purpose_family_id"]) for index in indices}
        if len(families) != 1:
            raise ValueError(f"pair unit crosses purpose families: {key}")
        units[key] = (by_label["deployment"], by_label["evaluation"])
    return units


def topic_block_folds(rows: Sequence[dict[str, Any]], n_folds: int) -> dict[str, int]:
    topic_by_block: dict[str, str] = {}
    for row in rows:
        block = str(row["payload_block_id"])
        topic = str(row["topic_id"])
        previous = topic_by_block.setdefault(block, topic)
        if previous != topic:
            raise ValueError(f"payload block crosses topics: {block}")
    blocks_by_topic: dict[str, list[str]] = defaultdict(list)
    for block, topic in topic_by_block.items():
        blocks_by_topic[topic].append(block)
    assignment: dict[str, int] = {}
    for topic, blocks in sorted(blocks_by_topic.items()):
        ordered = sorted(blocks, key=lambda block: stable_digest("payload_fold", topic, block))
        for rank, block in enumerate(ordered):
            assignment[block] = rank % n_folds
    return assignment


def balanced_environment_assignment(
    blocks: Sequence[str], topic_by_block: dict[str, str], held_family: str, seed: int, split_tag: str
) -> dict[str, str]:
    by_topic: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        by_topic[topic_by_block[block]].append(block)
    assignment: dict[str, str] = {}
    for topic, topic_blocks in sorted(by_topic.items()):
        ordered = sorted(
            topic_blocks,
            key=lambda block: stable_digest("mixed", seed, held_family, split_tag, topic, block),
        )
        if len(ordered) % 2:
            raise ValueError(f"cannot exactly balance odd topic block count {topic}: {len(ordered)}")
        half = len(ordered) // 2
        assignment.update({block: ("benchmark" if rank < half else "casual") for rank, block in enumerate(ordered)})
    return assignment


def build_splits(
    rows: Sequence[dict[str, Any]], protocol: dict[str, Any], *, double_heldout: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units = canonical_pair_units(rows)
    families = [str(value) for value in protocol["dataset"]["purpose_families"]]
    seeds = [int(value) for value in protocol["probe"]["assignment_seeds"]]
    topic_by_block: dict[str, str] = {}
    family_by_unit: dict[tuple[str, str, str, str], str] = {}
    for key, (deployment_index, evaluation_index) in units.items():
        block = key[0]
        topic = str(rows[deployment_index]["topic_id"])
        topic_by_block.setdefault(block, topic)
        family_by_unit[key] = str(rows[evaluation_index]["purpose_family_id"])
    block_folds = topic_block_folds(rows, 5)
    splits: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    held_block_fold_values: Iterable[int | None] = range(5) if double_heldout else (None,)
    for held_family in families:
        for held_block_fold in held_block_fold_values:
            split_tag = f"family={held_family};block_fold={held_block_fold}"
            train_units = {
                key: indices
                for key, indices in units.items()
                if family_by_unit[key] != held_family
                and (held_block_fold is None or block_folds[key[0]] != held_block_fold)
            }
            test_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(rows)
                    if row["render_slot"] == "stated"
                    and not bool(row["legacy"])
                    and str(row["purpose_family_id"]) == held_family
                    and (held_block_fold is None or block_folds[str(row["payload_block_id"])] == held_block_fold)
                ],
                dtype=np.int32,
            )
            by_block_environment: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
            for key in train_units:
                by_block_environment[(key[0], key[2])].append(key)
            blocks = sorted({key[0] for key in train_units})
            selected: dict[str, dict[str, list[tuple[str, str, str, str]]]] = {
                environment: {} for environment in ENVIRONMENTS
            }
            k_by_block: dict[str, int] = {}
            for block in blocks:
                counts = {environment: len(by_block_environment[(block, environment)]) for environment in ENVIRONMENTS}
                k = min(counts.values())
                if k <= 0:
                    raise ValueError(f"missing environment in training block {split_tag} {block}: {counts}")
                k_by_block[block] = k
                for environment in ENVIRONMENTS:
                    candidates = by_block_environment[(block, environment)]
                    ordered = sorted(
                        candidates,
                        key=lambda key: stable_digest("pair_select", protocol["protocol_id"], split_tag, *key),
                    )
                    selected[environment][block] = ordered[:k]
            assignments = {
                seed: balanced_environment_assignment(blocks, topic_by_block, held_family, seed, split_tag)
                for seed in seeds
            }
            regime_pairs: dict[str, list[tuple[tuple[str, str, str, str], int, int]]] = {}
            for environment, regime in zip(ENVIRONMENTS, REGIME_CONTROLS):
                regime_pairs[regime] = [
                    (key, *train_units[key]) for block in blocks for key in selected[environment][block]
                ]
            for seed in seeds:
                regime_pairs[f"mixed_seed_{seed}"] = [
                    (key, *train_units[key])
                    for block in blocks
                    for key in selected[assignments[seed][block]][block]
                ]
            lengths = {name: 2 * len(pairs) for name, pairs in regime_pairs.items()}
            if len(set(lengths.values())) != 1:
                raise AssertionError(f"training row imbalance in {split_tag}: {lengths}")
            for name, pairs in regime_pairs.items():
                indices = [index for _, deployment_index, evaluation_index in pairs for index in (deployment_index, evaluation_index)]
                labels = [0, 1] * len(pairs)
                if len(indices) != lengths[name] or sum(labels) * 2 != len(labels):
                    raise AssertionError(f"class imbalance in {split_tag} {name}")
            if len(np.unique(test_indices)) != len(test_indices):
                raise AssertionError(f"duplicate test rows in {split_tag}")
            split = {
                "held_family": held_family,
                "held_block_fold": held_block_fold,
                "split_tag": split_tag,
                "test_indices": test_indices,
                "regime_pairs": regime_pairs,
            }
            splits.append(split)
            assignment_hashes = {
                str(seed): canonical_json_sha256(assignments[seed]) for seed in seeds
            }
            audit_rows.append(
                {
                    "split_tag": split_tag,
                    "n_train_blocks": len(blocks),
                    "n_test_rows": int(len(test_indices)),
                    "n_train_rows_each_regime": next(iter(lengths.values())),
                    "k_min": int(min(k_by_block.values())),
                    "k_max": int(max(k_by_block.values())),
                    "assignment_sha256": assignment_hashes,
                    "selected_pair_sha256": {
                        environment: canonical_json_sha256(
                            [key for block in blocks for key in selected[environment][block]]
                        )
                        for environment in ENVIRONMENTS
                    },
                }
            )
    canonical_test = sorted(index for split in splits for index in split["test_indices"].tolist())
    expected_test = sorted(
        index
        for index, row in enumerate(rows)
        if row["render_slot"] == "stated" and not bool(row["legacy"])
    )
    if canonical_test != expected_test:
        raise AssertionError("split test rows do not partition canonical stated rows exactly")
    audit = {
        "double_heldout": double_heldout,
        "n_splits": len(splits),
        "canonical_test_rows": len(expected_test),
        "split_records": audit_rows,
        "split_records_sha256": canonical_json_sha256(audit_rows),
    }
    return splits, audit


def train_indices_and_labels(
    pairs: Sequence[tuple[tuple[str, str, str, str], int, int]], null_seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(2 * len(pairs), dtype=np.int32)
    labels = np.empty(2 * len(pairs), dtype=np.int8)
    for pair_index, (key, deployment_index, evaluation_index) in enumerate(pairs):
        indices[2 * pair_index : 2 * pair_index + 2] = (deployment_index, evaluation_index)
        swap = null_seed is not None and (stable_digest("label_swap", null_seed, *key)[0] & 1) == 1
        labels[2 * pair_index : 2 * pair_index + 2] = (1, 0) if swap else (0, 1)
    return indices, labels


def normalized_rows(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    norms = np.sqrt(np.sum(values * values, axis=1, dtype=np.float64)).astype(np.float32)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("row normalization requires finite nonzero norms")
    return np.asarray(values / norms[:, None], dtype=np.float32)


def diag_lda_scores(
    data: np.ndarray,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    test_indices: np.ndarray,
    ridge_fractions: Sequence[float],
) -> dict[float, np.ndarray]:
    if len(train_indices) != len(train_labels) or set(np.unique(train_labels).tolist()) != {0, 1}:
        raise ValueError("invalid training labels")
    class0 = np.asarray(data[train_indices[train_labels == 0]], dtype=np.float32)
    class1 = np.asarray(data[train_indices[train_labels == 1]], dtype=np.float32)
    if len(class0) != len(class1):
        raise ValueError("diagonal LDA requires exact class balance")
    mu0 = np.mean(class0, axis=0, dtype=np.float64)
    mu1 = np.mean(class1, axis=0, dtype=np.float64)
    var0 = np.var(class0, axis=0, dtype=np.float64)
    var1 = np.var(class1, axis=0, dtype=np.float64)
    pooled = 0.5 * (var0 + var1)
    delta = mu1 - mu0
    global_mean = 0.5 * (mu0 + mu1)
    global_var = pooled + 0.25 * delta * delta
    scale = np.sqrt(np.maximum(global_var, np.finfo(np.float64).tiny))
    pooled_scaled = pooled / (scale * scale)
    positive = pooled_scaled[np.isfinite(pooled_scaled) & (pooled_scaled > 0)]
    if not len(positive):
        raise ValueError("no positive pooled variances")
    median_variance = float(np.median(positive))
    test = np.asarray(data[test_indices], dtype=np.float32)
    outputs: dict[float, np.ndarray] = {}
    for ridge_fraction in ridge_fractions:
        variance = pooled_scaled + float(ridge_fraction) * median_variance
        delta_scaled = delta / scale
        midpoint_scaled = (0.5 * (mu1 + mu0) - global_mean) / scale
        weight_scaled = delta_scaled / variance
        intercept = -float(np.dot(weight_scaled, midpoint_scaled))
        weight_raw = weight_scaled / scale
        raw_intercept = intercept - float(np.dot(weight_raw, global_mean))
        scores = np.asarray(test @ weight_raw + raw_intercept, dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError("nonfinite diagonal-LDA scores")
        outputs[float(ridge_fraction)] = scores.astype(np.float32)
    return outputs


def fit_oof_scores(
    layer_data: np.ndarray,
    splits: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    *,
    include_robustness: bool,
    null_seed: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    n_rows = layer_data.shape[0]
    variants = {PRIMARY_VARIANT: (False, 0.1)}
    if include_robustness and null_seed is None:
        variants.update(
            {
                "raw_ridge_0.01": (False, 0.01),
                "raw_ridge_1": (False, 1.0),
                "row_l2_ridge_0.1": (True, 0.1),
            }
        )
    raw_ridges = sorted({ridge for normalize, ridge in variants.values() if not normalize})
    norm_ridges = sorted({ridge for normalize, ridge in variants.values() if normalize})
    normalized = normalized_rows(layer_data) if norm_ridges else None
    regime_names = [*REGIME_CONTROLS, *(f"mixed_seed_{seed}" for seed in seeds)]
    outputs = {
        variant: {name: np.full(n_rows, np.nan, dtype=np.float32) for name in regime_names}
        for variant in variants
    }
    for split in splits:
        test_indices = split["test_indices"]
        for regime_name in regime_names:
            train_indices, labels = train_indices_and_labels(split["regime_pairs"][regime_name], null_seed)
            raw_scores = diag_lda_scores(layer_data, train_indices, labels, test_indices, raw_ridges)
            for variant, (normalize, ridge) in variants.items():
                if not normalize:
                    outputs[variant][regime_name][test_indices] = raw_scores[ridge]
            if norm_ridges:
                assert normalized is not None
                norm_scores = diag_lda_scores(normalized, train_indices, labels, test_indices, norm_ridges)
                for variant, (normalize, ridge) in variants.items():
                    if normalize:
                        outputs[variant][regime_name][test_indices] = norm_scores[ridge]
    for variant_outputs in outputs.values():
        for regime_name, scores in variant_outputs.items():
            finite = np.isfinite(scores)
            expected = sum(len(split["test_indices"]) for split in splits)
            if int(finite.sum()) != expected:
                raise AssertionError(f"missing OOF scores for {regime_name}: {finite.sum()} != {expected}")
    return outputs


def metric_bundle(y: np.ndarray, formats: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    per_format: dict[str, dict[str, float]] = {}
    for environment in ENVIRONMENTS:
        selected = mask & (formats == environment)
        if set(np.unique(y[selected]).tolist()) != {0, 1}:
            raise ValueError(f"metric cell lacks both labels: {environment}")
        per_format[environment] = {
            "roc_auc": float(roc_auc_score(y[selected], scores[selected])),
            "balanced_accuracy": float(balanced_accuracy_score(y[selected], scores[selected] > 0)),
        }
    result: dict[str, float] = {}
    for metric in ("roc_auc", "balanced_accuracy"):
        values = [per_format[environment][metric] for environment in ENVIRONMENTS]
        result[f"benchmark_{metric}"] = values[0]
        result[f"casual_{metric}"] = values[1]
        result[f"macro_format_{metric}"] = float(np.mean(values))
        result[f"worst_format_{metric}"] = float(np.min(values))
    return result


def summarize_point_records(records: Sequence[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    endpoint_names = ("macro_format_roc_auc", "macro_format_balanced_accuracy")
    primary = [record for record in records if record["variant"] == PRIMARY_VARIANT and not record["double_heldout"]]
    double = [record for record in records if record["variant"] == PRIMARY_VARIANT and record["double_heldout"]]

    def cell_contrasts(group: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        by_cell: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            by_cell[(record["model"], record["normalized_depth"])].append(record)
        output: list[dict[str, Any]] = []
        for (model, depth), cell in sorted(by_cell.items()):
            lookup = {(record["regime"], record["seed"]): record for record in cell}
            controls = [lookup[(regime, None)] for regime in REGIME_CONTROLS]
            for seed in seeds:
                mixed = lookup[("mixed", seed)]
                row = {"model": model, "normalized_depth": depth, "seed": seed}
                for endpoint in endpoint_names:
                    row[endpoint] = mixed[endpoint] - max(control[endpoint] for control in controls)
                output.append(row)
        return output

    contrasts = cell_contrasts(primary)
    double_contrasts = cell_contrasts(double) if double else []
    aggregate = {
        endpoint: float(np.mean([row[endpoint] for row in contrasts])) for endpoint in endpoint_names
    }
    by_model = {
        model: {
            endpoint: float(np.mean([row[endpoint] for row in contrasts if row["model"] == model]))
            for endpoint in endpoint_names
        }
        for model in sorted({row["model"] for row in contrasts})
    }
    by_seed = {
        str(seed): {
            endpoint: float(np.mean([row[endpoint] for row in contrasts if row["seed"] == seed]))
            for endpoint in endpoint_names
        }
        for seed in seeds
    }
    by_depth = {
        str(depth): {
            endpoint: float(np.mean([row[endpoint] for row in contrasts if row["normalized_depth"] == depth]))
            for endpoint in endpoint_names
        }
        for depth in sorted({row["normalized_depth"] for row in contrasts})
    }
    robustness: dict[str, dict[str, float]] = {}
    for variant in sorted({record["variant"] for record in records if not record["double_heldout"]}):
        variant_contrasts = cell_contrasts(
            [record for record in records if record["variant"] == variant and not record["double_heldout"]]
        )
        robustness[variant] = {
            endpoint: float(np.mean([row[endpoint] for row in variant_contrasts]))
            for endpoint in endpoint_names
        }
    double_summary = {
        endpoint: float(np.mean([row[endpoint] for row in double_contrasts]))
        for endpoint in endpoint_names
    } if double_contrasts else {}
    return {
        "aggregate": aggregate,
        "by_model": by_model,
        "by_seed": by_seed,
        "by_depth": by_depth,
        "method_robustness": robustness,
        "double_heldout_payload": double_summary,
        "cell_contrasts": contrasts,
        "double_heldout_cell_contrasts": double_contrasts,
    }


def bootstrap_counts(blocks: Sequence[str], n_resamples: int, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    ordered = sorted(set(blocks))
    lookup = {block: index for index, block in enumerate(ordered)}
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(ordered), size=(n_resamples, len(ordered)), endpoint=False)
    counts = np.zeros((n_resamples, len(ordered)), dtype=np.int16)
    for draw in range(n_resamples):
        counts[draw] = np.bincount(chosen[draw], minlength=len(ordered))
    return counts, lookup


def weighted_auc_draws(
    labels: np.ndarray,
    scores: np.ndarray,
    block_columns: np.ndarray,
    counts: np.ndarray,
    batch_size: int = 100,
) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(bool)
    sorted_blocks = block_columns[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    outputs = np.empty(len(counts), dtype=np.float64)
    for begin in range(0, len(counts), batch_size):
        draw_counts = counts[begin : begin + batch_size]
        weights = draw_counts[:, sorted_blocks].astype(np.float64, copy=False)
        positive_groups = np.add.reduceat(weights * sorted_labels[None, :], starts, axis=1)
        negative_groups = np.add.reduceat(weights * (~sorted_labels)[None, :], starts, axis=1)
        negative_before = np.cumsum(negative_groups, axis=1) - negative_groups
        numerator = np.sum(positive_groups * (negative_before + 0.5 * negative_groups), axis=1)
        positives = np.sum(positive_groups, axis=1)
        negatives = np.sum(negative_groups, axis=1)
        outputs[begin : begin + len(draw_counts)] = numerator / (positives * negatives)
    if not np.all(np.isfinite(outputs)):
        raise ValueError("nonfinite bootstrap AUC")
    return outputs


def balanced_accuracy_draws(
    labels: np.ndarray, scores: np.ndarray, block_columns: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    prediction = scores > 0
    n_blocks = counts.shape[1]
    tp = np.bincount(block_columns, weights=(prediction & (labels == 1)), minlength=n_blocks)
    fn = np.bincount(block_columns, weights=((~prediction) & (labels == 1)), minlength=n_blocks)
    tn = np.bincount(block_columns, weights=((~prediction) & (labels == 0)), minlength=n_blocks)
    fp = np.bincount(block_columns, weights=(prediction & (labels == 0)), minlength=n_blocks)
    tp_draw = counts @ tp
    fn_draw = counts @ fn
    tn_draw = counts @ tn
    fp_draw = counts @ fp
    output = 0.5 * (tp_draw / (tp_draw + fn_draw) + tn_draw / (tn_draw + fp_draw))
    if not np.all(np.isfinite(output)):
        raise ValueError("nonfinite bootstrap balanced accuracy")
    return output


def bootstrap_primary(
    score_store: dict[tuple[str, float, str, int | None], np.ndarray],
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    counts: np.ndarray,
    block_lookup: dict[str, int],
) -> dict[str, Any]:
    labels_all = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats_all = np.asarray([str(row["format"]) for row in rows])
    canonical = np.asarray([row["render_slot"] == "stated" and not bool(row["legacy"]) for row in rows])
    block_columns_all = np.asarray([block_lookup[str(row["payload_block_id"])] for row in rows], dtype=np.int32)
    metric_cache: dict[tuple[str, float, str, int | None, str, str], np.ndarray] = {}
    for key, scores in score_store.items():
        for environment in ENVIRONMENTS:
            mask = canonical & (formats_all == environment)
            labels = labels_all[mask]
            selected_scores = scores[mask].astype(np.float64)
            blocks = block_columns_all[mask]
            metric_cache[(*key, environment, "roc_auc")] = weighted_auc_draws(labels, selected_scores, blocks, counts)
            metric_cache[(*key, environment, "balanced_accuracy")] = balanced_accuracy_draws(labels, selected_scores, blocks, counts)
    models = sorted({key[0] for key in score_store})
    depths = sorted({key[1] for key in score_store})
    endpoint_draws: dict[str, list[np.ndarray]] = {"macro_format_roc_auc": [], "macro_format_balanced_accuracy": []}
    model_draws: dict[str, dict[str, list[np.ndarray]]] = {
        model: {endpoint: [] for endpoint in endpoint_draws} for model in models
    }
    for model in models:
        for depth in depths:
            for endpoint, metric in (("macro_format_roc_auc", "roc_auc"), ("macro_format_balanced_accuracy", "balanced_accuracy")):
                control_values = []
                for regime in REGIME_CONTROLS:
                    control_values.append(
                        0.5 * sum(metric_cache[(model, depth, regime, None, environment, metric)] for environment in ENVIRONMENTS)
                    )
                better_control = np.maximum(control_values[0], control_values[1])
                mixed_values = []
                for seed in seeds:
                    mixed_values.append(
                        0.5 * sum(metric_cache[(model, depth, "mixed", seed, environment, metric)] for environment in ENVIRONMENTS)
                    )
                contrast = np.mean(mixed_values, axis=0) - better_control
                endpoint_draws[endpoint].append(contrast)
                model_draws[model][endpoint].append(contrast)
    aggregate: dict[str, dict[str, float]] = {}
    for endpoint, values in endpoint_draws.items():
        draws = np.mean(values, axis=0)
        aggregate[endpoint] = {
            "bootstrap_mean": float(np.mean(draws)),
            "ci_2.5": float(np.quantile(draws, 0.025)),
            "ci_50": float(np.quantile(draws, 0.5)),
            "ci_97.5": float(np.quantile(draws, 0.975)),
        }
    by_model = {}
    for model in models:
        by_model[model] = {}
        for endpoint, values in model_draws[model].items():
            draws = np.mean(values, axis=0)
            by_model[model][endpoint] = {
                "ci_2.5": float(np.quantile(draws, 0.025)),
                "ci_50": float(np.quantile(draws, 0.5)),
                "ci_97.5": float(np.quantile(draws, 0.975)),
            }
    return {"aggregate": aggregate, "by_model": by_model}


def plot_results(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = sorted(float(value) for value in summary["by_depth"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for endpoint, axis, title in (
        ("macro_format_roc_auc", axes[0], "Macro-format AUC gain"),
        ("macro_format_balanced_accuracy", axes[1], "Macro-format balanced-accuracy gain"),
    ):
        values = [summary["by_depth"][str(depth)][endpoint] for depth in depths]
        axis.plot(depths, values, marker="o", linewidth=2)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xlabel("Normalized depth")
        axis.set_ylabel("Mixed − better single environment")
        axis.set_title(title)
    variant_names = list(summary["method_robustness"])
    variant_values = [summary["method_robustness"][name]["macro_format_roc_auc"] for name in variant_names]
    axes[2].barh(range(len(variant_names)), variant_values)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_yticks(range(len(variant_names)), labels=variant_names)
    axes[2].set_xlabel("Aggregate AUC gain")
    axes[2].set_title("Predeclared method robustness")
    fig.suptitle("Strict equal-row training-environment coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def output_hash(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") not in {
        "full320_strict_equal_rows_v1",
        "full320_strict_equal_rows_8b_scale_v1",
    } or config.get("schema_version") != 2:
        raise ValueError("unexpected protocol")
    data_root = args.data_root.resolve()
    items_path = data_root / config["dataset"]["items_path"]
    rows = parse_rows(items_path, config["dataset"]["items_sha256"])
    if len(rows) != config["dataset"]["n_rows"]:
        raise ValueError("item row count mismatch")
    main_splits, main_design_audit = build_splits(rows, config, double_heldout=False)
    double_splits, double_design_audit = build_splits(rows, config, double_heldout=True)
    seeds = [int(value) for value in config["probe"]["assignment_seeds"]]
    normalized_depths = [float(value) for value in config["depths"]["normalized"]]
    canonical_mask = np.asarray([row["render_slot"] == "stated" and not bool(row["legacy"]) for row in rows])
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in rows])
    output_dir = args.out_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"stale incomplete output exists: {incomplete}")
    incomplete.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    score_store: dict[tuple[str, float, str, int | None], np.ndarray] = {}
    score_arrays: dict[str, np.ndarray] = {}
    score_index: list[dict[str, Any]] = []
    null_sums = np.zeros(len(config["falsification"]["training_pair_label_swap_seeds"]), dtype=np.float64)
    null_cell_count = 0
    model_audits: list[dict[str, Any]] = []
    try:
        for model_index, model_config in enumerate(config["models"]):
            model_id = str(model_config["model_id"])
            npz_path = data_root / model_config["activation_path"]
            actual_hash = sha256_file(npz_path)
            if actual_hash != model_config["activation_sha256"]:
                raise ValueError(f"activation hash mismatch for {model_id}: {actual_hash}")
            item_ids = [str(value) for value in npz_small_array(npz_path, "item_ids")]
            if item_ids != [str(row["item_id"]) for row in rows]:
                raise ValueError(f"activation item IDs mismatch for {model_id}")
            expected_shape = [len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"])]
            x_path = extract_x_memmap(npz_path, args.memmap_root.resolve(), expected_shape)
            X = np.load(x_path, mmap_mode="r", allow_pickle=False)
            layer_map = {depth: rounded_layer(depth, int(model_config["n_layers"])) for depth in normalized_depths}
            model_audits.append(
                {
                    "model": model_id,
                    "revision": model_config["revision"],
                    "activation_sha256": actual_hash,
                    "item_ids_sha256": canonical_json_sha256(item_ids),
                    "shape": list(X.shape),
                    "layers": {str(depth): layer for depth, layer in layer_map.items()},
                }
            )
            for depth, layer in layer_map.items():
                layer_data = X[:, layer, :]
                observed = fit_oof_scores(layer_data, main_splits, seeds, include_robustness=True)
                for variant, regime_scores in observed.items():
                    for regime_name, scores in regime_scores.items():
                        if regime_name.startswith("mixed_seed_"):
                            regime = "mixed"
                            seed: int | None = int(regime_name.rsplit("_", 1)[1])
                        else:
                            regime = regime_name
                            seed = None
                        metrics = metric_bundle(labels, formats, canonical_mask, scores)
                        record = {
                            "model": model_id,
                            "revision": model_config["revision"],
                            "layer": layer,
                            "normalized_depth": depth,
                            "variant": variant,
                            "regime": regime,
                            "seed": seed,
                            "double_heldout": False,
                            **metrics,
                        }
                        records.append(record)
                        if variant == PRIMARY_VARIANT:
                            key = (model_id, depth, regime, seed)
                            score_store[key] = scores
                            array_name = f"score_{len(score_arrays):04d}"
                            score_arrays[array_name] = scores[canonical_mask]
                            score_index.append({"array": array_name, "model": model_id, "depth": depth, "regime": regime, "seed": seed})
                if math.isclose(depth, float(config["falsification"]["double_heldout_payload_robustness"]["depth"])):
                    double_scores = fit_oof_scores(layer_data, double_splits, seeds, include_robustness=False)
                    for regime_name, scores in double_scores[PRIMARY_VARIANT].items():
                        if regime_name.startswith("mixed_seed_"):
                            regime = "mixed"
                            seed = int(regime_name.rsplit("_", 1)[1])
                        else:
                            regime = regime_name
                            seed = None
                        metrics = metric_bundle(labels, formats, canonical_mask, scores)
                        records.append(
                            {
                                "model": model_id,
                                "revision": model_config["revision"],
                                "layer": layer,
                                "normalized_depth": depth,
                                "variant": PRIMARY_VARIANT,
                                "regime": regime,
                                "seed": seed,
                                "double_heldout": True,
                                **metrics,
                            }
                        )
                if not args.skip_null:
                    for null_index, null_seed in enumerate(config["falsification"]["training_pair_label_swap_seeds"]):
                        null_outputs = fit_oof_scores(
                            layer_data, main_splits, seeds, include_robustness=False, null_seed=int(null_seed)
                        )[PRIMARY_VARIANT]
                        null_metrics: dict[tuple[str, int | None], dict[str, float]] = {}
                        for regime_name, scores in null_outputs.items():
                            if regime_name.startswith("mixed_seed_"):
                                regime = "mixed"
                                seed = int(regime_name.rsplit("_", 1)[1])
                            else:
                                regime = regime_name
                                seed = None
                            null_metrics[(regime, seed)] = metric_bundle(labels, formats, canonical_mask, scores)
                        controls = [null_metrics[(regime, None)]["macro_format_roc_auc"] for regime in REGIME_CONTROLS]
                        mixed_mean = float(np.mean([null_metrics[("mixed", seed)]["macro_format_roc_auc"] for seed in seeds]))
                        null_sums[null_index] += mixed_mean - max(controls)
                    null_cell_count += 1
            mmap_handle = X._mmap
            del layer_data
            del X
            mmap_handle.close()
            if not args.keep_memmap:
                x_path.unlink(missing_ok=True)
        point_summary = summarize_point_records(records, seeds)
        blocks = [str(row["payload_block_id"]) for row in rows]
        counts, block_lookup = bootstrap_counts(
            blocks,
            int(config["uncertainty"]["bootstrap_resamples"]),
            int(config["uncertainty"]["bootstrap_seed"]),
        )
        bootstrap_summary = bootstrap_primary(score_store, rows, seeds, counts, block_lookup)
        null_values = (null_sums / null_cell_count).tolist() if null_cell_count else []
        observed_auc = point_summary["aggregate"]["macro_format_roc_auc"]
        null_p = (
            (1 + sum(value >= observed_auc for value in null_values)) / (1 + len(null_values))
            if null_values
            else None
        )
        criteria = {
            "primary_auc": bootstrap_summary["aggregate"]["macro_format_roc_auc"]["ci_2.5"] > 0,
            "primary_balanced_accuracy": bootstrap_summary["aggregate"]["macro_format_balanced_accuracy"]["ci_2.5"] > 0,
            "checkpoint_replication": all(
                point_summary["by_model"][model][endpoint] > 0
                for model in point_summary["by_model"]
                for endpoint in ("macro_format_roc_auc", "macro_format_balanced_accuracy")
            ),
            "assignment_stability": all(
                values["macro_format_roc_auc"] > 0 for values in point_summary["by_seed"].values()
            ),
            "label_null": null_p is not None and null_p <= 0.05,
            "robustness": all(
                values["macro_format_roc_auc"] > 0 for values in point_summary["method_robustness"].values()
            ) and all(value > 0 for value in point_summary["double_heldout_payload"].values()),
        }
        decision = "positive" if all(criteria.values()) else "mixed"
        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "completed_preregistered_replication_extension",
            "analysis": config["protocol_id"],
            "repo_commit": repo_commit(),
            "protocol": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "evidence_boundary": config["evidence_boundary"],
            "design_audit": {"main": main_design_audit, "double_heldout": double_design_audit},
            "models": model_audits,
            "point_summary": point_summary,
            "bootstrap": bootstrap_summary,
            "label_swap_null": {
                "n": len(null_values),
                "values": null_values,
                "q_2.5": float(np.quantile(null_values, 0.025)) if null_values else None,
                "q_97.5": float(np.quantile(null_values, 0.975)) if null_values else None,
                "empirical_p_greater_equal": null_p,
            },
            "criteria": criteria,
            "decision": decision,
            "records": records,
            "scores": {
                "path": "primary_scores.npz",
                "index": score_index,
                "canonical_row_indices": np.flatnonzero(canonical_mask).tolist(),
            },
            "runtime_seconds": time.time() - started,
            "limitations": config["evidence_boundary"]["limitations"],
        }
        result_path = incomplete / config["required_outputs"]["result_json"]
        tsv_path = incomplete / config["required_outputs"]["metrics_tsv"]
        figure_path = incomplete / config["required_outputs"]["figure"]
        scores_path = incomplete / "primary_scores.npz"
        manifest_path = incomplete / config["required_outputs"]["manifest_json"]
        result_path.write_text(strict_json_text(result), encoding="utf-8")
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(records[0])
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        np.savez_compressed(scores_path, **score_arrays)
        plot_results(point_summary, figure_path)
        manifest = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "completed_preregistered_replication_extension",
            "analysis": config["protocol_id"],
            "repo_commit": result["repo_commit"],
            "protocol_sha256": result["protocol"]["sha256"],
            "input_hashes": {
                "items": config["dataset"]["items_sha256"],
                **{model["model_id"]: model["activation_sha256"] for model in config["models"]},
            },
            "output_hashes": {
                path.name: output_hash(path) for path in (result_path, tsv_path, figure_path, scores_path)
            },
            "decision": decision,
            "criteria": criteria,
            "evidence_grade": config["evidence_boundary"]["grade"],
            "prohibited_claims": config["evidence_boundary"]["prohibited_claims"],
        }
        manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")
        os.replace(incomplete, output_dir)
        return result
    except BaseException:
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keep-memmap", action="store_true")
    parser.add_argument("--skip-null", action="store_true", help="Development smoke only; cannot satisfy positive decision")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(strict_json_text({"run_id": result["run_id"], "decision": result["decision"], "criteria": result["criteria"]}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
