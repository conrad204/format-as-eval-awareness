#!/usr/bin/env python3
"""Preregistered raw/template regime-sensitivity analysis for devbunova activations.

The input cells are independent prompt distributions. Only identical full item IDs
across model sizes and stored regimes are paired. Results are development-only and
observational; this module does not infer undocumented regime semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CORE_CONDITION_TO_CELL = {
    "bench_eval": "EB",
    "bench_deploy_rewritten": "DB",
    "casual_eval_mutual": "EC",
    "casual_deploy_matched": "DC",
}
TASKS = {
    "purpose_benchmark": ("EB", "DB"),
    "purpose_casual": ("EC", "DC"),
    "format_evaluation": ("EB", "EC"),
    "format_deployment": ("DB", "DC"),
}
TRANSFER_CODES = ("RR", "RT", "TT", "TR")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_json_text(value: Any) -> str:
    """Serialize strict JSON and reject NaN or infinity anywhere in the output."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def bf16_u16_to_float32(values: np.ndarray) -> np.ndarray:
    """Decode uint16 BF16 bit patterns into a new float32 array."""
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32)
    return np.left_shift(bits, 16).view(np.float32)


def parse_item_id(item_id: str) -> dict[str, Any]:
    """Parse and validate ``dev2x2::<condition>::<variant>::<index>``."""
    parts = str(item_id).split("::")
    if len(parts) != 4 or parts[0] != "dev2x2":
        raise ValueError(f"invalid dev2x2 item ID: {item_id!r}")
    condition, variant, index_text = parts[1:]
    if not condition or variant not in {"asis", "stripped"}:
        raise ValueError(f"invalid dev2x2 item ID: {item_id!r}")
    if len(index_text) != 4 or not index_text.isdigit():
        raise ValueError(f"invalid dev2x2 item index: {item_id!r}")
    return {
        "dataset": "dev2x2",
        "condition": condition,
        "variant": variant,
        "index": int(index_text),
    }


def _stable_rank(seed: int, item_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{item_id}".encode("utf-8")).digest()


def select_balanced_core(
    item_ids: Sequence[str], n_per_cell: int = 204, seed: int = 20260822
) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly ``n_per_cell`` asis IDs per core cell by stable hash rank."""
    normalized_ids = [str(item_id) for item_id in item_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("duplicate full item IDs are forbidden")
    grouped: dict[str, list[tuple[bytes, int]]] = {cell: [] for cell in ("EB", "DB", "EC", "DC")}
    for row_index, item_id in enumerate(normalized_ids):
        parsed = parse_item_id(item_id)
        cell = CORE_CONDITION_TO_CELL.get(parsed["condition"])
        if cell is not None and parsed["variant"] == "asis":
            grouped[cell].append((_stable_rank(seed, str(item_id)), row_index))
    chosen: list[tuple[int, str]] = []
    for cell in ("EB", "DB", "EC", "DC"):
        rows = sorted(grouped[cell])
        if len(rows) < n_per_cell:
            raise ValueError(f"cell {cell} has {len(rows)} eligible rows; requires {n_per_cell}")
        chosen.extend((row_index, cell) for _, row_index in rows[:n_per_cell])
    chosen.sort()
    indices = np.asarray([row for row, _ in chosen], dtype=np.int64)
    cells = np.asarray([cell for _, cell in chosen], dtype="U2")
    return indices, cells


def fixed_depth_layers(n_layers: int, normalized_depths: Sequence[float]) -> list[int]:
    """Map normalized depths to zero-based layers with round-half-up semantics."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    layers: list[int] = []
    for depth in normalized_depths:
        value = float(depth)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"normalized depth outside [0,1]: {value}")
        layer = int(math.floor(value * (n_layers - 1) + 0.5))
        if layer in layers:
            raise ValueError(f"normalized depths collide at layer {layer}")
        layers.append(layer)
    return layers


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC AUC with average ranks for tied scores."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.all(np.isfinite(s)):
        raise ValueError("labels and finite scores must be aligned one-dimensional arrays")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be binary")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> float:
    """Balanced accuracy at a fixed score threshold."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or not np.all(np.isin(y, (0, 1))) or not np.all(np.isfinite(s)):
        raise ValueError("invalid labels or scores")
    positive = y == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("balanced accuracy requires both classes")
    prediction = s >= float(threshold)
    return float(0.5 * (prediction[positive].mean() + (~prediction[negative]).mean()))


def fit_diag_lda(
    features: np.ndarray, labels: np.ndarray, ridge_fraction: float = 0.1
) -> dict[str, Any]:
    """Fit fixed-ridge diagonal LDA using only the supplied training rows."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not np.all(np.isfinite(x)):
        raise ValueError("invalid feature matrix")
    x0, x1 = x[y == 0], x[y == 1]
    if min(len(x0), len(x1)) < 2:
        raise ValueError("diagonal LDA requires at least two rows per class")
    mean0 = x0.mean(axis=0, dtype=np.float64)
    mean1 = x1.mean(axis=0, dtype=np.float64)
    var0 = x0.var(axis=0, ddof=1, dtype=np.float64)
    var1 = x1.var(axis=0, ddof=1, dtype=np.float64)
    pooled = ((len(x0) - 1) * var0 + (len(x1) - 1) * var1) / (len(x0) + len(x1) - 2)
    eligible = pooled[np.isfinite(pooled) & (pooled > 0)]
    if eligible.size == 0:
        raise ValueError("all pooled feature variances are zero or non-finite")
    ridge = float(ridge_fraction * np.median(eligible))
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("invalid ridge")
    direction = (mean1 - mean0) / (pooled + ridge)
    intercept = float(-0.5 * np.dot(mean1 + mean0, direction))
    if not np.all(np.isfinite(direction)) or not np.isfinite(intercept):
        raise ValueError("non-finite LDA parameters")
    return {"direction": direction, "intercept": intercept, "ridge": ridge}


def score_diag_lda(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Score rows with a model returned by :func:`fit_diag_lda`."""
    scores = np.asarray(features, dtype=np.float32) @ np.asarray(model["direction"], dtype=np.float64)
    scores = scores + float(model["intercept"])
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite LDA scores")
    return np.asarray(scores, dtype=np.float64)


def stratified_cell_folds(
    item_ids: Sequence[str], cells: Sequence[str], n_folds: int = 5, seed: int = 20260822
) -> np.ndarray:
    """Assign deterministic round-robin folds independently within each cell."""
    ids = np.asarray(item_ids)
    strata = np.asarray(cells)
    if ids.ndim != 1 or strata.shape != ids.shape or n_folds < 2:
        raise ValueError("invalid fold inputs")
    folds = np.full(ids.size, -1, dtype=np.int16)
    for cell in sorted(set(strata.tolist())):
        indices = np.flatnonzero(strata == cell).tolist()
        indices.sort(key=lambda index: _stable_rank(seed, str(ids[index])))
        for rank, index in enumerate(indices):
            folds[index] = rank % n_folds
    if np.any(folds < 0):
        raise AssertionError("unassigned fold")
    return folds


def _metric(metric: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if metric == "auc":
        return roc_auc(labels, scores)
    if metric == "balanced_accuracy":
        return balanced_accuracy(labels, scores)
    raise ValueError(f"unknown metric: {metric}")


def paired_bootstrap_summary(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    seed: int = 20260823,
    n_resamples: int = 1000,
    metric: str = "auc",
) -> dict[str, Any]:
    """Paired class-stratified bootstrap for two aligned score vectors."""
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("bootstrap inputs must align")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == value) for value in (0, 1)]
    draws = np.empty(n_resamples, dtype=np.float64)
    for sample in range(n_resamples):
        chosen = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        draws[sample] = _metric(metric, y[chosen], a[chosen]) - _metric(metric, y[chosen], b[chosen])
    return {
        "a": _metric(metric, y, a),
        "b": _metric(metric, y, b),
        "difference": _metric(metric, y, a) - _metric(metric, y, b),
        "difference_ci_2.5": float(np.quantile(draws, 0.025)),
        "difference_ci_97.5": float(np.quantile(draws, 0.975)),
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_draws": draws,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _read_chunk_metadata(paths: Sequence[Path]) -> dict[str, Any]:
    ids: list[str] = []
    lengths: list[int] = []
    top1: list[int] = []
    chunk_rows: list[int] = []
    row_starts: list[int] = []
    layer_values: np.ndarray | None = None
    model_id: str | None = None
    regime: str | None = None
    required = {
        "X_bf16_u16",
        "item_ids",
        "layers",
        "token_lengths",
        "top1_token_ids",
        "row_start",
        "model_id",
        "regime",
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as chunk:
            missing = required - set(chunk.files)
            if missing:
                raise ValueError(f"{path} missing arrays {sorted(missing)}")
            raw_ids = chunk["item_ids"]
            raw_lengths = chunk["token_lengths"]
            raw_top1 = chunk["top1_token_ids"]
            raw_layers = chunk["layers"]
            raw_row_start = chunk["row_start"]
            raw_model = chunk["model_id"]
            raw_regime = chunk["regime"]
            if raw_ids.ndim != 1 or raw_ids.dtype.kind not in {"U", "S"}:
                raise ValueError(f"item_ids contract mismatch in {path}")
            rows = len(raw_ids)
            if (
                raw_lengths.shape != (rows,)
                or raw_top1.shape != (rows,)
                or raw_lengths.dtype.kind not in {"i", "u"}
                or raw_top1.dtype.kind not in {"i", "u"}
            ):
                raise ValueError(f"token metadata contract mismatch in {path}")
            if raw_layers.ndim != 1 or raw_layers.dtype.kind not in {"i", "u"}:
                raise ValueError(f"layer metadata contract mismatch in {path}")
            if raw_row_start.shape != () or raw_row_start.dtype.kind not in {"i", "u"}:
                raise ValueError(f"row_start contract mismatch in {path}")
            if raw_model.shape != () or raw_regime.shape != ():
                raise ValueError(f"scalar model/regime contract mismatch in {path}")
            chunk_ids = raw_ids.astype(str)
            ids.extend(chunk_ids.tolist())
            lengths.extend(raw_lengths.astype(np.int64).tolist())
            top1.extend(raw_top1.astype(np.int64).tolist())
            chunk_rows.append(rows)
            row_starts.append(int(raw_row_start.item()))
            current_layers = raw_layers.astype(np.int64)
            current_model = str(raw_model.item())
            current_regime = str(raw_regime.item())
            if layer_values is None:
                layer_values = current_layers
                model_id = current_model
                regime = current_regime
            elif (
                not np.array_equal(layer_values, current_layers)
                or model_id != current_model
                or regime != current_regime
            ):
                raise ValueError(f"inconsistent metadata in {path}")
    assert layer_values is not None and model_id is not None and regime is not None
    if any(right <= left for left, right in zip(row_starts, row_starts[1:])):
        raise ValueError("row_start values must increase strictly in chunk filename order")
    return {
        "item_ids": np.asarray(ids, dtype=str),
        "token_lengths": np.asarray(lengths, dtype=np.int64),
        "top1_token_ids": np.asarray(top1, dtype=np.int64),
        "chunk_rows": chunk_rows,
        "row_starts": row_starts,
        "layers": layer_values,
        "model_id": model_id,
        "regime": regime,
    }


def _chunk_paths(data_root: Path, model: str, regime: str, indices: Sequence[int]) -> list[Path]:
    paths = [data_root / model / regime / "chunks" / f"chunk_{index:04d}.npz" for index in indices]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing preregistered chunks: {missing[:3]}")
    present = sorted((data_root / model / regime / "chunks").glob("chunk_*.npz"))
    if [path.name for path in present] != [path.name for path in paths]:
        raise ValueError(f"chunk set for {model}/{regime} differs from preregistration")
    return paths


def _verify_and_index(config: dict[str, Any], data_root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    indices = config["dataset"]["selected_chunk_indices"]
    metadata: dict[str, Any] = {}
    reference_ids: np.ndarray | None = None
    reference_chunk_rows: list[int] | None = None
    reference_row_starts: list[int] | None = None
    for model, spec in config["dataset"]["expected_models"].items():
        metadata[model] = {}
        for regime in config["dataset"]["regimes"]:
            paths = _chunk_paths(data_root, model, regime, indices)
            current = _read_chunk_metadata(paths)
            current["paths"] = paths
            if current["model_id"] != spec["model_id"] or current["regime"] != regime:
                raise ValueError(f"model/regime metadata mismatch for {model}/{regime}")
            expected_layers = np.arange(spec["n_layers"], dtype=np.int64)
            if not np.array_equal(current["layers"], expected_layers):
                raise ValueError(f"layer metadata mismatch for {model}/{regime}")
            if len(current["item_ids"]) != config["dataset"]["expected_items_per_model_regime"]:
                raise ValueError(f"item count mismatch for {model}/{regime}")
            if reference_ids is None:
                reference_ids = current["item_ids"]
                reference_chunk_rows = current["chunk_rows"]
                reference_row_starts = current["row_starts"]
                if len(set(reference_ids.tolist())) != len(reference_ids):
                    raise ValueError("duplicate full item IDs are forbidden")
            elif not np.array_equal(reference_ids, current["item_ids"]):
                raise ValueError(f"ordered item IDs differ for {model}/{regime}")
            elif (
                current["chunk_rows"] != reference_chunk_rows
                or current["row_starts"] != reference_row_starts
            ):
                raise ValueError(f"chunk row metadata differs for {model}/{regime}")
            metadata[model][regime] = current
    assert reference_ids is not None
    selected_indices, cells = select_balanced_core(
        reference_ids,
        n_per_cell=config["selection"]["n_per_cell"],
        seed=config["selection"]["seed"],
    )
    if len(np.unique(selected_indices)) != len(selected_indices):
        raise ValueError("selected row indices must be unique")
    return metadata, selected_indices, cells


def _load_selected_layers(
    metadata: dict[str, Any],
    selected_indices: np.ndarray,
    selected_layers: Sequence[int],
    hidden_size: int,
) -> np.ndarray:
    output = np.empty(
        (len(selected_indices), len(selected_layers), hidden_size), dtype=np.uint16
    )
    offset = 0
    for path, rows in zip(metadata["paths"], metadata["chunk_rows"], strict=True):
        positions = np.flatnonzero(
            (selected_indices >= offset) & (selected_indices < offset + rows)
        )
        if positions.size:
            local_rows = selected_indices[positions] - offset
            with np.load(path, allow_pickle=False) as chunk:
                tensor = chunk["X_bf16_u16"]
                expected = (rows, len(metadata["layers"]), hidden_size)
                if tensor.shape != expected or tensor.dtype != np.uint16:
                    raise ValueError(f"activation contract mismatch in {path}: {tensor.shape} {tensor.dtype}")
                output[positions] = tensor[local_rows][:, selected_layers, :]
        offset += rows
    if offset != len(metadata["item_ids"]):
        raise AssertionError("chunk row accounting mismatch")
    return output


def _task_rows(cells: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    positive, negative = TASKS[task]
    rows = np.flatnonzero((cells == positive) | (cells == negative))
    labels = (cells[rows] == positive).astype(np.int8)
    return rows, labels


def _cross_validated_scores(
    raw: np.ndarray,
    template: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    ridge_fraction: float,
) -> tuple[dict[str, np.ndarray], list[float]]:
    scores = {code: np.empty(len(labels), dtype=np.float64) for code in TRANSFER_CODES}
    direction_cosines: list[float] = []
    for fold in sorted(np.unique(folds).tolist()):
        train = folds != fold
        test = ~train
        raw_model = fit_diag_lda(raw[train], labels[train], ridge_fraction)
        template_model = fit_diag_lda(template[train], labels[train], ridge_fraction)
        scores["RR"][test] = score_diag_lda(raw_model, raw[test])
        scores["RT"][test] = score_diag_lda(raw_model, template[test])
        scores["TT"][test] = score_diag_lda(template_model, template[test])
        scores["TR"][test] = score_diag_lda(template_model, raw[test])
        direction_cosines.append(
            _direction_cosine(raw_model["direction"], template_model["direction"])
        )
    return scores, direction_cosines


def _bootstrap_symmetric_penalty(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    seed: int,
    n_resamples: int,
) -> np.ndarray:
    """Bootstrap the four-score penalty with one shared paired item draw."""
    y = np.asarray(labels, dtype=np.int8)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == value) for value in (0, 1)]
    draws = np.empty(n_resamples, dtype=np.float64)
    for sample in range(n_resamples):
        chosen = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        draws[sample] = 0.5 * (
            roc_auc(y[chosen], scores["RR"][chosen])
            - roc_auc(y[chosen], scores["RT"][chosen])
            + roc_auc(y[chosen], scores["TT"][chosen])
            - roc_auc(y[chosen], scores["TR"][chosen])
        )
    return draws


def _record_metrics(
    model: str,
    layer: int,
    depth: float,
    task: str,
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    direction_cosines: Sequence[float],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model,
        "layer": int(layer),
        "normalized_depth": float(depth),
        "task": task,
        "n": int(len(labels)),
        "direction_cosine_mean": float(np.nanmean(direction_cosines)),
        "raw_template_oof_score_correlation": _safe_corr(scores["RR"], scores["TT"]),
    }
    for code in TRANSFER_CODES:
        result[f"{code}_auc"] = roc_auc(labels, scores[code])
        result[f"{code}_balanced_accuracy"] = balanced_accuracy(labels, scores[code])
    penalty_draws = _bootstrap_symmetric_penalty(
        labels, scores, bootstrap_seed, bootstrap_resamples
    )
    result.update(
        {
            "symmetric_penalty_auc": 0.5
            * ((result["RR_auc"] - result["RT_auc"]) + (result["TT_auc"] - result["TR_auc"])),
            "symmetric_penalty_auc_ci_2.5": float(np.quantile(penalty_draws, 0.025)),
            "symmetric_penalty_auc_ci_97.5": float(np.quantile(penalty_draws, 0.975)),
            "template_minus_raw_auc": result["TT_auc"] - result["RR_auc"],
        }
    )
    return result


def _permutation_null(
    raw: np.ndarray,
    template: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    count: int,
    seed: int,
    ridge_fraction: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    penalties: list[float] = []
    for _ in range(count):
        permuted = labels.copy()
        rng.shuffle(permuted)
        scores, _ = _cross_validated_scores(raw, template, permuted, folds, ridge_fraction)
        penalty = 0.5 * (
            (roc_auc(permuted, scores["RR"]) - roc_auc(permuted, scores["RT"]))
            + (roc_auc(permuted, scores["TT"]) - roc_auc(permuted, scores["TR"]))
        )
        penalties.append(float(penalty))
    values = np.asarray(penalties, dtype=np.float64)
    return {
        "n": count,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if count > 1 else 0.0,
        "q_2.5": float(np.quantile(values, 0.025)),
        "q_97.5": float(np.quantile(values, 0.975)),
        "values": penalties,
    }


def _token_diagnostics(
    model_metadata: dict[str, Any], selected_indices: np.ndarray, cells: np.ndarray
) -> dict[str, Any]:
    raw_lengths = model_metadata["raw"]["token_lengths"][selected_indices]
    template_lengths = model_metadata["template"]["token_lengths"][selected_indices]
    raw_top1 = model_metadata["raw"]["top1_token_ids"][selected_indices]
    template_top1 = model_metadata["template"]["top1_token_ids"][selected_indices]
    output: dict[str, Any] = {}
    for cell in ("EB", "DB", "EC", "DC", "ALL"):
        mask = np.ones(len(cells), dtype=bool) if cell == "ALL" else cells == cell
        differences = template_lengths[mask] - raw_lengths[mask]
        output[cell] = {
            "n": int(mask.sum()),
            "template_minus_raw_length_mean": float(differences.mean()),
            "template_minus_raw_length_min": int(differences.min()),
            "template_minus_raw_length_max": int(differences.max()),
            "top1_agreement": float((raw_top1[mask] == template_top1[mask]).mean()),
        }
    return output


def _bootstrap_primary(
    score_store: dict[tuple[str, str, float, str], np.ndarray],
    selected_cells: np.ndarray,
    primary_models: Sequence[str],
    primary_depths: Sequence[float],
    primary_tasks: Sequence[str],
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    task_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task in primary_tasks:
        rows, labels = _task_rows(selected_cells, task)
        task_data[task] = (rows, labels)
    point_values = []
    for model in primary_models:
        for depth in primary_depths:
            for task in primary_tasks:
                _, labels = task_data[task]
                values = score_store
                point_values.append(
                    0.5
                    * (
                        roc_auc(labels, values[(model, task, depth, "RR")])
                        - roc_auc(labels, values[(model, task, depth, "RT")])
                        + roc_auc(labels, values[(model, task, depth, "TT")])
                        - roc_auc(labels, values[(model, task, depth, "TR")])
                    )
                )
    draws = np.empty(n_resamples, dtype=np.float64)
    for sample in range(n_resamples):
        task_indices: dict[str, np.ndarray] = {}
        for task, (_, labels) in task_data.items():
            class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
            task_indices[task] = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
            )
        sampled_values = []
        for model in primary_models:
            for depth in primary_depths:
                for task in primary_tasks:
                    _, labels = task_data[task]
                    chosen = task_indices[task]
                    sampled_scores = score_store
                    sampled_values.append(
                        0.5
                        * (
                            roc_auc(labels[chosen], sampled_scores[(model, task, depth, "RR")][chosen])
                            - roc_auc(labels[chosen], sampled_scores[(model, task, depth, "RT")][chosen])
                            + roc_auc(labels[chosen], sampled_scores[(model, task, depth, "TT")][chosen])
                            - roc_auc(labels[chosen], sampled_scores[(model, task, depth, "TR")][chosen])
                        )
                    )
        draws[sample] = float(np.mean(sampled_values))
    return {
        "estimate": float(np.mean(point_values)),
        "ci_2.5": float(np.quantile(draws, 0.025)),
        "ci_97.5": float(np.quantile(draws, 0.975)),
        "bootstrap_mean": float(draws.mean()),
        "n_model_depth_task_cells": len(point_values),
        "bootstrap_resamples": n_resamples,
    }


def _make_figures(results: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = results["metrics"]
    model_order = ["8b", "70b", "405b"]
    colors = {"8b": "#0072B2", "70b": "#D55E00", "405b": "#009E73"}
    figures: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for axis, task in zip(axes, ("purpose_benchmark", "purpose_casual"), strict=True):
        for model in model_order:
            rows = sorted(
                [row for row in records if row["task"] == task and row["model"] == model],
                key=lambda row: row["normalized_depth"],
            )
            depth = [row["normalized_depth"] for row in rows]
            axis.plot(depth, [row["RR_auc"] for row in rows], color=colors[model], label=f"{model} raw")
            axis.plot(depth, [row["TT_auc"] for row in rows], color=colors[model], linestyle="--", label=f"{model} template")
        axis.axhline(0.5, color="0.5", linewidth=0.8)
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("normalized depth")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("held-out ROC AUC")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = output_dir / "auc_trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for axis, task in zip(axes, ("purpose_benchmark", "purpose_casual"), strict=True):
        for model in model_order:
            rows = sorted(
                [row for row in records if row["task"] == task and row["model"] == model],
                key=lambda row: row["normalized_depth"],
            )
            axis.plot(
                [row["normalized_depth"] for row in rows],
                [row["symmetric_penalty_auc"] for row in rows],
                marker="o",
                markersize=3,
                color=colors[model],
                label=model,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("normalized depth")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("symmetric cross-regime AUC penalty")
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "transfer_penalty.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, axis = plt.subplots(figsize=(5.2, 4.0))
    scale = results["scale_summary"]
    xs = [math.log10(scale[model]["parameter_count"]) for model in model_order]
    ys = [scale[model]["primary_penalty"] for model in model_order]
    axis.plot(xs, ys, marker="o", color="#CC79A7")
    for x, y, model in zip(xs, ys, model_order, strict=True):
        axis.annotate(model, (x, y), xytext=(4, 4), textcoords="offset points")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("log10 parameter count")
    axis.set_ylabel("primary-window regime penalty")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "scale_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return figures


def _write_metrics_tsv(records: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "layer",
        "normalized_depth",
        "task",
        "n",
        "RR_auc",
        "RT_auc",
        "TT_auc",
        "TR_auc",
        "RR_balanced_accuracy",
        "RT_balanced_accuracy",
        "TT_balanced_accuracy",
        "TR_balanced_accuracy",
        "symmetric_penalty_auc",
        "symmetric_penalty_auc_ci_2.5",
        "symmetric_penalty_auc_ci_97.5",
        "template_minus_raw_auc",
        "direction_cosine_mean",
        "raw_template_oof_score_correlation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        raise ValueError(f"manifest path must be repository-relative: {resolved}") from None


def _validate_config_contract(config: dict[str, Any]) -> None:
    expected_tasks = {
        "purpose_benchmark": {"positive_cells": ["EB"], "negative_cells": ["DB"]},
        "purpose_casual": {"positive_cells": ["EC"], "negative_cells": ["DC"]},
        "format_evaluation": {"positive_cells": ["EB"], "negative_cells": ["EC"]},
        "format_deployment": {"positive_cells": ["DB"], "negative_cells": ["DC"]},
    }
    configured_tasks = {
        **config["tasks"]["primary"],
        **config["tasks"]["secondary_controls"],
    }
    if configured_tasks != expected_tasks:
        raise ValueError("configured task definitions differ from implemented task contract")
    if config["selection"]["variant"] != "asis":
        raise ValueError("implemented selector requires variant=asis")
    if config["selection"]["expected_selected_rows"] != 4 * config["selection"]["n_per_cell"]:
        raise ValueError("expected selected-row count conflicts with four-cell selection")
    if list(config["dataset"]["expected_models"]) != ["8b", "70b", "405b"]:
        raise ValueError("implemented plotting/scale contract requires 8b,70b,405b order")
    if config["dataset"]["regimes"] != ["raw", "template"]:
        raise ValueError("implemented transfer contract requires raw,template order")
    output_names = [
        config["outputs"]["result_json"],
        config["outputs"]["table_tsv"],
        config["outputs"]["manifest_json"],
        *config["outputs"]["figures"],
    ]
    if len(output_names) != len(set(output_names)) or any(
        Path(name).name != name for name in output_names
    ):
        raise ValueError("output names must be unique basenames")


def run(config_path: Path, data_root: Path, out_dir: Path, run_id: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "devbunova_wrapper_scale_v1":
        raise ValueError("unexpected protocol")
    _validate_config_contract(config)
    incomplete = out_dir.with_name(out_dir.name + ".incomplete")
    if out_dir.exists() or incomplete.exists():
        raise FileExistsError(f"immutable output path already exists: {out_dir} or {incomplete}")

    metadata, selected_indices, cells = _verify_and_index(config, data_root)
    reference_ids = metadata["8b"]["raw"]["item_ids"]
    selected_ids = reference_ids[selected_indices]
    expected_cell_count = config["selection"]["n_per_cell"]
    selected_counts = {cell: int(np.sum(cells == cell)) for cell in ("EB", "DB", "EC", "DC")}
    if any(count != expected_cell_count for count in selected_counts.values()):
        raise ValueError(f"selected cell imbalance: {selected_counts}")
    if len(selected_ids) != config["selection"]["expected_selected_rows"]:
        raise ValueError("selected-row count differs from config")
    folds_all = stratified_cell_folds(
        selected_ids, cells, config["probe"]["folds"], config["probe"]["fold_seed"]
    )

    incomplete.mkdir(parents=True)
    try:
        metrics: list[dict[str, Any]] = []
        score_store: dict[tuple[str, str, float, str], np.ndarray] = {}
        nulls: dict[str, Any] = {}
        token_diagnostics: dict[str, Any] = {}
        depths = [float(value) for value in config["depths"]["normalized_grid"]]
        ridge_fraction = 0.1
        primary_tasks = list(config["tasks"]["primary"])
        all_tasks = primary_tasks + list(config["tasks"]["secondary_controls"])
        permutation_depth = float(config["depths"]["permutation_control_depth"])

        for model, spec in config["dataset"]["expected_models"].items():
            layers = fixed_depth_layers(spec["n_layers"], depths)
            raw_u16 = _load_selected_layers(
                metadata[model]["raw"], selected_indices, layers, spec["hidden_size"]
            )
            template_u16 = _load_selected_layers(
                metadata[model]["template"], selected_indices, layers, spec["hidden_size"]
            )
            token_diagnostics[model] = _token_diagnostics(metadata[model], selected_indices, cells)
            nulls[model] = {}
            for layer_slot, (layer, depth) in enumerate(zip(layers, depths, strict=True)):
                raw = bf16_u16_to_float32(raw_u16[:, layer_slot, :])
                template = bf16_u16_to_float32(template_u16[:, layer_slot, :])
                if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(template)):
                    raise ValueError(f"non-finite activation at {model} layer {layer}")
                for task in all_tasks:
                    task_rows, labels = _task_rows(cells, task)
                    task_folds = folds_all[task_rows]
                    scores, cosines = _cross_validated_scores(
                        raw[task_rows], template[task_rows], labels, task_folds, ridge_fraction
                    )
                    metrics.append(
                        _record_metrics(
                            model,
                            layer,
                            depth,
                            task,
                            labels,
                            scores,
                            cosines,
                            config["uncertainty"]["bootstrap_seed"] + layer + len(metrics),
                            config["uncertainty"]["bootstrap_resamples"],
                        )
                    )
                    for code in TRANSFER_CODES:
                        score_store[(model, task, depth, code)] = scores[code]
                    if depth == permutation_depth:
                        nulls[model][task] = _permutation_null(
                            raw[task_rows],
                            template[task_rows],
                            labels,
                            task_folds,
                            config["uncertainty"]["label_permutations"],
                            config["uncertainty"]["permutation_seed"]
                            + list(config["dataset"]["expected_models"]).index(model) * 100
                            + all_tasks.index(task),
                            ridge_fraction,
                        )
                del raw, template
            del raw_u16, template_u16

        primary_depths = [
            depth
            for depth in depths
            if config["depths"]["primary_window"][0]
            <= depth
            <= config["depths"]["primary_window"][1]
        ]
        models = list(config["dataset"]["expected_models"])
        primary = _bootstrap_primary(
            score_store,
            cells,
            models,
            primary_depths,
            primary_tasks,
            config["uncertainty"]["bootstrap_seed"] + 100000,
            config["uncertainty"]["bootstrap_resamples"],
        )

        scale_summary: dict[str, Any] = {}
        for model, spec in config["dataset"]["expected_models"].items():
            values = [
                record["symmetric_penalty_auc"]
                for record in metrics
                if record["model"] == model
                and record["task"] in primary_tasks
                and record["normalized_depth"] in primary_depths
            ]
            scale_summary[model] = {
                "parameter_count": spec["parameter_count"],
                "primary_penalty": float(np.mean(values)),
            }
        x = np.asarray(
            [math.log10(scale_summary[model]["parameter_count"]) for model in models],
            dtype=np.float64,
        )
        y = np.asarray([scale_summary[model]["primary_penalty"] for model in models])
        slope, intercept = np.polyfit(x, y, 1)
        scale_fit = {
            "slope_per_log10_parameter": float(slope),
            "intercept": float(intercept),
            "n_models": 3,
            "interpretation": "descriptive_only_unknown_model_revisions",
        }

        cross_model_correlations: list[dict[str, Any]] = []
        for task in all_tasks:
            for depth in depths:
                for code in TRANSFER_CODES:
                    for left_index, left in enumerate(models):
                        for right in models[left_index + 1 :]:
                            cross_model_correlations.append(
                                {
                                    "left_model": left,
                                    "right_model": right,
                                    "task": task,
                                    "normalized_depth": depth,
                                    "transfer_code": code,
                                    "score_correlation": _safe_corr(
                                        score_store[(left, task, depth, code)],
                                        score_store[(right, task, depth, code)],
                                    ),
                                }
                            )

        primary_task_model_means = {
            (model, task): float(
                np.mean(
                    [
                        record["symmetric_penalty_auc"]
                        for record in metrics
                        if record["model"] == model
                        and record["task"] == task
                        and record["normalized_depth"] in primary_depths
                    ]
                )
            )
            for model in models
            for task in primary_tasks
        }
        observed_q75 = float(
            np.mean(
                [
                    record["symmetric_penalty_auc"]
                    for record in metrics
                    if record["task"] in primary_tasks
                    and record["normalized_depth"] == permutation_depth
                ]
            )
        )
        label_null_values = np.mean(
            [
                nulls[model][task]["values"]
                for model in models
                for task in primary_tasks
            ],
            axis=0,
        )
        label_null_summary = {
            "observed_q75": observed_q75,
            "null_mean": float(np.mean(label_null_values)),
            "null_q_2.5": float(np.quantile(label_null_values, 0.025)),
            "null_q_97.5": float(np.quantile(label_null_values, 0.975)),
            "observed_exceeds_null_q97.5": bool(
                observed_q75 > np.quantile(label_null_values, 0.975)
            ),
        }
        all_task_model_positive = all(
            value > 0 for value in primary_task_model_means.values()
        )
        decision = "reversed_or_mixed"
        if (
            primary["ci_2.5"] > 0
            and all_task_model_positive
            and all(scale_summary[model]["primary_penalty"] > 0 for model in models)
        ):
            decision = "regime_specific"
        elif (
            primary["ci_2.5"] <= 0 <= primary["ci_97.5"]
            and not label_null_summary["observed_exceeds_null_q97.5"]
        ):
            decision = "regime_invariant"

        result: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed_development_observational",
            "analysis": "devbunova_wrapper_scale_v1",
            "decision": decision,
            "evidence_boundary": config["evidence_boundary"],
            "selection": {
                "available_rows": len(reference_ids),
                "selected_rows": len(selected_ids),
                "selected_counts": selected_counts,
                "ordered_selected_item_ids_sha256": canonical_json_sha256(selected_ids.tolist()),
            },
            "primary": primary,
            "scale_summary": scale_summary,
            "scale_fit": scale_fit,
            "token_diagnostics": token_diagnostics,
            "permutation_nulls": nulls,
            "label_permutation_aggregate": label_null_summary,
            "primary_task_model_means": {
                f"{model}:{task}": value
                for (model, task), value in primary_task_model_means.items()
            },
            "metrics": metrics,
            "cross_model_score_correlations": cross_model_correlations,
            "limitations": [
                "Core cells are independent prompt distributions rather than matched factorial payloads.",
                "The operational semantics of raw/template and asis are undocumented in the activation repository.",
                "Model revisions and extraction hook semantics are undocumented.",
                "The three-point parameter-count trajectory is descriptive and not a causal scale estimate.",
                "Top-token changes can alter the represented token position across regimes.",
                "No behavioral or interventional endpoint is evaluated.",
            ],
        }

        result_path = incomplete / config["outputs"]["result_json"]
        result_path.write_text(strict_json_text(result), encoding="utf-8")
        table_path = incomplete / config["outputs"]["table_tsv"]
        _write_metrics_tsv(metrics, table_path)
        figure_paths = _make_figures(result, incomplete)

        selected_file_hashes: dict[str, dict[str, Any]] = {}
        for model in models:
            for regime in config["dataset"]["regimes"]:
                for path in metadata[model][regime]["paths"]:
                    relative = path.relative_to(data_root).as_posix()
                    selected_file_hashes[relative] = {
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
        output_paths = [result_path, table_path, *figure_paths]
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed_development_observational",
            "analysis": "devbunova_wrapper_scale_v1",
            "repo_commit": _repo_commit(),
            "protocol_path": _repo_relative(config_path),
            "protocol_sha256": sha256_file(config_path),
            "code_path": _repo_relative(Path(__file__)),
            "code_sha256": sha256_file(Path(__file__)),
            "dataset_repo_id": config["dataset"]["repo_id"],
            "dataset_revision": config["dataset"]["revision"],
            "selected_file_sha256": selected_file_hashes,
            "ordered_all_item_ids_sha256": canonical_json_sha256(reference_ids.tolist()),
            "ordered_selected_item_ids_sha256": canonical_json_sha256(selected_ids.tolist()),
            "models": config["dataset"]["expected_models"],
            "stored_dtype": config["dataset"]["stored_dtype"],
            "ordered_item_ids_sha256": canonical_json_sha256(reference_ids.tolist()),
            "layer_semantics": config["evidence_boundary"]["layer_warning"],
            "token_position": "undocumented",
            "split_definition": config["probe"]["fold_assignment"],
            "regularization": config["probe"]["ridge"],
            "seeds": {
                "selection": config["selection"]["seed"],
                "fold": config["probe"]["fold_seed"],
                "bootstrap": config["uncertainty"]["bootstrap_seed"],
                "permutation": config["uncertainty"]["permutation_seed"],
            },
            "input_hashes": {
                "protocol": sha256_file(config_path),
                "code": sha256_file(Path(__file__)),
                "all_item_ids": canonical_json_sha256(reference_ids.tolist()),
                "selected_item_ids": canonical_json_sha256(selected_ids.tolist()),
            },
            "output_hashes": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in output_paths
            },
            "limitations": result["limitations"],
        }
        manifest_path = incomplete / config["outputs"]["manifest_json"]
        manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")
        incomplete.replace(out_dir)
        return result
    except BaseException:
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.config, args.data_root, args.out_dir, args.run_id)
    print(args.out_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
