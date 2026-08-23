#!/usr/bin/env python3
"""Post-outcome common-mask robustness checks for wrapper-scale analysis v1."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.devbunova_wrapper_scale import (
        _cross_validated_scores,
        _load_selected_layers,
        _repo_relative,
        _task_rows,
        _verify_and_index,
        bf16_u16_to_float32,
        canonical_json_sha256,
        fixed_depth_layers,
        roc_auc,
        sha256_file,
        stratified_cell_folds,
        strict_json_text,
    )
except ImportError:
    from devbunova_wrapper_scale import (
        _cross_validated_scores,
        _load_selected_layers,
        _repo_relative,
        _task_rows,
        _verify_and_index,
        bf16_u16_to_float32,
        canonical_json_sha256,
        fixed_depth_layers,
        roc_auc,
        sha256_file,
        strict_json_text,
        stratified_cell_folds,
    )


def row_l2_normalize(features: np.ndarray) -> np.ndarray:
    """Return row-normalized float32 features, failing on zero/non-finite norms."""
    values = np.asarray(features, dtype=np.float32)
    values64 = values.astype(np.float64)
    norms = np.sqrt(np.sum(values64 * values64, axis=1))
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("row L2 normalization requires finite nonzero norms")
    return np.asarray(values / norms[:, None], dtype=np.float32)


def paired_regime_swap(
    raw: np.ndarray, template: np.ndarray, swap_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Swap raw/template rows within selected item pairs according to ``swap_mask``."""
    left = np.asarray(raw)
    right = np.asarray(template)
    mask = np.asarray(swap_mask, dtype=bool)
    if left.shape != right.shape or mask.shape != (left.shape[0],):
        raise ValueError("paired swap inputs do not align")
    swapped_raw = left.copy()
    swapped_template = right.copy()
    swapped_raw[mask] = right[mask]
    swapped_template[mask] = left[mask]
    return swapped_raw, swapped_template


def symmetric_penalty(labels: np.ndarray, scores: dict[str, np.ndarray]) -> float:
    """Calculate the fixed symmetric cross-regime ROC-AUC penalty."""
    return float(
        0.5
        * (
            roc_auc(labels, scores["RR"])
            - roc_auc(labels, scores["RT"])
            + roc_auc(labels, scores["TT"])
            - roc_auc(labels, scores["TR"])
        )
    )


def evaluate(
    raw: np.ndarray,
    template: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    ridge_fraction: float,
) -> dict[str, Any]:
    scores, cosines = _cross_validated_scores(
        raw, template, labels, folds, ridge_fraction
    )
    return {
        "penalty": symmetric_penalty(labels, scores),
        "RR_auc": roc_auc(labels, scores["RR"]),
        "RT_auc": roc_auc(labels, scores["RT"]),
        "TT_auc": roc_auc(labels, scores["TT"]),
        "TR_auc": roc_auc(labels, scores["TR"]),
        "direction_cosine_mean": float(np.nanmean(cosines)),
    }


def _repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _plot_matrix(records: Sequence[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["8b", "70b", "405b"]
    combinations = [
        (normalization, ridge)
        for normalization in ("none", "l2")
        for ridge in (0.01, 0.1, 1.0)
    ]
    matrix = np.empty((len(models), len(combinations)), dtype=np.float64)
    for model_index, model in enumerate(models):
        for combination_index, (normalization, ridge) in enumerate(combinations):
            values = [
                row["penalty"]
                for row in records
                if row["analysis"] == "ridge_normalization"
                and row["model"] == model
                and row["task"] in {"purpose_benchmark", "purpose_casual"}
                and row["normalization"] == normalization
                and row["ridge_fraction"] == ridge
            ]
            matrix[model_index, combination_index] = float(np.mean(values))
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    image = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    axes[0].set_yticks(range(len(models)), models)
    axes[0].set_xticks(
        range(len(combinations)),
        [f"{normalization}\nr={ridge:g}" for normalization, ridge in combinations],
        rotation=30,
        ha="right",
    )
    axes[0].set_title("Purpose penalty sensitivity")
    figure.colorbar(image, ax=axes[0], shrink=0.8)

    null = np.asarray(summary["paired_regime_swap_null"]["aggregate_values"])
    axes[1].hist(null, bins=12, color="0.65", edgecolor="white")
    axes[1].axvline(summary["paired_regime_swap_null"]["observed_aggregate"], color="#D55E00", linewidth=2, label="observed")
    axes[1].set_xlabel("aggregate q=.75 penalty")
    axes[1].set_ylabel("swap-null count")
    axes[1].set_title("Paired regime-swap null")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    config_path: Path,
    parent_result_path: Path,
    parent_manifest_path: Path,
    data_root: Path,
    out_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "devbunova_wrapper_scale_robustness_v3":
        raise ValueError("unexpected Monte Carlo precision protocol")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if _repo_relative(parent_result_path) != config["parent_result"]:
        raise ValueError("parent result path differs from corrected protocol")
    if _repo_relative(parent_manifest_path) != config["parent_manifest"]:
        raise ValueError("parent manifest path differs from corrected protocol")
    if sha256_file(parent_result_path) != config["parent_result_sha256"]:
        raise ValueError("parent result hash mismatch")
    if sha256_file(parent_manifest_path) != config["parent_manifest_sha256"]:
        raise ValueError("parent manifest hash mismatch")
    if parent_result.get("run_id") != config["parent_run_id"]:
        raise ValueError("parent run mismatch")
    incomplete = out_dir.with_name(out_dir.name + ".incomplete")
    if out_dir.exists() or incomplete.exists():
        raise FileExistsError(f"immutable output already exists: {out_dir} or {incomplete}")

    parent_protocol_path = Path(parent_manifest["protocol_path"])
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    if sha256_file(parent_protocol_path) != parent_manifest["protocol_sha256"]:
        raise ValueError("parent protocol hash mismatch")
    parent_code_path = Path(parent_manifest["code_path"])
    if sha256_file(parent_code_path) != parent_manifest["code_sha256"]:
        raise ValueError("parent code hash mismatch")
    if parent_protocol["dataset"]["revision"] != config["data_contract"]["revision"]:
        raise ValueError("dataset revision mismatch")
    metadata, selected_indices, cells = _verify_and_index(parent_protocol, data_root)
    selected_ids = metadata["8b"]["raw"]["item_ids"][selected_indices]
    selected_hash = canonical_json_sha256(selected_ids.tolist())
    if selected_hash != parent_result["selection"]["ordered_selected_item_ids_sha256"]:
        raise ValueError("parent selected-item hash mismatch")
    for relative, expected in parent_manifest["selected_file_sha256"].items():
        path = data_root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"parent input size/path mismatch: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"parent input hash mismatch: {relative}")

    incomplete.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    swap_values: dict[str, dict[str, list[float]]] = {}
    task_order = list(config["fixed_endpoint"]["tasks"])
    base_seed = parent_protocol["probe"]["fold_seed"]
    fixed_depth = float(config["fixed_endpoint"]["normalized_depth"])
    swap_count = config["checks"]["paired_regime_swap_permutations"]
    swap_masks: dict[str, list[np.ndarray]] = {}
    for task_index, task in enumerate(task_order):
        task_rows, labels = _task_rows(cells, task)
        rng = np.random.default_rng(
            config["checks"]["paired_regime_swap_seed"] + task_index
        )
        swap_masks[task] = [
            rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
            for _ in range(swap_count)
        ]

    for model, spec in parent_protocol["dataset"]["expected_models"].items():
        layer = fixed_depth_layers(spec["n_layers"], [fixed_depth])[0]
        raw_u16 = _load_selected_layers(
            metadata[model]["raw"], selected_indices, [layer], spec["hidden_size"]
        )[:, 0, :]
        template_u16 = _load_selected_layers(
            metadata[model]["template"], selected_indices, [layer], spec["hidden_size"]
        )[:, 0, :]
        raw = bf16_u16_to_float32(raw_u16)
        template = bf16_u16_to_float32(template_u16)
        normalized = {
            "none": (raw, template),
            "l2": (row_l2_normalize(raw), row_l2_normalize(template)),
        }
        swap_values[model] = {}

        for task in task_order:
            task_rows, labels = _task_rows(cells, task)
            task_ids = selected_ids[task_rows]
            task_cells = cells[task_rows]
            for fold_seed in config["checks"]["fold_seeds"]:
                folds = stratified_cell_folds(
                    task_ids,
                    task_cells,
                    parent_protocol["probe"]["folds"],
                    fold_seed,
                )
                outcome = evaluate(
                    raw[task_rows], template[task_rows], labels, folds, 0.1
                )
                records.append(
                    {
                        "analysis": "fold_seed",
                        "model": model,
                        "layer": layer,
                        "normalized_depth": fixed_depth,
                        "task": task,
                        "fold_seed": fold_seed,
                        "normalization": "none",
                        "ridge_fraction": 0.1,
                        **outcome,
                    }
                )

            base_folds = stratified_cell_folds(
                task_ids,
                task_cells,
                parent_protocol["probe"]["folds"],
                base_seed,
            )
            for normalization in config["checks"]["row_normalizations"]:
                current_raw, current_template = normalized[normalization]
                for ridge in config["checks"]["ridge_fractions"]:
                    outcome = evaluate(
                        current_raw[task_rows],
                        current_template[task_rows],
                        labels,
                        base_folds,
                        float(ridge),
                    )
                    records.append(
                        {
                            "analysis": "ridge_normalization",
                            "model": model,
                            "layer": layer,
                            "normalized_depth": fixed_depth,
                            "task": task,
                            "fold_seed": base_seed,
                            "normalization": normalization,
                            "ridge_fraction": float(ridge),
                            **outcome,
                        }
                    )

            task_masks = swap_masks[task]
            task_raw = raw[task_rows]
            task_template = template[task_rows]
            task_swap_values: list[float] = []
            for mask in task_masks:
                swapped_raw, swapped_template = paired_regime_swap(
                    task_raw, task_template, mask
                )
                scores, _ = _cross_validated_scores(
                    swapped_raw, swapped_template, labels, base_folds, 0.1
                )
                task_swap_values.append(symmetric_penalty(labels, scores))
            swap_values[model][task] = task_swap_values
        del raw_u16, template_u16, raw, template, normalized

    primary_tasks = set(config["fixed_endpoint"]["primary_tasks_for_aggregate"])
    models = list(parent_protocol["dataset"]["expected_models"])
    base_records = [
        row
        for row in records
        if row["analysis"] == "ridge_normalization"
        and row["fold_seed"] == base_seed
        and row["normalization"] == "none"
        and row["ridge_fraction"] == 0.1
        and row["task"] in primary_tasks
    ]
    observed_aggregate = float(np.mean([row["penalty"] for row in base_records]))
    aggregate_null = np.mean(
        [swap_values[model][task] for model in models for task in sorted(primary_tasks)],
        axis=0,
    )
    permutation_p = float(
        (1 + np.sum(aggregate_null >= observed_aggregate)) / (1 + swap_count)
    )

    ridge_normalization_model = {}
    fold_seed_model = {}
    all_sensitivity_positive = True
    all_seed_medians_positive = True
    for model in models:
        combination_values = {}
        for normalization in config["checks"]["row_normalizations"]:
            for ridge in config["checks"]["ridge_fractions"]:
                values = [
                    row["penalty"]
                    for row in records
                    if row["analysis"] == "ridge_normalization"
                    and row["model"] == model
                    and row["task"] in primary_tasks
                    and row["normalization"] == normalization
                    and row["ridge_fraction"] == ridge
                ]
                aggregate = float(np.mean(values))
                combination_values[f"{normalization}_ridge_{ridge:g}"] = aggregate
                all_sensitivity_positive &= aggregate > 0
        ridge_normalization_model[model] = combination_values
        seed_values = []
        for fold_seed in config["checks"]["fold_seeds"]:
            values = [
                row["penalty"]
                for row in records
                if row["analysis"] == "fold_seed"
                and row["model"] == model
                and row["task"] in primary_tasks
                and row["fold_seed"] == fold_seed
            ]
            seed_values.append(float(np.mean(values)))
        median = float(np.median(seed_values))
        fold_seed_model[model] = {
            "values": seed_values,
            "median": median,
            "min": float(np.min(seed_values)),
            "max": float(np.max(seed_values)),
        }
        all_seed_medians_positive &= median > 0

    swap_robust = observed_aggregate > float(np.quantile(aggregate_null, 0.975))
    decision = (
        "robust_positive"
        if all_sensitivity_positive and all_seed_medians_positive and swap_robust
        else "mixed"
    )
    summary = {
        "decision": decision,
        "observed_aggregate": observed_aggregate,
        "ridge_normalization_by_model": ridge_normalization_model,
        "fold_seed_by_model": fold_seed_model,
        "paired_regime_swap_null": {
            "observed_aggregate": observed_aggregate,
            "aggregate_values": aggregate_null.tolist(),
            "mean": float(np.mean(aggregate_null)),
            "q_2.5": float(np.quantile(aggregate_null, 0.025)),
            "q_97.5": float(np.quantile(aggregate_null, 0.975)),
            "empirical_p_greater_equal": permutation_p,
            "n": swap_count,
            "by_model_task": swap_values,
        },
        "criteria": {
            "all_ridge_normalization_model_aggregates_positive": bool(all_sensitivity_positive),
            "all_model_fold_seed_medians_positive": bool(all_seed_medians_positive),
            "observed_exceeds_swap_null_q97.5": bool(swap_robust),
        },
    }
    result = {
        "schema_version": 3,
        "run_id": run_id,
        "status": "completed_post_outcome_monte_carlo_precision",
        "analysis": config["protocol_id"],
        "parent_run_id": config["parent_run_id"],
        "parent_result_sha256": sha256_file(parent_result_path),
        "selected_item_ids_sha256": selected_hash,
        "summary": summary,
        "records": records,
        "evidence_boundary": config["evidence_boundary"],
    }

    result_path = incomplete / config["outputs"]["result_json"]
    result_path.write_text(strict_json_text(result), encoding="utf-8")
    table_path = incomplete / config["outputs"]["table_tsv"]
    fields = [
        "analysis",
        "model",
        "layer",
        "normalized_depth",
        "task",
        "fold_seed",
        "normalization",
        "ridge_fraction",
        "penalty",
        "RR_auc",
        "RT_auc",
        "TT_auc",
        "TR_auc",
        "direction_cosine_mean",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    figure_path = incomplete / config["outputs"]["figure"]
    _plot_matrix(records, summary, figure_path)

    manifest = {
        "schema_version": 3,
        "run_id": run_id,
        "status": "completed_post_outcome_monte_carlo_precision",
        "analysis": config["protocol_id"],
        "repo_commit": _repo_commit(),
        "protocol_path": _repo_relative(config_path),
        "protocol_sha256": sha256_file(config_path),
        "code_path": _repo_relative(Path(__file__)),
        "code_sha256": sha256_file(Path(__file__)),
        "parent_run_id": config["parent_run_id"],
        "parent_result_sha256": sha256_file(parent_result_path),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "dataset_repo_id": config["data_contract"]["repo_id"],
        "dataset_revision": config["data_contract"]["revision"],
        "models": parent_protocol["dataset"]["expected_models"],
        "stored_dtype": config["data_contract"]["stored_dtype"],
        "layer_semantics": config["data_contract"]["layer_semantics"],
        "token_position": config["data_contract"]["token_position"],
        "split_definition": parent_protocol["probe"]["fold_assignment"],
        "regularization": parent_protocol["probe"]["ridge"],
        "seeds": {
            "folds": config["checks"]["fold_seeds"],
            "swap": config["checks"]["paired_regime_swap_seed"],
        },
        "selected_file_sha256": parent_manifest["selected_file_sha256"],
        "selected_item_ids_sha256": selected_hash,
        "input_hashes": {
            "parent_result": sha256_file(parent_result_path),
            "parent_manifest": sha256_file(parent_manifest_path),
            "protocol": sha256_file(config_path),
            "code": sha256_file(Path(__file__)),
            "selected_item_ids": selected_hash,
        },
        "output_hashes": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (result_path, table_path, figure_path)
        },
        "evidence_boundary": config["evidence_boundary"],
    }
    manifest_path = incomplete / config["outputs"]["manifest_json"]
    manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")
    incomplete.replace(out_dir)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        args.config,
        args.parent_result,
        args.parent_manifest,
        args.data_root,
        args.out_dir,
        args.run_id,
    )
    print(args.out_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
