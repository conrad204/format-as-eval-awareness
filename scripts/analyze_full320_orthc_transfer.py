#!/usr/bin/env python3
"""Analyze complete full320 orthogonalized-C transfer runs under frozen rules."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

PROTOCOL_ID = "full320_orthogonalized_c_transfer_v1"
ENDPOINTS = ("shared", "interaction", "format")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def response(scores: dict[str, float]) -> dict[str, float]:
    benchmark = float(scores["purpose_benchmark"])
    casual = float(scores["purpose_casual"])
    return {
        "shared": (benchmark + casual) / 2.0,
        "interaction": (benchmark - casual) / 2.0,
        "format": float(scores["format"]),
    }


def finite_difference(plus: dict[str, Any], minus: dict[str, Any]) -> dict[str, float]:
    if plus["format"] != minus["format"] or plus["item_id"] != minus["item_id"]:
        raise ValueError("finite-difference pair mismatch")
    positive = response(plus["readout_scores"])
    negative = response(minus["readout_scores"])
    return {endpoint: (positive[endpoint] - negative[endpoint]) / 2.0 for endpoint in ENDPOINTS}


def percentile_ci(values: np.ndarray, rng: np.random.Generator, resamples: int) -> list[float]:
    n = len(values)
    draws = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, n, size=(count, n))
        draws[start:start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def one_sided_t(values: np.ndarray) -> float:
    result = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
    return float(result.pvalue)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def load_complete(run_dir: Path, expected_model: dict[str, Any], config_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_dir / "runner_manifest.json"
    raw_path = run_dir / "causal_raw.jsonl"
    if not manifest_path.is_file() or not raw_path.is_file() or (run_dir / "INVALID.json").exists():
        raise ValueError(f"run is not complete and valid: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_uninspected_full320_orthc_transfer_raw"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("model_id") != expected_model["model_id"]
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("records") != manifest.get("expected_records")
        or manifest.get("raw_sha256") != sha256_file(raw_path)
    ):
        raise ValueError(f"run manifest contract mismatch: {run_dir}")
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != manifest["records"]:
        raise ValueError(f"raw record count mismatch: {run_dir}")
    return records, manifest


def model_block_effects(records: Sequence[dict[str, Any]], model: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], np.ndarray]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (record["fold_id"], int(record["source_layer"]), int(record["row_index"]), record["condition_id"])
        if key in by_key:
            raise ValueError(f"duplicate raw key: {key}")
        by_key[key] = record
    row_values: list[dict[str, Any]] = []
    block_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        condition = record["condition_id"]
        if record["sign"] != 1:
            continue
        if record["condition_type"] == "component" and record["component"] == "C":
            minus_condition = condition.replace("C_p_", "C_m_")
            family = "C"
            control = None
        elif record["condition_type"] == "random":
            minus_condition = condition.replace("_p_alpha1", "_m_alpha1")
            family = "random"
            control = int(record["random_seed"])
        else:
            continue
        minus_key = (record["fold_id"], int(record["source_layer"]), int(record["row_index"]), minus_condition)
        if minus_key not in by_key:
            raise ValueError(f"missing sign pair: {minus_key}")
        values = finite_difference(record, by_key[minus_key])
        alpha = float(record["alpha"])
        depth = record["source_depth_label"]
        for endpoint, point in values.items():
            block_values[(family, control, alpha, depth, endpoint, record["payload_block_id"])].append(point)
            row_values.append({
                "model_id": model["model_id"],
                "payload_block_id": record["payload_block_id"],
                "family": family,
                "control_seed": control,
                "alpha": alpha,
                "depth": depth,
                "endpoint": endpoint,
                "point": point,
            })
    collapsed = {key: np.asarray(values, dtype=np.float64).mean() for key, values in block_values.items()}
    return row_values, collapsed


def vector_by_blocks(collapsed: dict[tuple[Any, ...], float], family: str, control: int | None, alpha: float, depth: str, endpoint: str) -> tuple[list[str], np.ndarray]:
    pairs = sorted(
        (key[-1], value) for key, value in collapsed.items()
        if key[:5] == (family, control, alpha, depth, endpoint)
    )
    if len(pairs) != 80:
        raise ValueError(f"expected 80 payload-block values, got {len(pairs)} for {(family, control, alpha, depth, endpoint)}")
    return [block for block, _ in pairs], np.asarray([value for _, value in pairs], dtype=np.float64)


def analyze(config_path: Path, run_dirs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("analysis protocol mismatch")
    analysis_contract = protocol["analysis"]
    if analysis_contract["script_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("analysis script differs from frozen hash")
    config_sha256 = sha256_file(config_path)
    rng = np.random.default_rng(int(analysis_contract["bootstrap_seed"]))
    model_results: dict[str, Any] = {}
    all_block_rows: list[dict[str, Any]] = []
    primary_p: dict[str, float] = {}
    high_p: dict[str, float] = {}
    primary_vectors: dict[str, tuple[list[str], np.ndarray]] = {}
    manifests: dict[str, Any] = {}

    for model in protocol["models"]:
        model_id = model["model_id"]
        records, manifest = load_complete(run_dirs[model_id], model, config_sha256)
        manifests[model_id] = manifest
        _, collapsed = model_block_effects(records, model)
        blocks, high = vector_by_blocks(collapsed, "C", None, 1.0, "high_interaction", "interaction")
        low_blocks, low = vector_by_blocks(collapsed, "C", None, 1.0, "low_interaction", "interaction")
        if blocks != low_blocks:
            raise ValueError("high/low payload blocks differ")
        difference = high - low
        primary_vectors[model_id] = (blocks, difference)
        primary_p[model_id] = one_sided_t(difference)
        high_p[model_id] = one_sided_t(high)
        random_means = []
        random_rows = []
        for seed in protocol["interventions"]["random_seeds"]:
            random_blocks, random_high = vector_by_blocks(collapsed, "random", int(seed), 1.0, "high_interaction", "interaction")
            low_random_blocks, random_low = vector_by_blocks(collapsed, "random", int(seed), 1.0, "low_interaction", "interaction")
            if random_blocks != blocks or low_random_blocks != blocks:
                raise ValueError("random/C payload blocks differ")
            random_difference = random_high - random_low
            random_means.append(float(random_difference.mean()))
            random_rows.extend({
                "model_id": model_id, "payload_block_id": block, "family": "random_high_minus_low",
                "control_seed": int(seed), "alpha": 1.0, "depth": "high_minus_low",
                "endpoint": "interaction", "point": float(value),
            } for block, value in zip(blocks, random_difference, strict=True))
        random_means_array = np.asarray(random_means)
        _, half_high = vector_by_blocks(collapsed, "C", None, 0.5, "high_interaction", "interaction")
        _, half_low = vector_by_blocks(collapsed, "C", None, 0.5, "low_interaction", "interaction")
        half_difference = half_high - half_low
        specificity_threshold = float(np.quantile(np.abs(random_means_array), 0.975))
        collateral: dict[str, Any] = {}
        collateral_vectors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for endpoint in ("shared", "format"):
            endpoint_blocks, endpoint_high = vector_by_blocks(
                collapsed, "C", None, 1.0, "high_interaction", endpoint
            )
            endpoint_low_blocks, endpoint_low = vector_by_blocks(
                collapsed, "C", None, 1.0, "low_interaction", endpoint
            )
            if endpoint_blocks != blocks or endpoint_low_blocks != blocks:
                raise ValueError("collateral/C payload blocks differ")
            endpoint_difference = endpoint_high - endpoint_low
            collateral[endpoint] = {
                "high_mean": float(endpoint_high.mean()),
                "low_mean": float(endpoint_low.mean()),
                "high_minus_low_mean": float(endpoint_difference.mean()),
                "high_minus_low_sd": float(endpoint_difference.std(ddof=1)),
            }
            collateral_vectors[endpoint] = (endpoint_high, endpoint_low, endpoint_difference)
        model_results[model_id] = {
            "n_payload_blocks": len(blocks),
            "primary_high_minus_low": {
                "mean": float(difference.mean()),
                "sd": float(difference.std(ddof=1)),
                "ci95_payload_bootstrap": percentile_ci(difference, rng, int(analysis_contract["bootstrap_resamples"])),
                "one_sided_t_p_unadjusted": primary_p[model_id],
                "all_blocks_positive": bool(np.all(difference > 0)),
            },
            "high_layer_C_interaction": {
                "mean": float(high.mean()),
                "sd": float(high.std(ddof=1)),
                "ci95_payload_bootstrap": percentile_ci(high, rng, int(analysis_contract["bootstrap_resamples"])),
                "one_sided_t_p_unadjusted": high_p[model_id],
            },
            "low_layer_C_interaction": {
                "mean": float(low.mean()),
                "sd": float(low.std(ddof=1)),
                "ci95_payload_bootstrap": percentile_ci(low, rng, int(analysis_contract["bootstrap_resamples"])),
            },
            "half_alpha_high_minus_low": {
                "mean": float(half_difference.mean()),
                "sd": float(half_difference.std(ddof=1)),
                "ratio_to_alpha1": float(half_difference.mean() / difference.mean()) if difference.mean() != 0 else None,
            },
            "random_control_calibration": {
                "n_directions": len(random_means),
                "high_minus_low_means": random_means,
                "absolute_mean_97_5_percentile": specificity_threshold,
                "C_exceeds_absolute_random_97_5_percentile": bool(difference.mean() > specificity_threshold),
            },
            "collateral_endpoints_descriptive": collateral,
        }
        for block, high_value, low_value, diff_value, half_value in zip(blocks, high, low, difference, half_difference, strict=True):
            all_block_rows.extend([
                {"model_id": model_id, "payload_block_id": block, "family": "C", "control_seed": "", "alpha": 1.0, "depth": "high_interaction", "endpoint": "interaction", "point": float(high_value)},
                {"model_id": model_id, "payload_block_id": block, "family": "C", "control_seed": "", "alpha": 1.0, "depth": "low_interaction", "endpoint": "interaction", "point": float(low_value)},
                {"model_id": model_id, "payload_block_id": block, "family": "C_high_minus_low", "control_seed": "", "alpha": 1.0, "depth": "high_minus_low", "endpoint": "interaction", "point": float(diff_value)},
                {"model_id": model_id, "payload_block_id": block, "family": "C_high_minus_low", "control_seed": "", "alpha": 0.5, "depth": "high_minus_low", "endpoint": "interaction", "point": float(half_value)},
            ])
        all_block_rows.extend(random_rows)
        for endpoint, (endpoint_high, endpoint_low, endpoint_difference) in collateral_vectors.items():
            for block, high_value, low_value, diff_value in zip(
                blocks, endpoint_high, endpoint_low, endpoint_difference, strict=True
            ):
                all_block_rows.extend([
                    {"model_id": model_id, "payload_block_id": block, "family": "C", "control_seed": "", "alpha": 1.0, "depth": "high_interaction", "endpoint": endpoint, "point": float(high_value)},
                    {"model_id": model_id, "payload_block_id": block, "family": "C", "control_seed": "", "alpha": 1.0, "depth": "low_interaction", "endpoint": endpoint, "point": float(low_value)},
                    {"model_id": model_id, "payload_block_id": block, "family": "C_high_minus_low", "control_seed": "", "alpha": 1.0, "depth": "high_minus_low", "endpoint": endpoint, "point": float(diff_value)},
                ])

    primary_adjusted = holm_adjust(primary_p)
    high_adjusted = holm_adjust(high_p)
    for model_id in model_results:
        model_results[model_id]["primary_high_minus_low"]["holm_p_two_models"] = primary_adjusted[model_id]
        model_results[model_id]["high_layer_C_interaction"]["holm_p_two_models"] = high_adjusted[model_id]
    model_ids = [model["model_id"] for model in protocol["models"]]
    blocks_a, vector_a = primary_vectors[model_ids[0]]
    blocks_b, vector_b = primary_vectors[model_ids[1]]
    if blocks_a != blocks_b:
        raise ValueError("cross-model payload block mismatch")
    pearson = stats.pearsonr(vector_a, vector_b)
    spearman = stats.spearmanr(vector_a, vector_b)

    alpha = float(analysis_contract["familywise_alpha"])
    positive = all(
        model_results[model_id]["primary_high_minus_low"]["mean"] > 0
        and primary_adjusted[model_id] <= alpha
        and model_results[model_id]["high_layer_C_interaction"]["mean"] > 0
        and high_adjusted[model_id] <= alpha
        and model_results[model_id]["half_alpha_high_minus_low"]["mean"] > 0
        and model_results[model_id]["random_control_calibration"]["C_exceeds_absolute_random_97_5_percentile"]
        for model_id in model_ids
    )
    result = {
        "schema_version": 1,
        "status": "complete_frozen_transfer_analysis",
        "decision": "positive_replication" if positive else "clean_negative_or_mixed_replication",
        "protocol_id": PROTOCOL_ID,
        "config_path": config_path.as_posix(),
        "config_sha256": config_sha256,
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "decision_rule": analysis_contract["decision_rule"],
        "models": model_results,
        "cross_model": {
            "n_paired_payload_blocks": len(blocks_a),
            "primary_effect_pearson_r": float(pearson.statistic),
            "primary_effect_pearson_p_two_sided": float(pearson.pvalue),
            "primary_effect_spearman_r": float(spearman.statistic),
            "primary_effect_spearman_p_two_sided": float(spearman.pvalue),
        },
        "run_manifests": manifests,
        "claim_scope": protocol["claim_scope"],
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    results_path = out_dir / "transfer_results.json"
    blocks_path = out_dir / "payload_block_effects.csv"
    atomic_json(results_path, result)
    with blocks_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["model_id", "payload_block_id", "family", "control_seed", "alpha", "depth", "endpoint", "point"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_block_rows)
    analysis_manifest = {
        "schema_version": 1,
        "status": "completed_frozen_transfer_analysis",
        "decision": result["decision"],
        "config_sha256": config_sha256,
        "analysis_script_sha256": result["analysis_script_sha256"],
        "results_path": results_path.name,
        "results_sha256": sha256_file(results_path),
        "block_effects_path": blocks_path.name,
        "block_effects_sha256": sha256_file(blocks_path),
    }
    atomic_json(out_dir / "analysis_manifest.json", analysis_manifest)
    print(strict_json(analysis_manifest))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-8b", type=Path, required=True)
    parser.add_argument("--run-70b", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    ids = [model["model_id"] for model in protocol["models"]]
    if len(ids) != 2:
        raise ValueError("frozen transfer analysis requires exactly two models")
    analyze(args.config.resolve(), {ids[0]: args.run_8b.resolve(), ids[1]: args.run_70b.resolve()}, args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
