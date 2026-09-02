#!/usr/bin/env python3
"""Analyze the complete prospective Llama-3.1-70B orthogonalized-C extension."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts import analyze_full320_orthc_transfer as base

PROTOCOL_ID = "full320_orthogonalized_c_transfer_model_extension_v1"


def analyze(config_path: Path, plan_json_path: Path, run_dir: Path, out_dir: Path) -> dict[str, Any]:
    base.PROTOCOL_ID = PROTOCOL_ID
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID or len(protocol.get("models", [])) != 1:
        raise ValueError("model-extension protocol mismatch")
    analysis_contract = protocol["analysis"]
    if analysis_contract["script_sha256"] != base.sha256_file(Path(__file__)):
        raise ValueError("analysis script differs from frozen hash")
    config_sha256 = base.sha256_file(config_path)
    plan = json.loads(plan_json_path.read_text(encoding="utf-8"))
    if plan.get("config_sha256") != config_sha256 or plan.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("analysis plan/config mismatch")
    model = protocol["models"][0]
    records, run_manifest = base.load_complete(run_dir, model, config_sha256)
    if run_manifest["plan_json_sha256"] != base.sha256_file(plan_json_path):
        raise ValueError("run used a different transfer plan")
    _, collapsed = base.model_block_effects(records, model)
    blocks, high = base.vector_by_blocks(collapsed, "C", None, 1.0, "high_interaction", "interaction")
    low_blocks, low = base.vector_by_blocks(collapsed, "C", None, 1.0, "low_interaction", "interaction")
    if blocks != low_blocks:
        raise ValueError("high/low payload-block mismatch")
    difference = high - low
    _, half_high = base.vector_by_blocks(collapsed, "C", None, 0.5, "high_interaction", "interaction")
    _, half_low = base.vector_by_blocks(collapsed, "C", None, 0.5, "low_interaction", "interaction")
    half_difference = half_high - half_low
    random_means: list[float] = []
    random_rows: list[dict[str, Any]] = []
    for seed in protocol["interventions"]["random_seeds"]:
        random_blocks, random_high = base.vector_by_blocks(collapsed, "random", int(seed), 1.0, "high_interaction", "interaction")
        low_random_blocks, random_low = base.vector_by_blocks(collapsed, "random", int(seed), 1.0, "low_interaction", "interaction")
        if random_blocks != blocks or low_random_blocks != blocks:
            raise ValueError("random/C payload-block mismatch")
        random_difference = random_high - random_low
        random_means.append(float(random_difference.mean()))
        random_rows.extend({
            "payload_block_id": block,
            "family": "random_high_minus_low",
            "control_seed": int(seed),
            "alpha": 1.0,
            "depth": "high_minus_low",
            "endpoint": "interaction",
            "point": float(value),
        } for block, value in zip(blocks, random_difference, strict=True))
    random_threshold = float(np.quantile(np.abs(np.asarray(random_means)), 0.975))

    items_path = Path(plan["items_path"])
    if not items_path.exists():
        items_path = Path(protocol["dataset"]["items_path"])
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    topic_by_block: dict[str, str] = {}
    for row in items:
        block = row["payload_block_id"]
        topic = row["topic_id"]
        if block in topic_by_block and topic_by_block[block] != topic:
            raise ValueError("payload block spans topics")
        topic_by_block[block] = topic
    topics = sorted(set(topic_by_block.values()))
    if len(topics) != 16 or set(blocks) != set(topic_by_block):
        raise ValueError("topic/block inventory mismatch")
    topic_means = {
        topic: float(np.mean([difference[index] for index, block in enumerate(blocks) if topic_by_block[block] == topic]))
        for topic in topics
    }
    leave_one_topic_out = {
        topic: float(np.mean([difference[index] for index, block in enumerate(blocks) if topic_by_block[block] != topic]))
        for topic in topics
    }
    rng = np.random.default_rng(int(analysis_contract["bootstrap_seed"]))
    primary_p = base.one_sided_t(difference)
    high_p = base.one_sided_t(high)
    positive = (
        float(difference.mean()) > 0
        and primary_p <= float(analysis_contract["alpha"])
        and float(high.mean()) > 0
        and high_p <= float(analysis_contract["alpha"])
        and float(half_difference.mean()) > 0
        and float(difference.mean()) > random_threshold
    )
    prior = analysis_contract["prior_two_model_reference"]
    result = {
        "schema_version": 1,
        "status": "complete_frozen_model_extension_analysis",
        "decision": "positive_model_extension" if positive else "clean_negative_or_mixed_model_extension",
        "protocol_id": PROTOCOL_ID,
        "config_sha256": config_sha256,
        "analysis_script_sha256": base.sha256_file(Path(__file__)),
        "model_id": model["model_id"],
        "n_payload_blocks": len(blocks),
        "primary_high_minus_low": {
            "mean": float(difference.mean()),
            "sd": float(difference.std(ddof=1)),
            "ci95_payload_bootstrap": base.percentile_ci(difference, rng, int(analysis_contract["bootstrap_resamples"])),
            "one_sided_t_p": primary_p,
            "all_blocks_positive": bool(np.all(difference > 0)),
        },
        "high_layer_C_interaction": {
            "mean": float(high.mean()),
            "sd": float(high.std(ddof=1)),
            "ci95_payload_bootstrap": base.percentile_ci(high, rng, int(analysis_contract["bootstrap_resamples"])),
            "one_sided_t_p": high_p,
        },
        "low_layer_C_interaction": {
            "mean": float(low.mean()),
            "sd": float(low.std(ddof=1)),
            "ci95_payload_bootstrap": base.percentile_ci(low, rng, int(analysis_contract["bootstrap_resamples"])),
        },
        "half_alpha_high_minus_low": {
            "mean": float(half_difference.mean()),
            "sd": float(half_difference.std(ddof=1)),
            "ratio_to_alpha1": float(half_difference.mean() / difference.mean()) if difference.mean() != 0 else None,
        },
        "random_control_calibration": {
            "n_directions": len(random_means),
            "high_minus_low_means": random_means,
            "absolute_mean_97_5_percentile": random_threshold,
            "C_exceeds_absolute_random_97_5_percentile": bool(difference.mean() > random_threshold),
        },
        "topic_robustness_descriptive": {
            "topic_means": topic_means,
            "all_topic_means_positive": all(value > 0 for value in topic_means.values()),
            "leave_one_topic_out_means": leave_one_topic_out,
            "all_leave_one_topic_out_means_positive": all(value > 0 for value in leave_one_topic_out.values()),
        },
        "prior_two_model_comparison_descriptive": {
            "prior_model_means": prior,
            "extension_minus_prior_range": [
                float(difference.mean() - max(prior.values())),
                float(difference.mean() - min(prior.values())),
            ],
        },
        "decision_rule": analysis_contract["decision_rule"],
        "run_manifest": run_manifest,
        "claim_scope": protocol["claim_scope"],
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    results_path = out_dir / "extension_results.json"
    effects_path = out_dir / "payload_block_effects.csv"
    base.atomic_json(results_path, result)
    rows: list[dict[str, Any]] = []
    for block, high_value, low_value, full_value, half_value in zip(blocks, high, low, difference, half_difference, strict=True):
        rows.extend([
            {"payload_block_id": block, "topic_id": topic_by_block[block], "family": "C", "control_seed": "", "alpha": 1.0, "depth": "high_interaction", "endpoint": "interaction", "point": float(high_value)},
            {"payload_block_id": block, "topic_id": topic_by_block[block], "family": "C", "control_seed": "", "alpha": 1.0, "depth": "low_interaction", "endpoint": "interaction", "point": float(low_value)},
            {"payload_block_id": block, "topic_id": topic_by_block[block], "family": "C_high_minus_low", "control_seed": "", "alpha": 1.0, "depth": "high_minus_low", "endpoint": "interaction", "point": float(full_value)},
            {"payload_block_id": block, "topic_id": topic_by_block[block], "family": "C_high_minus_low", "control_seed": "", "alpha": 0.5, "depth": "high_minus_low", "endpoint": "interaction", "point": float(half_value)},
        ])
    for row in random_rows:
        row["topic_id"] = topic_by_block[row["payload_block_id"]]
    rows.extend(random_rows)
    fields = ["payload_block_id", "topic_id", "family", "control_seed", "alpha", "depth", "endpoint", "point"]
    with effects_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "status": "completed_frozen_model_extension_analysis",
        "decision": result["decision"],
        "config_sha256": config_sha256,
        "analysis_script_sha256": result["analysis_script_sha256"],
        "results_path": results_path.name,
        "results_sha256": base.sha256_file(results_path),
        "block_effects_path": effects_path.name,
        "block_effects_sha256": base.sha256_file(effects_path),
    }
    base.atomic_json(out_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analyze(args.config.resolve(), args.plan_json.resolve(), args.run_dir.resolve(), args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
