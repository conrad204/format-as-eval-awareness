#!/usr/bin/env python3
"""Exact matched-pair replication of strict probe-training environment coverage."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.full320_strict_equal_rows import (
        ENVIRONMENTS,
        PRIMARY_VARIANT,
        REGIME_CONTROLS,
        bootstrap_counts,
        bootstrap_primary,
        canonical_json_sha256,
        extract_x_memmap,
        fit_oof_scores,
        metric_bundle,
        npz_small_array,
        output_hash,
        parse_rows,
        plot_results,
        repo_commit,
        rounded_layer,
        sha256_file,
        stable_digest,
        strict_json_text,
        summarize_point_records,
    )
except ImportError:
    from full320_strict_equal_rows import (
        ENVIRONMENTS,
        PRIMARY_VARIANT,
        REGIME_CONTROLS,
        bootstrap_counts,
        bootstrap_primary,
        canonical_json_sha256,
        extract_x_memmap,
        fit_oof_scores,
        metric_bundle,
        npz_small_array,
        output_hash,
        parse_rows,
        plot_results,
        repo_commit,
        rounded_layer,
        sha256_file,
        stable_digest,
        strict_json_text,
        summarize_point_records,
    )


def matched_units(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["render_slot"] == "stated":
            grouped[(str(row["payload_block_id"]), str(row["purpose_pair_id"]))].append(index)
    units: dict[tuple[str, str], dict[str, Any]] = {}
    required = {
        ("deployment", "benchmark"),
        ("evaluation", "benchmark"),
        ("deployment", "casual"),
        ("evaluation", "casual"),
    }
    for key, indices in grouped.items():
        cells = {
            (str(rows[index]["intended_purpose"]), str(rows[index]["format"])): index
            for index in indices
        }
        if set(cells) != required or len(indices) != 4:
            raise ValueError(f"incomplete matched group {key}: {sorted(cells)}")
        families = {str(rows[index]["purpose_family_id"]) for index in indices}
        topics = {str(rows[index]["topic_id"]) for index in indices}
        if len(families) != 1 or len(topics) != 1:
            raise ValueError(f"matched group crosses family/topic: {key}")
        units[key] = {
            "family": next(iter(families)),
            "topic": next(iter(topics)),
            "cells": {
                "benchmark": (cells[("deployment", "benchmark")], cells[("evaluation", "benchmark")]),
                "casual": (cells[("deployment", "casual")], cells[("evaluation", "casual")]),
            },
        }
    return units


def payload_folds(units: dict[tuple[str, str], dict[str, Any]], n_folds: int) -> dict[str, int]:
    topic_by_block: dict[str, str] = {}
    for (block, _), unit in units.items():
        previous = topic_by_block.setdefault(block, unit["topic"])
        if previous != unit["topic"]:
            raise ValueError(f"block crosses topic: {block}")
    by_topic: dict[str, list[str]] = defaultdict(list)
    for block, topic in topic_by_block.items():
        by_topic[topic].append(block)
    result: dict[str, int] = {}
    for topic, blocks in sorted(by_topic.items()):
        ordered = sorted(blocks, key=lambda block: stable_digest("pilot_payload_fold", topic, block))
        for rank, block in enumerate(ordered):
            result[block] = rank % n_folds
    return result


def matched_assignment(
    keys: Sequence[tuple[str, str]], units: dict[tuple[str, str], dict[str, Any]], seed: int, split_tag: str
) -> dict[tuple[str, str], str]:
    by_topic: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        by_topic[units[key]["topic"]].append(key)
    assignment: dict[tuple[str, str], str] = {}
    leftovers: list[tuple[str, str]] = []
    for topic, topic_keys in sorted(by_topic.items()):
        ordered = sorted(topic_keys, key=lambda key: stable_digest("pilot_mixed", seed, split_tag, topic, *key))
        half = len(ordered) // 2
        for key in ordered[:half]:
            assignment[key] = "benchmark"
        for key in ordered[half : 2 * half]:
            assignment[key] = "casual"
        leftovers.extend(ordered[2 * half :])
    leftovers.sort(key=lambda key: stable_digest("pilot_mixed_leftover", seed, split_tag, *key))
    for rank, key in enumerate(leftovers):
        assignment[key] = ENVIRONMENTS[rank % 2]
    counts = {environment: sum(value == environment for value in assignment.values()) for environment in ENVIRONMENTS}
    if abs(counts["benchmark"] - counts["casual"]) > 1:
        raise AssertionError(f"global assignment imbalance: {counts}")
    return assignment


def build_matched_splits(
    rows: Sequence[dict[str, Any]], config: dict[str, Any], *, double_heldout: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units = matched_units(rows)
    folds = payload_folds(units, 5)
    families = [str(value) for value in config["dataset"]["purpose_families"]]
    seeds = [int(value) for value in config["probe"]["assignment_seeds"]]
    held_payload_values: Sequence[int | None] = tuple(range(5)) if double_heldout else (None,)
    splits: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for family in families:
        for held_payload in held_payload_values:
            split_tag = f"family={family};payload_fold={held_payload}"
            train_keys = sorted(
                key
                for key, unit in units.items()
                if unit["family"] != family and (held_payload is None or folds[key[0]] != held_payload)
            )
            test_keys = sorted(
                key
                for key, unit in units.items()
                if unit["family"] == family and (held_payload is None or folds[key[0]] == held_payload)
            )
            assignments = {
                seed: matched_assignment(train_keys, units, seed, split_tag) for seed in seeds
            }
            regime_pairs: dict[str, list[tuple[tuple[str, str], int, int]]] = {
                "benchmark_only": [(key, *units[key]["cells"]["benchmark"]) for key in train_keys],
                "casual_only": [(key, *units[key]["cells"]["casual"]) for key in train_keys],
            }
            for seed in seeds:
                regime_pairs[f"mixed_seed_{seed}"] = [
                    (key, *units[key]["cells"][assignments[seed][key]]) for key in train_keys
                ]
            lengths = {name: 2 * len(pairs) for name, pairs in regime_pairs.items()}
            if len(set(lengths.values())) != 1:
                raise AssertionError(f"row imbalance: {split_tag} {lengths}")
            test_indices = np.asarray(
                [index for key in test_keys for environment in ENVIRONMENTS for index in units[key]["cells"][environment]],
                dtype=np.int32,
            )
            train_blocks = {key[0] for key in train_keys}
            test_blocks = {key[0] for key in test_keys}
            if double_heldout and not train_blocks.isdisjoint(test_blocks):
                raise AssertionError(f"payload leakage in {split_tag}")
            splits.append(
                {
                    "held_family": family,
                    "held_block_fold": held_payload,
                    "split_tag": split_tag,
                    "test_indices": test_indices,
                    "regime_pairs": regime_pairs,
                }
            )
            audits.append(
                {
                    "split_tag": split_tag,
                    "n_train_groups": len(train_keys),
                    "n_train_rows_each_regime": next(iter(lengths.values())),
                    "n_test_groups": len(test_keys),
                    "n_test_rows": len(test_indices),
                    "train_group_sha256": canonical_json_sha256(train_keys),
                    "test_group_sha256": canonical_json_sha256(test_keys),
                    "assignment_sha256": {
                        str(seed): canonical_json_sha256(
                            [[*key, assignments[seed][key]] for key in train_keys]
                        )
                        for seed in seeds
                    },
                }
            )
    seen = sorted(index for split in splits for index in split["test_indices"].tolist())
    expected = sorted(
        index for index, row in enumerate(rows) if row["render_slot"] == "stated"
    )
    if seen != expected:
        raise AssertionError("matched test splits do not partition stated rows")
    return splits, {
        "double_heldout": double_heldout,
        "n_splits": len(splits),
        "n_complete_matched_groups": len(units),
        "n_stated_test_rows": len(expected),
        "records": audits,
        "records_sha256": canonical_json_sha256(audits),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "pilot80_matched_equal_rows_v1":
        raise ValueError("unexpected matched replication protocol")
    data_root = args.data_root.resolve()
    rows = parse_rows(data_root / config["dataset"]["items_path"], config["dataset"]["items_sha256"])
    for row in rows:
        row["legacy"] = False
    main_splits, main_audit = build_matched_splits(rows, config, double_heldout=False)
    double_splits, double_audit = build_matched_splits(rows, config, double_heldout=True)
    seeds = [int(value) for value in config["probe"]["assignment_seeds"]]
    depths = [float(value) for value in config["depths"]["normalized"]]
    canonical = np.asarray([row["render_slot"] == "stated" for row in rows])
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in rows])
    output_dir = args.out_dir.resolve()
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or incomplete.exists():
        raise FileExistsError(f"immutable output path already exists: {output_dir} or {incomplete}")
    incomplete.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    score_store: dict[tuple[str, float, str, int | None], np.ndarray] = {}
    score_arrays: dict[str, np.ndarray] = {}
    score_index: list[dict[str, Any]] = []
    null_seeds = [int(value) for value in config["falsification"]["training_pair_label_swap_seeds"]]
    null_sums = np.zeros(len(null_seeds), dtype=np.float64)
    null_cells = 0
    model_audits = []
    for model_config in config["models"]:
        model_id = str(model_config["model_id"])
        npz_path = data_root / model_config["activation_path"]
        actual_hash = sha256_file(npz_path)
        if actual_hash != model_config["activation_sha256"]:
            raise ValueError(f"activation hash mismatch for {model_id}")
        ids = [str(value) for value in npz_small_array(npz_path, "item_ids")]
        if ids != [str(row["item_id"]) for row in rows]:
            raise ValueError(f"item IDs mismatch for {model_id}")
        expected_shape = [len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"])]
        x_path = extract_x_memmap(npz_path, args.memmap_root.resolve(), expected_shape)
        X = np.load(x_path, mmap_mode="r", allow_pickle=False)
        layer_map = {depth: rounded_layer(depth, int(model_config["n_layers"])) for depth in depths}
        model_audits.append(
            {
                "model": model_id,
                "recorded_revision": model_config["recorded_revision"],
                "activation_sha256": actual_hash,
                "item_ids_sha256": canonical_json_sha256(ids),
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
                    metrics = metric_bundle(labels, formats, canonical, scores)
                    records.append(
                        {
                            "model": model_id,
                            "revision": model_config["recorded_revision"],
                            "layer": layer,
                            "normalized_depth": depth,
                            "variant": variant,
                            "regime": regime,
                            "seed": seed,
                            "double_heldout": False,
                            **metrics,
                        }
                    )
                    if variant == PRIMARY_VARIANT:
                        score_store[(model_id, depth, regime, seed)] = scores
                        name = f"score_{len(score_arrays):04d}"
                        score_arrays[name] = scores[canonical]
                        score_index.append({"array": name, "model": model_id, "depth": depth, "regime": regime, "seed": seed})
            if math.isclose(depth, float(config["falsification"]["double_heldout_payload"]["depth"])):
                double = fit_oof_scores(layer_data, double_splits, seeds, include_robustness=False)[PRIMARY_VARIANT]
                for regime_name, scores in double.items():
                    if regime_name.startswith("mixed_seed_"):
                        regime = "mixed"
                        seed = int(regime_name.rsplit("_", 1)[1])
                    else:
                        regime = regime_name
                        seed = None
                    records.append(
                        {
                            "model": model_id,
                            "revision": model_config["recorded_revision"],
                            "layer": layer,
                            "normalized_depth": depth,
                            "variant": PRIMARY_VARIANT,
                            "regime": regime,
                            "seed": seed,
                            "double_heldout": True,
                            **metric_bundle(labels, formats, canonical, scores),
                        }
                    )
            if not args.skip_null:
                for null_index, null_seed in enumerate(null_seeds):
                    null_outputs = fit_oof_scores(
                        layer_data, main_splits, seeds, include_robustness=False, null_seed=null_seed
                    )[PRIMARY_VARIANT]
                    metric_lookup: dict[tuple[str, int | None], dict[str, float]] = {}
                    for regime_name, scores in null_outputs.items():
                        if regime_name.startswith("mixed_seed_"):
                            regime = "mixed"
                            seed = int(regime_name.rsplit("_", 1)[1])
                        else:
                            regime = regime_name
                            seed = None
                        metric_lookup[(regime, seed)] = metric_bundle(labels, formats, canonical, scores)
                    control = max(metric_lookup[(regime, None)]["macro_format_roc_auc"] for regime in REGIME_CONTROLS)
                    mixed = float(np.mean([metric_lookup[("mixed", seed)]["macro_format_roc_auc"] for seed in seeds]))
                    null_sums[null_index] += mixed - control
                null_cells += 1
        mmap_handle = X._mmap
        del layer_data
        del X
        mmap_handle.close()
        if not args.keep_memmap:
            x_path.unlink(missing_ok=True)
    point_summary = summarize_point_records(records, seeds)
    counts, block_lookup = bootstrap_counts(
        [str(row["payload_block_id"]) for row in rows],
        int(config["uncertainty"]["bootstrap_resamples"]),
        int(config["uncertainty"]["bootstrap_seed"]),
    )
    bootstrap = bootstrap_primary(score_store, rows, seeds, counts, block_lookup)
    null_values = (null_sums / null_cells).tolist() if null_cells else []
    observed_auc = point_summary["aggregate"]["macro_format_roc_auc"]
    null_p = (1 + sum(value >= observed_auc for value in null_values)) / (1 + len(null_values)) if null_values else None
    exact_model = "meta-llama/Llama-3.3-70B-Instruct"
    criteria = {
        "aggregate_auc": bootstrap["aggregate"]["macro_format_roc_auc"]["ci_2.5"] > 0,
        "aggregate_balanced_accuracy": bootstrap["aggregate"]["macro_format_balanced_accuracy"]["ci_2.5"] > 0,
        "checkpoint_replication": all(
            point_summary["by_model"][model][endpoint] > 0
            for model in point_summary["by_model"]
            for endpoint in ("macro_format_roc_auc", "macro_format_balanced_accuracy")
        ),
        "exact_revision_replication": bootstrap["by_model"][exact_model]["macro_format_roc_auc"]["ci_2.5"] > 0,
        "assignment_stability": all(value["macro_format_roc_auc"] > 0 for value in point_summary["by_seed"].values()),
        "label_null": null_p is not None and null_p <= 0.05,
        "robustness": all(value["macro_format_roc_auc"] > 0 for value in point_summary["method_robustness"].values())
        and all(value > 0 for value in point_summary["double_heldout_payload"].values()),
    }
    decision = "positive" if all(criteria.values()) else "mixed"
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_external_bank_replication",
        "analysis": config["protocol_id"],
        "repo_commit": repo_commit(),
        "protocol": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "evidence_boundary": config["evidence_boundary"],
        "design_audit": {"main": main_audit, "double_heldout": double_audit},
        "models": model_audits,
        "point_summary": point_summary,
        "bootstrap": bootstrap,
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
        "scores": {"path": "primary_scores.npz", "index": score_index, "stated_row_indices": np.flatnonzero(canonical).tolist()},
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
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(scores_path, **score_arrays)
    plot_results(point_summary, figure_path)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_external_bank_replication",
        "analysis": config["protocol_id"],
        "repo_commit": result["repo_commit"],
        "protocol_sha256": result["protocol"]["sha256"],
        "input_hashes": {
            "items": config["dataset"]["items_sha256"],
            **{model["model_id"]: model["activation_sha256"] for model in config["models"]},
        },
        "output_hashes": {path.name: output_hash(path) for path in (result_path, tsv_path, figure_path, scores_path)},
        "decision": decision,
        "criteria": criteria,
        "evidence_grade": config["evidence_boundary"]["grade"],
        "prohibited_claims": config["evidence_boundary"]["prohibited_claims"],
    }
    manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")
    os.replace(incomplete, output_dir)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keep-memmap", action="store_true")
    parser.add_argument("--skip-null", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(strict_json_text({"run_id": result["run_id"], "decision": result["decision"], "criteria": result["criteria"]}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
