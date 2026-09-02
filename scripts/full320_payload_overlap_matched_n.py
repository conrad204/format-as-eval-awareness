#!/usr/bin/env python3
"""Matched-N confirmation that same-payload exposure changes mixed-format probe gains."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.full320_strict_equal_rows import (
        ENVIRONMENTS,
        REGIME_CONTROLS,
        balanced_accuracy_draws,
        bootstrap_counts,
        canonical_json_sha256,
        canonical_pair_units,
        extract_x_memmap,
        fit_oof_scores,
        metric_bundle,
        output_hash,
        parse_rows,
        repo_commit,
        rounded_layer,
        sha256_file,
        stable_digest,
        strict_json_text,
        topic_block_folds,
        weighted_auc_draws,
    )
except ImportError:
    from full320_strict_equal_rows import (
        ENVIRONMENTS,
        REGIME_CONTROLS,
        balanced_accuracy_draws,
        bootstrap_counts,
        canonical_json_sha256,
        canonical_pair_units,
        extract_x_memmap,
        fit_oof_scores,
        metric_bundle,
        output_hash,
        parse_rows,
        repo_commit,
        rounded_layer,
        sha256_file,
        stable_digest,
        strict_json_text,
        topic_block_folds,
        weighted_auc_draws,
    )

CONDITIONS = ("payload_heldout", "payload_exposed")
ENDPOINTS = ("macro_format_roc_auc", "macro_format_balanced_accuracy")
PRIMARY_VARIANT = "raw_ridge_0.1"
def normalized_layer_map(n_layers: int, depths: Sequence[float]) -> dict[float, int]:
    return {float(depth): rounded_layer(float(depth), n_layers) for depth in depths}




def _topic_slots(
    blocks: Sequence[str], folds: dict[str, int], topic_by_block: dict[str, str], test_fold: int, placebo_fold: int
) -> list[tuple[str, str, str]]:
    """Return (slot, heldout-block, exposed-block), pairing swapped folds within topic."""
    by_topic_fold: dict[tuple[str, int], list[str]] = defaultdict(list)
    for block in blocks:
        by_topic_fold[(topic_by_block[block], folds[block])].append(block)
    slots: list[tuple[str, str, str]] = []
    for topic in sorted(set(topic_by_block.values())):
        test_blocks = sorted(
            by_topic_fold[(topic, test_fold)], key=lambda block: stable_digest("overlap_pair", topic, test_fold, block)
        )
        placebo_blocks = sorted(
            by_topic_fold[(topic, placebo_fold)],
            key=lambda block: stable_digest("overlap_pair", topic, test_fold, block),
        )
        if len(test_blocks) != len(placebo_blocks):
            raise ValueError(f"payload folds cannot be paired in {topic}: {len(test_blocks)} != {len(placebo_blocks)}")
        for rank, (test_block, placebo_block) in enumerate(zip(test_blocks, placebo_blocks)):
            slots.append((f"{topic}:swap:{rank}", placebo_block, test_block))
        for block in sorted(
            block
            for block in blocks
            if topic_by_block[block] == topic and folds[block] not in {test_fold, placebo_fold}
        ):
            slots.append((f"{topic}:common:{block}", block, block))
    return slots


def _slot_assignments(slots: Sequence[tuple[str, str, str]], seed: int, held_family: str, test_fold: int) -> dict[str, str]:
    by_topic: dict[str, list[str]] = defaultdict(list)
    for slot, _, _ in slots:
        by_topic[slot.split(":", 1)[0]].append(slot)
    assignments: dict[str, str] = {}
    for topic, topic_slots in sorted(by_topic.items()):
        ordered = sorted(
            topic_slots,
            key=lambda slot: stable_digest("overlap_environment", seed, held_family, test_fold, topic, slot),
        )
        if len(ordered) % 2:
            raise ValueError(f"odd slot count prevents exact environment balance in {topic}")
        half = len(ordered) // 2
        for rank, slot in enumerate(ordered):
            assignments[slot] = ENVIRONMENTS[rank >= half]
    return assignments


def build_paired_splits(rows: Sequence[dict[str, Any]], protocol: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    units = canonical_pair_units(rows)
    folds = topic_block_folds(rows, 5)
    families = [str(value) for value in protocol["dataset"]["purpose_families"]]
    seeds = [int(value) for value in protocol["probe"]["assignment_seeds"]]
    topic_by_block: dict[str, str] = {}
    family_by_unit: dict[tuple[str, str, str, str], str] = {}
    for key, (_, evaluation_index) in units.items():
        topic_by_block.setdefault(key[0], str(rows[evaluation_index]["topic_id"]))
        family_by_unit[key] = str(rows[evaluation_index]["purpose_family_id"])
    all_blocks = sorted(topic_by_block)
    splits: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    audit: list[dict[str, Any]] = []
    for held_family in families:
        train_units = {key: value for key, value in units.items() if family_by_unit[key] != held_family}
        candidates: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
        for key in train_units:
            candidates[(key[0], key[2])].append(key)
        for test_fold in range(5):
            placebo_fold = (test_fold + 1) % 5
            slots = _topic_slots(all_blocks, folds, topic_by_block, test_fold, placebo_fold)
            assignments = {
                seed: _slot_assignments(slots, seed, held_family, test_fold) for seed in seeds
            }
            selected: dict[str, dict[str, dict[str, list[tuple[str, str, str, str]]]]] = {
                condition: {environment: {} for environment in ENVIRONMENTS} for condition in CONDITIONS
            }
            slot_audit = []
            for slot, heldout_block, exposed_block in slots:
                block_by_condition = {
                    "payload_heldout": heldout_block,
                    "payload_exposed": exposed_block,
                }
                k = min(
                    len(candidates[(block, environment)])
                    for block in block_by_condition.values()
                    for environment in ENVIRONMENTS
                )
                if k <= 0:
                    raise ValueError(f"missing paired environment units: {held_family} {test_fold} {slot}")
                for condition, block in block_by_condition.items():
                    for environment in ENVIRONMENTS:
                        ordered = sorted(
                            candidates[(block, environment)],
                            key=lambda key: stable_digest(
                                "overlap_pair_select", protocol["protocol_id"], held_family, test_fold, slot, environment, *key
                            ),
                        )
                        selected[condition][environment][slot] = ordered[:k]
                slot_audit.append(
                    {"slot": slot, "payload_heldout_block": heldout_block, "payload_exposed_block": exposed_block, "k": k}
                )
            test_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(rows)
                    if row["render_slot"] == "stated"
                    and not bool(row["legacy"])
                    and str(row["purpose_family_id"]) == held_family
                    and folds[str(row["payload_block_id"])] == test_fold
                ],
                dtype=np.int32,
            )
            condition_hashes = {}
            for condition in CONDITIONS:
                regime_pairs: dict[str, list[tuple[tuple[str, str, str, str], int, int]]] = {}
                for environment, regime in zip(ENVIRONMENTS, REGIME_CONTROLS):
                    regime_pairs[regime] = [
                        (key, *train_units[key]) for slot, _, _ in slots for key in selected[condition][environment][slot]
                    ]
                for seed in seeds:
                    regime_pairs[f"mixed_seed_{seed}"] = [
                        (key, *train_units[key])
                        for slot, _, _ in slots
                        for key in selected[condition][assignments[seed][slot]][slot]
                    ]
                lengths = {name: 2 * len(pairs) for name, pairs in regime_pairs.items()}
                if len(set(lengths.values())) != 1:
                    raise AssertionError(f"row imbalance in {held_family}/{test_fold}/{condition}: {lengths}")
                splits[condition].append(
                    {
                        "held_family": held_family,
                        "held_block_fold": test_fold,
                        "split_tag": f"family={held_family};test_fold={test_fold};condition={condition}",
                        "test_indices": test_indices,
                        "regime_pairs": regime_pairs,
                    }
                )
                condition_hashes[condition] = canonical_json_sha256(
                    {name: [[*key, dep, eva] for key, dep, eva in pairs] for name, pairs in regime_pairs.items()}
                )
            heldout_rows = len(splits["payload_heldout"][-1]["regime_pairs"][REGIME_CONTROLS[0]]) * 2
            exposed_rows = len(splits["payload_exposed"][-1]["regime_pairs"][REGIME_CONTROLS[0]]) * 2
            if heldout_rows != exposed_rows:
                raise AssertionError("paired conditions differ in training rows")
            audit.append(
                {
                    "held_family": held_family,
                    "test_fold": test_fold,
                    "placebo_fold": placebo_fold,
                    "n_test_rows": len(test_indices),
                    "n_train_rows_each_regime_condition": heldout_rows,
                    "slot_sha256": canonical_json_sha256(slot_audit),
                    "condition_regime_sha256": condition_hashes,
                }
            )
    expected = sorted(
        index for index, row in enumerate(rows) if row["render_slot"] == "stated" and not bool(row["legacy"])
    )
    for condition in CONDITIONS:
        seen = sorted(index for split in splits[condition] for index in split["test_indices"].tolist())
        if seen != expected:
            raise AssertionError(f"{condition} test splits do not partition canonical rows")
    return splits, {
        "n_splits_each_condition": len(splits[CONDITIONS[0]]),
        "n_canonical_test_rows": len(expected),
        "records": audit,
        "records_sha256": canonical_json_sha256(audit),
    }


def _point_summary(records: Sequence[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    models = sorted({str(row["model"]) for row in records})
    depths = sorted({float(row["normalized_depth"]) for row in records})
    lookup = {
        (row["model"], row["normalized_depth"], row["condition"], row["regime"], row["seed"]): row
        for row in records
    }
    cells = []
    for model in models:
        for depth in depths:
            for seed in seeds:
                result = {"model": model, "normalized_depth": depth, "seed": seed}
                for endpoint in ENDPOINTS:
                    gains = {}
                    for condition in CONDITIONS:
                        controls = [lookup[(model, depth, condition, regime, None)][endpoint] for regime in REGIME_CONTROLS]
                        mixed = lookup[(model, depth, condition, "mixed", seed)][endpoint]
                        gains[condition] = mixed - max(controls)
                        result[f"{condition}_{endpoint}"] = gains[condition]
                    result[f"exposure_effect_{endpoint}"] = gains["payload_exposed"] - gains["payload_heldout"]
                cells.append(result)
    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, float]:
        return {
            endpoint: float(np.mean([row[f"exposure_effect_{endpoint}"] for row in group])) for endpoint in ENDPOINTS
        }
    return {
        "aggregate": summarize(cells),
        "by_model": {model: summarize([row for row in cells if row["model"] == model]) for model in models},
        "by_depth": {str(depth): summarize([row for row in cells if row["normalized_depth"] == depth]) for depth in depths},
        "condition_gains": {
            condition: {
                endpoint: float(np.mean([row[f"{condition}_{endpoint}"] for row in cells])) for endpoint in ENDPOINTS
            }
            for condition in CONDITIONS
        },
        "cells": cells,
    }


def _metric_draw_cache(
    score_store: dict[tuple[str, float, str, str, int | None], np.ndarray],
    rows: Sequence[dict[str, Any]],
    counts: np.ndarray,
    block_lookup: dict[str, int],
) -> dict[tuple[str, float, str, str, int | None, str, str], np.ndarray]:
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in rows])
    canonical = np.asarray([row["render_slot"] == "stated" and not bool(row["legacy"]) for row in rows])
    block_columns = np.asarray([block_lookup[str(row["payload_block_id"])] for row in rows], dtype=np.int32)
    cache = {}
    for key, scores in score_store.items():
        for environment in ENVIRONMENTS:
            mask = canonical & (formats == environment)
            cache[(*key, environment, "roc_auc")] = weighted_auc_draws(
                labels[mask], scores[mask].astype(np.float64), block_columns[mask], counts
            )
            cache[(*key, environment, "balanced_accuracy")] = balanced_accuracy_draws(
                labels[mask], scores[mask], block_columns[mask], counts
            )
    return cache


def _bootstrap_summary(
    score_store: dict[tuple[str, float, str, str, int | None], np.ndarray],
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    counts: np.ndarray,
    block_lookup: dict[str, int],
) -> dict[str, Any]:
    cache = _metric_draw_cache(score_store, rows, counts, block_lookup)
    models = sorted({key[0] for key in score_store})
    depths = sorted({key[1] for key in score_store})
    cell_draws: dict[tuple[str, float, str], np.ndarray] = {}
    for model in models:
        for depth in depths:
            for endpoint, metric in ((ENDPOINTS[0], "roc_auc"), (ENDPOINTS[1], "balanced_accuracy")):
                gains = {}
                for condition in CONDITIONS:
                    controls = [
                        0.5 * sum(cache[(model, depth, condition, regime, None, env, metric)] for env in ENVIRONMENTS)
                        for regime in REGIME_CONTROLS
                    ]
                    mixed = np.mean(
                        [
                            0.5
                            * sum(cache[(model, depth, condition, "mixed", seed, env, metric)] for env in ENVIRONMENTS)
                            for seed in seeds
                        ],
                        axis=0,
                    )
                    gains[condition] = mixed - np.maximum(controls[0], controls[1])
                cell_draws[(model, depth, endpoint)] = gains["payload_exposed"] - gains["payload_heldout"]
    def interval(draws: np.ndarray) -> dict[str, float]:
        return {
            "bootstrap_mean": float(np.mean(draws)),
            "ci_2.5": float(np.quantile(draws, 0.025)),
            "ci_50": float(np.quantile(draws, 0.5)),
            "ci_97.5": float(np.quantile(draws, 0.975)),
        }
    return {
        "aggregate": {
            endpoint: interval(np.mean([cell_draws[(model, depth, endpoint)] for model in models for depth in depths], axis=0))
            for endpoint in ENDPOINTS
        },
        "by_model": {
            model: {
                endpoint: interval(np.mean([cell_draws[(model, depth, endpoint)] for depth in depths], axis=0))
                for endpoint in ENDPOINTS
            }
            for model in models
        },
    }


def _block_randomization(
    score_store: dict[tuple[str, float, str, str, int | None], np.ndarray],
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    models = sorted({key[0] for key in score_store})
    depths = sorted({key[1] for key in score_store})
    blocks = sorted({str(row["payload_block_id"]) for row in rows})
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in rows])
    block_values = []
    for block in blocks:
        block_mask = np.asarray(
            [row["render_slot"] == "stated" and not bool(row["legacy"]) and str(row["payload_block_id"]) == block for row in rows]
        )
        cell_effects = []
        for model in models:
            for depth in depths:
                gains = {}
                for condition in CONDITIONS:
                    control_values = []
                    for regime in REGIME_CONTROLS:
                        metrics = metric_bundle(labels, formats, block_mask, score_store[(model, depth, condition, regime, None)])
                        control_values.append(metrics[ENDPOINTS[0]])
                    mixed_values = [
                        metric_bundle(labels, formats, block_mask, score_store[(model, depth, condition, "mixed", assignment_seed)])[ENDPOINTS[0]]
                        for assignment_seed in seeds
                    ]
                    gains[condition] = float(np.mean(mixed_values)) - max(control_values)
                cell_effects.append(gains["payload_exposed"] - gains["payload_heldout"])
        block_values.append(float(np.mean(cell_effects)))
    values = np.asarray(block_values, dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(draws, len(values)), endpoint=False, dtype=np.int8) * 2 - 1
    null = np.mean(signs * values[None, :], axis=1)
    return {
        "unit": "complete_payload_block",
        "n_blocks": len(values),
        "n_draws": draws,
        "observed_mean_block_effect": observed,
        "null_q_2.5": float(np.quantile(null, 0.025)),
        "null_q_97.5": float(np.quantile(null, 0.975)),
        "empirical_p_greater_equal": float((1 + np.sum(null >= observed)) / (draws + 1)),
        "block_effect_sha256": canonical_json_sha256(dict(zip(blocks, block_values))),
    }


def _plot(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summary["by_model"])
    depths = sorted(summary["by_depth"], key=float)
    labels = ["aggregate", *[model.rsplit("/", 1)[-1].replace("-Instruct", "") for model in models], *[f"q={d}" for d in depths]]
    values = [
        summary["aggregate"][ENDPOINTS[0]],
        *[summary["by_model"][model][ENDPOINTS[0]] for model in models],
        *[summary["by_depth"][depth][ENDPOINTS[0]] for depth in depths],
    ]
    colors = ["#2f6f9f" if value >= 0 else "#b04a3a" for value in values]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    ax.set_ylabel("Exposed − heldout mixed-format gain (macro ROC AUC)")
    ax.set_title("Matched-N effect of same-payload exposure")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "full320_payload_overlap_matched_n_v1":
        raise ValueError("unexpected payload-overlap protocol")
    data_root = args.data_root.resolve()
    rows = parse_rows(data_root / config["dataset"]["items_path"], config["dataset"]["items_sha256"])
    paired_splits, split_audit = build_paired_splits(rows, config)
    seeds = [int(value) for value in config["probe"]["assignment_seeds"]]
    depths = [float(value) for value in config["depths"]["normalized"]]
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in rows])
    canonical = np.asarray([row["render_slot"] == "stated" and not bool(row["legacy"]) for row in rows])
    output_dir = args.out_dir.resolve()
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or incomplete.exists():
        raise FileExistsError(f"immutable output path already exists: {output_dir} or {incomplete}")
    incomplete.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    score_store: dict[tuple[str, float, str, str, int | None], np.ndarray] = {}
    model_audits = []
    for model_config in config["models"]:
        model = str(model_config["model_id"])
        npz_path = data_root / model_config["activation_path"]
        if sha256_file(npz_path) != model_config["activation_sha256"]:
            raise ValueError(f"activation hash mismatch for {model}")
        expected_shape = [len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"])]
        x_path = extract_x_memmap(npz_path, args.memmap_root.resolve(), expected_shape)
        X = np.load(x_path, mmap_mode="r", allow_pickle=False)
        layer_map = normalized_layer_map(int(model_config["n_layers"]), depths)
        model_audits.append({"model": model, "activation_sha256": sha256_file(npz_path), "layer_map": layer_map})
        for depth, layer in layer_map.items():
            layer_data = X[:, layer, :]
            for condition in CONDITIONS:
                outputs = fit_oof_scores(layer_data, paired_splits[condition], seeds, include_robustness=False)[PRIMARY_VARIANT]
                for regime_name, scores in outputs.items():
                    if regime_name.startswith("mixed_seed_"):
                        regime = "mixed"
                        assignment_seed: int | None = int(regime_name.rsplit("_", 1)[1])
                    else:
                        regime = regime_name
                        assignment_seed = None
                    metrics = metric_bundle(labels, formats, canonical, scores)
                    record = {
                        "model": model,
                        "revision": model_config["revision"],
                        "layer": layer,
                        "normalized_depth": depth,
                        "condition": condition,
                        "regime": regime,
                        "seed": assignment_seed,
                        **metrics,
                    }
                    records.append(record)
                    score_store[(model, depth, condition, regime, assignment_seed)] = scores
        mmap_handle = X._mmap
        del layer_data
        del X
        mmap_handle.close()
        if not args.keep_memmap:
            x_path.unlink(missing_ok=True)
    point = _point_summary(records, seeds)
    block_counts, block_lookup = bootstrap_counts(
        [str(row["payload_block_id"]) for row in rows],
        int(config["uncertainty"]["bootstrap_resamples"]),
        int(config["uncertainty"]["bootstrap_seed"]),
    )
    bootstrap = _bootstrap_summary(score_store, rows, seeds, block_counts, block_lookup)
    randomization = _block_randomization(
        score_store,
        rows,
        seeds,
        int(config["falsification"]["paired_sign_flip_draws"]),
        int(config["falsification"]["paired_sign_flip_seed"]),
    )
    auc_point = point["aggregate"][ENDPOINTS[0]]
    criteria = {
        "aggregate_ci_positive": bootstrap["aggregate"][ENDPOINTS[0]]["ci_2.5"] > 0,
        "sign_flip": randomization["empirical_p_greater_equal"] <= 0.05,
        "both_checkpoints": all(value[ENDPOINTS[0]] > 0 for value in point["by_model"].values()),
        "depth_replication": sum(value[ENDPOINTS[0]] > 0 for value in point["by_depth"].values()) >= 3,
    }
    decision = "positive" if all(criteria.values()) else ("mixed" if auc_point > 0 else "negative")
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_confirmation",
        "analysis": config["protocol_id"],
        "decision": decision,
        "criteria": criteria,
        "point_summary": point,
        "bootstrap": bootstrap,
        "paired_sign_flip": randomization,
        "split_audit": split_audit,
        "model_audits": model_audits,
        "records": records,
        "limitations": [
            "The endpoint is observational probe ranking, not model behavior or causal steering.",
            "The two checkpoints and frozen item bank are fixed case studies, not sampled populations.",
            "Benchmark/casual are confounded stored format-template environments.",
            "ROC AUC was selected for independent confirmation after a pilot ranking/threshold dissociation; balanced accuracy is descriptive.",
        ],
    }
    result_path = incomplete / config["outputs"]["result_json"]
    table_path = incomplete / config["outputs"]["table_tsv"]
    figure_path = incomplete / config["outputs"]["figure"]
    manifest_path = incomplete / config["outputs"]["manifest_json"]
    result_path.write_text(strict_json_text(result), encoding="utf-8")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    _plot(point, figure_path)
    output_paths = [result_path, table_path, figure_path]
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_confirmation",
        "analysis": config["protocol_id"],
        "repo_commit": repo_commit(),
        "protocol_path": config_path.as_posix(),
        "protocol_sha256": sha256_file(config_path),
        "dataset_repo_id": "conrad204/format-as-eval-awareness",
        "dataset_revision": repo_commit(),
        "items_sha256": config["dataset"]["items_sha256"],
        "split_audit_sha256": split_audit["records_sha256"],
        "elapsed_seconds": time.time() - started,
        "outputs": {path.name: output_hash(path) for path in output_paths},
        "decision": decision,
        "criteria": criteria,
    }
    manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")
    incomplete.replace(output_dir)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keep-memmap", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"run_id": result["run_id"], "decision": result["decision"], "criteria": result["criteria"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
