#!/usr/bin/env python3
"""Compute the preregistered Llama-3.1 8B/70B strict equal-row interaction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.full320_strict_equal_rows import (
        ENVIRONMENTS,
        REGIME_CONTROLS,
        balanced_accuracy_draws,
        bootstrap_counts,
        output_hash,
        repo_commit,
        sha256_file,
        strict_json_text,
        weighted_auc_draws,
    )
except ImportError:
    from full320_strict_equal_rows import (
        ENVIRONMENTS,
        REGIME_CONTROLS,
        balanced_accuracy_draws,
        bootstrap_counts,
        output_hash,
        repo_commit,
        sha256_file,
        strict_json_text,
        weighted_auc_draws,
    )

ENDPOINTS = ("macro_format_roc_auc", "macro_format_balanced_accuracy")


def load_run(
    result_path: Path, expected_model: str
) -> tuple[dict[str, Any], dict[tuple[float, str, int | None], np.ndarray]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    archive_path = result_path.parent / result["scores"]["path"]
    archive = np.load(archive_path, allow_pickle=False)
    models = {str(row["model"]) for row in result["scores"]["index"]}
    if expected_model not in models:
        raise ValueError(f"expected model missing from scale-interaction input: {expected_model}")
    scores = {
        (float(row["depth"]), str(row["regime"]), None if row["seed"] is None else int(row["seed"])): np.asarray(
            archive[str(row["array"])], dtype=np.float32
        )
        for row in result["scores"]["index"]
        if row["model"] == expected_model
    }
    lengths = {len(value) for value in scores.values()}
    if len(lengths) != 1 or not all(np.all(np.isfinite(value)) for value in scores.values()):
        raise ValueError("invalid saved OOF score arrays")
    return result, scores


def point_effects(
    scores_by_size: dict[str, dict[tuple[float, str, int | None], np.ndarray]],
    labels: np.ndarray,
    formats: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    depths = sorted({key[0] for scores in scores_by_size.values() for key in scores})

    def metrics(scores: np.ndarray) -> dict[str, float]:
        per_environment = {}
        for environment in ENVIRONMENTS:
            mask = formats == environment
            per_environment[environment] = {
                "roc_auc": float(roc_auc_score(labels[mask], scores[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(labels[mask], scores[mask] > 0)),
            }
        return {
            "macro_format_roc_auc": float(np.mean([per_environment[env]["roc_auc"] for env in ENVIRONMENTS])),
            "macro_format_balanced_accuracy": float(
                np.mean([per_environment[env]["balanced_accuracy"] for env in ENVIRONMENTS])
            ),
        }

    effects: dict[str, dict[float, dict[str, float]]] = {size: {} for size in scores_by_size}
    absolute = []
    for size, score_store in scores_by_size.items():
        for depth in depths:
            control_metrics = {regime: metrics(score_store[(depth, regime, None)]) for regime in REGIME_CONTROLS}
            mixed_metrics = {seed: metrics(score_store[(depth, "mixed", seed)]) for seed in seeds}
            effects[size][depth] = {
                endpoint: float(np.mean([mixed_metrics[seed][endpoint] for seed in seeds]))
                - max(control_metrics[regime][endpoint] for regime in REGIME_CONTROLS)
                for endpoint in ENDPOINTS
            }
            for regime, values in control_metrics.items():
                absolute.append({"size": size, "depth": depth, "regime": regime, "seed": None, **values})
            for seed, values in mixed_metrics.items():
                absolute.append({"size": size, "depth": depth, "regime": "mixed", "seed": seed, **values})
    by_depth = {
        str(depth): {
            endpoint: effects["8B"][depth][endpoint] - effects["70B"][depth][endpoint] for endpoint in ENDPOINTS
        }
        for depth in depths
    }
    aggregate = {
        endpoint: float(np.mean([by_depth[str(depth)][endpoint] for depth in depths])) for endpoint in ENDPOINTS
    }
    return {
        "aggregate_interaction": aggregate,
        "by_depth_interaction": by_depth,
        "by_size_effect": {
            size: {
                endpoint: float(np.mean([effects[size][depth][endpoint] for depth in depths])) for endpoint in ENDPOINTS
            }
            for size in ("8B", "70B")
        },
        "absolute_metrics": absolute,
    }


def bootstrap_interaction(
    scores_by_size: dict[str, dict[tuple[float, str, int | None], np.ndarray]],
    labels: np.ndarray,
    formats: np.ndarray,
    blocks: np.ndarray,
    seeds: Sequence[int],
    counts: np.ndarray,
    block_lookup: dict[str, int],
) -> dict[str, Any]:
    block_columns = np.asarray([block_lookup[str(block)] for block in blocks], dtype=np.int32)
    cache: dict[tuple[str, float, str, int | None, str, str], np.ndarray] = {}
    for size, score_store in scores_by_size.items():
        for (depth, regime, seed), scores in score_store.items():
            for environment in ENVIRONMENTS:
                mask = formats == environment
                cache[(size, depth, regime, seed, environment, "roc_auc")] = weighted_auc_draws(
                    labels[mask], scores[mask].astype(np.float64), block_columns[mask], counts
                )
                cache[(size, depth, regime, seed, environment, "balanced_accuracy")] = balanced_accuracy_draws(
                    labels[mask], scores[mask], block_columns[mask], counts
                )
    depths = sorted({key[1] for key in cache})
    interactions: dict[tuple[float, str], np.ndarray] = {}
    size_draws: dict[tuple[str, float, str], np.ndarray] = {}
    for depth in depths:
        for endpoint, metric in ((ENDPOINTS[0], "roc_auc"), (ENDPOINTS[1], "balanced_accuracy")):
            for size in ("8B", "70B"):
                controls = [
                    0.5 * sum(cache[(size, depth, regime, None, env, metric)] for env in ENVIRONMENTS)
                    for regime in REGIME_CONTROLS
                ]
                mixed = np.mean(
                    [
                        0.5 * sum(cache[(size, depth, "mixed", seed, env, metric)] for env in ENVIRONMENTS)
                        for seed in seeds
                    ],
                    axis=0,
                )
                size_draws[(size, depth, endpoint)] = mixed - np.maximum(controls[0], controls[1])
            interactions[(depth, endpoint)] = size_draws[("8B", depth, endpoint)] - size_draws[("70B", depth, endpoint)]

    def interval(draws: np.ndarray) -> dict[str, float]:
        return {
            "bootstrap_mean": float(np.mean(draws)),
            "ci_2.5": float(np.quantile(draws, 0.025)),
            "ci_50": float(np.quantile(draws, 0.5)),
            "ci_97.5": float(np.quantile(draws, 0.975)),
        }

    return {
        "aggregate_interaction": {
            endpoint: interval(np.mean([interactions[(depth, endpoint)] for depth in depths], axis=0))
            for endpoint in ENDPOINTS
        },
        "by_depth_interaction": {
            str(depth): {endpoint: interval(interactions[(depth, endpoint)]) for endpoint in ENDPOINTS}
            for depth in depths
        },
        "by_size_effect": {
            size: {
                endpoint: interval(np.mean([size_draws[(size, depth, endpoint)] for depth in depths], axis=0))
                for endpoint in ENDPOINTS
            }
            for size in ("8B", "70B")
        },
    }


def plot_interaction(point: dict[str, Any], bootstrap: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = sorted(point["by_depth_interaction"], key=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, endpoint, label in zip(axes, ENDPOINTS, ("ROC AUC", "Balanced accuracy")):
        values = [point["by_depth_interaction"][depth][endpoint] for depth in depths]
        low = [bootstrap["by_depth_interaction"][depth][endpoint]["ci_2.5"] for depth in depths]
        high = [bootstrap["by_depth_interaction"][depth][endpoint]["ci_97.5"] for depth in depths]
        x = np.asarray([float(depth) for depth in depths])
        ax.errorbar(x, values, yerr=[np.asarray(values) - low, np.asarray(high) - values], marker="o", capsize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel(f"8B − 70B mixed-coverage effect ({label})")
    fig.suptitle("Within-family size-associated reversal under exact row balance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "full320_strict_equal_rows_8b_scale_v1":
        raise ValueError("unexpected scale-interaction protocol")
    expected_8b = "meta-llama/Llama-3.1-8B-Instruct"
    expected_70b = "meta-llama/Llama-3.1-70B-Instruct"
    result_8b, scores_8b = load_run(args.result_8b.resolve(), expected_8b)
    result_70b, scores_70b = load_run(args.result_70b.resolve(), expected_70b)
    if set(result_8b["point_summary"]["by_model"]) != {expected_8b}:
        raise ValueError("unexpected 8B run model")
    if result_8b["scores"]["canonical_row_indices"] != result_70b["scores"]["canonical_row_indices"]:
        raise ValueError("8B and 70B canonical row order differs")
    rows = [
        json.loads(line)
        for line in (args.data_root.resolve() / config["dataset"]["items_path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    canonical_indices = result_8b["scores"]["canonical_row_indices"]
    canonical_rows = [rows[index] for index in canonical_indices]
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in canonical_rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in canonical_rows])
    blocks = np.asarray([str(row["payload_block_id"]) for row in canonical_rows])
    scores_by_size = {"8B": scores_8b, "70B": scores_70b}
    seeds = [int(seed) for seed in config["probe"]["assignment_seeds"]]
    point = point_effects(scores_by_size, labels, formats, seeds)
    counts, block_lookup = bootstrap_counts(
        blocks,
        int(config["uncertainty"]["bootstrap_resamples"]),
        int(config["uncertainty"]["bootstrap_seed"]),
    )
    bootstrap = bootstrap_interaction(scores_by_size, labels, formats, blocks, seeds, counts, block_lookup)
    interaction_primary = all(
        bootstrap["aggregate_interaction"][endpoint]["ci_2.5"] > 0 for endpoint in ENDPOINTS
    )
    interaction_depth = all(
        sum(point["by_depth_interaction"][depth][endpoint] > 0 for depth in point["by_depth_interaction"]) >= 4
        for endpoint in ENDPOINTS
    )
    criteria = {
        "8b_replication": all(bool(value) for value in result_8b["criteria"].values()),
        "scale_interaction_primary": interaction_primary,
        "scale_interaction_depth_replication": interaction_depth,
    }
    aggregate_positive = all(point["aggregate_interaction"][endpoint] > 0 for endpoint in ENDPOINTS)
    decision = "positive" if all(criteria.values()) else ("mixed" if aggregate_positive else "negative")
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_scale_interaction",
        "analysis": config["protocol_id"],
        "decision": decision,
        "criteria": criteria,
        "point_summary": point,
        "bootstrap": bootstrap,
        "source_runs": {
            "8B": {"path": args.result_8b.resolve().as_posix(), "sha256": sha256_file(args.result_8b.resolve())},
            "70B": {"path": args.result_70b.resolve().as_posix(), "sha256": sha256_file(args.result_70b.resolve())},
        },
        "limitations": config["evidence_boundary"]["limitations"],
    }
    output_dir = args.out_dir.resolve()
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or incomplete.exists():
        raise FileExistsError(f"immutable output exists: {output_dir} or {incomplete}")
    incomplete.mkdir(parents=True)
    output_config = config["scale_interaction_outputs"]
    result_path = incomplete / output_config["result_json"]
    table_path = incomplete / output_config["table_tsv"]
    figure_path = incomplete / output_config["figure"]
    manifest_path = incomplete / output_config["manifest_json"]
    result_path.write_text(strict_json_text(result), encoding="utf-8")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(point["absolute_metrics"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(point["absolute_metrics"])
    plot_interaction(point, bootstrap, figure_path)
    outputs = [result_path, table_path, figure_path]
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_scale_interaction",
        "analysis": config["protocol_id"],
        "repo_commit": repo_commit(),
        "protocol_sha256": sha256_file(config_path),
        "source_run_hashes": result["source_runs"],
        "output_hashes": {path.name: output_hash(path) for path in outputs},
        "decision": decision,
        "criteria": criteria,
        "evidence_grade": config["evidence_boundary"]["grade"],
        "prohibited_claims": config["evidence_boundary"]["prohibited_claims"],
    }
    manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")
    incomplete.replace(output_dir)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-8b", type=Path, required=True)
    parser.add_argument("--result-70b", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"run_id": result["run_id"], "decision": result["decision"], "criteria": result["criteria"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
