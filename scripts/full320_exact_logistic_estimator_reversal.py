#!/usr/bin/env python3
"""Test estimator dependence of exact-row mixed-environment probe conclusions."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from scripts.full320_strict_equal_rows import (
        ENVIRONMENTS,
        REGIME_CONTROLS,
        balanced_accuracy_draws,
        bootstrap_counts,
        build_splits,
        canonical_json_sha256,
        extract_x_memmap,
        metric_bundle,
        output_hash,
        parse_rows,
        repo_commit,
        sha256_file,
        strict_json_text,
        train_indices_and_labels,
        weighted_auc_draws,
    )
except ImportError:
    from full320_strict_equal_rows import (
        ENVIRONMENTS,
        REGIME_CONTROLS,
        balanced_accuracy_draws,
        bootstrap_counts,
        build_splits,
        canonical_json_sha256,
        extract_x_memmap,
        metric_bundle,
        output_hash,
        parse_rows,
        repo_commit,
        sha256_file,
        strict_json_text,
        train_indices_and_labels,
        weighted_auc_draws,
    )

ENDPOINTS = ("macro_format_roc_auc", "macro_format_balanced_accuracy")
ESTIMATORS = ("logistic", "diagonal_lda")


def fit_logistic_oof(
    layer_data: np.ndarray,
    splits: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    probe: dict[str, Any],
) -> dict[str, np.ndarray]:
    regime_names = [*REGIME_CONTROLS, *(f"mixed_seed_{seed}" for seed in seeds)]
    outputs = {name: np.full(layer_data.shape[0], np.nan, dtype=np.float32) for name in regime_names}
    for split in splits:
        test_indices = split["test_indices"]
        test = np.asarray(layer_data[test_indices], dtype=np.float32)
        for regime in regime_names:
            train_indices, labels = train_indices_and_labels(split["regime_pairs"][regime], None)
            train = np.asarray(layer_data[train_indices], dtype=np.float32)
            scaler = StandardScaler(copy=False).fit(train)
            train = scaler.transform(train)
            model = LogisticRegression(
                C=float(probe["C"]),
                penalty=str(probe["penalty"]),
                solver=str(probe["solver"]),
                max_iter=int(probe["max_iter"]),
                tol=float(probe["tol"]),
                random_state=int(probe["random_state"]),
            ).fit(train, labels)
            outputs[regime][test_indices] = np.asarray(
                model.decision_function(scaler.transform(test.copy())), dtype=np.float32
            )
    expected = sum(len(split["test_indices"]) for split in splits)
    for regime, scores in outputs.items():
        if int(np.isfinite(scores).sum()) != expected:
            raise AssertionError(f"missing logistic OOF scores for {regime}")
    return outputs


def load_lda_scores(
    result_path: Path, model_id: str, depth: float, n_rows: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    archive_path = result_path.parent / result["scores"]["path"]
    archive = np.load(archive_path, allow_pickle=False)
    canonical_indices = np.asarray(result["scores"]["canonical_row_indices"], dtype=np.int32)
    selected = [
        row
        for row in result["scores"]["index"]
        if row["model"] == model_id and float(row["depth"]) == depth
    ]
    if not selected:
        raise ValueError(f"missing LDA comparator scores for {model_id} depth={depth}")
    outputs = {}
    for row in selected:
        regime = str(row["regime"])
        name = regime if regime in REGIME_CONTROLS else f"mixed_seed_{int(row['seed'])}"
        full = np.full(n_rows, np.nan, dtype=np.float32)
        full[canonical_indices] = np.asarray(archive[str(row["array"])], dtype=np.float32)
        outputs[name] = full
    return outputs, {
        "path": result_path.resolve().as_posix(),
        "sha256": sha256_file(result_path),
        "score_path": archive_path.resolve().as_posix(),
        "score_sha256": sha256_file(archive_path),
        "source_run_id": result["run_id"],
        "source_analysis": result["analysis"],
        "source_split_audit_sha256": result["design_audit"]["main"]["split_records_sha256"],
    }


def records_and_store(
    outputs: dict[str, np.ndarray],
    estimator: str,
    model_id: str,
    revision: str,
    layer: int,
    depth: float,
    labels: np.ndarray,
    formats: np.ndarray,
    canonical: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int | None], np.ndarray]]:
    records = []
    store = {}
    for regime_name, scores in outputs.items():
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
                "revision": revision,
                "layer": layer,
                "normalized_depth": depth,
                "estimator": estimator,
                "regime": regime,
                "seed": seed,
                **metrics,
            }
        )
        store[(estimator, regime, seed)] = scores
    return records, store


def point_summary(records: Sequence[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    models = sorted({str(row["model"]) for row in records})
    lookup = {
        (row["model"], row["estimator"], row["regime"], row["seed"]): row for row in records
    }
    cells = []
    for model in models:
        for estimator in ESTIMATORS:
            controls = [lookup[(model, estimator, regime, None)] for regime in REGIME_CONTROLS]
            for seed in seeds:
                mixed = lookup[(model, estimator, "mixed", seed)]
                cells.append(
                    {
                        "model": model,
                        "estimator": estimator,
                        "seed": seed,
                        **{
                            endpoint: mixed[endpoint] - max(control[endpoint] for control in controls)
                            for endpoint in ENDPOINTS
                        },
                    }
                )
    by_model = {}
    for model in models:
        by_model[model] = {}
        for estimator in ESTIMATORS:
            selected = [row for row in cells if row["model"] == model and row["estimator"] == estimator]
            by_model[model][estimator] = {
                endpoint: float(np.mean([row[endpoint] for row in selected])) for endpoint in ENDPOINTS
            }
        by_model[model]["interaction"] = {
            endpoint: by_model[model]["logistic"][endpoint] - by_model[model]["diagonal_lda"][endpoint]
            for endpoint in ENDPOINTS
        }
    aggregate = {
        estimator: {
            endpoint: float(np.mean([by_model[model][estimator][endpoint] for model in models]))
            for endpoint in ENDPOINTS
        }
        for estimator in ESTIMATORS
    }
    aggregate["interaction"] = {
        endpoint: aggregate["logistic"][endpoint] - aggregate["diagonal_lda"][endpoint] for endpoint in ENDPOINTS
    }
    by_seed = {
        str(seed): {
            estimator: {
                endpoint: float(
                    np.mean(
                        [
                            row[endpoint]
                            for row in cells
                            if row["seed"] == seed and row["estimator"] == estimator
                        ]
                    )
                )
                for endpoint in ENDPOINTS
            }
            for estimator in ESTIMATORS
        }
        for seed in seeds
    }
    return {"aggregate": aggregate, "by_model": by_model, "by_seed": by_seed, "cell_contrasts": cells}


def bootstrap_summary(
    score_store: dict[tuple[str, str, str, int | None], np.ndarray],
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    counts: np.ndarray,
    block_lookup: dict[str, int],
) -> dict[str, Any]:
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
    models = sorted({key[0] for key in score_store})
    draws: dict[tuple[str, str, str], np.ndarray] = {}
    for model in models:
        for estimator in ESTIMATORS:
            for endpoint, metric in ((ENDPOINTS[0], "roc_auc"), (ENDPOINTS[1], "balanced_accuracy")):
                controls = [
                    0.5 * sum(cache[(model, estimator, regime, None, env, metric)] for env in ENVIRONMENTS)
                    for regime in REGIME_CONTROLS
                ]
                mixed = np.mean(
                    [
                        0.5 * sum(cache[(model, estimator, "mixed", seed, env, metric)] for env in ENVIRONMENTS)
                        for seed in seeds
                    ],
                    axis=0,
                )
                draws[(model, estimator, endpoint)] = mixed - np.maximum(controls[0], controls[1])

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "bootstrap_mean": float(np.mean(values)),
            "ci_2.5": float(np.quantile(values, 0.025)),
            "ci_50": float(np.quantile(values, 0.5)),
            "ci_97.5": float(np.quantile(values, 0.975)),
        }

    aggregate = {}
    by_model = {}
    for estimator in ESTIMATORS:
        aggregate[estimator] = {
            endpoint: interval(np.mean([draws[(model, estimator, endpoint)] for model in models], axis=0))
            for endpoint in ENDPOINTS
        }
    aggregate["interaction"] = {
        endpoint: interval(
            np.mean(
                [draws[(model, "logistic", endpoint)] - draws[(model, "diagonal_lda", endpoint)] for model in models],
                axis=0,
            )
        )
        for endpoint in ENDPOINTS
    }
    for model in models:
        by_model[model] = {
            estimator: {endpoint: interval(draws[(model, estimator, endpoint)]) for endpoint in ENDPOINTS}
            for estimator in ESTIMATORS
        }
        by_model[model]["interaction"] = {
            endpoint: interval(draws[(model, "logistic", endpoint)] - draws[(model, "diagonal_lda", endpoint)])
            for endpoint in ENDPOINTS
        }
    return {"aggregate": aggregate, "by_model": by_model}


def plot_summary(point: dict[str, Any], bootstrap: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(point["by_model"])
    labels = [model.rsplit("/", 1)[-1].replace("-Instruct", "") for model in models]
    x = np.arange(len(models))
    width = 0.34
    logistic = [point["by_model"][model]["logistic"][ENDPOINTS[0]] for model in models]
    lda = [point["by_model"][model]["diagonal_lda"][ENDPOINTS[0]] for model in models]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, logistic, width, label="L2 logistic")
    ax.bar(x + width / 2, lda, width, label="Diagonal LDA")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mixed − best specialist macro ROC AUC")
    ax.set_title("Exact-row probe conclusion by estimator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "full320_exact_logistic_estimator_reversal_v1":
        raise ValueError("unexpected estimator-reversal protocol")
    data_root = args.data_root.resolve()
    rows = parse_rows(data_root / config["dataset"]["items_path"], config["dataset"]["items_sha256"])
    split_audits = {}
    seeds = [int(seed) for seed in config["probe"]["assignment_seeds"]]
    labels = np.asarray([1 if row["intended_purpose"] == "evaluation" else 0 for row in rows], dtype=np.int8)
    formats = np.asarray([str(row["format"]) for row in rows])
    canonical = np.asarray([row["render_slot"] == "stated" and not bool(row["legacy"]) for row in rows])
    records = []
    score_store: dict[tuple[str, str, str, int | None], np.ndarray] = {}
    source_runs = {}
    model_audits = []
    for model_config in config["models"]:
        model_id = str(model_config["model_id"])
        npz_path = data_root / model_config["activation_path"]
        if sha256_file(npz_path) != model_config["activation_sha256"]:
            raise ValueError(f"activation hash mismatch for {model_id}")
        expected_shape = [len(rows), int(model_config["n_layers"]), int(model_config["hidden_size"])]
        x_path = extract_x_memmap(npz_path, args.memmap_root.resolve(), expected_shape)
        X = np.load(x_path, mmap_mode="r", allow_pickle=False)
        layer = int(config["depth"]["layers"][str(model_config["n_layers"])])
        lda_outputs, source = load_lda_scores(
            data_root / model_config["lda_result_path"], model_id, float(config["depth"]["normalized"]), len(rows)
        )
        split_protocol = {
            "protocol_id": source["source_analysis"],
            "dataset": config["dataset"],
            "probe": config["probe"],
        }
        splits, split_audit = build_splits(rows, split_protocol, double_heldout=False)
        if split_audit["split_records_sha256"] != source["source_split_audit_sha256"]:
            raise ValueError(f"logistic/LDA split audit mismatch for {model_id}")
        split_audits[model_id] = split_audit
        logistic_outputs = fit_logistic_oof(X[:, layer, :], splits, seeds, config["probe"])
        logistic_records, logistic_store = records_and_store(
            logistic_outputs,
            "logistic",
            model_id,
            str(model_config["revision"]),
            layer,
            float(config["depth"]["normalized"]),
            labels,
            formats,
            canonical,
        )
        lda_records, lda_store = records_and_store(
            lda_outputs,
            "diagonal_lda",
            model_id,
            str(model_config["revision"]),
            layer,
            float(config["depth"]["normalized"]),
            labels,
            formats,
            canonical,
        )
        records.extend(logistic_records)
        records.extend(lda_records)
        score_store.update({(model_id, *key): value for key, value in logistic_store.items()})
        score_store.update({(model_id, *key): value for key, value in lda_store.items()})
        source_runs[model_id] = source
        model_audits.append(
            {"model": model_id, "revision": model_config["revision"], "layer": layer, "activation_sha256": sha256_file(npz_path)}
        )
        mmap_handle = X._mmap
        del X
        mmap_handle.close()
        if not args.keep_memmap:
            x_path.unlink(missing_ok=True)
    point = point_summary(records, seeds)
    counts, block_lookup = bootstrap_counts(
        [str(row["payload_block_id"]) for row in rows],
        int(config["uncertainty"]["bootstrap_resamples"]),
        int(config["uncertainty"]["bootstrap_seed"]),
    )
    bootstrap = bootstrap_summary(score_store, rows, seeds, counts, block_lookup)
    criteria = {
        "logistic_auc_ci_positive": bootstrap["aggregate"]["logistic"][ENDPOINTS[0]]["ci_2.5"] > 0,
        "interaction_auc_ci_positive": bootstrap["aggregate"]["interaction"][ENDPOINTS[0]]["ci_2.5"] > 0,
        "logistic_model_replication": sum(
            value["logistic"][ENDPOINTS[0]] > 0 for value in point["by_model"].values()
        ) >= 2,
        "interaction_model_replication": all(
            value["interaction"][ENDPOINTS[0]] > 0 for value in point["by_model"].values()
        ),
        "assignment_stability": sum(
            value["logistic"][ENDPOINTS[0]] > 0 for value in point["by_seed"].values()
        ) >= 7,
    }
    logistic_point = point["aggregate"]["logistic"][ENDPOINTS[0]]
    interaction_point = point["aggregate"]["interaction"][ENDPOINTS[0]]
    decision = "positive" if all(criteria.values()) else (
        "mixed" if logistic_point > 0 and interaction_point > 0 else "negative"
    )
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_estimator_reversal",
        "analysis": config["protocol_id"],
        "decision": decision,
        "criteria": criteria,
        "point_summary": point,
        "bootstrap": bootstrap,
        "split_audits": split_audits,
        "split_audit_sha256": canonical_json_sha256(split_audits),
        "models": model_audits,
        "source_lda_runs": source_runs,
        "records": records,
        "limitations": [
            "The logistic estimator was chosen after inspecting LDA outcomes and a loose 8B logistic precursor; this is a targeted confirmation.",
            "The fixed q=0.375 depth matches the known precursor rather than a full layer trajectory.",
            "Estimator dependence does not establish which probe is semantically correct.",
            "The endpoint is observational decoding, not behavior or a causal model mechanism.",
        ],
    }
    output_dir = args.out_dir.resolve()
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or incomplete.exists():
        raise FileExistsError(f"immutable output exists: {output_dir} or {incomplete}")
    incomplete.mkdir(parents=True)
    outputs = config["outputs"]
    result_path = incomplete / outputs["result_json"]
    table_path = incomplete / outputs["table_tsv"]
    figure_path = incomplete / outputs["figure"]
    manifest_path = incomplete / outputs["manifest_json"]
    result_path.write_text(strict_json_text(result), encoding="utf-8")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plot_summary(point, bootstrap, figure_path)
    output_paths = [result_path, table_path, figure_path]
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_preregistered_estimator_reversal",
        "analysis": config["protocol_id"],
        "repo_commit": repo_commit(),
        "protocol_sha256": sha256_file(config_path),
        "items_sha256": config["dataset"]["items_sha256"],
        "source_lda_runs": source_runs,
        "output_hashes": {path.name: output_hash(path) for path in output_paths},
        "decision": decision,
        "criteria": criteria,
        "claim_scope": config["claim_scope"],
        "elapsed_seconds": time.time() - started,
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
