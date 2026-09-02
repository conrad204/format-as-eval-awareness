from __future__ import annotations


import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.full320_strict_equal_rows import (
    PRIMARY_VARIANT,
    balanced_accuracy_draws,
    bootstrap_counts,
    build_splits,
    canonical_pair_units,
    fit_oof_scores,
    metric_bundle,
    weighted_auc_draws,
)


FAMILIES = ("fg1", "fg2", "fg3", "fg4", "fg5")
SEEDS = (101, 102)


def synthetic_protocol() -> dict:
    return {
        "protocol_id": "synthetic_strict_equal_rows",
        "dataset": {"purpose_families": list(FAMILIES)},
        "probe": {"assignment_seeds": list(SEEDS)},
    }


def synthetic_rows(*, null: bool = False) -> tuple[list[dict], np.ndarray]:
    rows: list[dict] = []
    features: list[np.ndarray] = []
    for topic_index in range(2):
        for block_index in range(10):
            block = f"block_{topic_index}_{block_index}"
            topic = f"topic_{topic_index}"
            noise = (block_index - 4.5) / 20.0
            for family_index, family in enumerate(FAMILIES):
                for environment in ("benchmark", "casual"):
                    template = f"{environment}_template_{family_index % 2}"
                    pair = f"pair_{family}"
                    for purpose, sign in (("deployment", -1.0), ("evaluation", 1.0)):
                        rows.append(
                            {
                                "item_id": f"item_{len(rows)}",
                                "payload_block_id": block,
                                "purpose_pair_id": pair,
                                "format": environment,
                                "template_id": template,
                                "render_slot": "stated",
                                "legacy": False,
                                "intended_purpose": purpose,
                                "purpose_family_id": family,
                                "topic_id": topic,
                            }
                        )
                        shared = 0.0 if null else sign
                        interaction = 0.0 if null else sign * (1.5 if environment == "benchmark" else -1.5)
                        features.append(
                            np.asarray(
                                [shared, interaction, noise, family_index / 10.0, topic_index / 10.0, 1.0 + noise * noise],
                                dtype=np.float32,
                            )
                        )
    return rows, np.stack(features)


def test_pair_unit_includes_template_and_exact_labels() -> None:
    rows, _ = synthetic_rows()
    units = canonical_pair_units(rows)
    assert len(units) == 2 * 10 * 5 * 2
    assert all(len(key) == 4 for key in units)
    for deployment_index, evaluation_index in units.values():
        assert rows[deployment_index]["intended_purpose"] == "deployment"
        assert rows[evaluation_index]["intended_purpose"] == "evaluation"


def test_splits_exactly_balance_rows_blocks_and_classes() -> None:
    rows, _ = synthetic_rows()
    splits, audit = build_splits(rows, synthetic_protocol(), double_heldout=False)
    assert audit["n_splits"] == 5
    assert audit["canonical_test_rows"] == len(rows)
    for split in splits:
        lengths = []
        blocks = []
        for pairs in split["regime_pairs"].values():
            lengths.append(2 * len(pairs))
            blocks.append({key[0] for key, _, _ in pairs})
            labels = [label for _, _, _ in pairs for label in (0, 1)]
            assert sum(labels) * 2 == len(labels)
        assert len(set(lengths)) == 1
        assert all(value == blocks[0] for value in blocks[1:])


def test_double_heldout_splits_exclude_test_payload_blocks() -> None:
    rows, _ = synthetic_rows()
    splits, audit = build_splits(rows, synthetic_protocol(), double_heldout=True)
    assert audit["n_splits"] == 25
    seen_test: list[int] = []
    for split in splits:
        test_blocks = {rows[index]["payload_block_id"] for index in split["test_indices"]}
        seen_test.extend(split["test_indices"].tolist())
        for pairs in split["regime_pairs"].values():
            train_blocks = {key[0] for key, _, _ in pairs}
            assert train_blocks.isdisjoint(test_blocks)
    assert sorted(seen_test) == list(range(len(rows)))


def test_mixed_environment_recovers_shared_signal() -> None:
    rows, features = synthetic_rows()
    splits, _ = build_splits(rows, synthetic_protocol(), double_heldout=False)
    outputs = fit_oof_scores(features, splits, SEEDS, include_robustness=False)[PRIMARY_VARIANT]
    labels = np.asarray([row["intended_purpose"] == "evaluation" for row in rows], dtype=np.int8)
    formats = np.asarray([row["format"] for row in rows])
    mask = np.ones(len(rows), dtype=bool)
    benchmark = metric_bundle(labels, formats, mask, outputs["benchmark_only"])
    casual = metric_bundle(labels, formats, mask, outputs["casual_only"])
    for seed in SEEDS:
        mixed = metric_bundle(labels, formats, mask, outputs[f"mixed_seed_{seed}"])
        assert mixed["macro_format_roc_auc"] > max(
            benchmark["macro_format_roc_auc"], casual["macro_format_roc_auc"]
        )
        assert mixed["macro_format_balanced_accuracy"] > max(
            benchmark["macro_format_balanced_accuracy"], casual["macro_format_balanced_accuracy"]
        )


def test_null_features_do_not_create_positive_decoding() -> None:
    rows, features = synthetic_rows(null=True)
    splits, _ = build_splits(rows, synthetic_protocol(), double_heldout=False)
    outputs = fit_oof_scores(features, splits, SEEDS, include_robustness=False)[PRIMARY_VARIANT]
    labels = np.asarray([row["intended_purpose"] == "evaluation" for row in rows], dtype=np.int8)
    formats = np.asarray([row["format"] for row in rows])
    mask = np.ones(len(rows), dtype=bool)
    for scores in outputs.values():
        metrics = metric_bundle(labels, formats, mask, scores)
        assert metrics["macro_format_roc_auc"] == pytest.approx(0.5)
        assert metrics["macro_format_balanced_accuracy"] == pytest.approx(0.5)


def test_weighted_cluster_metrics_match_explicit_resampling() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([-1.0, 0.2, -0.4, 0.2, 0.1, 1.2, -0.8, 0.7])
    blocks = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    counts, lookup = bootstrap_counts(blocks.tolist(), 20, 991)
    columns = np.asarray([lookup[value] for value in blocks], dtype=np.int32)
    auc_draws = weighted_auc_draws(labels, scores, columns, counts, batch_size=5)
    ba_draws = balanced_accuracy_draws(labels, scores, columns, counts)
    for draw, block_counts in enumerate(counts):
        indices = np.concatenate(
            [np.flatnonzero(blocks == block) for block, column in lookup.items() for _ in range(int(block_counts[column]))]
        )
        assert auc_draws[draw] == pytest.approx(roc_auc_score(labels[indices], scores[indices]))
        assert ba_draws[draw] == pytest.approx(
            balanced_accuracy_score(labels[indices], scores[indices] > 0)
        )
