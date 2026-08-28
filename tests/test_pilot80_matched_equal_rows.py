from __future__ import annotations

import numpy as np
import pytest

from scripts.full320_strict_equal_rows import PRIMARY_VARIANT, fit_oof_scores, metric_bundle
from scripts.pilot80_matched_equal_rows import build_matched_splits, matched_units


FAMILIES = ("fg1", "fg2", "fg3", "fg4", "fg5")
SEEDS = (201, 202)


def protocol() -> dict:
    return {
        "dataset": {"purpose_families": list(FAMILIES)},
        "probe": {"assignment_seeds": list(SEEDS)},
    }


def matched_synthetic_rows() -> tuple[list[dict], np.ndarray]:
    rows: list[dict] = []
    features: list[np.ndarray] = []
    for topic_index in range(2):
        for block_index in range(5):
            block = f"block_{topic_index}_{block_index}"
            topic = f"topic_{topic_index}"
            for family_index, family in enumerate(FAMILIES):
                pair = f"pair_{family}"
                for environment in ("benchmark", "casual"):
                    for purpose, sign in (("deployment", -1.0), ("evaluation", 1.0)):
                        rows.append(
                            {
                                "item_id": f"item_{len(rows)}",
                                "payload_block_id": block,
                                "purpose_pair_id": pair,
                                "format": environment,
                                "render_slot": "stated",
                                "intended_purpose": purpose,
                                "purpose_family_id": family,
                                "topic_id": topic,
                            }
                        )
                        interaction = sign * (1.5 if environment == "benchmark" else -1.5)
                        features.append(
                            np.asarray(
                                [sign, interaction, block_index / 20.0, family_index / 20.0, topic_index / 10.0, 1.0],
                                dtype=np.float32,
                            )
                        )
    return rows, np.stack(features)


def test_matched_units_require_complete_factorial_cells() -> None:
    rows, _ = matched_synthetic_rows()
    units = matched_units(rows)
    assert len(units) == 2 * 5 * 5
    assert all(set(unit["cells"]) == {"benchmark", "casual"} for unit in units.values())
    with pytest.raises(ValueError, match="incomplete matched group"):
        matched_units(rows[:-1])


def test_matched_splits_use_identical_groups_and_rows() -> None:
    rows, _ = matched_synthetic_rows()
    splits, audit = build_matched_splits(rows, protocol(), double_heldout=False)
    assert audit["n_splits"] == 5
    assert audit["n_complete_matched_groups"] == 50
    assert audit["n_stated_test_rows"] == len(rows)
    for split in splits:
        group_sets = [{key for key, _, _ in pairs} for pairs in split["regime_pairs"].values()]
        assert all(groups == group_sets[0] for groups in group_sets[1:])
        assert len({2 * len(pairs) for pairs in split["regime_pairs"].values()}) == 1


def test_matched_double_holdout_excludes_payloads() -> None:
    rows, _ = matched_synthetic_rows()
    splits, audit = build_matched_splits(rows, protocol(), double_heldout=True)
    assert audit["n_splits"] == 25
    for split in splits:
        test_blocks = {rows[index]["payload_block_id"] for index in split["test_indices"]}
        for pairs in split["regime_pairs"].values():
            train_blocks = {key[0] for key, _, _ in pairs}
            assert train_blocks.isdisjoint(test_blocks)


def test_matched_mixed_training_recovers_shared_signal() -> None:
    rows, features = matched_synthetic_rows()
    splits, _ = build_matched_splits(rows, protocol(), double_heldout=False)
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
