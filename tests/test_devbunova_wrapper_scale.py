from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.devbunova_wrapper_scale import (
    balanced_accuracy,
    bf16_u16_to_float32,
    fit_diag_lda,
    fixed_depth_layers,
    paired_bootstrap_summary,
    parse_item_id,
    roc_auc,
    run,
    score_diag_lda,
    select_balanced_core,
    stratified_cell_folds,
)


def _score_train_test(train_x, train_y, test_x):
    return score_diag_lda(fit_diag_lda(train_x, train_y), test_x)


def test_bf16_decode_matches_known_float32_prefix_bits():
    expected = np.asarray([0.0, 1.0, -2.0, 3.5, math.inf, -math.inf], dtype=np.float32)
    encoded = (expected.view(np.uint32) >> 16).astype(np.uint16)
    decoded = bf16_u16_to_float32(encoded)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)


def test_parse_item_id_accepts_contract_and_rejects_malformed_values():
    parsed = parse_item_id("dev2x2::bench_eval::asis::0042")
    assert parsed == {
        "dataset": "dev2x2",
        "condition": "bench_eval",
        "variant": "asis",
        "index": 42,
    }
    for malformed in (
        "bench_eval::asis::0042",
        "dev2x2::bench_eval::unknown::0042",
        "dev2x2::bench_eval::asis::42",
        "dev2x2::::asis::0042",
    ):
        with pytest.raises(ValueError):
            parse_item_id(malformed)


def test_balanced_core_selection_is_exact_stable_and_asis_only():
    conditions = (
        "bench_eval",
        "bench_deploy_rewritten",
        "casual_eval_mutual",
        "casual_deploy_matched",
    )
    item_ids = [
        f"dev2x2::{condition}::asis::{index:04d}"
        for condition in conditions
        for index in range(210)
    ]
    item_ids += [
        f"dev2x2::{condition}::stripped::{index:04d}"
        for condition in conditions
        for index in range(210, 220)
    ]
    indices_a, cells_a = select_balanced_core(item_ids, n_per_cell=204, seed=17)
    indices_b, cells_b = select_balanced_core(item_ids, n_per_cell=204, seed=17)
    np.testing.assert_array_equal(indices_a, indices_b)
    np.testing.assert_array_equal(cells_a, cells_b)
    assert len(indices_a) == 816
    assert {cell: int(np.sum(cells_a == cell)) for cell in set(cells_a)} == {
        "EB": 204,
        "DB": 204,
        "EC": 204,
        "DC": 204,
    }
    assert all("::asis::" in item_ids[index] for index in indices_a)


def test_fixed_depth_layers_uses_round_half_up_and_rejects_collisions():
    assert fixed_depth_layers(32, [0, 0.125, 0.5, 0.75, 1]) == [0, 4, 16, 23, 31]
    assert fixed_depth_layers(126, [0, 0.125, 0.5, 0.75, 1]) == [0, 16, 63, 94, 125]
    with pytest.raises(ValueError):
        fixed_depth_layers(2, [0.0, 0.1])


def test_auc_handles_ties_and_balanced_accuracy_uses_fixed_threshold():
    labels = np.asarray([0, 0, 1, 1])
    assert roc_auc(labels, np.asarray([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert roc_auc(labels, np.zeros(4)) == 0.5
    assert balanced_accuracy(labels, np.asarray([-2.0, 0.2, 0.1, 3.0])) == 0.75


def test_diagonal_lda_detects_a_only_signal_and_sign_reversal():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1], 100)
    features = rng.normal(0, 1, size=(200, 12)).astype(np.float32)
    features[:, 0] += np.where(labels == 1, 2.0, -2.0)
    scores = _score_train_test(features, labels, features)
    assert roc_auc(labels, scores) > 0.99
    reversed_scores = _score_train_test(features, 1 - labels, features)
    assert roc_auc(labels, reversed_scores) < 0.01


def test_identical_regimes_have_zero_symmetric_transfer_penalty():
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 80)
    raw = rng.normal(size=(160, 20)).astype(np.float32)
    raw[:, 0] += np.where(labels == 1, 1.5, -1.5)
    template = raw.copy()
    model = fit_diag_lda(raw, labels)
    rr = roc_auc(labels, score_diag_lda(model, raw))
    rt = roc_auc(labels, score_diag_lda(model, template))
    template_model = fit_diag_lda(template, labels)
    tt = roc_auc(labels, score_diag_lda(template_model, template))
    tr = roc_auc(labels, score_diag_lda(template_model, raw))
    assert 0.5 * ((rr - rt) + (tt - tr)) == pytest.approx(0.0)


def test_rotated_wrapper_specific_signal_has_positive_transfer_penalty():
    rng = np.random.default_rng(3)
    labels = np.repeat([0, 1], 120)
    sign = np.where(labels == 1, 1.0, -1.0)
    raw = rng.normal(0, 1, size=(240, 24)).astype(np.float32)
    template = rng.normal(0, 1, size=(240, 24)).astype(np.float32)
    raw[:, 0] += 3.0 * sign
    template[:, 1] += 3.0 * sign
    raw_model = fit_diag_lda(raw, labels)
    template_model = fit_diag_lda(template, labels)
    rr = roc_auc(labels, score_diag_lda(raw_model, raw))
    rt = roc_auc(labels, score_diag_lda(raw_model, template))
    tt = roc_auc(labels, score_diag_lda(template_model, template))
    tr = roc_auc(labels, score_diag_lda(template_model, raw))
    assert rr > 0.99 and tt > 0.99
    assert 0.5 * ((rr - rt) + (tt - tr)) > 0.4


def test_b_only_format_signal_does_not_create_purpose_signal():
    rng = np.random.default_rng(4)
    n = 400
    purpose = np.tile(np.repeat([0, 1], 100), 2)
    format_label = np.repeat([0, 1], 200)
    features = rng.normal(size=(n, 16)).astype(np.float32)
    features[:, 0] += np.where(format_label == 1, 3.0, -3.0)
    train = np.arange(n) % 2 == 0
    purpose_scores = _score_train_test(features[train], purpose[train], features[~train])
    format_scores = _score_train_test(features[train], format_label[train], features[~train])
    assert 0.4 < roc_auc(purpose[~train], purpose_scores) < 0.6
    assert roc_auc(format_label[~train], format_scores) > 0.99


def test_c_like_opposite_purpose_codes_produce_opposite_directions():
    rng = np.random.default_rng(5)
    labels = np.repeat([0, 1], 100)
    sign = np.where(labels == 1, 1.0, -1.0)
    benchmark = rng.normal(size=(200, 10)).astype(np.float32)
    casual = rng.normal(size=(200, 10)).astype(np.float32)
    benchmark[:, 0] += 2.5 * sign
    casual[:, 0] -= 2.5 * sign
    benchmark_model = fit_diag_lda(benchmark, labels)
    casual_model = fit_diag_lda(casual, labels)
    cosine = np.dot(benchmark_model["direction"], casual_model["direction"]) / (
        np.linalg.norm(benchmark_model["direction"])
        * np.linalg.norm(casual_model["direction"])
    )
    assert cosine < -0.8
    assert roc_auc(labels, score_diag_lda(benchmark_model, casual)) < 0.05


def test_null_noise_heldout_auc_remains_near_chance():
    rng = np.random.default_rng(6)
    labels = np.repeat([0, 1], 300)
    features = rng.normal(size=(600, 8)).astype(np.float32)
    train = np.arange(600) % 2 == 0
    scores = _score_train_test(features[train], labels[train], features[~train])
    assert 0.4 < roc_auc(labels[~train], scores) < 0.6


def test_cell_stratified_folds_are_disjoint_exhaustive_and_balanced():
    ids = []
    cells = []
    for cell in ("EB", "DB", "EC", "DC"):
        for index in range(204):
            ids.append(f"dev2x2::{cell.lower()}::asis::{index:04d}")
            cells.append(cell)
    folds_a = stratified_cell_folds(ids, cells, n_folds=5, seed=19)
    folds_b = stratified_cell_folds(ids, cells, n_folds=5, seed=19)
    np.testing.assert_array_equal(folds_a, folds_b)
    assert set(folds_a.tolist()) == set(range(5))
    for cell in set(cells):
        counts = np.bincount(folds_a[np.asarray(cells) == cell], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_paired_bootstrap_is_deterministic_and_preserves_score_pairing():
    labels = np.repeat([0, 1], 100)
    perfect = labels.astype(float)
    reversed_scores = 1.0 - perfect
    summary_a = paired_bootstrap_summary(
        labels, perfect, reversed_scores, seed=20, n_resamples=100
    )
    summary_b = paired_bootstrap_summary(
        labels, perfect, reversed_scores, seed=20, n_resamples=100
    )
    assert summary_a["difference"] == pytest.approx(1.0)
    assert summary_a["difference_ci_2.5"] == pytest.approx(1.0)
    np.testing.assert_array_equal(summary_a["bootstrap_draws"], summary_b["bootstrap_draws"])


def test_end_to_end_tiny_run_produces_immutable_evidence(tmp_path: Path):
    conditions = (
        "bench_eval",
        "bench_deploy_rewritten",
        "casual_eval_mutual",
        "casual_deploy_matched",
    )
    item_ids = np.asarray(
        [
            f"dev2x2::{condition}::asis::{index:04d}"
            for condition in conditions
            for index in range(4)
        ]
    )
    data_root = tmp_path / "data"
    rng = np.random.default_rng(21)
    model_ids = {
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
        "405b": "meta-llama/Llama-3.1-405B-Instruct",
    }
    purpose = np.asarray([1] * 4 + [0] * 4 + [1] * 4 + [0] * 4)
    format_label = np.asarray([1] * 8 + [0] * 8)
    for model_index, (model, model_id) in enumerate(model_ids.items()):
        for regime in ("raw", "template"):
            directory = data_root / model / regime / "chunks"
            directory.mkdir(parents=True)
            activations = rng.normal(0, 0.5, size=(16, 9, 4)).astype(np.float32)
            signal_feature = 0 if regime == "raw" else 1
            activations[:, :, signal_feature] += (
                (2 * purpose - 1)[:, None] * (1.0 + 0.2 * model_index)
            )
            activations[:, :, 2] += (2 * format_label - 1)[:, None]
            encoded = (activations.view(np.uint32) >> 16).astype(np.uint16)
            np.savez(
                directory / "chunk_0000.npz",
                X_bf16_u16=encoded,
                item_ids=item_ids,
                layers=np.arange(9, dtype=np.int16),
                token_lengths=np.arange(16, dtype=np.int32)
                + (20 if regime == "template" else 0),
                top1_token_ids=np.arange(16, dtype=np.int64)
                + (1 if regime == "template" else 0),
                row_start=np.asarray(0, dtype=np.int64),
                model_id=np.asarray(model_id),
                regime=np.asarray(regime),
            )

    protocol = json.loads(
        Path("configs/devbunova_wrapper_scale_v1.json").read_text(encoding="utf-8")
    )
    protocol["dataset"]["selected_chunk_indices"] = [0]
    protocol["dataset"]["expected_items_per_model_regime"] = 16
    for spec in protocol["dataset"]["expected_models"].values():
        spec["n_layers"] = 9
        spec["hidden_size"] = 4
    protocol["selection"]["n_per_cell"] = 4
    protocol["selection"]["expected_selected_rows"] = 16
    protocol["uncertainty"]["bootstrap_resamples"] = 5
    protocol["uncertainty"]["label_permutations"] = 2
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(protocol), encoding="utf-8")
    output_path = tmp_path / "evidence"

    result = run(config_path, data_root, output_path, "synthetic_test")

    assert result["status"] == "completed_development_observational"
    assert result["selection"]["selected_rows"] == 16
    assert (output_path / "results.json").is_file()
    assert (output_path / "metrics.tsv").is_file()
    assert (output_path / "run_manifest.json").is_file()
    assert (output_path / "auc_trajectories.png").is_file()
    with pytest.raises(FileExistsError):
        run(config_path, data_root, output_path, "synthetic_test_repeat")
