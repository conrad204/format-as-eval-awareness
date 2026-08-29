import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from full320_v3_factorial_residual_bootstrap import (  # noqa: E402
    prepare_auc_order,
    specificity_components,
    weighted_auc_from_order,
)
from full320_v3_factorial_residual_sweep import (  # noqa: E402
    captured_energy,
    combined_basis,
    factorial_effects,
    load_partial_layer,
    random_basis_orthogonal_to,
    write_partial_layer,
)
from full320_v3_interchange_stratified_bootstrap import summarize  # noqa: E402
from format_erasure_development import orth_basis, remove  # noqa: E402


def factorial_fixture(n_groups=3, d=8):
    rng = np.random.default_rng(7)
    B = rng.normal(size=(n_groups, d))
    C = rng.normal(size=(n_groups, d))
    rows, values = [], []
    cells = (
        ("evaluation", "benchmark", 1, 1),
        ("evaluation", "casual", 1, -1),
        ("deployment", "benchmark", -1, 1),
        ("deployment", "casual", -1, -1),
    )
    for group in range(n_groups):
        h = rng.normal(size=d)
        a = rng.normal(size=d)
        for purpose, fmt, p, f in cells:
            rows.append(
                {
                    "payload_block_id": f"pb{group}",
                    "purpose_pair_id": f"pair{group}",
                    "purpose_family_id": f"fg{group % 2}",
                    "intended_purpose": purpose,
                    "format": fmt,
                    "render_slot": "stated",
                }
            )
            values.append(h + p * a + f * B[group] + p * f * C[group])
    return np.asarray(values), rows, B, C


def test_factorial_b_c_and_natural_identities_are_exact():
    X, rows, expected_b, expected_c = factorial_fixture()
    B, C, feval, fdeploy, _, error = factorial_effects(X, rows, np.ones(len(rows), dtype=bool))
    np.testing.assert_allclose(B, expected_b, atol=1e-12)
    np.testing.assert_allclose(C, expected_c, atol=1e-12)
    np.testing.assert_allclose(feval, B + C, atol=1e-12)
    np.testing.assert_allclose(fdeploy, B - C, atol=1e-12)
    assert error < 1e-12


def test_factorial_groups_fail_closed_across_families():
    X, rows, _, _ = factorial_fixture(n_groups=1)
    rows[-1]["purpose_family_id"] = "held-out"
    with pytest.raises(ValueError, match="spans purpose_family_id"):
        factorial_effects(X, rows, np.ones(len(rows), dtype=bool))


def test_residualized_c_is_orthogonal_and_sequential_equals_joint():
    X, rows, _, _ = factorial_fixture(n_groups=6, d=16)
    B, C, _, _, _, _ = factorial_effects(X, rows, np.ones(len(rows), dtype=bool))
    q_b = orth_basis(B, 4)
    q_cperp = orth_basis(remove(C, q_b), 4)
    q_joint = combined_basis(q_b, q_cperp)
    assert np.max(np.abs(q_b.T @ q_cperp)) < 1e-10
    sequential = remove(remove(X, q_b), q_cperp)
    direct = remove(X, q_joint)
    np.testing.assert_allclose(sequential, direct, atol=1e-10)



def test_c_capture_uses_c_own_squared_norm():
    C = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    q_b = np.asarray([[1.0], [0.0], [0.0]])
    numerator, denominator, fraction = captured_energy(C, q_b)
    assert numerator == pytest.approx(9.0)
    assert denominator == pytest.approx(50.0)
    assert fraction == pytest.approx(9.0 / 50.0)

def test_random_residual_control_is_seeded_rank_matched_and_b_orthogonal():
    rng = np.random.default_rng(19)
    q_b = orth_basis(rng.normal(size=(5, 32)), 5)
    first = random_basis_orthogonal_to(32, 7, q_b, seed=991)
    second = random_basis_orthogonal_to(32, 7, q_b, seed=991)
    assert first.shape == (32, 7)
    np.testing.assert_allclose(first, second, atol=0)
    np.testing.assert_allclose(first.T @ first, np.eye(7), atol=1e-10)
    assert np.max(np.abs(q_b.T @ first)) < 1e-10


def test_weighted_auc_matches_explicit_block_resampling():
    scores = np.asarray(
        [
            [-2.1, -0.7, 0.2, 1.4, 2.8, 3.3],
            [-1.2, 0.4, -0.8, 2.1, 1.7, 3.9],
        ]
    )
    y = np.asarray([0, 0, 1, 0, 1, 1])
    item_block = np.asarray([0, 0, 1, 1, 2, 2])
    block_counts = np.asarray([2, 0, 3])
    order, positive = prepare_auc_order(scores, y)
    actual = weighted_auc_from_order(order, positive, block_counts[item_block])
    expanded = np.concatenate(
        [np.flatnonzero(item_block == block) for block in [0, 0, 2, 2, 2]]
    )
    expected = np.asarray([roc_auc_score(y[expanded], row[expanded]) for row in scores])
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_specific_c_subtracts_rank_matched_residual_control():
    extra_c, extra_random, specific = specificity_components(0.40, 0.10, 0.25)
    assert extra_c == pytest.approx(0.30)
    assert extra_random == pytest.approx(0.15)
    assert specific == pytest.approx(0.15)


def test_interchange_strata_reuse_blocks_and_preserve_gap_signs():
    records = []
    row_id = 0
    for layer in (0, 1):
        for block in ("pb0", "pb1"):
            for purpose in ("evaluation", "deployment"):
                for direction in ("b2c", "c2b"):
                    for condition in ("real_swap", "random_control"):
                        records.append(
                            {
                                "layer": layer,
                                "payload_block_id": block,
                                "intended_purpose": purpose,
                                "direction": direction,
                                "condition": condition,
                                "purpose_score_baseline": 0.0,
                                "purpose_score_patched": 1.0 if condition == "real_swap" else 2.0,
                                "format_score_baseline_src": 0.0,
                                "format_score_baseline_tgt": 1.0,
                                "format_score_patched": 1.0 if condition == "real_swap" else -1.0,
                                "src_row": row_id,
                            }
                        )
                        row_id += 1
    result = summarize(records, n_boot=1000)
    assert result["n_layers"] == 2
    assert result["n_blocks"] == 2
    for groups in result["by_layer"].values():
        for group in groups.values():
            assert group["metrics"]["purpose_abs_shift"]["gap"] == pytest.approx(1.0)
            assert group["metrics"]["format_iia"]["gap"] == pytest.approx(1.0)


def test_partial_layer_checkpoint_publishes_atomically(tmp_path):
    partial_dir = tmp_path / "partial"
    ids = np.asarray(["item-a", "item-b"])
    layer_scores = np.zeros((9, 4, 3, len(ids)), dtype=np.float32)
    write_partial_layer(
        partial_dir,
        "llama31_8b",
        2,
        layer_scores,
        {"max_factorial_identity_error": 0.0},
        1.25,
        ids,
        [2],
        4,
    )

    restored_scores, restored_sanity, restored_elapsed = load_partial_layer(
        partial_dir, 2, "llama31_8b", ids
    )
    np.testing.assert_array_equal(restored_scores, layer_scores)
    assert restored_sanity == {"max_factorial_identity_error": 0.0}
    assert restored_elapsed == pytest.approx(1.25)
    manifest = json.loads((partial_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["completed_layers"] == [2]
    assert not list(partial_dir.glob(".*.tmp"))
