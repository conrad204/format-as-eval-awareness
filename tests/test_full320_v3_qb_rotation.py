import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from full320_v3_emergency_cperp_sweep import RANK, gram_orth_basis  # noqa: E402
from full320_v3_factorial_residual_sweep import captured_energy, factorial_effects  # noqa: E402
from full320_v3_qb_rotation_sweep import (  # noqa: E402
    SUB_RANK,
    containment_error,
    init_worker,
    layer_job,
    random_subspace_inside,
    rotate_qb_by_c_alignment,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_full320_v3_emergency_cperp import emergency_fixture  # noqa: E402


def fitted_bases(n_families=3, groups_per_family=24, d=192):
    X, rows = emergency_fixture(n_families=n_families, groups_per_family=groups_per_family, d=d)
    train_mask = np.asarray([r["purpose_family_id"] != "fg0" for r in rows])
    B, C, _, _, _, _ = factorial_effects(X, rows, train_mask)
    return X, rows, gram_orth_basis(B, RANK), C


def test_rotation_is_orthonormal_and_stays_inside_qb():
    _, _, q_b, C = fitted_bases()
    q_rot, singular = rotate_qb_by_c_alignment(q_b, C)
    assert q_rot.shape == q_b.shape
    np.testing.assert_allclose(q_rot.T @ q_rot, np.eye(q_rot.shape[1]), atol=1e-10)
    # A rotation of Q_B spans exactly Q_B: same projector, zero containment error.
    np.testing.assert_allclose(q_rot @ q_rot.T, q_b @ q_b.T, atol=1e-10)
    assert containment_error(q_rot, q_b) < 1e-10
    assert np.all(np.diff(singular) <= 1e-12), "singular values must be descending"


def test_top_block_captures_more_training_c_than_bottom_block():
    _, _, q_b, C = fitted_bases()
    q_rot, _ = rotate_qb_by_c_alignment(q_b, C)
    top, bottom = q_rot[:, :SUB_RANK], q_rot[:, -SUB_RANK:]
    # The rotation is defined by exactly this ordering, so it must hold on train C.
    assert np.sum((C @ top) ** 2) > np.sum((C @ bottom) ** 2)
    # Both blocks together cannot beat the parent subspace they came from.
    assert np.sum((C @ q_b) ** 2) >= np.sum((C @ top) ** 2) - 1e-9


def test_top_bottom_and_random_blocks_are_rank_matched_and_inside_qb():
    _, _, q_b, C = fitted_bases()
    q_rot, _ = rotate_qb_by_c_alignment(q_b, C)
    random_block = random_subspace_inside(q_b, SUB_RANK, seed=123)
    for block in (q_rot[:, :SUB_RANK], q_rot[:, -SUB_RANK:], random_block):
        assert block.shape[1] == SUB_RANK
        np.testing.assert_allclose(block.T @ block, np.eye(SUB_RANK), atol=1e-10)
        assert containment_error(block, q_b) < 1e-10
    # Seeded control is reproducible.
    np.testing.assert_allclose(random_block, random_subspace_inside(q_b, SUB_RANK, seed=123), atol=0)
    assert not np.allclose(random_block, random_subspace_inside(q_b, SUB_RANK, seed=124))


def test_rotation_rejects_a_parent_subspace_too_small_to_split():
    rng = np.random.default_rng(5)
    q_small = np.linalg.qr(rng.normal(size=(64, 2 * SUB_RANK - 1)))[0]
    with pytest.raises(ValueError, match="too small for disjoint"):
        rotate_qb_by_c_alignment(q_small, rng.normal(size=(40, 64)))


def test_layer_job_emits_three_rank16_conditions_with_free_diagnostics(tmp_path):
    X, rows = emergency_fixture(n_families=3, groups_per_family=24, d=192)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.save(shard_dir / "layer_000.f32.npy", np.ascontiguousarray(X, dtype=np.float32))
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    y = np.asarray([int(r["label"]) for r in rows])
    init_worker(
        {
            "rows": rows,
            "y": y,
            "fam": fam,
            "is_bench": fmt == "benchmark",
            "is_casual": fmt == "casual",
            "shard_dir": shard_dir,
            "basis_backend": "gram",
        }
    )
    layer, scores, sanity, _ = layer_job(0)

    assert layer == 0
    assert scores.shape == (3, 3, len(y))
    assert sanity["held_family_basis_violations"] == 0
    assert sanity["max_orthonormality_error"] < 1e-10
    assert sanity["max_subspace_containment_error"] < 1e-6 * max(1.0, sanity["max_abs_activation"])
    assert set(sanity["c_capture"]) == {"topC16", "bottomC16", "randomB16"}
    assert set(sanity["energy_removed"]) == {"topC16", "bottomC16", "randomB16"}
    for name in ("topC16", "bottomC16", "randomB16"):
        assert 0.0 < sanity["energy_removed"][name]["pooled_fraction_all"] < 1.0
        assert 0.0 <= sanity["c_capture"][name]["cross_fitted_fraction"] <= 1.0
    # Held-out C capture ordering is an empirical claim, but train-side alignment
    # energy must be ordered by construction in every fold.
    for fold in sanity["c_alignment_spectrum"]:
        assert fold["top16_alignment_energy_fraction"] >= fold["bottom16_alignment_energy_fraction"]
    joint_i = 0
    for condition_i in range(3):
        assert np.all(np.isfinite(scores[condition_i, joint_i]))


def test_captured_energy_of_top_block_is_bounded_by_parent():
    X, rows, q_b, _ = fitted_bases()
    test_mask = np.asarray([r["purpose_family_id"] == "fg0" for r in rows])
    _, C_holdout, _, _, _, _ = factorial_effects(X, rows, test_mask)
    train_mask = ~test_mask
    B_train, C_train, _, _, _, _ = factorial_effects(X, rows, train_mask)
    q_rot, _ = rotate_qb_by_c_alignment(gram_orth_basis(B_train, RANK), C_train)
    parent = captured_energy(C_holdout, q_b)[2]
    top = captured_energy(C_holdout, q_rot[:, :SUB_RANK])[2]
    assert top <= parent + 1e-12
