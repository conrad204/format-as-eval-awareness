import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from format_erasure_development import orth_basis  # noqa: E402
from full320_v3_emergency_cperp_bootstrap import (  # noqa: E402
    analyze,
    prepare_tie_aware_order,
    weighted_auc_with_ties,
)
from full320_v3_emergency_cperp_sweep import (  # noqa: E402
    CONDITIONS,
    RANK,
    REGIMES,
    gram_orth_basis,
    init_worker,
    layer_job,
    prepare_shards,
    shard_path,
    validate_basis_backend,
)
from full320_v3_factorial_residual_sweep import captured_energy, factorial_effects  # noqa: E402


def emergency_fixture(n_families=3, groups_per_family=8, d=128, seed=11):
    """Matched 2x2 stated bank with a purpose effect, a format effect, and interaction."""
    rng = np.random.default_rng(seed)
    cells = (
        ("evaluation", "benchmark", 1, 1),
        ("evaluation", "casual", 1, -1),
        ("deployment", "benchmark", -1, 1),
        ("deployment", "casual", -1, -1),
    )
    rows, values = [], []
    for family in range(n_families):
        for group in range(groups_per_family):
            key = f"fg{family}-pb{group}"
            payload = rng.normal(size=d)
            purpose = rng.normal(size=d)
            fmt_effect = rng.normal(size=d)
            interaction = rng.normal(size=d)
            for intended, fmt, p, f in cells:
                rows.append(
                    {
                        "item_id": f"item::{key}::{intended}::{fmt}",
                        "payload_block_id": key,
                        "purpose_pair_id": f"pair-{key}",
                        "purpose_family_id": f"fg{family}",
                        "intended_purpose": intended,
                        "format": fmt,
                        "render_slot": "stated",
                        "label": 1 if intended == "evaluation" else 0,
                    }
                )
                values.append(
                    payload
                    + p * purpose
                    + f * fmt_effect
                    + p * f * interaction
                    + rng.normal(0.0, 0.05, size=d)
                )
    return np.asarray(values), rows


def test_prepare_shards_writes_exact_atomic_layer_files(tmp_path):
    rng = np.random.default_rng(2)
    n_items, n_layers, d = 6, 3, 5
    X = rng.normal(size=(n_items, n_layers, d)).astype(np.float32)
    npz_path = tmp_path / "activations_all_layers.npz"
    np.savez_compressed(
        npz_path, X=X, item_ids=np.asarray([f"item{i}" for i in range(n_items)])
    )
    keep_mask = np.ones(n_items, dtype=bool)
    keep_mask[0] = False
    shard_dir = tmp_path / "shards"

    layers, hidden, seconds = prepare_shards(npz_path, keep_mask, shard_dir)
    assert (layers, hidden) == (n_layers, d)
    assert seconds > 0
    assert not list(shard_dir.glob(".*tmp*")), "atomic temp files must not survive"
    for layer in range(n_layers):
        shard = np.load(shard_path(shard_dir, layer))
        assert shard.dtype == np.float32
        np.testing.assert_array_equal(shard, X[keep_mask][:, layer, :])

    # Second call is a no-op that reports zero expansion time.
    assert prepare_shards(npz_path, keep_mask, shard_dir) == (n_layers, d, 0.0)


def test_prepare_shards_streams_in_blocks_without_materializing_x(tmp_path):
    """Streaming expansion must equal direct slicing, including across chunk edges."""
    rng = np.random.default_rng(9)
    n_items, n_layers, d = 11, 4, 6
    X = rng.normal(size=(n_items, n_layers, d)).astype(np.float32)
    npz_path = tmp_path / "activations_all_layers.npz"
    np.savez_compressed(
        npz_path, X=X, item_ids=np.asarray([f"item{i}" for i in range(n_items)])
    )
    keep_mask = np.ones(n_items, dtype=bool)
    keep_mask[[0, 5, 6, 10]] = False  # drops rows inside and at the edges of chunks
    shard_dir = tmp_path / "shards"

    row_bytes = n_layers * d * 4
    layers, hidden, _ = prepare_shards(
        npz_path, keep_mask, shard_dir, target_chunk_bytes=3 * row_bytes
    )
    assert (layers, hidden) == (n_layers, d)
    assert not list(shard_dir.glob(".*tmp*"))
    for layer in range(n_layers):
        shard = np.load(shard_path(shard_dir, layer))
        assert shard.shape == (int(keep_mask.sum()), d)
        assert shard.dtype == np.float32
        np.testing.assert_array_equal(shard, X[keep_mask][:, layer, :])


def test_prepare_shards_can_expand_only_requested_layers(tmp_path):
    rng = np.random.default_rng(12)
    n_items, n_layers, d = 8, 5, 4
    X = rng.normal(size=(n_items, n_layers, d)).astype(np.float32)
    npz_path = tmp_path / "activations_all_layers.npz"
    np.savez_compressed(
        npz_path, X=X, item_ids=np.asarray([f"item{i}" for i in range(n_items)])
    )
    keep_mask = np.ones(n_items, dtype=bool)
    shard_dir = tmp_path / "shards"

    prepare_shards(npz_path, keep_mask, shard_dir, layers=[1, 3])
    assert sorted(p.name for p in shard_dir.glob("layer_*.npy")) == [
        "layer_001.f32.npy",
        "layer_003.f32.npy",
    ]
    for layer in (1, 3):
        np.testing.assert_array_equal(np.load(shard_path(shard_dir, layer)), X[:, layer, :])

    # A later call for a different slice adds only what is missing.
    _, _, seconds = prepare_shards(npz_path, keep_mask, shard_dir, layers=[3, 4])
    assert seconds > 0
    np.testing.assert_array_equal(np.load(shard_path(shard_dir, 4)), X[:, 4, :])
    assert prepare_shards(npz_path, keep_mask, shard_dir, layers=[1, 3, 4])[2] == 0.0
    with pytest.raises(ValueError, match="outside 0"):
        prepare_shards(npz_path, keep_mask, shard_dir, layers=[99])


def test_prepare_shards_rejects_a_mask_of_the_wrong_length(tmp_path):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(4, 2, 3)).astype(np.float32)
    npz_path = tmp_path / "activations_all_layers.npz"
    np.savez_compressed(npz_path, X=X, item_ids=np.asarray(["a", "b", "c", "d"]))
    with pytest.raises(ValueError, match="keep_mask length"):
        prepare_shards(npz_path, np.ones(3, dtype=bool), tmp_path / "shards")


def test_tie_aware_weighted_auc_equals_sklearn_with_sample_weights():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(31)
    n = 240
    y = rng.integers(0, 2, size=n)
    # Rounding forces many exact ties, which is what float32 OOF scores produce.
    scores = np.asarray(
        [np.round(rng.normal(size=n) + 0.8 * y, 1).astype(np.float32).astype(np.float64) for _ in range(6)]
    )
    assert np.any(np.diff(np.sort(scores[0])) == 0), "fixture must contain exact ties"

    prepared = prepare_tie_aware_order(scores, y)
    for draw in range(6):
        weights = (
            np.ones(n)
            if draw == 0
            else rng.integers(0, 4, size=n).astype(np.float64)  # bootstrap-style counts incl. zeros
        )
        if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
            continue
        actual = weighted_auc_with_ties(prepared, weights)
        expected = [roc_auc_score(y, row, sample_weight=weights) for row in scores]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_bootstrap_matrix_layout_matches_point_aucs_under_unit_weights():
    """A unit-weight replicate must equal the sklearn point AUCs, row for row.

    This pins the (condition, layer) -> matrix-row mapping: indexing a boolean
    item mask alongside an integer regime index silently moves the masked axis to
    the front, which scrambles rows without changing any shape.
    """
    from full320_v3_factorial_residual_bootstrap import point_aucs

    rng = np.random.default_rng(47)
    n_layers, n_items = 4, 160
    y = rng.integers(0, 2, size=n_items)
    mask = np.zeros(n_items, dtype=bool)
    mask[::2] = True  # a cross-format regime scores only half the rows
    scores = np.full((len(CONDITIONS), 1, len(REGIMES), n_layers, n_items), np.nan)
    for condition_i in range(len(CONDITIONS)):
        for layer in range(n_layers):
            # Distinct separability per (condition, layer) so any row swap shows up.
            strength = 0.2 + 0.37 * condition_i + 0.11 * layer
            values = rng.normal(size=n_items) + strength * (y * 2 - 1)
            scores[condition_i, 0, REGIMES.index("joint"), layer] = values
            scores[condition_i, 0, REGIMES.index("b2c"), layer, mask] = values[mask]
            scores[condition_i, 0, REGIMES.index("c2b"), layer, ~mask] = values[~mask]
    regime_i = REGIMES.index("b2c")

    n_cells = len(CONDITIONS) * n_layers
    matrix = scores[:, 0, regime_i, :, :][:, :, mask].reshape(n_cells, int(mask.sum()))
    prepared = prepare_tie_aware_order(matrix, y[mask])
    actual = weighted_auc_with_ties(prepared, np.ones(int(mask.sum())))
    expected = point_aucs(scores, y)[:, 0, regime_i, :].reshape(n_cells)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_layer_job_reports_residual_energy_removed_for_cperp_and_random(tmp_path):
    X, rows = emergency_fixture()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.save(shard_path(shard_dir, 0), np.ascontiguousarray(X, dtype=np.float32))
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    init_worker(
        {
            "rows": rows,
            "y": np.asarray([int(r["label"]) for r in rows]),
            "fam": fam,
            "is_bench": fmt == "benchmark",
            "is_casual": fmt == "casual",
            "shard_dir": shard_dir,
            "basis_backend": "gram",
        }
    )
    _, _, sanity, _ = layer_job(0)

    energy = sanity["residual_energy_removed"]
    assert set(energy) >= {"b", "cperp", "rperp", "cperp_over_rperp_all", "cperp_over_rperp_heldout"}
    # The control is rank-matched to Q_C|B, so an energy gap is about direction, not rank.
    assert energy["cperp"]["mean_rank"] == energy["rperp"]["mean_rank"]
    for name in ("b", "cperp", "rperp"):
        for scope in ("all", "heldout"):
            fraction = energy[name][f"pooled_fraction_{scope}"]
            assert 0.0 < fraction < 1.0, (name, scope, fraction)
        assert len(energy[name]["by_fold"]) == len(set(fam))
    ratio = energy["cperp_over_rperp_all"]
    assert ratio == pytest.approx(
        energy["cperp"]["pooled_fraction_all"] / energy["rperp"]["pooled_fraction_all"]
    )


def test_gram_basis_reproduces_orth_basis_subspace_on_real_effect_matrices():
    X, rows = emergency_fixture()
    train_mask = np.asarray([r["purpose_family_id"] != "fg0" for r in rows])
    checks = validate_basis_backend(X, rows, train_mask, RANK, 1e-8)
    assert checks["passed"], checks
    for key in ("B", "C_perp_B"):
        assert checks[key]["min_principal_cosine"] > 1.0 - 1e-8
        assert checks[key]["max_projector_error"] < 1e-8
        assert checks[key]["max_transformed_error"] < 1e-8
        assert checks[key]["fast_rank"] == checks[key]["reference_rank"]


def test_gram_basis_matches_reference_rank_truncation_when_rows_limit_rank():
    rng = np.random.default_rng(3)
    V = rng.normal(size=(5, 64))
    fast = gram_orth_basis(V, RANK)
    reference = orth_basis(V, RANK)
    assert fast.shape == reference.shape == (64, 5)
    np.testing.assert_allclose(fast @ fast.T, reference @ reference.T, atol=1e-10)
    np.testing.assert_allclose(np.linalg.svd(fast.T @ reference, compute_uv=False), np.ones(5), atol=1e-10)


def test_gram_basis_is_orthonormal_and_spans_row_space():
    rng = np.random.default_rng(5)
    V = rng.normal(size=(12, 96))
    q = gram_orth_basis(V, 8)
    assert q.shape == (96, 8)
    np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-10)
    # Rank-8 truncation must beat any 8 columns of an arbitrary orthonormal basis.
    captured = np.sum((V @ q) ** 2)
    other = np.linalg.qr(rng.normal(size=(96, 8)))[0]
    assert captured > np.sum((V @ other) ** 2)


def test_capture_prefixes_are_nested_and_reach_full_rank_value():
    X, rows = emergency_fixture()
    train_mask = np.asarray([r["purpose_family_id"] != "fg0" for r in rows])
    test_mask = ~train_mask
    B, _, _, _, _, _ = factorial_effects(X, rows, train_mask)
    _, C_holdout, _, _, _, _ = factorial_effects(X, rows, test_mask)
    q_b = gram_orth_basis(B, RANK)
    fractions = [
        captured_energy(C_holdout, q_b[:, : min(prefix, q_b.shape[1])])[2] for prefix in (1, 4, 16, 64)
    ]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(fractions, fractions[1:]))
    assert fractions[-1] == pytest.approx(captured_energy(C_holdout, q_b)[2])


def test_layer_job_is_leakage_free_and_writes_regime_specific_scores(tmp_path):
    X, rows = emergency_fixture()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.save(shard_path(shard_dir, 0), np.ascontiguousarray(X, dtype=np.float32))

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
    layer, scores, sanity, elapsed = layer_job(0)

    assert layer == 0
    assert elapsed > 0
    assert scores.shape == (len(CONDITIONS), len(REGIMES), len(y))
    # float32 probe dtype: the identity residual is rounding, so gate it on scale.
    assert sanity["max_abs_activation"] == pytest.approx(float(np.max(np.abs(X.astype(np.float32)))))
    assert sanity["max_factorial_identity_error"] < 1e-6 * max(1.0, sanity["max_abs_activation"])
    assert sanity["held_family_basis_violations"] == 0
    assert sanity["max_qb_t_qcperp"] < 1e-10
    assert sanity["max_qb_t_rperp"] < 1e-10
    assert sanity["min_training_groups_per_fold"] > 0
    assert sorted(sanity["c_capture_by_rank"]) == sorted(["1", "4", "16", "64"])

    joint_i, b2c_i, c2b_i = (REGIMES.index(name) for name in ("joint", "b2c", "c2b"))
    for condition_i in range(len(CONDITIONS)):
        assert np.all(np.isfinite(scores[condition_i, joint_i]))
        # Cross-format regimes only score the opposite-format held-out rows.
        np.testing.assert_array_equal(np.isfinite(scores[condition_i, b2c_i]), fmt == "casual")
        np.testing.assert_array_equal(np.isfinite(scores[condition_i, c2b_i]), fmt == "benchmark")


def test_bootstrap_reports_specific_c_as_paired_difference(tmp_path):
    rng = np.random.default_rng(23)
    n_layers, n_blocks = 3, 40
    blocks = np.repeat([f"pb{i}" for i in range(n_blocks)], 4)
    y = np.tile([1, 1, 0, 0], n_blocks)
    fmt = np.tile(["benchmark", "casual", "benchmark", "casual"], n_blocks)
    n_items = len(y)
    scores = np.full((len(CONDITIONS), 1, len(REGIMES), n_layers, n_items), np.nan)
    # Overlapping distributions keep every AUC strictly inside (0.5, 1); only the
    # cross-format penalty varies by condition, so the gap ordering is the design.
    cross_penalty = {"raw": 0.9, "minus_b": 0.6, "minus_b_rperp": 0.5, "minus_b_cperp": 0.2}
    for condition_i, condition in enumerate(CONDITIONS):
        for layer in range(n_layers):
            base = rng.normal(size=n_items) + (y * 2 - 1)
            scores[condition_i, 0, REGIMES.index("joint"), layer] = base
            weak = base - cross_penalty[condition] * (y * 2 - 1)
            scores[condition_i, 0, REGIMES.index("b2c"), layer, fmt == "casual"] = weak[fmt == "casual"]
            scores[condition_i, 0, REGIMES.index("c2b"), layer, fmt == "benchmark"] = weak[
                fmt == "benchmark"
            ]

    scores_path = tmp_path / "emergency_cperp_scores_test.npz"
    np.savez_compressed(
        scores_path,
        schema_version=np.asarray(1),
        model=np.asarray("llama31_8b"),
        ranks=np.asarray([RANK], dtype=np.int16),
        conditions=np.asarray(CONDITIONS),
        regimes=np.asarray(REGIMES),
        n_layers=np.asarray(n_layers),
        item_id=np.asarray([f"item{i}" for i in range(n_items)]),
        payload_block_id=blocks,
        purpose_family_id=np.asarray(["fg1"] * n_items),
        fold_id=np.asarray(["fg1"] * n_items),
        format=fmt,
        y=y,
        scores=scores,
    )
    layer_checks = {
        str(layer): {
            "c_capture_by_rank": {
                str(prefix): {
                    "cross_fitted_fraction": 0.01 * prefix,
                    "nominal_isotropic_chance": prefix / 4096,
                }
                for prefix in (1, 4, 16, 64)
            }
        }
        for layer in range(n_layers)
    }
    scores_path.with_suffix(".sanity.json").write_text(
        json.dumps({"layer_checks": layer_checks}), encoding="utf-8"
    )

    result = analyze(scores_path, n_boot=200)
    decision = result["decision"]
    integrated = result["integrated_gap"]
    assert decision["extra_C"] == pytest.approx(integrated["minus_b"] - integrated["minus_b_cperp"])
    assert decision["extra_random"] == pytest.approx(integrated["minus_b"] - integrated["minus_b_rperp"])
    assert decision["specific_C"] == pytest.approx(decision["extra_C"] - decision["extra_random"])
    # Construction removes more gap with C_perp than with the random control.
    assert decision["specific_C"] > 0
    assert decision["specific_C_ci95"][0] < decision["specific_C"] < decision["specific_C_ci95"][1]
    assert result["c_capture"]["64"]["integrated_c_capture"] == pytest.approx(0.64)
