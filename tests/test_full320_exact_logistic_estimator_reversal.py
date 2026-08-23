from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.full320_exact_logistic_estimator_reversal import (
    ENDPOINTS,
    fit_logistic_oof,
    load_lda_scores,
    point_summary,
)


def test_fit_logistic_oof_separates_synthetic_pairs() -> None:
    data = np.asarray([[-2.0], [2.0], [-1.0], [1.0], [-0.5], [0.5]], dtype=np.float32)
    pair_a = (("b1", "p1", "benchmark", "t1"), 0, 1)
    pair_b = (("b2", "p2", "benchmark", "t2"), 2, 3)
    split = {
        "held_family": "fg-test",
        "test_indices": np.asarray([4, 5], dtype=np.int32),
        "regime_pairs": {
            "benchmark_only": [pair_a, pair_b],
            "casual_only": [pair_a, pair_b],
            "mixed_seed_7": [pair_a, pair_b],
        },
    }
    probe = {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "max_iter": 3000, "tol": 1e-4, "random_state": 0}
    outputs, diagnostics = fit_logistic_oof(data, [split], [7], probe)
    assert outputs["mixed_seed_7"][5] > outputs["mixed_seed_7"][4]
    assert np.isfinite(outputs["mixed_seed_7"][4:]).all()
    assert len(diagnostics) == 3
    assert all(row["converged"] for row in diagnostics)


def test_point_summary_computes_logistic_minus_lda_interaction() -> None:
    records = []
    for estimator, mixed in (("logistic", 0.70), ("diagonal_lda", 0.52)):
        for regime, value in (("benchmark_only", 0.55), ("casual_only", 0.50)):
            records.append({"model": "m", "estimator": estimator, "regime": regime, "seed": None, ENDPOINTS[0]: value, ENDPOINTS[1]: value})
        records.append({"model": "m", "estimator": estimator, "regime": "mixed", "seed": 7, ENDPOINTS[0]: mixed, ENDPOINTS[1]: mixed})
    summary = point_summary(records, [7])
    assert summary["aggregate"]["logistic"][ENDPOINTS[0]] == pytest.approx(0.15)
    assert summary["aggregate"]["diagonal_lda"][ENDPOINTS[0]] == pytest.approx(-0.03)
    assert summary["aggregate"]["interaction"][ENDPOINTS[0]] == pytest.approx(0.18)


def test_load_lda_scores_filters_model_and_depth() -> None:
    config = json.loads(Path("configs/full320_exact_logistic_estimator_reversal_v1.json").read_text())
    model_config = config["models"][1]
    path = Path(model_config["lda_result_path"])
    source_result = json.loads(path.read_text())
    canonical = np.asarray(source_result["scores"]["canonical_row_indices"], dtype=np.int32)
    seeds = config["probe"]["assignment_seeds"]
    outputs, source = load_lda_scores(path, model_config, 0.375, 11520, canonical, seeds)
    assert len(outputs) == 10
    assert source["source_run_id"] == "full320_strict_equal_rows_v1_20260823_r2"
    assert all(int(np.isfinite(scores).sum()) == 7040 for scores in outputs.values())
    assert source["source_analysis"] == "full320_strict_equal_rows_v1"
    assert source["source_split_audit_sha256"]


def test_load_lda_scores_rejects_canonical_permutation() -> None:
    config = json.loads(Path("configs/full320_exact_logistic_estimator_reversal_v1.json").read_text())
    model_config = config["models"][1]
    path = Path(model_config["lda_result_path"])
    source_result = json.loads(path.read_text())
    canonical = np.asarray(source_result["scores"]["canonical_row_indices"], dtype=np.int32)
    with pytest.raises(ValueError, match="canonical row mismatch"):
        load_lda_scores(
            path,
            model_config,
            0.375,
            11520,
            canonical[::-1],
            config["probe"]["assignment_seeds"],
        )
