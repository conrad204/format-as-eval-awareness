from __future__ import annotations

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
        "test_indices": np.asarray([4, 5], dtype=np.int32),
        "regime_pairs": {
            "benchmark_only": [pair_a, pair_b],
            "casual_only": [pair_a, pair_b],
            "mixed_seed_7": [pair_a, pair_b],
        },
    }
    probe = {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "max_iter": 3000, "tol": 1e-4, "random_state": 0}
    outputs = fit_logistic_oof(data, [split], [7], probe)
    assert outputs["mixed_seed_7"][5] > outputs["mixed_seed_7"][4]
    assert np.isfinite(outputs["mixed_seed_7"][4:]).all()


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
    path = Path("artifacts/exploratory/full320_strict_equal_rows_v1_20260823_r2/strict_equal_rows_results.json")
    outputs, source = load_lda_scores(path, "meta-llama/Llama-3.1-70B-Instruct", 0.375, 11520)
    assert len(outputs) == 10
    assert source["source_run_id"] == "full320_strict_equal_rows_v1_20260823_r2"
    assert all(int(np.isfinite(scores).sum()) == 7040 for scores in outputs.values())
