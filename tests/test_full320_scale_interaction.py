from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.full320_scale_interaction import ENDPOINTS, load_run, point_effects


def _store(control: np.ndarray, mixed: np.ndarray, seeds: list[int]) -> dict[tuple[float, str, int | None], np.ndarray]:
    return {
        (0.5, "benchmark_only", None): control,
        (0.5, "casual_only", None): control,
        **{(0.5, "mixed", seed): mixed for seed in seeds},
    }


def test_point_effects_computes_positive_size_interaction() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    formats = np.asarray(["benchmark"] * 4 + ["casual"] * 4)
    seeds = [1, 2]
    imperfect = np.asarray([0.0, 2.0, 1.0, 3.0] * 2, dtype=np.float32)
    perfect = np.asarray([-2.0, -1.0, 1.0, 2.0] * 2, dtype=np.float32)
    summary = point_effects(
        {"8B": _store(imperfect, perfect, seeds), "70B": _store(perfect, imperfect, seeds)},
        labels,
        formats,
        seeds,
    )
    assert summary["by_size_effect"]["8B"][ENDPOINTS[0]] == pytest.approx(0.25)
    assert summary["by_size_effect"]["70B"][ENDPOINTS[0]] == pytest.approx(-0.25)
    assert summary["aggregate_interaction"][ENDPOINTS[0]] == pytest.approx(0.5)


def test_load_run_filters_requested_model_from_two_model_archive() -> None:
    result_path = Path(
        "artifacts/exploratory/full320_strict_equal_rows_v1_20260823_r2/strict_equal_rows_results.json"
    )
    result, scores = load_run(result_path, "meta-llama/Llama-3.1-70B-Instruct")
    assert result["analysis"] == "full320_strict_equal_rows_v1"
    assert len(scores) == 50
    assert {key[0] for key in scores} == {0.25, 0.375, 0.5, 0.625, 0.75}
