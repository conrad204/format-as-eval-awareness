from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.full320_payload_overlap_matched_n import (
    CONDITIONS,
    ENDPOINTS,
    _point_summary,
    build_paired_splits,
    normalized_layer_map,
)
from scripts.full320_strict_equal_rows import REGIME_CONTROLS, parse_rows


def test_normalized_layer_map_uses_depth_before_layer_count() -> None:
    assert normalized_layer_map(80, [0.25, 0.375, 0.5, 0.625, 0.75]) == {
        0.25: 20,
        0.375: 30,
        0.5: 40,
        0.625: 49,
        0.75: 59,
    }



def test_real_splits_exactly_match_rows_and_payload_exposure() -> None:
    config = json.loads(Path("configs/full320_payload_overlap_matched_n_v1.json").read_text(encoding="utf-8"))
    rows = parse_rows(Path(config["dataset"]["items_path"]), config["dataset"]["items_sha256"])
    splits, audit = build_paired_splits(rows, config)
    assert audit["n_splits_each_condition"] == 25
    assert audit["n_canonical_test_rows"] == 7040
    for index in range(25):
        heldout = splits["payload_heldout"][index]
        exposed = splits["payload_exposed"][index]
        assert heldout["test_indices"].tolist() == exposed["test_indices"].tolist()
        test_blocks = {str(rows[row_index]["payload_block_id"]) for row_index in heldout["test_indices"]}
        heldout_blocks = {pair[0][0] for pair in heldout["regime_pairs"][REGIME_CONTROLS[0]]}
        exposed_blocks = {pair[0][0] for pair in exposed["regime_pairs"][REGIME_CONTROLS[0]]}
        assert test_blocks.isdisjoint(heldout_blocks)
        assert test_blocks <= exposed_blocks
        heldout_lengths = {len(pairs) for pairs in heldout["regime_pairs"].values()}
        exposed_lengths = {len(pairs) for pairs in exposed["regime_pairs"].values()}
        assert len(heldout_lengths) == len(exposed_lengths) == 1
        assert heldout_lengths == exposed_lengths


def test_point_summary_computes_exposed_minus_heldout_gain() -> None:
    records = []
    for condition in CONDITIONS:
        controls = (0.50, 0.55)
        mixed = 0.60 if condition == "payload_heldout" else 0.68
        for regime, value in zip(REGIME_CONTROLS, controls):
            records.append(
                {
                    "model": "model",
                    "normalized_depth": 0.5,
                    "condition": condition,
                    "regime": regime,
                    "seed": None,
                    ENDPOINTS[0]: value,
                    ENDPOINTS[1]: value,
                }
            )
        records.append(
            {
                "model": "model",
                "normalized_depth": 0.5,
                "condition": condition,
                "regime": "mixed",
                "seed": 7,
                ENDPOINTS[0]: mixed,
                ENDPOINTS[1]: mixed,
            }
        )
    summary = _point_summary(records, [7])
    assert summary["condition_gains"]["payload_heldout"][ENDPOINTS[0]] == pytest.approx(0.05)
    assert summary["condition_gains"]["payload_exposed"][ENDPOINTS[0]] == pytest.approx(0.13)
    assert summary["aggregate"][ENDPOINTS[0]] == pytest.approx(0.08)
