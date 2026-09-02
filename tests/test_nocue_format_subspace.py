import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "full320_v3_nocue_format_subspace.py"
SPEC = importlib.util.spec_from_file_location("nocue_subspace", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_nocue_contrast_uses_within_payload_format_centroids():
    rows = [
        {"render_slot": "no_cue", "payload_block_id": "p1", "format": "benchmark"},
        {"render_slot": "no_cue", "payload_block_id": "p1", "format": "benchmark"},
        {"render_slot": "no_cue", "payload_block_id": "p1", "format": "casual"},
        {"render_slot": "no_cue", "payload_block_id": "p2", "format": "benchmark"},
    ]
    X = np.array([[3.0, 1.0], [5.0, 3.0], [1.0, 2.0], [99.0, 99.0]])
    contrasts = MOD.nocue_payload_contrasts(X, rows)
    assert set(contrasts) == {"p1"}
    np.testing.assert_allclose(contrasts["p1"], [3.0, 0.0])


def test_training_group_keys_exclude_held_fold_and_nocue_rows():
    rows = [
        {"render_slot": "stated", "payload_block_id": "p1", "purpose_pair_id": "v1"},
        {"render_slot": "stated", "payload_block_id": "p1", "purpose_pair_id": "v1"},
        {"render_slot": "stated", "payload_block_id": "p1", "purpose_pair_id": "v2"},
        {"render_slot": "no_cue", "payload_block_id": "p1", "purpose_pair_id": "none"},
    ]
    keys = MOD.training_group_keys(rows, np.array([True, True, False, True]))
    assert keys == [("p1", "v1")]
