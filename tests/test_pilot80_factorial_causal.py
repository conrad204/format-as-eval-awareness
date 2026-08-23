from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_pilot80_factorial_causal import response_values
from scripts.build_pilot80_factorial_causal_plan import (
    component_arrays,
    payload_folds,
    random_directions,
    validate_groups,
)
from scripts.run_pilot80_factorial_causal import (
    atomic_json,
    condition_specs,
    encode_rows,
    load_baseline_metrics,
    main,
    load_resume,
    output_hidden,
    replace_hidden,
)


def protocol() -> dict:
    return {
        "interventions": {
            "components": ["A", "B", "C"],
            "alphas": [0.5, 1.0],
            "signs": [-1, 1],
            "random_seeds": list(range(20261201, 20261233)),
        }
    }


def exact_rows(n_groups: int = 2) -> list[dict]:
    rows = []
    for group in range(n_groups):
        for purpose in ("evaluation", "deployment"):
            for format_name in ("benchmark", "casual"):
                rows.append({
                    "item_id": f"i{group}-{purpose}-{format_name}",
                    "payload_block_id": f"pb{group}",
                    "purpose_pair_id": f"vp{group}",
                    "purpose_family_id": "fg1",
                    "topic_id": "t1",
                    "render_slot": "stated",
                    "intended_purpose": purpose,
                    "format": format_name,
                    "text": "x",
                })
    return rows


def test_condition_bank_is_exact_and_sign_symmetric() -> None:
    specs = condition_specs(protocol())
    assert len(specs) == 85
    assert len({spec["condition_id"] for spec in specs}) == 85
    assert sum(spec["condition_type"] == "zero" for spec in specs) == 1
    assert sum(spec["condition_type"] == "component" for spec in specs) == 12
    assert sum(spec["condition_type"] == "random" for spec in specs) == 64
    assert sum(spec["condition_type"] == "cube" for spec in specs) == 8
    for component in "ABC":
        for alpha in (0.5, 1.0):
            signs = {spec["sign"] for spec in specs if spec["component"] == component and spec["alpha"] == alpha}
            assert signs == {-1, 1}


def test_effect_coded_component_formulas_and_scale() -> None:
    rows = exact_rows(2)
    A = np.array([2.0, 0.0, 0.0])
    B = np.array([0.0, 3.0, 0.0])
    C = np.array([0.0, 0.0, 4.0])
    X = []
    for group in range(2):
        mu = np.array([10.0 + group, -2.0, 1.0])
        for purpose in (1, -1):
            for format_sign in (1, -1):
                X.append(mu + purpose * A + format_sign * B + purpose * format_sign * C)
    fitted_A, fitted_B, fitted_C, scale, n_groups = component_arrays(np.asarray(X), rows, list(range(8)))
    np.testing.assert_allclose(fitted_A, A)
    np.testing.assert_allclose(fitted_B, B)
    np.testing.assert_allclose(fitted_C, C)
    assert scale == pytest.approx(2.0)
    assert n_groups == 2


def test_group_validation_fails_closed() -> None:
    rows = exact_rows(1)
    validate_groups(rows)
    with pytest.raises(ValueError, match="incomplete"):
        validate_groups(rows[:-1])
    inconsistent = exact_rows(1)
    inconsistent[0]["purpose_family_id"] = "fg2"
    with pytest.raises(ValueError, match="spans purpose_family_id"):
        validate_groups(inconsistent)


def test_chat_template_disables_special_token_readdition() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: list[dict], **kwargs: object) -> list[int]:
            assert messages == [{"role": "user", "content": "x"}]
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "add_special_tokens": False,
            }
            return [1, 2, 3]

    assert encode_rows(Tokenizer(), exact_rows(1), [0]) == {0: [1, 2, 3]}


def test_payload_folds_are_topic_stratified_and_deterministic() -> None:
    rows = []
    for topic in ("t1", "t2"):
        for block in range(10):
            row = exact_rows(1)[0]
            row = {**row, "payload_block_id": f"{topic}-pb{block}", "topic_id": topic, "item_id": f"{topic}-{block}"}
            rows.append(row)
    first = payload_folds(rows)
    second = payload_folds(list(reversed(rows)))
    assert first == second
    for topic in ("t1", "t2"):
        counts = np.bincount([fold for block, fold in first.items() if block.startswith(topic)], minlength=5)
        assert counts.tolist() == [2, 2, 2, 2, 2]


def test_random_controls_are_unit_and_component_orthogonal() -> None:
    rng = np.random.default_rng(9)
    components = rng.normal(size=(3, 64)).astype(np.float32)
    randoms = random_directions(*components, range(100, 132))
    assert randoms.shape == (32, 64)
    np.testing.assert_allclose(np.linalg.norm(randoms, axis=1), 1.0, atol=1e-6)
    q, _ = np.linalg.qr(components.T.astype(np.float64))
    assert np.max(np.abs(randoms @ q)) < 1e-6


def test_endpoint_pairing_uses_format_coded_interaction() -> None:
    plus = {
        "format": "benchmark",
        "readout_scores": {"purpose_benchmark": 3.0, "purpose_casual": 99.0, "format": 2.0},
    }
    minus = {
        "format": "benchmark",
        "readout_scores": {"purpose_benchmark": 1.0, "purpose_casual": -99.0, "format": -2.0},
    }
    assert response_values(plus, minus) == {
        "shared_purpose": 1.0, "purpose_interaction": 1.0, "format": 2.0,
    }
    plus["format"] = minus["format"] = "casual"
    plus["readout_scores"]["purpose_casual"] = 5.0
    minus["readout_scores"]["purpose_casual"] = 1.0
    assert response_values(plus, minus)["purpose_interaction"] == -2.0


def test_resume_rejects_duplicates_and_recovers_zero_metrics(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    record = {
        "model_id": "m", "source_layer": 5, "fold_id": "fg1_pf0", "row_index": 3,
        "condition_id": "zero", "source_baseline_max_abs": 0.1, "source_baseline_cosine": 1.0,
        "target_baseline_max_abs": 0.1, "target_baseline_cosine": 1.0,
    }
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_resume(raw) == {("m", 5, "fg1_pf0", 3, "zero")}
    assert load_baseline_metrics(raw)[("fg1_pf0", 5, 3)]["target_baseline_cosine"] == 1.0
    raw.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_resume(raw)


def test_atomic_json_writes_strict_complete_marker(tmp_path: Path) -> None:
    marker = tmp_path / "INVALID.json"
    atomic_json(marker, {"status": "invalid", "use_for_inference": False})
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "status": "invalid",
        "use_for_inference": False,
    }
    assert not marker.with_name("INVALID.json.tmp").exists()




def test_main_marks_preflight_failure_invalid(tmp_path: Path) -> None:
    out_dir = tmp_path / "failed"
    argv = [
        "--config", str(tmp_path / "missing-config.json"),
        "--plan-json", str(tmp_path / "missing-plan.json"),
        "--plan-npz", str(tmp_path / "missing-plan.npz"),
        "--model-id", "test/model",
        "--out-dir", str(out_dir),
    ]
    with pytest.raises(FileNotFoundError):
        main(argv)
    marker = json.loads((out_dir / "INVALID.json").read_text(encoding="utf-8"))
    assert marker["status"] == "invalid_incomplete_causal_run"
    assert marker["use_for_inference"] is False
    assert marker["raw_exists"] is False
    assert marker["partial_record_count"] == 0
def test_tuple_hook_output_replacement_preserves_auxiliary_values() -> None:
    hidden = np.ones((2, 3))
    output = (hidden, "cache")
    assert output_hidden(output) is hidden
    replacement = np.zeros((2, 3))
    changed = replace_hidden(output, replacement)
    assert changed[0] is replacement
    assert changed[1] == "cache"
