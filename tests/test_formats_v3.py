"""Acceptance and seeded-mutation tests for the v3 format-catalog lane.

Covers format_catalog_v2_plan.md sections 4-7 and the validator plan's gates:
lint teeth (F1/F2 seeded violations), renderer determinism, structural and
F4 fidelity checks, contract compatibility, reserved/anchor policies, the F3
surface battery, and the frozen-chain regression (pilot80 still validates
under the untouched v2 code path).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval_format_mvp.catalogs import load_catalogs
from eval_format_mvp.formats_v2 import (
    LABELED_OPTION_STYLES,
    FormatsV2,
    lint_formats_v2,
    load_formats_v2,
)
from eval_format_mvp.frames_v2 import load_frames_v2
from eval_format_mvp.io import MVPError, load_jsonl
from eval_format_mvp.render import _options
from eval_format_mvp.render_v2 import validate_rendered_v2
from eval_format_mvp.render_v3 import (
    ITEMS_PER_BLOCK_V3,
    build_dataset_plan_v3,
    render_dataset_v3,
    validate_rendered_v3,
)
from eval_format_mvp.surface_v3 import cue_excised_text, surface_probe_battery

ROOT = Path(__file__).resolve().parents[1]
FULL80 = ROOT / "data" / "payloads.grok43-medium_gpt54.full80.jsonl"
PILOT80_FROZEN = ROOT / "data" / "rendered_items_v2_pilot80_8-16-2026.jsonl"


@pytest.fixture(scope="module")
def frames():
    return load_frames_v2(ROOT)


@pytest.fixture(scope="module")
def catalog():
    return load_formats_v2(ROOT)


@pytest.fixture(scope="module")
def payloads():
    return load_jsonl(FULL80)


@pytest.fixture(scope="module")
def small(payloads):
    topics = {"t001", "t002", "t035"}
    return [p for p in payloads if p["topic_id"] in topics]


@pytest.fixture(scope="module")
def rendered(small, frames, catalog):
    return render_dataset_v3(small, frames, catalog)


def _mutate_template(catalog: FormatsV2, template_id: str, **changes) -> FormatsV2:
    templates = []
    for template in catalog.templates:
        row = copy.deepcopy(template)
        if str(row["template_id"]) == template_id:
            for key, value in changes.items():
                if key == "template":
                    row["template"] = value
                else:
                    row["dimensions"][key] = value
        templates.append(row)
    return FormatsV2(
        tuple(templates),
        catalog.contracts,
        catalog.f1_exceptions,
        catalog.banned_stem_manual,
        catalog.f2_generic_lines,
    )


# --- Lint teeth ------------------------------------------------------------


def test_catalog_lints_clean(catalog, frames) -> None:
    report = lint_formats_v2(catalog, frames)
    assert report["status"] == "passed"
    assert "bench_preamble_v2::with" in report["exceptions_used"]


def test_f1_catches_banned_stem(catalog, frames) -> None:
    mutated = _mutate_template(
        catalog,
        "deploy_bare_v2",
        template="{purpose}\n\nA reviewer will check this.\n{body}\n\n{contract}",
    )
    with pytest.raises(MVPError, match="F1 deploy_bare_v2"):
        lint_formats_v2(mutated, frames)


def test_f1_exception_does_not_leak_scope(catalog, frames) -> None:
    # "following" is excepted only for bench_preamble_v2.
    mutated = _mutate_template(
        catalog,
        "deploy_lead_in_v2",
        template="{purpose}\n\nHere's the following information:\n{body}\n\n{contract}",
    )
    with pytest.raises(MVPError, match="following"):
        lint_formats_v2(mutated, frames)


def test_f2_catches_duplicated_distinctive_line(catalog, frames) -> None:
    mutated = _mutate_template(
        catalog,
        "deploy_bare_v2",
        template="{purpose}\n\nHere's the information:\n{body}\n\n{contract}",
    )
    with pytest.raises(MVPError, match="F2 line"):
        lint_formats_v2(mutated, frames)


# --- Renderer + structural validator --------------------------------------


def test_block_composition_and_determinism(small, frames, catalog, rendered) -> None:
    by_block: dict[str, list[dict]] = {}
    for row in rendered:
        by_block.setdefault(row["payload_block_id"], []).append(row)
    assert len(by_block) == len(small)
    for block in by_block.values():
        assert len(block) == ITEMS_PER_BLOCK_V3
        strata = [r["format"] for r in block if not r["legacy"]]
        assert strata.count("benchmark") == 9 and strata.count("casual") == 9
        anchors = {r["template_id"] for r in block if r["legacy"]}
        assert len(anchors) == 1
    again = render_dataset_v3(small, frames, catalog)
    assert again == rendered


def test_validator_passes_and_report_shape(small, frames, catalog, rendered) -> None:
    report = validate_rendered_v3(rendered, small, frames, catalog)
    assert report["status"] == "passed"
    assert report["rendered_items"] == len(small) * ITEMS_PER_BLOCK_V3
    assert not any(t.endswith("reserved_v2") for t in report["items_per_template"])


def test_reconstruction_catches_mutation(small, frames, catalog, rendered) -> None:
    tampered = copy.deepcopy(rendered)
    tampered[7]["text"] += " "
    tampered[7]["rendered_text_sha256"] = "0" * 64
    with pytest.raises(MVPError):
        validate_rendered_v3(tampered, small, frames, catalog)


def test_f4_catches_wrong_option_style(small, frames, catalog, rendered) -> None:
    tampered = copy.deepcopy(rendered)
    victim = next(
        r for r in tampered if r["option_style"] == "letter_dot" and not r["legacy"]
    )
    victim["option_style"] = "bullet"
    with pytest.raises(MVPError, match="option style|deterministically"):
        validate_rendered_v3(tampered, small, frames, catalog)


def test_contract_compatibility(rendered, catalog) -> None:
    for row in rendered:
        if row["contract_id"] == "C2":
            assert row["option_style"] in LABELED_OPTION_STYLES
        if row["legacy"]:
            assert row["contract_id"] is None
        if row["template_id"] == "bench_qa_record_v2":
            assert row["contract_id"] != "C2"
            if row["template_mode"] == "gsm8k":
                assert row["option_style"] == "absent" and row["answer_key"] is None


def test_json_template_round_trips(rendered) -> None:
    for row in rendered:
        if row["template_id"] != "deploy_json_v2":
            continue
        payload_object = json.loads(row["text"].splitlines()[-1])
        assert isinstance(payload_object["input"], str)
        assert len(payload_object["choices"]) == 4
        assert row["correct_answer"] in payload_object["choices"]
        if row["contract_id"] == "C0":
            assert "response_instruction" not in payload_object


def test_anchor_rows_bridge_to_v1(small, rendered) -> None:
    payload_by_block = {p["payload_block_id"]: p for p in small}
    for row in rendered:
        if not row["legacy"]:
            continue
        assert row["contract_id"] is None
        if row["template_id"] == "benchmark_mcq_v1":
            options_block, key = _options(payload_by_block[row["payload_block_id"]])
            assert options_block in row["text"]
            assert row["answer_key"] == key
            assert "Respond with only the letter of the correct option." in row["text"]
        else:
            assert "Could you give me the answer in a short phrase?" in row["text"]
            assert row["answer_key"] is None


def test_answer_keys_uniformish_on_full80(payloads, frames, catalog) -> None:
    rows = render_dataset_v3(payloads, frames, catalog)
    plan = build_dataset_plan_v3(payloads, frames, catalog)
    assert sum(len(slots) for slots in plan.values()) == 7 * len(payloads)
    keys: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["answer_key"] is not None and not row["legacy"]:
            keys.setdefault(row["template_id"], {}).setdefault(row["answer_key"], 0)
            keys[row["template_id"]][row["answer_key"]] += 1
    for template_id, distribution in keys.items():
        labels = LABELED_OPTION_STYLES[
            str(catalog.template(template_id)["dimensions"]["option_style"])
        ]
        assert set(distribution) == set(labels), template_id


# --- Surface battery -------------------------------------------------------


def test_cue_excision_makes_twins_identical(rendered) -> None:
    by_key: dict[tuple, dict[str, str]] = {}
    for row in rendered:
        if row["render_slot"] != "stated":
            continue
        key = (row["payload_block_id"], row["template_id"])
        by_key.setdefault(key, {})[row["intended_purpose"]] = cue_excised_text(row)
    for pair in by_key.values():
        assert pair["evaluation"] == pair["deployment"]


def test_surface_battery_gates_pass(rendered) -> None:
    report = surface_probe_battery(rendered)
    assert report["gates_passed"]
    word_gate = report["expectations"]["purpose_held_template_word"]
    assert word_gate["role"] == "hard_gate" and word_gate["near_chance"]


# --- Frozen-chain regression (validator plan section 8 acceptance) ---------


@pytest.mark.skipif(not PILOT80_FROZEN.exists(), reason="frozen pilot80 not present")
def test_pilot80_still_validates_under_v2_code(frames) -> None:
    catalogs = load_catalogs(ROOT)
    payload_rows = load_jsonl(ROOT / "data" / "payloads.grok43-medium_gpt54.full80.jsonl")
    frozen_rows = load_jsonl(PILOT80_FROZEN)
    report = validate_rendered_v2(frozen_rows, payload_rows, frames, catalogs)
    assert report["status"] == "passed"


def test_full_crossing_schedule(small, frames, catalog) -> None:
    from eval_format_mvp.render_v3 import ITEMS_PER_BLOCK_FULL

    rows = render_dataset_v3(small, frames, catalog, schedule="full")
    by_block: dict[str, list[dict]] = {}
    for row in rows:
        by_block.setdefault(row["payload_block_id"], []).append(row)
    rendered_ids = {str(t["template_id"]) for t in catalog.rendered_templates}
    for block in by_block.values():
        assert len(block) == ITEMS_PER_BLOCK_FULL
        assert rendered_ids <= {r["template_id"] for r in block}
        assert all(r["frame_assignment_scheme"] == "crossed_formats_v3_full" for r in block)
    report = validate_rendered_v3(rows, small, frames, catalog)
    assert report["status"] == "passed"
    assert report["rendered_items"] == len(small) * ITEMS_PER_BLOCK_FULL
    # Every rendered template appears on every block: within-block pairing.
    for template_id in rendered_ids:
        assert report["items_per_template"][template_id] == len(small) * 3
    assert render_dataset_v3(small, frames, catalog, schedule="full") == rows
    # Rotation rows must NOT validate as full and vice versa (scheme-locked).
    rotation_rows = render_dataset_v3(small, frames, catalog, schedule="rotation")
    assert {r["frame_assignment_scheme"] for r in rotation_rows} == {"crossed_formats_v3"}
