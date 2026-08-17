"""Acceptance tests for the pre-scaling housekeeping bundle.

Covers: versioned frame assignment (stable under insertion/deletion, legacy
scheme reproduces the frozen dataset), cue-span annotations, enforced manifest
hashes, near-duplicate detection, and quota-refill request scaffolding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_format_mvp.catalogs import load_catalogs
from eval_format_mvp.dedup import (
    DEFAULT_NEAR_DUP_THRESHOLD,
    near_duplicate_clusters,
    near_duplicate_pairs,
)
from eval_format_mvp.generation import generate_validated_payloads
from eval_format_mvp.io import MVPError, load_jsonl
from eval_format_mvp.payloads import build_refill_requests
from eval_format_mvp.render import (
    render_dataset,
    verify_render_manifest,
    write_rendered_dataset,
)
from eval_format_mvp.validate import validate_rendered_dataset

from test_generation import GENERATOR, VALIDATOR, _judgment, _payload, _result


ROOT = Path(__file__).resolve().parents[1]
AUDITED = ROOT / "data" / "payloads.grok43-medium_gpt54.full.audited.jsonl"
PRE_AUDIT = ROOT / "data" / "payloads.grok43-medium_gpt54.full.jsonl"
FROZEN_ITEMS = ROOT / "data" / "rendered_items_v1_8-14-2026.jsonl"


def _pair_by_payload(rows: list[dict]) -> dict[str, str]:
    return {row["payload_block_id"]: row["purpose_pair_id"] for row in rows}


def test_stable_hash_assignment_survives_insertion_and_deletion() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(ROOT / "data" / "example_payloads.jsonl")
    full = _pair_by_payload(render_dataset(payloads, catalogs))
    without_first = _pair_by_payload(render_dataset(payloads[1:], catalogs))
    removed = payloads[0]["payload_block_id"]
    assert without_first == {k: v for k, v in full.items() if k != removed}

    # The legacy scheme is exactly the landmine the stable scheme removes:
    # deleting one payload reshuffles the frames of the payloads after it.
    legacy_full = _pair_by_payload(
        render_dataset(payloads, catalogs, frame_assignment="index_mod_v1")
    )
    legacy_without = _pair_by_payload(
        render_dataset(payloads[1:], catalogs, frame_assignment="index_mod_v1")
    )
    assert legacy_without != {k: v for k, v in legacy_full.items() if k != removed}

    # Annotated stable-hash rows pass schema and reproduction validation.
    report = validate_rendered_dataset(
        render_dataset(payloads, catalogs), payloads, catalogs
    )
    assert report["status"] == "passed"
    assert report["frame_assignment"] == "stable_hash_v1"


def test_legacy_scheme_reproduces_frozen_rendered_items() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(AUDITED)
    frozen = load_jsonl(FROZEN_ITEMS)
    report = validate_rendered_dataset(frozen, payloads, catalogs)
    assert report["status"] == "passed"
    assert report["frame_assignment"] == "index_mod_v1 (legacy rows)"
    assert report["rendered_items"] == 288


def test_cue_span_annotations_locate_the_cue() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(ROOT / "data" / "example_payloads.jsonl")
    pairs = {row["pair_id"]: row for row in catalogs.purpose_pairs}
    rows = render_dataset(payloads, catalogs)
    assert rows
    for row in rows:
        cue = row["text"][row["cue_char_start"]:row["cue_char_end"]]
        assert cue == row["cue_text"]
        expected = str(
            pairs[row["purpose_pair_id"]]["en"][row["intended_purpose"]]
        ).strip()
        assert cue == expected
        assert row["text"].count(cue) == 1
        assert row["cue_position"] == "start"
        assert row["message_role"] == "user"


def test_enforced_manifest_hash_mismatch_aborts(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)
    payload_path = ROOT / "data" / "example_payloads.jsonl"
    payloads = load_jsonl(payload_path)
    out = tmp_path / "rendered.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = write_rendered_dataset(
        payload_path=payload_path,
        output_path=out,
        manifest_path=manifest_path,
        payloads=payloads,
        catalogs=catalogs,
    )
    assert manifest["hash_policy"] == "enforced"
    assert manifest["frame_assignment"] == "stable_hash_v1"
    assert verify_render_manifest(manifest_path)["status"] == "passed"

    out.write_text(out.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(MVPError, match="hash verification failed"):
        verify_render_manifest(manifest_path)
    report = verify_render_manifest(manifest_path, strict=False)
    assert report["status"] == "hash_mismatch"
    assert [row["artifact"] for row in report["mismatches"]] == ["output"]


def test_near_duplicate_clustering_flags_t035_paraphrases() -> None:
    payloads = load_jsonl(AUDITED)
    clusters = near_duplicate_clusters(payloads)
    t035 = [group for group in clusters if group[0].startswith("pb_t035")]
    assert t035 and set(t035[0]) == {
        "pb_t035_01",
        "pb_t035_02",
        "pb_t035_03",
        "pb_t035_04",
        "pb_t035_05",
    }
    # The default threshold never links payloads across topics.
    for pair in near_duplicate_pairs(payloads):
        assert pair["a"].split("_")[1] == pair["b"].split("_")[1]


def test_refill_requests_fill_documented_shortfall() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(AUDITED)
    reserved = {str(row["payload_block_id"]) for row in load_jsonl(PRE_AUDIT)}
    report, requests = build_refill_requests(
        payloads, catalogs, per_topic=5, reserved_ids=reserved
    )
    shortfall = {
        topic: entry["missing"] for topic, entry in report["topics_below_quota"].items()
    }
    assert shortfall == {"t001": 2, "t002": 2, "t004": 1, "t018": 1, "t032": 2}
    assert report["total_missing"] == 8
    assert len(requests) == 8
    new_ids = {str(row["payload_block_id"]) for row in requests}
    assert len(new_ids) == 8
    existing = {str(row["payload_block_id"]) for row in payloads}
    assert not new_ids & (existing | reserved)
    for row in requests:
        assert row["payload_block_id"] in row["request_id"]
        assert row["topic_id"] in row["user"]


def test_generation_rejects_near_duplicates_without_validator_spend(
    tmp_path: Path,
) -> None:
    catalogs = load_catalogs(ROOT)
    calls: list[str] = []

    def fake_caller(profile, **kwargs):
        calls.append(profile["role"])
        if profile["role"] == "generator":
            return _result(_payload(), profile, len(calls))
        return _result(_judgment(passed=True), profile, len(calls))

    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=tmp_path / "payloads.jsonl",
        run_dir=tmp_path / "run",
        per_topic=1,
        max_generation_attempts=2,
        max_semantic_attempts=2,
        limit=2,
        caller=fake_caller,
    )
    # Slot 1 is accepted; every candidate for slot 2 is the same payload, so it
    # is rejected as a near-duplicate before any validator call is spent.
    assert report["accepted_payloads"] == 1
    assert report["status"] == "completed_with_exhausted"
    assert report["exhaustion_reasons"] == {
        report["exhausted_payloads"][0]: "generation_budget"
    }
    assert calls.count("validator") == 1
    attempts = load_jsonl(tmp_path / "run" / "attempts.jsonl")
    statuses = [row["status"] for row in attempts]
    assert statuses == [
        "accepted",
        "near_duplicate_rejection",
        "near_duplicate_rejection",
    ]
    duplicate = attempts[1]["near_duplicate"]
    assert duplicate["similarity"] == 1.0
    assert duplicate["payload_block_id"] == attempts[0]["payload_block_id"]
    assert "near-duplicate" in attempts[1]["error"].lower()
