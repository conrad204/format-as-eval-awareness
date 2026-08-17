"""Acceptance tests for the v2 crossed, vocabulary-disjoint frame redesign.

Covers proposedchanges.md items 1+8 (disjoint fold lexicons, frames crossed
with payloads), 3 (locus crossed with fold and payload), 5 (no-cue controls),
and 6 (second-person class kept separate).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from eval_format_mvp.catalogs import load_catalogs
from eval_format_mvp.frames_v2 import (
    FOLD_IDS,
    FramesV2,
    lint_frames_v2,
    load_frames_v2,
)
from eval_format_mvp.io import MVPError, load_jsonl
from eval_format_mvp.render_v2 import (
    ITEMS_PER_BLOCK,
    assign_block_folds,
    render_dataset_v2,
    validate_rendered_v2,
    verify_render_manifest_v2,
    write_rendered_dataset_v2,
)

ROOT = Path(__file__).resolve().parents[1]
AUDITED = ROOT / "tests" / "data" / "payloads.grok43-medium_gpt54.full.audited.jsonl"


@pytest.fixture(scope="module")
def frames() -> FramesV2:
    return load_frames_v2(ROOT)


@pytest.fixture(scope="module")
def payloads() -> list[dict]:
    return load_jsonl(AUDITED)


@pytest.fixture(scope="module")
def small(payloads) -> list[dict]:
    topics = {"t001", "t002", "t035"}
    return [p for p in payloads if p["topic_id"] in topics]


def test_catalog_lints_clean(frames) -> None:
    report = lint_frames_v2(frames)
    assert report["status"] == "passed"
    assert report["pairs"] == 50
    assert report["warnings"] == []
    declared = report["declared_stems_per_fold"]
    all_stems = [s for fold in FOLD_IDS for s in declared[fold]]
    assert len(all_stems) == len(set(all_stems))


def test_lint_rejects_cross_fold_vocabulary_reuse(frames) -> None:
    pairs = copy.deepcopy([dict(p) for p in frames.pairs])
    for pair in pairs:
        # Give fg2 an fg1-owned cue word ("reviewer") both in text and books.
        if pair["pair_id"] == "vp13":
            pair["en"]["evaluation"] = "A reviewer will read this reply."
            pair["cue_tokens"]["evaluation"] = ["reviewer"]
    with pytest.raises(MVPError, match="R2"):
        lint_frames_v2(FramesV2(tuple(pairs), frames.folds))


def test_lint_rejects_undeclared_asymmetry(frames) -> None:
    pairs = copy.deepcopy([dict(p) for p in frames.pairs])
    for pair in pairs:
        if pair["pair_id"] == "vp13":
            # Sneak an undeclared eval-side word into the sentence.
            pair["en"]["evaluation"] = "A careful grader will read this reply."
    with pytest.raises(MVPError, match="R1"):
        lint_frames_v2(FramesV2(tuple(pairs), frames.folds))


def test_ngram_holdout_is_near_chance_on_catalog_and_far_from_it_on_v1(frames) -> None:
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from eval_format_mvp.frames_v2 import ngram_holdout_check

    assert ngram_holdout_check(frames)["near_chance"]

    # The v1 catalog fails the same functional test: its held-family cue
    # sentences are lexically separable. This is the regression the v2
    # redesign exists to fix, so keep the contrast pinned.
    catalogs = load_catalogs(ROOT)
    v1_meta = [
        (str(pair["en"][purpose]), str(pair["family_id"]), 1 if purpose == "evaluation" else 0)
        for pair in catalogs.purpose_pairs
        for purpose in ("evaluation", "deployment")
    ]
    v1 = ngram_holdout_check(frames, texts_and_meta=v1_meta)
    assert v1["auc"] > 0.65


def test_fold_assignment_covers_every_topic_and_is_topic_local(payloads) -> None:
    assignment = assign_block_folds(payloads)
    by_topic: dict[str, set[str]] = {}
    for payload in payloads:
        folds = assignment[payload["payload_block_id"]]
        assert len(set(folds)) == 3
        by_topic.setdefault(payload["topic_id"], set()).update(folds)
    for topic, folds in by_topic.items():
        assert folds == set(FOLD_IDS), f"{topic} misses folds {set(FOLD_IDS) - folds}"

    # Deleting a block reassigns at most its own topic; every other topic's
    # blocks keep their folds (the v2 analogue of stable_hash_v1).
    removed = payloads[0]
    rest = [p for p in payloads if p is not removed]
    after = assign_block_folds(rest)
    for payload in rest:
        if payload["topic_id"] != removed["topic_id"]:
            assert after[payload["payload_block_id"]] == assignment[payload["payload_block_id"]]


def test_render_shape_and_determinism(small, frames) -> None:
    catalogs = load_catalogs(ROOT)
    rows = render_dataset_v2(small, frames, catalogs)
    assert len(rows) == ITEMS_PER_BLOCK * len(small)
    report = validate_rendered_v2(rows, small, frames, catalogs)
    assert report["status"] == "passed"
    assert report["items_per_slot"]["no_cue"] == 2 * len(small)
    assert report["items_per_slot"]["stated"] == 12 * len(small)


def test_no_cue_rows_are_unlabeled_and_unannotated(small, frames) -> None:
    catalogs = load_catalogs(ROOT)
    rows = render_dataset_v2(small, frames, catalogs)
    no_cue = [r for r in rows if r["render_slot"] == "no_cue"]
    assert no_cue
    for row in no_cue:
        assert row["label"] is None
        assert row["intended_purpose"] == "none"
        assert row["purpose_pair_id"] == "none"
        assert "cue_text" not in row and "cue_char_start" not in row
        # The payload survives intact with the purpose line simply absent.
        assert not row["text"].startswith("\n")
        first_line = row["text"].partition("\n")[0]
        assert first_line in ("Question:", "I'm trying to figure something out:")


def test_cue_rows_carry_spans_and_second_person_is_separable(small, frames) -> None:
    catalogs = load_catalogs(ROOT)
    rows = render_dataset_v2(small, frames, catalogs)
    cue_rows = [r for r in rows if r["render_slot"] != "no_cue"]
    for row in cue_rows:
        assert row["text"][row["cue_char_start"]:row["cue_char_end"]] == row["cue_text"]
    second = {r["purpose_pair_id"] for r in rows if r["render_slot"] == "second_person"}
    third = {r["purpose_pair_id"] for r in rows if r["render_slot"] != "second_person"}
    assert second and not second & third


def test_manifest_roundtrip_and_tamper_detection(small, frames, tmp_path) -> None:
    catalogs = load_catalogs(ROOT)
    out = tmp_path / "rendered_v2.jsonl"
    manifest_path = tmp_path / "manifest_v2.json"
    manifest = write_rendered_dataset_v2(
        payload_path=AUDITED,
        output_path=out,
        manifest_path=manifest_path,
        payloads=small,
        frames=frames,
        catalogs=catalogs,
    )
    assert manifest["hash_policy"] == "enforced"
    assert verify_render_manifest_v2(manifest_path)["status"] == "passed"
    out.write_text(out.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(MVPError, match="hash verification failed"):
        verify_render_manifest_v2(manifest_path)
