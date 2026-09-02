"""Crossed frames-by-payloads rendering (v2, proposedchanges.md items 5, 6, 8).

The v1 renderer married each payload block to exactly one purpose pair, so
family folds held out frames and payloads simultaneously. Here every payload
renders under three of the five vocabulary folds (a topic-local balanced
incomplete block design), with cue locus rotated across its three stated
slots, plus one indirect render, one second-person render, and one no-cue
control render:

  per block: 3 stated x 4 items + indirect x 4 + second-person x 4 + no-cue x 2
           = 22 items.

Assignment is deterministic and topic-local: blocks are ordered inside their
topic by sha256(payload_block_id), fold triples step by 3 (mod 5) from a
topic-hash offset, which guarantees every topic reaches every fold with as
few as three blocks (union of {s, s+1, s+2} over starts s, s+3, s+6 covers all
residues mod 5). Editing one topic's payloads never reassigns any other topic.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .catalogs import Catalogs
from .frames_v2 import FOLD_IDS, FramesV2, lint_frames_v2
from .io import (
    MVPError,
    PROJECT_ROOT,
    load_json,
    sha256_file,
    sha256_object,
    sha256_text,
    write_json,
    write_jsonl,
)
from .render import PURPOSES, FORMATS, _cue_annotations, _options
from .schema import validate_against_schema


SCHEME_V2 = "crossed_bibd_v2"
STATED_LOCI = ("both", "noun_only", "verb_only")
ITEMS_PER_BLOCK = 22


def _digest(text: str) -> int:
    return int(sha256_text(text)[:16], 16)


def assign_block_folds(payloads: list[dict[str, Any]]) -> dict[str, tuple[str, str, str]]:
    """Topic-local BIBD: each block gets three of the five folds."""
    by_topic: dict[str, list[str]] = {}
    for payload in payloads:
        by_topic.setdefault(str(payload["topic_id"]), []).append(
            str(payload["payload_block_id"])
        )
    assignment: dict[str, tuple[str, str, str]] = {}
    for topic_id, block_ids in by_topic.items():
        offset = _digest(topic_id) % len(FOLD_IDS)
        for index, block_id in enumerate(sorted(block_ids, key=sha256_text)):
            start = (offset + 3 * index) % len(FOLD_IDS)
            assignment[block_id] = tuple(
                FOLD_IDS[(start + k) % len(FOLD_IDS)] for k in range(3)
            )
    return assignment


def _pick(pairs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ordered = sorted(pairs, key=lambda row: str(row["pair_id"]))
    if not ordered:
        raise MVPError(f"No candidate pairs for {key}")
    return ordered[_digest(key) % len(ordered)]


def build_block_plan(
    block_id: str, folds: tuple[str, str, str], frames: FramesV2
) -> dict[str, Any]:
    """Deterministic slot plan for one payload block."""
    by_fold = frames.by_fold
    locus_offset = _digest(f"{block_id}::locus") % 3
    stated: list[dict[str, Any]] = []
    for slot, fold_id in enumerate(folds):
        locus = STATED_LOCI[(slot + locus_offset) % 3]
        candidates = [
            pair
            for pair in by_fold[fold_id]
            if pair["cue_style"] == "stated"
            and pair["cue_perspective"] == "third"
            and pair["cue_locus"] == locus
        ]
        stated.append(_pick(candidates, f"{block_id}::{fold_id}::{locus}"))
    indirect_fold = folds[_digest(f"{block_id}::indirect") % 3]
    indirect = _pick(
        [p for p in by_fold[indirect_fold] if p["cue_style"] == "indirect"],
        f"{block_id}::indirect_pair",
    )
    second_fold = folds[_digest(f"{block_id}::second") % 3]
    second = _pick(
        [p for p in by_fold[second_fold] if p["cue_perspective"] == "second"],
        f"{block_id}::second_pair",
    )
    return {"stated": stated, "indirect": indirect, "second_person": second}


def _base_row(
    payload: dict[str, Any], payload_hash: str, format_row: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "payload_block_id": payload["payload_block_id"],
        "topic_id": payload["topic_id"],
        "task_family": payload["task_family"],
        "format": format_row["format"],
        "format_variant_id": format_row["format_variant_id"],
        "format_family_id": format_row["family_id"],
        "correct_answer": payload["correct_answer"],
        "canonical_payload_sha256": payload_hash,
        "split_group_id": payload["payload_block_id"],
        "frame_assignment_scheme": SCHEME_V2,
    }


def render_payload_v2(
    payload: dict[str, Any],
    folds: tuple[str, str, str],
    frames: FramesV2,
    catalogs: Catalogs,
) -> list[dict[str, Any]]:
    plan = build_block_plan(str(payload["payload_block_id"]), folds, frames)
    payload_hash = sha256_object(payload)
    body = f"{str(payload['context']).strip()}\n\n{str(payload['question']).strip()}"
    options, answer_key = _options(payload)
    rows: list[dict[str, Any]] = []

    cue_renders = [("stated", pair) for pair in plan["stated"]]
    cue_renders.append(("indirect", plan["indirect"]))
    cue_renders.append(("second_person", plan["second_person"]))
    for slot_name, pair in cue_renders:
        for purpose in PURPOSES:
            purpose_text = str(pair["en"][purpose]).strip()
            for format_name in FORMATS:
                format_row = catalogs.formats[format_name]
                text = str(format_row["template"]).format(
                    purpose=purpose_text, body=body, options=options
                ).strip()
                item_id = (
                    f"item::{payload['payload_block_id']}::{pair['pair_id']}::"
                    f"{'ev' if purpose == 'evaluation' else 'dp'}::"
                    f"{'bm' if format_name == 'benchmark' else 'cs'}"
                )
                row = _base_row(payload, payload_hash, format_row)
                row.update(
                    {
                        "item_id": item_id,
                        "intended_purpose": purpose,
                        "label": 1 if purpose == "evaluation" else 0,
                        "render_slot": slot_name,
                        "purpose_pair_id": pair["pair_id"],
                        "purpose_family_id": pair["fold_id"],
                        "cue_style": pair["cue_style"],
                        "cue_perspective": pair["cue_perspective"],
                        "cue_locus": pair["cue_locus"],
                        "text": text,
                        "answer_key": answer_key if format_name == "benchmark" else None,
                        "rendered_text_sha256": sha256_text(text),
                    }
                )
                row.update(_cue_annotations(text, purpose_text, item_id))
                validate_against_schema(row, "rendered_item_v2.schema.json", label=item_id)
                rows.append(row)

    # No-cue control (item 5): identical payload and format, purpose line empty.
    for format_name in FORMATS:
        format_row = catalogs.formats[format_name]
        text = str(format_row["template"]).format(
            purpose="", body=body, options=options
        ).strip()
        item_id = (
            f"item::{payload['payload_block_id']}::none::nc::"
            f"{'bm' if format_name == 'benchmark' else 'cs'}"
        )
        row = _base_row(payload, payload_hash, format_row)
        row.update(
            {
                "item_id": item_id,
                "intended_purpose": "none",
                "label": None,
                "render_slot": "no_cue",
                "purpose_pair_id": "none",
                "purpose_family_id": "none",
                "cue_style": "none",
                "cue_perspective": "none",
                "cue_locus": "none",
                "text": text,
                "answer_key": answer_key if format_name == "benchmark" else None,
                "rendered_text_sha256": sha256_text(text),
            }
        )
        validate_against_schema(row, "rendered_item_v2.schema.json", label=item_id)
        rows.append(row)
    return rows


def render_dataset_v2(
    payloads: list[dict[str, Any]], frames: FramesV2, catalogs: Catalogs
) -> list[dict[str, Any]]:
    assignment = assign_block_folds(payloads)
    rows: list[dict[str, Any]] = []
    for payload in sorted(payloads, key=lambda row: str(row["payload_block_id"])):
        rows.extend(
            render_payload_v2(
                payload, assignment[str(payload["payload_block_id"])], frames, catalogs
            )
        )
    return rows


def validate_rendered_v2(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    frames: FramesV2,
    catalogs: Catalogs,
) -> dict[str, Any]:
    if not rows:
        raise MVPError("Rendered v2 dataset is empty")
    item_ids: set[str] = set()
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        validate_against_schema(
            row, "rendered_item_v2.schema.json", label=str(row.get("item_id"))
        )
        item_id = str(row["item_id"])
        if item_id in item_ids:
            raise MVPError(f"Duplicate rendered item_id {item_id}")
        item_ids.add(item_id)
        by_block.setdefault(str(row["payload_block_id"]), []).append(row)

    payload_ids = {str(row["payload_block_id"]) for row in payloads}
    if set(by_block) != payload_ids:
        raise MVPError("Rendered v2 blocks differ from canonical payloads")
    for block_id, block in by_block.items():
        if len(block) != ITEMS_PER_BLOCK:
            raise MVPError(f"Block {block_id} has {len(block)} items, expected {ITEMS_PER_BLOCK}")
        stated = [r for r in block if r["render_slot"] == "stated"]
        stated_folds = {r["purpose_family_id"] for r in stated}
        stated_loci = {r["cue_locus"] for r in stated}
        if len(stated_folds) != 3 or stated_loci != set(STATED_LOCI):
            raise MVPError(
                f"Block {block_id} stated slots must span 3 folds and all loci "
                f"(got folds={sorted(stated_folds)}, loci={sorted(stated_loci)})"
            )
        no_cue = [r for r in block if r["render_slot"] == "no_cue"]
        if len(no_cue) != 2 or {r["format"] for r in no_cue} != set(FORMATS):
            raise MVPError(f"Block {block_id} must have exactly one no-cue item per format")
        for slot_name in ("stated", "indirect", "second_person"):
            slot_rows = [r for r in block if r["render_slot"] == slot_name]
            for pair_id in {r["purpose_pair_id"] for r in slot_rows}:
                cells = {
                    (r["intended_purpose"], r["format"])
                    for r in slot_rows
                    if r["purpose_pair_id"] == pair_id
                }
                if len(cells) != 4:
                    raise MVPError(
                        f"Block {block_id} pair {pair_id} is not a complete 2x2"
                    )

    expected = render_dataset_v2(payloads, frames, catalogs)
    expected_by_id = {str(r["item_id"]): r for r in expected}
    actual_by_id = {str(r["item_id"]): r for r in rows}
    if actual_by_id != expected_by_id:
        missing = sorted(set(expected_by_id) - set(actual_by_id))[:3]
        extra = sorted(set(actual_by_id) - set(expected_by_id))[:3]
        changed = sorted(
            i for i in set(actual_by_id) & set(expected_by_id)
            if actual_by_id[i] != expected_by_id[i]
        )[:3]
        raise MVPError(
            "Rendered v2 rows do not reproduce deterministically: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    # Coverage (item 8 acceptance): every topic and task family reaches every
    # fold through its stated renders.
    topics = {str(p["topic_id"]): str(p["task_family"]) for p in payloads}
    stated_rows = [r for r in rows if r["render_slot"] == "stated"]
    topic_fold = {
        (r["topic_id"], r["purpose_family_id"]) for r in stated_rows
    }
    missing_cells = [
        (topic, fold)
        for topic in sorted(topics)
        for fold in FOLD_IDS
        if (topic, fold) not in topic_fold
    ]
    if missing_cells:
        raise MVPError(f"Topic x fold coverage holes: {missing_cells}")

    def _fold_counts(slot: str) -> dict[str, int]:
        counted = Counter(
            r["purpose_family_id"]
            for r in rows
            if r["render_slot"] == slot and r["intended_purpose"] == "evaluation"
            and r["format"] == "benchmark"
        )
        return {fold: counted.get(fold, 0) for fold in FOLD_IDS}

    locus_by_fold = Counter(
        (r["purpose_family_id"], r["cue_locus"])
        for r in stated_rows
        if r["intended_purpose"] == "evaluation" and r["format"] == "benchmark"
    )
    return {
        "status": "passed",
        "payload_blocks": len(by_block),
        "rendered_items": len(rows),
        "items_per_slot": dict(sorted(Counter(r["render_slot"] for r in rows).items())),
        "stated_blocks_per_fold": _fold_counts("stated"),
        "indirect_blocks_per_fold": _fold_counts("indirect"),
        "second_person_blocks_per_fold": _fold_counts("second_person"),
        "stated_fold_x_locus": {
            f"{fold}::{locus}": locus_by_fold.get((fold, locus), 0)
            for fold in FOLD_IDS
            for locus in STATED_LOCI
        },
        "task_family_coverage": "complete",
    }


def write_rendered_dataset_v2(
    *,
    payload_path: Path,
    output_path: Path,
    manifest_path: Path,
    payloads: list[dict[str, Any]],
    frames: FramesV2,
    catalogs: Catalogs,
) -> dict[str, Any]:
    lint = lint_frames_v2(frames)
    rows = render_dataset_v2(payloads, frames, catalogs)
    validation = validate_rendered_v2(rows, payloads, frames, catalogs)
    write_jsonl(output_path, rows)
    manifest = {
        "schema_version": 2,
        "manifest_kind": "render_v2",
        "status": "development_unlocked",
        "hash_policy": "enforced",
        "frame_assignment": SCHEME_V2,
        "inputs": {
            "payloads": {"path": str(payload_path), "sha256": sha256_file(payload_path)},
            "purpose_frames_v2": {
                "path": str(PROJECT_ROOT / "catalogs" / "purpose_frames_v2.yaml"),
                "sha256": sha256_file(PROJECT_ROOT / "catalogs" / "purpose_frames_v2.yaml"),
            },
            "formats": {
                "path": str(PROJECT_ROOT / "catalogs" / "formats.yaml"),
                "sha256": sha256_file(PROJECT_ROOT / "catalogs" / "formats.yaml"),
            },
            "topics": {
                "path": str(PROJECT_ROOT / "catalogs" / "topics.yaml"),
                "sha256": sha256_file(PROJECT_ROOT / "catalogs" / "topics.yaml"),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "frames_lint": lint,
        "validation": validation,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_render_manifest_v2(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("manifest_kind") != "render_v2":
        raise MVPError(f"{manifest_path} is not a render_v2 manifest")
    mismatches: list[dict[str, str]] = []
    checked: list[str] = []
    entries = dict(manifest["inputs"])
    entries["output"] = manifest["output"]
    for name, entry in entries.items():
        checked.append(name)
        path = Path(entry["path"])
        if not path.exists():
            mismatches.append({"artifact": name, "path": str(path), "error": "missing file"})
            continue
        actual = sha256_file(path)
        if actual != str(entry["sha256"]):
            mismatches.append(
                {
                    "artifact": name,
                    "path": str(path),
                    "expected_sha256": str(entry["sha256"]),
                    "actual_sha256": actual,
                }
            )
    if mismatches:
        details = "; ".join(
            f"{row['artifact']} ({row.get('error', 'sha256 mismatch')})" for row in mismatches
        )
        raise MVPError(f"Render v2 manifest hash verification failed: {details}")
    return {
        "manifest": str(manifest_path),
        "hash_policy": "enforced",
        "checked": checked,
        "status": "passed",
    }
