"""Deterministically render every payload into the complete purpose x format block."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .catalogs import Catalogs
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
from .schema import validate_against_schema


PURPOSES = ("evaluation", "deployment")
FORMATS = ("benchmark", "casual")

# Frame-assignment schemes. "index_mod_v1" is the original sorted-index modulo
# assignment: it reproduces the frozen 288-item dataset, but adding or removing
# one payload silently reshuffles the frame of every later payload.
# "stable_hash_v1" keys the choice on payload_block_id alone, so each payload
# keeps its frame no matter what else is in the file. New renders default to
# the stable scheme; the legacy scheme exists only to reproduce frozen data.
FRAME_ASSIGNMENT_SCHEMES = ("index_mod_v1", "stable_hash_v1")
DEFAULT_FRAME_ASSIGNMENT = "stable_hash_v1"

HASH_POLICIES = ("enforced", "observational_not_enforced")
DEFAULT_HASH_POLICY = "enforced"


def _options(payload: dict[str, Any]) -> tuple[str, str]:
    values = [str(payload["correct_answer"]), *map(str, payload["distractors"])]
    seed = int(sha256_text(str(payload["payload_block_id"]))[:16], 16)
    random.Random(seed).shuffle(values)
    letters = "ABCD"
    answer_index = values.index(str(payload["correct_answer"]))
    rendered = "\n".join(f"{letter}. {value}" for letter, value in zip(letters, values))
    return rendered, letters[answer_index]


def assign_purpose_pair(
    payload_block_id: str,
    payload_index: int,
    purpose_pairs: list[dict[str, Any]],
    scheme: str,
) -> dict[str, Any]:
    if scheme == "index_mod_v1":
        return purpose_pairs[payload_index % len(purpose_pairs)]
    if scheme == "stable_hash_v1":
        digest = int(sha256_text(str(payload_block_id))[:16], 16)
        return purpose_pairs[digest % len(purpose_pairs)]
    raise MVPError(
        f"Unknown frame-assignment scheme {scheme!r}; expected one of "
        f"{FRAME_ASSIGNMENT_SCHEMES}"
    )


def _cue_annotations(text: str, purpose_text: str, item_id: str) -> dict[str, Any]:
    occurrences = text.count(purpose_text)
    if occurrences != 1:
        raise MVPError(
            f"Cue sentence must appear exactly once in {item_id}, found {occurrences}"
        )
    start = text.index(purpose_text)
    return {
        "cue_text": purpose_text,
        "cue_char_start": start,
        "cue_char_end": start + len(purpose_text),
        "cue_position": "start" if start == 0 else "body",
        "message_role": "user",
    }


def render_payload(
    payload: dict[str, Any],
    purpose_pair: dict[str, Any],
    catalogs: Catalogs,
    *,
    frame_assignment: str = DEFAULT_FRAME_ASSIGNMENT,
    annotate: bool = True,
) -> list[dict[str, Any]]:
    payload_hash = sha256_object(payload)
    body = f"{str(payload['context']).strip()}\n\n{str(payload['question']).strip()}"
    options, answer_key = _options(payload)
    rows: list[dict[str, Any]] = []
    for purpose in PURPOSES:
        purpose_text = str(purpose_pair["en"][purpose]).strip()
        for format_name in FORMATS:
            format_row = catalogs.formats[format_name]
            text = str(format_row["template"]).format(
                purpose=purpose_text,
                body=body,
                options=options,
            ).strip()
            item_id = (
                f"item::{payload['payload_block_id']}::{purpose_pair['pair_id']}::"
                f"{'ev' if purpose == 'evaluation' else 'dp'}::"
                f"{'bm' if format_name == 'benchmark' else 'cs'}"
            )
            row = {
                "schema_version": 1,
                "item_id": item_id,
                "payload_block_id": payload["payload_block_id"],
                "topic_id": payload["topic_id"],
                "task_family": payload["task_family"],
                "intended_purpose": purpose,
                "label": 1 if purpose == "evaluation" else 0,
                "format": format_name,
                "purpose_pair_id": purpose_pair["pair_id"],
                "purpose_family_id": purpose_pair["family_id"],
                "cue_locus": purpose_pair["cue_locus"],
                "format_variant_id": format_row["format_variant_id"],
                "format_family_id": format_row["family_id"],
                "text": text,
                "correct_answer": payload["correct_answer"],
                "answer_key": answer_key if format_name == "benchmark" else None,
                "canonical_payload_sha256": payload_hash,
                "rendered_text_sha256": sha256_text(text),
                "split_group_id": payload["payload_block_id"],
            }
            if annotate:
                row["frame_assignment_scheme"] = frame_assignment
                row.update(_cue_annotations(text, purpose_text, item_id))
            validate_against_schema(row, "rendered_item.schema.json", label=item_id)
            rows.append(row)
    return rows


def render_dataset(
    payloads: list[dict[str, Any]],
    catalogs: Catalogs,
    *,
    frame_assignment: str = DEFAULT_FRAME_ASSIGNMENT,
    annotate: bool = True,
) -> list[dict[str, Any]]:
    purpose_pairs = sorted(catalogs.purpose_pairs, key=lambda row: str(row["pair_id"]))
    rows: list[dict[str, Any]] = []
    for index, payload in enumerate(sorted(payloads, key=lambda row: row["payload_block_id"])):
        pair = assign_purpose_pair(
            str(payload["payload_block_id"]), index, purpose_pairs, frame_assignment
        )
        rows.extend(
            render_payload(
                payload,
                pair,
                catalogs,
                frame_assignment=frame_assignment,
                annotate=annotate,
            )
        )
    return rows


def write_rendered_dataset(
    *,
    payload_path: Path,
    output_path: Path,
    manifest_path: Path,
    payloads: list[dict[str, Any]],
    catalogs: Catalogs,
    frame_assignment: str = DEFAULT_FRAME_ASSIGNMENT,
    hash_policy: str = DEFAULT_HASH_POLICY,
) -> dict[str, Any]:
    if hash_policy not in HASH_POLICIES:
        raise MVPError(f"Unknown hash policy {hash_policy!r}; expected one of {HASH_POLICIES}")
    rows = render_dataset(payloads, catalogs, frame_assignment=frame_assignment)
    write_jsonl(output_path, rows)
    counts = {
        "payload_blocks": len(payloads),
        "rendered_items": len(rows),
        "evaluation": sum(row["intended_purpose"] == "evaluation" for row in rows),
        "deployment": sum(row["intended_purpose"] == "deployment" for row in rows),
        "benchmark": sum(row["format"] == "benchmark" for row in rows),
        "casual": sum(row["format"] == "casual" for row in rows),
    }
    manifest = {
        "schema_version": 1,
        "status": "development_unlocked",
        "hash_policy": hash_policy,
        "frame_assignment": frame_assignment,
        "inputs": {
            "payloads": {"path": str(payload_path), "sha256": sha256_file(payload_path)},
            "topics": sha256_file(PROJECT_ROOT / "catalogs" / "topics.yaml"),
            "purpose_frames": sha256_file(
                PROJECT_ROOT / "catalogs" / "purpose_frames.yaml"
            ),
            "formats": sha256_file(PROJECT_ROOT / "catalogs" / "formats.yaml"),
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "counts": counts,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_render_manifest(
    manifest_path: Path, *, strict: bool | None = None
) -> dict[str, Any]:
    """Recompute every hash recorded in a render manifest.

    Under the "enforced" hash policy (or strict=True) any mismatch or missing
    artifact raises MVPError; the legacy observational policy only reports.
    """
    manifest = load_json(manifest_path)
    policy = str(manifest.get("hash_policy", "observational_not_enforced"))
    enforce = policy == "enforced" if strict is None else strict
    mismatches: list[dict[str, str]] = []
    checked: list[str] = []

    def check(artifact: str, path: Path, expected: str) -> None:
        checked.append(artifact)
        if not path.exists():
            mismatches.append(
                {"artifact": artifact, "path": str(path), "error": "missing file"}
            )
            return
        actual = sha256_file(path)
        if actual != str(expected):
            mismatches.append(
                {
                    "artifact": artifact,
                    "path": str(path),
                    "expected_sha256": str(expected),
                    "actual_sha256": actual,
                }
            )

    inputs = manifest["inputs"]
    check("payloads", Path(inputs["payloads"]["path"]), inputs["payloads"]["sha256"])
    for name in ("topics", "purpose_frames", "formats"):
        check(name, PROJECT_ROOT / "catalogs" / f"{name}.yaml", inputs[name])
    check("output", Path(manifest["output"]["path"]), manifest["output"]["sha256"])

    report = {
        "manifest": str(manifest_path),
        "hash_policy": policy,
        "enforced": enforce,
        "checked": checked,
        "mismatches": mismatches,
        "status": "passed" if not mismatches else "hash_mismatch",
    }
    if mismatches and enforce:
        details = "; ".join(
            f"{row['artifact']} ({row.get('error', 'sha256 mismatch')})"
            for row in mismatches
        )
        raise MVPError(f"Render manifest hash verification failed: {details}")
    return report
