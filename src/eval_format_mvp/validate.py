"""Structural checks for canonical payloads and complete rendered blocks."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .catalogs import Catalogs
from .io import MVPError
from .render import FRAME_ASSIGNMENT_SCHEMES, render_dataset
from .schema import validate_against_schema


EXPECTED_CELLS = {
    ("evaluation", "benchmark"),
    ("evaluation", "casual"),
    ("deployment", "benchmark"),
    ("deployment", "casual"),
}


def validate_rendered_dataset(
    rows: list[dict[str, Any]], payloads: list[dict[str, Any]], catalogs: Catalogs
) -> dict[str, Any]:
    if not rows:
        raise MVPError("Rendered dataset is empty")
    item_ids: set[str] = set()
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        validate_against_schema(row, "rendered_item.schema.json", label=str(row.get("item_id")))
        item_id = str(row["item_id"])
        if item_id in item_ids:
            raise MVPError(f"Duplicate rendered item_id {item_id}")
        item_ids.add(item_id)
        by_block.setdefault(str(row["payload_block_id"]), []).append(row)

    payload_ids = {str(row["payload_block_id"]) for row in payloads}
    if set(by_block) != payload_ids:
        raise MVPError(
            f"Rendered payload blocks differ from canonical payloads: "
            f"rendered={sorted(by_block)}, payloads={sorted(payload_ids)}"
        )
    for block_id, block in by_block.items():
        cells = {(row["intended_purpose"], row["format"]) for row in block}
        if len(block) != 4 or cells != EXPECTED_CELLS:
            raise MVPError(f"Block {block_id} is not a complete purpose x format 2x2")
        if len({row["canonical_payload_sha256"] for row in block}) != 1:
            raise MVPError(f"Block {block_id} does not share one canonical payload hash")
        if len({row["purpose_pair_id"] for row in block}) != 1:
            raise MVPError(f"Block {block_id} uses multiple purpose pairs")

    # Legacy files (rendered before frame-assignment versioning) carry no
    # scheme field and no cue annotations; reproduce them exactly as written.
    schemes = {row.get("frame_assignment_scheme") for row in rows}
    if len(schemes) != 1:
        raise MVPError(f"Rendered rows mix frame-assignment schemes: {sorted(map(str, schemes))}")
    scheme = schemes.pop()
    if scheme is None:
        expected = render_dataset(
            payloads, catalogs, frame_assignment="index_mod_v1", annotate=False
        )
    elif scheme in FRAME_ASSIGNMENT_SCHEMES:
        expected = render_dataset(payloads, catalogs, frame_assignment=scheme)
    else:
        raise MVPError(f"Unknown frame-assignment scheme {scheme!r} in rendered rows")
    expected_by_id = {str(row["item_id"]): row for row in expected}
    actual_by_id = {str(row["item_id"]): row for row in rows}
    if actual_by_id != expected_by_id:
        missing = sorted(set(expected_by_id) - set(actual_by_id))
        extra = sorted(set(actual_by_id) - set(expected_by_id))
        changed = sorted(
            item_id
            for item_id in set(actual_by_id) & set(expected_by_id)
            if actual_by_id[item_id] != expected_by_id[item_id]
        )
        raise MVPError(
            "Rendered rows do not reproduce deterministically from payloads/catalogs: "
            f"missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]}"
        )

    return {
        "status": "passed",
        "frame_assignment": scheme if scheme is not None else "index_mod_v1 (legacy rows)",
        "payload_blocks": len(by_block),
        "rendered_items": len(rows),
        "cells": dict(
            sorted(
                Counter(
                    f"{row['intended_purpose']}::{row['format']}" for row in rows
                ).items()
            )
        ),
        "purpose_families": dict(
            sorted(Counter(row["purpose_family_id"] for row in rows).items())
        ),
    }

