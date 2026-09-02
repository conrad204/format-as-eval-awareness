#!/usr/bin/env python3
"""Select a prospectively fixed, payload-disjoint full320 transfer bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

DOMAIN = "full320_orthc_transfer_bank_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(full_items: Path, pilot_items: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite immutable transfer-bank output")
    full_rows = load_jsonl(full_items)
    pilot_rows = load_jsonl(pilot_items)
    if len(full_rows) != 11520 or len({row["payload_block_id"] for row in full_rows}) != 320:
        raise ValueError("full320 source inventory mismatch")
    pilot_blocks = {row["payload_block_id"] for row in pilot_rows}
    full_blocks = {row["payload_block_id"] for row in full_rows}
    if len(pilot_blocks) != 80 or not pilot_blocks < full_blocks:
        raise ValueError("pilot80 payload blocks are not a strict 80-block subset of full320")

    representative: dict[str, dict[str, Any]] = {}
    for row in full_rows:
        representative.setdefault(row["payload_block_id"], row)
    remaining = full_blocks - pilot_blocks
    by_topic: dict[str, list[str]] = defaultdict(list)
    for block in remaining:
        by_topic[representative[block]["topic_id"]].append(block)
    if len(by_topic) != 16 or any(len(blocks) != 15 for blocks in by_topic.values()):
        raise ValueError("expected 16 topics with 15 pilot-disjoint candidate blocks each")

    selected: list[str] = []
    for topic, blocks in sorted(by_topic.items()):
        ordered = sorted(blocks, key=lambda block: stable_hash(DOMAIN, topic, block))
        selected.extend(ordered[:5])
    selected_set = set(selected)
    rows = [row for row in full_rows if row["payload_block_id"] in selected_set]
    stated = [row for row in rows if row["render_slot"] == "stated"]
    no_cue = [row for row in rows if row["render_slot"] == "no_cue"]
    families = sorted({row["purpose_family_id"] for row in stated})
    cells = Counter((row["intended_purpose"], row["format"]) for row in stated)
    if (
        len(selected_set) != 80
        or selected_set & pilot_blocks
        or len(rows) != 2880
        or len(stated) != 1920
        or len(no_cue) != 960
        or families != ["fg1", "fg2", "fg3", "fg4", "fg5"]
        or set(cells) != {
            ("evaluation", "benchmark"), ("evaluation", "casual"),
            ("deployment", "benchmark"), ("deployment", "casual"),
        }
        or len({row["item_id"] for row in rows}) != len(rows)
    ):
        raise ValueError("selected transfer-bank structure mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "prospectively_selected_uninspected_causal_transfer_bank",
        "selection_domain": DOMAIN,
        "selection_rule": "Exclude all pilot80 payload_block_id values; within each of 16 topics, SHA256-rank the 15 remaining blocks and take the first five.",
        "full_items_path": full_items.as_posix(),
        "full_items_sha256": sha256_file(full_items),
        "pilot_items_path": pilot_items.as_posix(),
        "pilot_items_sha256": sha256_file(pilot_items),
        "output_path": output.as_posix(),
        "output_sha256": sha256_file(output),
        "selected_payload_blocks": selected,
        "counts": {
            "rows": len(rows),
            "stated_rows": len(stated),
            "no_cue_rows": len(no_cue),
            "payload_blocks": len(selected_set),
            "topics": len(by_topic),
            "purpose_families": len(families),
            "stated_cells": {f"{purpose}__{fmt}": count for (purpose, fmt), count in sorted(cells.items())},
        },
        "known_prior_boundary": "The full320 bank has prior observational analyses. No causal intervention outcome on these selected payload-disjoint rows has been generated or inspected at freeze time.",
        "allowed_if_positive": "A pilot80-trained orthogonalized-C intervention transfers its format-conditioned internal readout response to a prospectively selected, pilot80-payload-disjoint full320 subset in both fixed models.",
        "prohibited": [
            "evaluation awareness", "behavioral steering", "independent item-bank confirmation",
            "universal model mechanism", "causal scaling law", "post-outcome block selection",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--pilot-items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.full_items, args.pilot_items, args.out, args.manifest)
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
