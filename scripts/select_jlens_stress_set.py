"""Select a balanced development-only J-lens stress set from pilot rows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--topics", type=int, default=5)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_block: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_block[row["payload_block_id"]].append(row)
    topic_blocks: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for block_id, block_rows in by_block.items():
        topics = {row["topic_id"] for row in block_rows}
        if len(topics) != 1:
            raise ValueError(f"block {block_id} has multiple topics")
        topic_blocks[next(iter(topics))][block_rows[0]["purpose_family_id"]] = block_rows

    complete_topics = [
        topic for topic, folds in sorted(topic_blocks.items())
        if len(folds) == 5 and all(len(block) == 22 for block in folds.values())
    ]
    selected_topics = complete_topics[: args.topics]
    if len(selected_topics) != args.topics:
        raise ValueError(f"only {len(selected_topics)} complete topics available")
    selected = []
    for topic in selected_topics:
        for fold in ("fg1", "fg2", "fg3", "fg4", "fg5"):
            selected.extend(topic_blocks[topic][fold])
    if len(selected) != args.topics * 5 * 22:
        raise AssertionError("unexpected stress-set size")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"rows": len(selected), "topics": selected_topics, "blocks": args.topics * 5, "folds": 5}))


if __name__ == "__main__":
    main()
