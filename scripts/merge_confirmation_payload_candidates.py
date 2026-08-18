"""Merge a failed payload batch replacement without editing candidate content."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.generate_confirmation_candidates import PAYLOAD_TOPIC_SCHEDULE
except ImportError:
    from generate_confirmation_candidates import PAYLOAD_TOPIC_SCHEDULE


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("replacement", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    original = load(args.original)
    replacement = load(args.replacement)
    if len(original) != 72 or len(replacement) != 8:
        raise ValueError(f"expected 72 original and 8 replacement payloads, got {len(original)} and {len(replacement)}")
    # The original generator log records batch 7 as the only rejected batch;
    # its accepted list is batches 0-6 followed by batches 8-9.
    merged = original[:56] + replacement + original[56:]
    topics = [str(row.get("topic_id", "")) for row in merged]
    if topics != list(PAYLOAD_TOPIC_SCHEDULE):
        raise ValueError("merged payload topics do not match the frozen 16-topic x 5-block schedule")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(merged), "out": str(args.out)}))


if __name__ == "__main__":
    main()
