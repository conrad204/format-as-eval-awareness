"""Report surface-statistic balance for generated purpose frame pairs."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def stats(text: str) -> dict[str, float]:
    words = re.findall(r"(?u)\b\w+\b", text.lower())
    return {
        "chars": len(text),
        "words": len(words),
        "punctuation": sum(not char.isalnum() and not char.isspace() for char in text),
        "sentences": len(re.findall(r"[.!?](?:\s|$)", text)),
        "past_markers": sum(word.endswith("ed") for word in words),
        "present_markers": sum(word in {"is", "are", "use", "uses", "helps", "enters"} for word in words),
        "future_markers": sum(word in {"will", "shall", "may"} for word in words),
        "passive_markers": sum(word in {"was", "were", "been", "becomes", "used"} for word in words),
        "active_markers": sum(word in {"decide", "decides", "helps", "serve", "serves", "answer", "answers"} for word in words),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    frames = json.loads(args.frames.read_text(encoding="utf-8"))
    rows = []
    aggregate = Counter()
    for frame in frames:
        evaluation = stats(str(frame["evaluation"]))
        deployment = stats(str(frame["deployment"]))
        differences = {key: abs(evaluation[key] - deployment[key]) for key in evaluation}
        rows.append({"fold_id": frame.get("fold_id"), "pair_id": frame.get("pair_id"), "evaluation": evaluation, "deployment": deployment, "absolute_difference": differences})
        for key, value in differences.items():
            aggregate[key] += value
    report = {
        "schema_version": 1,
        "frames": len(frames),
        "pair_reports": rows,
        "mean_absolute_difference": {key: value / max(1, len(rows)) for key, value in aggregate.items()},
        "required_balance_dimensions": ["chars", "words", "punctuation", "sentences", "past_markers", "present_markers", "future_markers", "passive_markers", "active_markers"],
        "status": "diagnostic; thresholds must be frozen before final bank generation",
    }
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(frames), "out": str(args.out)}))


if __name__ == "__main__":
    main()
