"""Fail-closed structural and explicit-label gate for confirmation frames."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BANNED = re.compile(r"\b(evaluation|evaluate|evaluative|deployment|deploy|benchmark|test|testing|exam|grading|assessment|assess|live[- ]use)\b", re.I)


def features(text: str) -> tuple:
    sentences = tuple(part.strip() for part in re.split(r"[.!?]+", text) if part.strip())
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    first = tuple(re.findall(r"[A-Za-z]+", sentence.lower())[:2] for sentence in sentences)
    punctuation = tuple(ch for ch in text if ch in ",;:!?-")
    return (len(sentences), len(words), first, punctuation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = json.loads(args.frames.read_text(encoding="utf-8"))
    if len(frames) != 50:
        raise SystemExit(f"expected 50 frames, got {len(frames)}")
    failures: list[str] = []
    signatures = []
    for index, frame in enumerate(frames):
        evaluation = str(frame.get("evaluation", ""))
        deployment = str(frame.get("deployment", ""))
        if not evaluation or not deployment:
            failures.append(f"empty pair at {index}")
        if BANNED.search(evaluation) or BANNED.search(deployment):
            failures.append(f"explicit purpose/format label at {index}")
        if evaluation.strip().casefold() == deployment.strip().casefold():
            failures.append(f"identical pair at {index}")
        signatures.append((features(evaluation), features(deployment)))
    fold_reports = {}
    for fold_index in range(5):
        start = fold_index * 10
        fold_signatures = signatures[start : start + 10]
        unique = len(set(fold_signatures))
        cue_loci = Counter(str(frame.get("cue_locus")) for frame in frames[start : start + 10])
        perspectives = Counter(str(frame.get("cue_perspective")) for frame in frames[start : start + 10])
        fold_reports[f"fg{fold_index + 1}"] = {
            "unique_pair_structures": unique,
            "cue_loci": dict(cue_loci),
            "perspectives": dict(perspectives),
            "pass": unique >= 4,
        }
        if unique < 4:
            failures.append(f"fold fg{fold_index + 1} has only {unique} unique pair structures")
    report = {
        "schema_version": 1,
        "frames": str(args.frames),
        "banned_label_regex": BANNED.pattern,
        "folds": fold_reports,
        "duplicate_pair_structures": len(signatures) - len(set(signatures)),
        "failures": failures,
        "pass": not failures,
    }
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "failures": len(failures), "out": str(args.out)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
