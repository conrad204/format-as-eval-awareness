"""Assemble model-generated candidates into the frozen 80-block bank shape.

This is an automatic development step. It rejects malformed or imbalanced
pools; it never edits individual examples to rescue a failed gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
try:
    from scripts import build_confirmation_bank as shape
    from scripts.generate_confirmation_candidates import PAYLOAD_TOPIC_SCHEDULE
except ImportError:
    import build_confirmation_bank as shape
    from generate_confirmation_candidates import PAYLOAD_TOPIC_SCHEDULE

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
EXPECTED = {
    ("noun_only", "third"): 2,
    ("verb_only", "third"): 2,
    ("both", "third"): 2,
    ("context", "third"): 2,
    ("context", "second"): 2,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list: {path}")
    return value


def normalize_frames(candidates: list[dict]) -> list[dict]:
    if len(candidates) != 50:
        raise ValueError(f"expected exactly 50 generated frame pairs, got {len(candidates)}")
    frames = []
    cursor = 0
    for fold in FOLDS:
        selected = candidates[cursor : cursor + 10]
        cursor += 10
        counts = {}
        for candidate in selected:
            locus = str(candidate.get("cue_locus"))
            perspective = str(candidate.get("cue_perspective"))
            key = ("context" if perspective in {"indirect", "second"} else locus, "second" if perspective == "second" else "third")
            counts[key] = counts.get(key, 0) + 1
        if counts != EXPECTED:
            raise ValueError(f"frame pool fold {fold} is not balanced: {counts}")
        ordinal = {key: 0 for key in EXPECTED}
        for candidate in selected:
            locus = str(candidate["cue_locus"])
            perspective = str(candidate["cue_perspective"])
            kind = "context" if perspective in {"indirect", "second"} else locus
            key = (kind, "second" if perspective == "second" else "third")
            ordinal[key] += 1
            if kind == "context" and perspective == "indirect":
                pair_id = f"{fold}_indirect_{ordinal[key]}"
                cue_locus = "context"
                cue_perspective = "third"
            elif kind == "context":
                pair_id = f"{fold}_second_{ordinal[key]}"
                cue_locus = "context"
                cue_perspective = "second"
            else:
                pair_id = f"{fold}_stated_{kind}_{ordinal[key]}"
                cue_locus = kind
                cue_perspective = "third"
            eval_words = shape.tokens(str(candidate["evaluation"]))
            deploy_words = shape.tokens(str(candidate["deployment"]))
            eval_only = sorted(eval_words - deploy_words)
            deploy_only = sorted(deploy_words - eval_words)
            if not eval_only and not deploy_only:
                eval_only = sorted(eval_words)
                deploy_only = sorted(deploy_words)
            frames.append({
                "fold_id": fold,
                "pair_id": pair_id,
                "cue_locus": cue_locus,
                "cue_perspective": cue_perspective,
                "evaluation": str(candidate["evaluation"]).strip(),
                "deployment": str(candidate["deployment"]).strip(),
                "evaluation_cue_tokens": eval_only,
                "deployment_cue_tokens": deploy_only,
                "semantic_rationale": str(candidate.get("semantic_rationale", "")),
                "lineage": "stockmark-2-100b-local-generation-candidate",
            })
    return frames


def normalize_payloads(candidates: list[dict]) -> list[dict]:
    if len(candidates) != 80:
        raise ValueError(f"expected exactly 80 generated payloads, got {len(candidates)}")
    raw_topics = [str(candidate.get("topic_id", "")).strip() for candidate in candidates]
    expected_topics = list(PAYLOAD_TOPIC_SCHEDULE)
    if raw_topics != expected_topics:
        raise ValueError("payload topic IDs/order do not match the frozen 16-topic x 5-block schedule")
    topic_counts: dict[str, int] = {}
    for topic in raw_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    if len(topic_counts) != 16 or set(topic_counts.values()) != {5}:
        raise ValueError(f"payload topic coverage must be 16 topics x 5 blocks: {topic_counts}")
    topic_map = {topic: f"ct_{index + 1:02d}" for index, topic in enumerate(topic_counts)}
    payloads = []
    for index, candidate in enumerate(candidates):
        required = {"context", "question", "correct_answer", "distractors", "task_family"}
        missing = sorted(required - set(candidate))
        if missing or not isinstance(candidate.get("distractors"), list) or len(candidate["distractors"]) != 3:
            raise ValueError(f"payload {index} malformed: {missing}")
        payload = dict(candidate)
        payload["payload_block_id"] = f"cb_{index + 1:03d}"
        payload["topic_id"] = topic_map[raw_topics[index]]
        payload["distractors"] = [str(value) for value in payload["distractors"]]
        payload["correct_answer"] = str(payload["correct_answer"])
        if len(set(payload["distractors"] + [payload["correct_answer"]])) != 4:
            raise ValueError(f"payload {index} has non-unique answer options")
        payloads.append(payload)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.out_dir}")
    payloads = normalize_payloads(load(args.payloads))
    frames = normalize_frames(load(args.frames))
    rows = shape.render_bank(payloads, frames)
    forbidden_data = json.loads(args.forbidden.read_text(encoding="utf-8"))
    forbidden = {str(row["stem"]) for row in forbidden_data["records"]}
    counts = shape.validate(payloads, frames, rows, forbidden)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "payloads.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in payloads), encoding="utf-8")
    (args.out_dir / "frames.json").write_text(json.dumps(frames, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "rendered_items.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "development_model_generated_candidate",
        "generator_lineage": "stockmark/Stockmark-2-100B-Instruct; local BF16; revision pinned by generation log",
        "validator_status": "pending",
        "counts": counts,
        "artifacts": {},
    }
    for name in ("payloads.jsonl", "frames.json", "rendered_items.jsonl"):
        path = args.out_dir / name
        manifest["artifacts"][name] = {"path": name, "sha256": sha256_file(path)}
    (args.out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), **counts}))


if __name__ == "__main__":
    main()
