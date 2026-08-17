"""Seal a validated confirmation bank exactly once.

The command is intentionally fail-closed: it requires immutable candidate
artifacts and pre-extraction gates, then creates a single exclusive manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = (
    "payloads.jsonl",
    "frames.json",
    "rendered_items.jsonl",
    "payloads_generation_log.jsonl",
    "frames_generation_log.jsonl",
    "payloads_semantic_validation.jsonl",
    "frames_semantic_validation.jsonl",
    "lexical_gate.json",
    "subword_gate.json",
    "red_team_lexical_report.json",
    "surface_balance.json",
    "structural_variation.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_validation_pass(path: Path) -> bool:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return bool(records) and all(bool(record.get("pass")) for record in records)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--lexical-gate-script", type=Path, required=True)
    parser.add_argument("--subword-gate-script", type=Path, required=True)
    parser.add_argument("--structural-gate-script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.candidate_dir / name).is_file()]
    if missing:
        raise SystemExit(f"missing seal inputs: {missing}")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite sealed manifest: {args.out}")
    for name in ("payloads_semantic_validation.jsonl", "frames_semantic_validation.jsonl"):
        if not all_validation_pass(args.candidate_dir / name):
            raise SystemExit(f"semantic validation failed: {name}")
    lexical = json.loads((args.candidate_dir / "lexical_gate.json").read_text(encoding="utf-8"))
    if not lexical.get("pass"):
        raise SystemExit("lexical gate failed")
    subword = json.loads((args.candidate_dir / "subword_gate.json").read_text(encoding="utf-8"))
    if not subword.get("pass"):
        raise SystemExit("subword gate failed")
    red_team = json.loads((args.candidate_dir / "red_team_lexical_report.json").read_text(encoding="utf-8"))
    if red_team.get("egregious"):
        raise SystemExit("red-team lexical audit failed")
    structural = json.loads((args.candidate_dir / "structural_variation.json").read_text(encoding="utf-8"))
    if not structural.get("pass"):
        raise SystemExit("structural variation gate failed")
    artifacts = {}
    for name in REQUIRED:
        path = args.candidate_dir / name
        artifacts[name] = {"path": str(path), "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "status": "sealed",
        "hash_policy": "immutable_sha256",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "generator_lineage": "stockmark/Stockmark-2-100B-Instruct",
        "validator_lineage": "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4",
        "artifacts": artifacts,
        "code": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "analysis_script": args.analysis_script,
            "renderer": args.renderer,
            "builder": args.builder,
            "lexical_gate_script": args.lexical_gate_script,
            "subword_gate_script": args.subword_gate_script,
            "structural_gate_script": args.structural_gate_script,
        }.items()},
        "activation_contract": {"position": "last_prompt_token_after_chat_template", "semantics": "zero_based_transformer_block_output_before_final_model_norm", "dtype": "float32", "inference_dtype": "bfloat16"},
        "statistical_plan": "scripts/confirmation_analysis.py --confirmatory; predeclared E1-E8; payload-block bootstrap; vocabulary-fold holdout",
        "extraction_allowed": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "sealed", "manifest": str(args.out), "artifact_count": len(artifacts)}))


if __name__ == "__main__":
    main()
