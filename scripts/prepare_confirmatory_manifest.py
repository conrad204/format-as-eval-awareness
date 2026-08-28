from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite manifest: {args.out}")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    if base.get("status") != "sealed":
        raise SystemExit("base manifest is not sealed")
    source_names = {
        "bank": "payloads.jsonl",
        "frames": "frames.json",
        "rendered_items": "rendered_items.jsonl",
        "semantic_validation": "payloads_semantic_validation.jsonl",
        "frames_semantic_validation": "frames_semantic_validation.jsonl",
        "lexical_gate": "lexical_gate.json",
        "subword_gate": "subword_gate.json",
        "surface_balance": "surface_balance.json",
        "red_team": "red_team_lexical_report.json",
        "payloads_generation_log": "payloads_generation_log.jsonl",
        "frames_generation_log": "frames_generation_log.jsonl",
    }
    artifacts = {}
    for name, filename in source_names.items():
        spec = base["artifacts"][filename]
        artifacts[name] = {"path": os.path.relpath(args.base.parent / spec["path"], args.out.parent), "sha256": spec["sha256"]}
    artifacts["analysis_script"] = {"path": os.path.relpath(args.analysis_script, args.out.parent), "sha256": sha256(args.analysis_script)}
    manifest = dict(base)
    manifest["artifacts"] = artifacts
    manifest["confirmatory_run"] = {
        "items": os.path.relpath(args.items, args.out.parent),
        "activations": os.path.relpath(args.activations, args.out.parent),
        "output": os.path.relpath(args.output, args.out.parent),
        "sentinel": os.path.relpath(args.sentinel, args.out.parent),
        "analysis_script": os.path.relpath(args.analysis_script, args.out.parent),
        "layer": args.layer,
        "model": args.model,
        "revision": args.revision,
        "bank_sha256": base["artifacts"]["rendered_items.jsonl"]["sha256"],
        "activations_sha256": sha256(args.activations),
        "analysis_sha256": sha256(args.analysis_script),
        "wrapper_sha256": sha256(args.wrapper),
        "n_layers": args.n_layers,
        "bootstrap": 10000,
        "seed": 0,
        "timeout_seconds": 7200,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"manifest": str(args.out), "model": args.model, "layer": args.layer}))


if __name__ == "__main__":
    main()
