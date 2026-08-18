"""Analyze fixed J-lens purpose-projection shifts across format families."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("assay", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = json.loads(args.assay.read_text(encoding="utf-8"))
    scores = result["scores"]
    families = sorted({r["format_family_id"] for r in scores})
    blocks = sorted({r["payload_block_id"] for r in scores})
    layers = result["source_layers"]
    report = {}
    for layer in layers:
        by_family = {family: {} for family in families}
        for row in scores:
            value = row["layers"][str(layer)]["purpose"]["score"]
            by_family[row["format_family_id"]][row["item_id"]] = value
        family_means = {family: float(np.mean(list(values.values()))) for family, values in by_family.items()}
        benchmark = [f for f in families if f in {"mcq_letter", "numbered_options", "json_schema", "exact_label"}]
        casual = [f for f in families if f in {"conversational", "email_style", "chat_explanation", "bullet_answer"}]
        paired = []
        for block in blocks:
            b = [by_family[f][next(r["item_id"] for r in scores if r["payload_block_id"] == block and r["format_family_id"] == f)] for f in benchmark]
            c = [by_family[f][next(r["item_id"] for r in scores if r["payload_block_id"] == block and r["format_family_id"] == f)] for f in casual]
            paired.append(float(np.mean(b) - np.mean(c)))
        report[str(layer)] = {"family_means": family_means, "benchmark_minus_casual": {"mean": float(np.mean(paired)), "sd": float(np.std(paired, ddof=1)), "n_blocks": len(paired)}, "leave_one_family_out": {family: float(np.mean([family_means[x] for x in benchmark if x != family]) - np.mean([family_means[x] for x in casual])) if family in benchmark else float(np.mean(benchmark and [family_means[x] for x in benchmark]) - np.mean([family_means[x] for x in casual if x != family])) for family in families}}
    out = {"schema_version": 1, "status": "development_only", "assay": {"path": str(args.assay), "sha256": sha256(args.assay)}, "layers": report, "interpretation": "fixed evaluation-minus-deployment J-lens projection; no fitting or family selection used"}
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layers": layers, "families": families, "blocks": blocks.__len__()}))


if __name__ == "__main__":
    main()
