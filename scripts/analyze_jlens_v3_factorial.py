"""Bootstrap factorial A/B/C summaries from a full320 jLens assay result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: np.ndarray, seed: int, resamples: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty block set")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return {
        "n_blocks": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "bootstrap_ci_2.5": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_50": float(np.quantile(draws, 0.50)),
        "bootstrap_ci_97.5": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    result = json.loads(args.assay.read_text(encoding="utf-8"))
    if result.get("status") != "development_only":
        raise ValueError("expected development-only assay result")
    summary = {}
    for layer, layer_result in result["summary"]["factorial"].items():
        blocks = layer_result.get("blocks", [])
        if not blocks:
            raise ValueError(f"layer {layer} has no complete factorial blocks")
        a = np.asarray([block["A"] for block in blocks])
        b = np.asarray([block["B"] for block in blocks])
        c = np.asarray([block["C"] for block in blocks])
        ratio = c / np.maximum(np.abs(a), 1e-12)
        summary[layer] = {
            "n_complete_blocks": len(blocks),
            "A": summarize(a, args.seed + int(layer) * 11 + 1, args.resamples),
            "B": summarize(b, args.seed + int(layer) * 11 + 2, args.resamples),
            "C": summarize(c, args.seed + int(layer) * 11 + 3, args.resamples),
            "C_over_abs_A": summarize(ratio, args.seed + int(layer) * 11 + 4, args.resamples),
        }
    output = {
        "schema_version": 1,
        "status": "development_only",
        "assay": {"path": str(args.assay), "sha256": sha256(args.assay)},
        "bootstrap_unit": "complete payload block",
        "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.resamples,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"layers": sorted(summary), "out": str(args.out)}))


if __name__ == "__main__":
    main()
