#!/usr/bin/env python3
"""Rebuild a resume manifest over layer checkpoints gathered from many boxes.

The rotation sweep was sharded across small CPU boxes, each publishing atomic
per-layer checkpoints for its own slice of the depth range. Once the files are
collected into one directory, this writes the manifest that
``full320_v3_qb_rotation_sweep.py --resume`` expects, after checking that every
checkpoint agrees on model identity and item identity.

Nothing here recomputes a score: it validates and indexes what the boxes wrote.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PARTIAL_SCHEMA_VERSION = 1
CONDITIONS = ("topC16", "bottomC16", "randomB16")
REGIMES = ("joint", "b2c", "c2b")
SUB_RANK = 16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partial-dir", required=True)
    ap.add_argument("--expect-layers", type=int, required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    partial_dir = Path(args.partial_dir)
    files = sorted(partial_dir.glob("layer_*.npz"))
    if not files:
        raise SystemExit(f"no layer checkpoints in {partial_dir}")

    layers, digests, models, shapes, elapsed = [], set(), set(), set(), {}
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            layer = int(payload["layer"].item())
            if int(payload["schema_version"].item()) != PARTIAL_SCHEMA_VERSION:
                raise SystemExit(f"{path.name}: unexpected schema version")
            if f"layer_{layer:03d}.npz" != path.name:
                raise SystemExit(f"{path.name}: filename disagrees with stored layer {layer}")
            layers.append(layer)
            digests.add(str(payload["item_id_digest"].item()))
            models.add(str(payload["model"].item()))
            shapes.add(tuple(int(v) for v in payload["scores"].shape))
            elapsed[str(layer)] = float(payload["elapsed"].item())
    if len(digests) != 1:
        raise SystemExit(f"checkpoints disagree on item identity: {sorted(digests)}")
    if len(models) != 1:
        raise SystemExit(f"checkpoints disagree on model: {sorted(models)}")
    if len(shapes) != 1:
        raise SystemExit(f"checkpoints disagree on score shape: {sorted(shapes)}")
    shape = shapes.pop()
    if shape[:2] != (len(CONDITIONS), len(REGIMES)):
        raise SystemExit(f"unexpected score shape {shape}")
    if len(set(layers)) != len(layers):
        raise SystemExit("duplicate layer checkpoints")

    missing = sorted(set(range(args.expect_layers)) - set(layers))
    if missing and not args.allow_incomplete:
        raise SystemExit(f"missing {len(missing)} layers: {missing}")

    manifest = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "status": "complete" if not missing else "running",
        "model": models.pop(),
        "n_layers": args.expect_layers,
        "n_items": int(shape[2]),
        "item_id_digest": digests.pop(),
        "conditions": list(CONDITIONS),
        "ranks": [SUB_RANK],
        "regimes": list(REGIMES),
        "completed_layers": sorted(layers),
    }
    (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "partial_dir": str(partial_dir),
                "layers_present": len(layers),
                "layers_expected": args.expect_layers,
                "missing": missing,
                "n_items": manifest["n_items"],
                "total_layer_seconds": round(sum(elapsed.values()), 1),
                "median_layer_seconds": round(float(np.median(list(elapsed.values()))), 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
