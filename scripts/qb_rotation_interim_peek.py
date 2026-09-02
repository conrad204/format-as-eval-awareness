#!/usr/bin/env python3
"""Interim within-Q_B readout from whatever rotation layers exist so far.

Matches the reused `raw` baseline to exactly the layers the fleet has published,
so `rescue` is computed on a common layer set rather than comparing a partial
depth range against a full one. Reports the mean gap over those layers (not the
full-depth integral) plus the paired payload-block bootstrap contrasts.

The reportable statistic remains the full-depth integral from
``full320_v3_qb_rotation_bootstrap.py`` once all layers exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full320_v3_factorial_residual_bootstrap import (  # noqa: E402
    ci,
    prepare_auc_order,
    weighted_auc_from_order,
)
from full320_v3_qb_rotation_sweep import CONDITIONS as ROT_CONDITIONS  # noqa: E402
from full320_v3_qb_rotation_sweep import REGIMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partial-dir", required=True)
    ap.add_argument("--emergency-scores", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    partial_dir = Path(args.partial_dir)
    files = sorted(partial_dir.glob("layer_*.npz"))
    layers = []
    stack = []
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            layers.append(int(payload["layer"].item()))
            stack.append(np.asarray(payload["scores"], dtype=np.float64))
    if not layers:
        raise SystemExit("no rotation checkpoints yet")
    order = np.argsort(layers)
    layers = [layers[i] for i in order]
    rotation = np.stack([stack[i] for i in order], axis=2)  # (cond, regime, layer, item)

    emergency = np.load(args.emergency_scores, allow_pickle=True)
    emergency_conditions = [str(v) for v in emergency["conditions"]]
    if [str(v) for v in emergency["regimes"]] != list(REGIMES):
        raise SystemExit("regime order differs between runs")
    y = emergency["y"].astype(np.int8)
    blocks = emergency["payload_block_id"].astype(str)
    raw = emergency["scores"][emergency_conditions.index("raw"), 0][:, layers, :].astype(np.float64)

    conditions = ["raw", *ROT_CONDITIONS]
    scores = np.concatenate([raw[None], rotation], axis=0)
    cidx = {name: i for i, name in enumerate(conditions)}
    ridx = {name: i for i, name in enumerate(REGIMES)}

    auc = np.full(scores.shape[:-1], np.nan)
    for index in np.ndindex(scores.shape[:-1]):
        values = scores[index]
        mask = np.isfinite(values)
        auc[index] = roc_auc_score(y[mask], values[mask])
    gap = auc[:, ridx["joint"]] - (auc[:, ridx["b2c"]] + auc[:, ridx["c2b"]]) / 2.0
    mean_gap = gap.mean(axis=1)

    unique_blocks = np.asarray(sorted(set(blocks)))
    block_pos = {block: i for i, block in enumerate(unique_blocks)}
    item_block = np.asarray([block_pos[b] for b in blocks], dtype=np.int32)
    prepared = {}
    for regime, regime_i in ridx.items():
        mask = np.isfinite(scores[0, regime_i, 0])
        matrix = scores[:, regime_i, :, :][:, :, mask].reshape(len(conditions) * len(layers), int(mask.sum()))
        prepared[regime] = (prepare_auc_order(matrix, y[mask]), item_block[mask])

    rng = np.random.default_rng(0)
    boot = np.empty((len(conditions), args.n_boot))
    for rep in range(args.n_boot):
        sampled = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        counts = np.bincount(sampled, minlength=len(unique_blocks)).astype(np.float64)
        rep_auc = np.empty((len(conditions), len(REGIMES), len(layers)))
        for regime, regime_i in ridx.items():
            regime_prepared, regime_item_block = prepared[regime]
            rep_auc[:, regime_i, :] = weighted_auc_from_order(
                regime_prepared, counts[regime_item_block]
            ).reshape(len(conditions), len(layers))
        rep_gap = rep_auc[:, ridx["joint"]] - (rep_auc[:, ridx["b2c"]] + rep_auc[:, ridx["c2b"]]) / 2.0
        boot[:, rep] = rep_gap.mean(axis=1)

    rescue = {name: float(mean_gap[cidx["raw"]] - mean_gap[cidx[name]]) for name in ROT_CONDITIONS}
    boot_rescue = {name: boot[cidx["raw"]] - boot[cidx[name]] for name in ROT_CONDITIONS}
    contrasts = {}
    for label, (a, b) in {
        "C_enrichment": ("topC16", "bottomC16"),
        "C_vs_random_inside_B": ("topC16", "randomB16"),
        "bottom_vs_random_inside_B": ("bottomC16", "randomB16"),
    }.items():
        draws = boot_rescue[a] - boot_rescue[b]
        interval = ci(draws)
        contrasts[label] = {
            "difference": rescue[a] - rescue[b],
            "ci95": interval,
            "excludes_zero": bool(interval[0] > 0.0 or interval[1] < 0.0),
            "boot_fraction_positive": float(np.mean(draws > 0.0)),
        }

    report = {
        "status": "INTERIM: mean gap over published layers only, not the full-depth integral",
        "model": str(emergency["model"]),
        "layers": layers,
        "n_layers_published": len(layers),
        "n_boot": args.n_boot,
        "mean_gap": {name: float(mean_gap[cidx[name]]) for name in conditions},
        "mean_gap_ci95": {name: ci(boot[cidx[name]]) for name in conditions},
        "rescue": rescue,
        "rescue_ci95": {name: ci(boot_rescue[name]) for name in ROT_CONDITIONS},
        "contrasts": contrasts,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
