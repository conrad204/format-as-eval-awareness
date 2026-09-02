#!/usr/bin/env python3
"""Interim readout from already-published emergency partial layer checkpoints.

Reports per-condition mean gap over the *completed* layers only, plus the
partial-depth specificity contrast and a payload-block bootstrap CI computed with
the frozen AUC machinery. This is a monitoring tool: the reportable statistic is
the full-depth integral produced by
``full320_v3_emergency_cperp_bootstrap.py`` once every layer exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full320_v3_emergency_cperp_sweep import CONDITIONS, MODELS, REGIMES  # noqa: E402
from full320_v3_emergency_cperp_bootstrap import (  # noqa: E402
    prepare_tie_aware_order,
    weighted_auc_with_ties,
)
from full320_v3_factorial_residual_bootstrap import ci  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--partial-dir", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--raw-reference", default=None)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    rows_by_id = {}
    for line in Path(args.items).open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("label") is not None:
                rows_by_id[row["item_id"]] = row
    with zipfile.ZipFile(args.npz) as archive:
        raw_ids = np.load(archive.open("item_ids.npy")).astype(str)
    ids = [iid for iid in raw_ids if iid in rows_by_id]
    y = np.asarray([int(rows_by_id[iid]["label"]) for iid in ids])
    blocks = np.asarray([rows_by_id[iid]["payload_block_id"] for iid in ids])

    partial_dir = Path(args.partial_dir)
    layers = sorted(int(p.stem.split("_")[1]) for p in partial_dir.glob("layer_*.npz"))
    if not layers:
        raise SystemExit("no completed layers yet")
    stack = np.stack(
        [np.load(partial_dir / f"layer_{layer:03d}.npz")["scores"] for layer in layers], axis=2
    )  # (condition, regime, layer, item)

    cidx = {name: i for i, name in enumerate(CONDITIONS)}
    ridx = {name: i for i, name in enumerate(REGIMES)}
    auc = np.full(stack.shape[:-1], np.nan)
    for index in np.ndindex(stack.shape[:-1]):
        values = stack[index]
        mask = np.isfinite(values)
        auc[index] = roc_auc_score(y[mask], values[mask])
    gap = auc[:, ridx["joint"]] - (auc[:, ridx["b2c"]] + auc[:, ridx["c2b"]]) / 2.0
    mean_gap = gap.mean(axis=1)

    unique_blocks = np.asarray(sorted(set(blocks)))
    block_pos = {block: i for i, block in enumerate(unique_blocks)}
    item_block = np.asarray([block_pos[b] for b in blocks], dtype=np.int32)
    prepared = {}
    for regime, regime_i in ridx.items():
        mask = np.isfinite(stack[0, regime_i, 0])
        matrix = stack[:, regime_i, :, :][:, :, mask].reshape(len(CONDITIONS) * len(layers), int(mask.sum()))
        prepared[regime] = (prepare_tie_aware_order(matrix, y[mask]), item_block[mask])

    rng = np.random.default_rng(0)
    boot = np.empty((len(CONDITIONS), args.n_boot))
    for rep in range(args.n_boot):
        sampled = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        counts = np.bincount(sampled, minlength=len(unique_blocks)).astype(np.float64)
        rep_auc = np.empty((len(CONDITIONS), len(REGIMES), len(layers)))
        for regime, regime_i in ridx.items():
            regime_prepared, regime_item_block = prepared[regime]
            rep_auc[:, regime_i, :] = weighted_auc_with_ties(
                regime_prepared, counts[regime_item_block]
            ).reshape(len(CONDITIONS), len(layers))
        rep_gap = rep_auc[:, ridx["joint"]] - (rep_auc[:, ridx["b2c"]] + rep_auc[:, ridx["c2b"]]) / 2.0
        boot[:, rep] = rep_gap.mean(axis=1)

    extra_c = mean_gap[cidx["minus_b"]] - mean_gap[cidx["minus_b_cperp"]]
    extra_random = mean_gap[cidx["minus_b"]] - mean_gap[cidx["minus_b_rperp"]]
    boot_specific = (boot[cidx["minus_b"]] - boot[cidx["minus_b_cperp"]]) - (
        boot[cidx["minus_b"]] - boot[cidx["minus_b_rperp"]]
    )
    report = {
        "model": args.model,
        "status": "INTERIM over completed layers only; not the reportable full-depth integral",
        "completed_layers": layers,
        "n_completed": len(layers),
        "mean_gap": {name: float(mean_gap[cidx[name]]) for name in CONDITIONS},
        "mean_gap_ci95": {name: ci(boot[cidx[name]]) for name in CONDITIONS},
        "mean_joint_auc": {name: float(auc[cidx[name], ridx["joint"]].mean()) for name in CONDITIONS},
        "mean_b2c_auc": {name: float(auc[cidx[name], ridx["b2c"]].mean()) for name in CONDITIONS},
        "mean_c2b_auc": {name: float(auc[cidx[name], ridx["c2b"]].mean()) for name in CONDITIONS},
        "extra_C": float(extra_c),
        "extra_random": float(extra_random),
        "specific_C": float(extra_c - extra_random),
        "specific_C_ci95": ci(boot_specific),
        "boot_fraction_specific_C_positive": float(np.mean(boot_specific > 0)),
        "n_boot": args.n_boot,
    }

    if args.raw_reference:
        reference = json.loads(Path(args.raw_reference).read_text())
        errors = [
            abs(auc[cidx["raw"], ridx[regime], position] - float(reference[layer][key]["auc"]))
            for position, layer in enumerate(layers)
            for regime, key in (("joint", "decorrelated"), ("b2c", "b2c"), ("c2b", "c2b"))
        ]
        report["raw_reference_max_abs_error_on_completed_layers"] = float(max(errors))

    sanity = [
        json.loads(str(np.load(partial_dir / f"layer_{layer:03d}.npz")["sanity_json"].item()))
        for layer in layers
    ]
    report["gates"] = {
        "max_factorial_identity_error": max(v["max_factorial_identity_error"] for v in sanity),
        "max_qb_t_qcperp": max(v["max_qb_t_qcperp"] for v in sanity),
        "max_qb_t_rperp": max(v["max_qb_t_rperp"] for v in sanity),
        "held_family_basis_violations": sum(v["held_family_basis_violations"] for v in sanity),
        "basis_backend": sorted({v["basis_backend"] for v in sanity}),
    }
    report["mean_c_capture"] = {
        prefix: float(np.mean([v["c_capture_by_rank"][prefix]["cross_fitted_fraction"] for v in sanity]))
        for prefix in ("1", "4", "16", "64")
    }
    report["mean_c_capture_chance_multiple"] = {
        prefix: float(np.mean([v["c_capture_by_rank"][prefix]["chance_multiple"] for v in sanity]))
        for prefix in ("1", "4", "16", "64")
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
