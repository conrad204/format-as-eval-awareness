#!/usr/bin/env python3
"""Per-layer, per-condition 95% CIs for the interchange-patching figure lines.

The existing bootstrap reducer
(scripts/full320_v3_interchange_patch_bootstrap.py) reports intervals on the
real_swap-minus-control GAP. The cross-model figure
(plots/jpp/interchange_patch_all_models) instead draws the two conditions as
separate lines, so it needs an interval per (layer, condition) line rather
than per contrast. This script produces exactly those, using the SAME
resampling convention as the gap reducer:

  * the resampling unit is the payload block, never the individual record;
  * ONE block resample per replicate is reused across EVERY layer, condition
    and metric, so the ribbons share the true cross-layer correlation
    structure instead of pretending layers are independent draws.

Ribbons are therefore descriptive uncertainty on each line. They are NOT a
per-layer significance test, and no layer count is thresholded off them.

Usage:
  python scripts/interchange_patch_line_ci.py llama33_70b \
      results/tae_ixpatch_full_llama33_70b.jsonl \
      results/tae_ixpatch_matched_llama33_70b.jsonl \
      -- results/tae_ixpatch_lineci_llama33_70b.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 0


def load_records(paths: list[Path]) -> list[dict]:
    recs: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def accumulate(records: list[dict]):
    """Per (layer, condition, block): summed metric and contributing count."""
    blocks = sorted({r["payload_block_id"] for r in records})
    bidx = {b: i for i, b in enumerate(blocks)}
    cells = sorted({(r["layer"], r["condition"]) for r in records})
    cidx = {c: i for i, c in enumerate(cells)}

    nC, nB = len(cells), len(blocks)
    iia_sum = np.zeros((nC, nB))
    iia_n = np.zeros((nC, nB))
    psh_sum = np.zeros((nC, nB))
    psh_n = np.zeros((nC, nB))

    for r in records:
        i = cidx[(r["layer"], r["condition"])]
        j = bidx[r["payload_block_id"]]
        target_gap = r["format_score_baseline_tgt"] - r["format_score_baseline_src"]
        actual = r["format_score_patched"] - r["format_score_baseline_src"]
        if target_gap != 0:
            iia_sum[i, j] += float(np.sign(actual) == np.sign(target_gap))
            iia_n[i, j] += 1.0
        psh_sum[i, j] += abs(r["purpose_score_patched"] - r["purpose_score_baseline"])
        psh_n[i, j] += 1.0

    return cells, blocks, iia_sum, iia_n, psh_sum, psh_n


def bootstrap_ci(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ratio-of-sums means under shared block weights -> (lo, hi) per cell."""
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (num @ weights) / (den @ weights)
    return (np.nanpercentile(draws, 2.5, axis=1), np.nanpercentile(draws, 97.5, axis=1))


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--")
    model, inputs, out_path = argv[0], [Path(p) for p in argv[1:sep]], Path(argv[sep + 1])

    records = load_records(inputs)
    cells, blocks, iia_sum, iia_n, psh_sum, psh_n = accumulate(records)
    nB = len(blocks)

    rng = np.random.default_rng(SEED)
    # One resample of blocks per replicate, shared across all cells/metrics.
    weights = rng.multinomial(nB, np.full(nB, 1.0 / nB), size=N_BOOT).T.astype(float)

    iia_lo, iia_hi = bootstrap_ci(iia_sum, iia_n, weights)
    psh_lo, psh_hi = bootstrap_ci(psh_sum, psh_n, weights)

    with np.errstate(invalid="ignore", divide="ignore"):
        iia_pt = iia_sum.sum(axis=1) / iia_n.sum(axis=1)
        psh_pt = psh_sum.sum(axis=1) / psh_n.sum(axis=1)

    by_layer: dict[str, dict] = defaultdict(dict)
    for i, (layer, cond) in enumerate(cells):
        by_layer[str(layer)][cond] = {
            "format_iia": float(iia_pt[i]),
            "format_iia_ci95": [float(iia_lo[i]), float(iia_hi[i])],
            "mean_abs_purpose_shift": float(psh_pt[i]),
            "mean_abs_purpose_shift_ci95": [float(psh_lo[i]), float(psh_hi[i])],
        }

    out = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "Per-(layer, condition) block-bootstrap CIs for figure ribbons. "
                   "Shared block resample across all layers/conditions/metrics. "
                   "Descriptive uncertainty, not a per-layer significance test.",
        "model": model,
        "n_boot": N_BOOT,
        "seed": SEED,
        "n_payload_blocks": nB,
        "n_layers": len({L for L, _ in cells}),
        "conditions": sorted({c for _, c in cells}),
        "inputs": [str(p) for p in inputs],
        "by_layer": dict(by_layer),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}  ({out['n_layers']} layers, {nB} blocks, {N_BOOT} reps)")


if __name__ == "__main__":
    main()
