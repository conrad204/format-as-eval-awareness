#!/usr/bin/env python3
"""Depth-integrated interchange gap (real_swap minus magnitude-matched control).

The existing reducer (scripts/full320_v3_interchange_patch_bootstrap.py) reports
a family-wise-valid JOINT WORST-LAYER statistic. That statistic is hostage to the
weakest layer, and layer 0 is null by construction (the format signal has not
formed at the embedding output), so it fails in the 80-layer checkpoints for a
reason that is not about the effect.

This script reports the complementary summary: the interchange gap integrated
over normalized depth, which is the same convention already used for the
cross-format transfer gap (see the erasure analyses) and is consistent with
treating layers as repeated observations along one trajectory rather than as
independent tests.

Convention matches the rest of the repo: the resampling unit is the payload
block, and ONE block resample per replicate is reused across every layer and
both conditions, so the integral carries the true cross-layer correlation.

Both statistics should be reported. The worst-layer statistic was the
pre-specified one; this integral is added post hoc and is labeled as such.

Usage:
  python scripts/interchange_integrated_gap.py llama33_70b 80 \
      results/tae_ixpatch_full_llama33_70b.jsonl \
      results/tae_ixpatch_matched_llama33_70b.jsonl \
      -- results/tae_ixpatch_integrated_llama33_70b.json
  # add --even-only for checkpoints swept on an even-layer grid
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 0
CONDS = ("real_swap", "matched_control")


def main() -> None:
    argv = [a for a in sys.argv[1:]]
    even_only = "--even-only" in argv
    if even_only:
        argv.remove("--even-only")
    sep = argv.index("--")
    model, n_layers = argv[0], int(argv[1])
    inputs = [Path(p) for p in argv[2:sep]]
    out_path = Path(argv[sep + 1])

    # Pass 1: index blocks and the layers present under real_swap.
    blocks: set[str] = set()
    layers: set[int] = set()
    for p in inputs:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if even_only and r["layer"] % 2:
                    continue
                blocks.add(r["payload_block_id"])
                if r["condition"] == "real_swap":
                    layers.add(r["layer"])
    blocks_l, layers_l = sorted(blocks), sorted(layers)
    bidx = {b: i for i, b in enumerate(blocks_l)}
    lidx = {L: i for i, L in enumerate(layers_l)}
    nL, nB = len(layers_l), len(blocks_l)

    # Pass 2: accumulate per (layer, block) sign-agreement hits and counts.
    S = {c: np.zeros((nL, nB)) for c in CONDS}
    N = {c: np.zeros((nL, nB)) for c in CONDS}
    for p in inputs:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                c = r["condition"]
                if c not in S or r["layer"] not in lidx:
                    continue
                if even_only and r["layer"] % 2:
                    continue
                tg = r["format_score_baseline_tgt"] - r["format_score_baseline_src"]
                if tg == 0:
                    continue
                i, j = lidx[r["layer"]], bidx[r["payload_block_id"]]
                hit = float(np.sign(r["format_score_patched"] - r["format_score_baseline_src"]) == np.sign(tg))
                S[c][i, j] += hit
                N[c][i, j] += 1.0

    q = np.array([L / (n_layers - 1) for L in layers_l], dtype=float)
    span = q[-1] - q[0]

    rng = np.random.default_rng(SEED)
    W = rng.multinomial(nB, np.full(nB, 1.0 / nB), size=N_BOOT).T.astype(float)
    real = (S["real_swap"] @ W) / (N["real_swap"] @ W)
    ctrl = (S["matched_control"] @ W) / (N["matched_control"] @ W)
    draws = np.trapezoid(real - ctrl, q, axis=0) / span

    pt_real = S["real_swap"].sum(1) / N["real_swap"].sum(1)
    pt_ctrl = S["matched_control"].sum(1) / N["matched_control"].sum(1)
    point = float(np.trapezoid(pt_real - pt_ctrl, q) / span)
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))

    out = {
        "schema_version": 1,
        "mode": "development",
        "statistic": "depth-integrated interchange gap (real_swap - matched_control), "
                     "mean over normalized depth; POST HOC, complements the pre-specified "
                     "joint worst-layer statistic",
        "model": model,
        "n_layers_tested": nL,
        "n_layers_total": n_layers,
        "even_layer_grid": even_only,
        "n_payload_blocks": nB,
        "n_boot": N_BOOT,
        "seed": SEED,
        "integrated_format_iia_gap": point,
        "integrated_format_iia_gap_ci95": [lo, hi],
        "inputs": [str(p) for p in inputs],
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"{model:12s} layers={nL:3d} blocks={nB}  integrated IIA gap = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
