"""Cheap extension of scripts/full_layer_gap_erasure_bootstrap.py: paired
payload-block-bootstrap significance statistics for the full-layer,
normalized-depth matched-format-PCA erasure dose-response, computed from the
SAME cached per-item OOF scores (scripts/full_layer_gap_erasure_sweep.py) --
no activation access, no probe refitting, no new model inference.

Because every bootstrap replicate reuses ONE payload-block resample across
every layer AND every rank (see full_layer_gap_erasure_bootstrap.py's
docstring), G(r) and G(r') are PAIRED within a replicate. That licenses
direct paired significance statements that a naive comparison of two
independent 95% CIs cannot support:

  - P(G(r) < G(0))                         -- is erasure at rank r a real
                                               reduction, not bootstrap noise?
  - P(G(r_hi) < G(r_lo)) for every adjacent -- is the rank dose-response
    pair r_lo < r_hi                          monotone, not just "trending"?
  - P(G(64) > 0)                           -- is the residual gap at maximal
                                               tested erasure significantly
                                               nonzero (erasure is incomplete)?
  - per-layer P(gap(q,64) < gap(q,0))      -- at which depths does erasure
                                               reliably reduce the gap, without
                                               singling out any one layer as a
                                               category?

Usage:
  python scripts/full_layer_gap_erasure_significance.py [--models llama31_8b ...]
Reads results/full_layer_gap_erasure_scores_<model>.npz.
Writes results/full_layer_gap_erasure_significance.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_layer_gap_erasure_bootstrap import RESULTS, RNG_SEED, N_BOOT, _fast_auc_matrix, _trapz  # noqa: E402

MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")


def analyze_model(model: str) -> dict:
    d = np.load(RESULTS / f"full_layer_gap_erasure_scores_{model}.npz", allow_pickle=True)
    ranks = [int(r) for r in d["ranks"]]
    n_ranks = len(ranks)
    n_layers = int(d["n_layers"])
    y = d["y"].astype(np.float64)
    blk = d["payload_block_id"]
    purpose_scores = d["purpose_scores"].astype(np.float64)
    b2c_scores = d["b2c_scores"].astype(np.float64)
    c2b_scores = d["c2b_scores"].astype(np.float64)
    q = np.arange(n_layers) / (n_layers - 1)

    mask_p = ~np.isnan(purpose_scores[0, 0])
    mask_b = ~np.isnan(b2c_scores[0, 0])
    mask_c = ~np.isnan(c2b_scores[0, 0])

    block_to_pos: dict = {}
    for i, b in enumerate(blk):
        block_to_pos.setdefault(b, []).append(i)
    blocks = np.array(sorted(block_to_pos))
    rng = np.random.default_rng(RNG_SEED)  # identical draw sequence to full_layer_gap_erasure_bootstrap.py

    boot_G = np.empty((n_ranks, N_BOOT))
    boot_gap_layer = np.empty((n_ranks, n_layers, N_BOOT))
    t0 = time.time()
    for rep in range(N_BOOT):
        sample_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([block_to_pos[b] for b in sample_blocks])
        idx_p = idx[mask_p[idx]]
        idx_b = idx[mask_b[idx]]
        idx_c = idx[mask_c[idx]]

        Pp = purpose_scores[:, :, idx_p].reshape(n_ranks * n_layers, len(idx_p))
        Pb = b2c_scores[:, :, idx_b].reshape(n_ranks * n_layers, len(idx_b))
        Pc = c2b_scores[:, :, idx_c].reshape(n_ranks * n_layers, len(idx_c))
        auc_p = _fast_auc_matrix(Pp, y[idx_p]).reshape(n_ranks, n_layers)
        auc_b = _fast_auc_matrix(Pb, y[idx_b]).reshape(n_ranks, n_layers)
        auc_c = _fast_auc_matrix(Pc, y[idx_c]).reshape(n_ranks, n_layers)
        gap_rep = auc_p - (auc_b + auc_c) / 2.0
        boot_gap_layer[:, :, rep] = gap_rep
        boot_G[:, rep] = _trapz(gap_rep, q, axis=1)
    print(f"  {model}: {N_BOOT} paired replicates in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    rank_pos = {r: i for i, r in enumerate(ranks)}
    r0 = rank_pos[0]

    contrasts_vs_raw = {}
    for r in ranks:
        if r == 0:
            continue
        ri = rank_pos[r]
        diff = boot_G[ri] - boot_G[r0]  # paired per-replicate difference
        p_reduced = float(np.mean(diff < 0))  # P(G(r) < G(0))
        contrasts_vs_raw[str(r)] = {
            "p_gap_reduced_vs_raw": p_reduced,
            "mean_reduction": float(np.mean(boot_G[r0] - boot_G[ri])),
            "reduction_ci95": [float(np.percentile(boot_G[r0] - boot_G[ri], 2.5)),
                                float(np.percentile(boot_G[r0] - boot_G[ri], 97.5))],
        }

    ranks_sorted = sorted(ranks)
    adjacent_monotone = {}
    for lo, hi in zip(ranks_sorted[:-1], ranks_sorted[1:]):
        d_ = boot_G[rank_pos[hi]] - boot_G[rank_pos[lo]]
        adjacent_monotone[f"{lo}->{hi}"] = {
            "p_gap_decreases": float(np.mean(d_ < 0)),
        }
    p_fully_monotone = float(np.mean(np.all(np.diff(boot_G[[rank_pos[r] for r in ranks_sorted]], axis=0) < 0, axis=0)))

    r64 = rank_pos.get(64)
    residual_at_max_rank = None
    if r64 is not None:
        residual_at_max_rank = {
            "p_G_positive": float(np.mean(boot_G[r64] > 0)),
            "boot_median": float(np.median(boot_G[r64])),
            "boot_ci95": [float(np.percentile(boot_G[r64], 2.5)), float(np.percentile(boot_G[r64], 97.5))],
        }

    layerwise_p_reduced_r64_vs_r0 = None
    if r64 is not None:
        diff_layer = boot_gap_layer[r64] - boot_gap_layer[r0]  # (n_layers, N_BOOT)
        layerwise_p_reduced_r64_vs_r0 = np.mean(diff_layer < 0, axis=1).tolist()

    return {
        "n_layers": n_layers,
        "ranks": ranks,
        "contrasts_vs_raw": contrasts_vs_raw,
        "adjacent_rank_monotonicity": adjacent_monotone,
        "p_fully_monotone_dose_response": p_fully_monotone,
        "residual_gap_at_rank64": residual_at_max_rank,
        "layerwise_p_gap_reduced_rank64_vs_rank0": layerwise_p_reduced_r64_vs_r0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    out_path = RESULTS / "full_layer_gap_erasure_significance.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "schema_version": 1,
        "purpose": (
            "Paired payload-block-bootstrap significance statistics for the "
            "full-layer normalized-depth matched-format-PCA erasure dose-response "
            "(rank contrasts vs. raw, adjacent-rank monotonicity, residual gap at "
            "rank 64, per-layer reduction probability), derived from the same "
            "cached OOF scores and same block-resample convention as "
            "scripts/full_layer_gap_erasure_bootstrap.py."
        ),
        "n_boot": N_BOOT, "rng_seed": RNG_SEED,
        "by_model": {},
    }
    for model in args.models:
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        out["by_model"][model] = analyze_model(model)

    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
