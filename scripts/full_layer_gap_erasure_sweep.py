"""Stage 1 (heavy compute) for the full-layer, normalized-depth matched-format-PCA
erasure gap analysis (plots/jpp/full_layer_gap_erasure/).

Replaces the arbitrary spike/min_gap/final 3-point depth convention
(scripts/full320_v3_erasure_dose_response_bootstrap.py) with the full continuous
layer trajectory, at every erasure rank r in {0,1,4,16,64}, for all 3
checkpoints. No new transformer forward passes: reads directly from the frozen
v3 activation tensors (v3-activations/activations/<model>/activations_all_layers.npz),
exactly as scripts/full320_v3_erasure_full_layer_sweep.py (rank=64-only) and
scripts/full320_v3_erasure_dose_response_bootstrap.py (3-depth, all ranks) already
do. Reuses the SAME helper functions from format_erasure_development.py
(PROBE, factorial_vectors, fit_probe, orth_basis, remove) -- same StandardScaler +
LogisticRegression(C=0.1) probe, same matched-format PCA basis fit only on
training folds, same held-purpose-family cross-fitting, same benchmark->casual /
casual->benchmark definitions.

Unlike full320_v3_erasure_full_layer_sweep.py (which only pools AUCs), this
script caches the per-item cross-fitted OOF decision-function scores for the
purpose / b2c / c2b regimes at every (layer, rank), so that stage 2
(scripts/full_layer_gap_erasure_bootstrap.py) can payload-block-bootstrap the
INTEGRATED gap curve (same block resample reused across all layers, all ranks,
and all three AUC regimes per replicate) instead of bootstrapping each layer's
AUC independently (which would discard the cross-layer/cross-regime
correlation and be pointwise-CI pseudoreplication).

Usage (one model per invocation, matches the existing cluster_run_*.sbatch
convention):
  python scripts/full_layer_gap_erasure_sweep.py --model llama31_8b
  python scripts/full_layer_gap_erasure_sweep.py --model llama31_70b
  python scripts/full_layer_gap_erasure_sweep.py --model llama33_70b

Output: results/full_layer_gap_erasure_scores_<model>.npz containing, per
(layer, rank) cell, the full-length (NaN-padded outside each regime's held-out
rows) OOF score arrays plus y / payload_block_id / purpose_family_id / format,
so stage 2 needs no further activation access.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import factorial_vectors, fit_probe, orth_basis, remove  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
    "llama31_70b": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
    "llama33_70b": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
}
RANKS = (0, 1, 4, 16, 64)
N_JOBS = int(os.environ.get("N_JOBS", "28"))


def oof_scores_layer_rank(Xl, rows, y, fam, is_bench, is_casual, rank):
    """Same 5-fold held-out-purpose-family cross-fitted matched_pca erasure +
    retrain convention as full320_v3_erasure_full_layer_sweep.score_layer /
    full320_v3_erasure_dose_response_bootstrap.score_fold, but returns the
    full per-item OOF score vectors instead of pooling directly to AUC, so
    stage 2 can bootstrap them layer-jointly."""
    folds = sorted(set(fam))
    n = len(y)
    s_purpose = np.full(n, np.nan)
    s_b2c = np.full(n, np.nan)
    s_c2b = np.full(n, np.nan)
    for fold in folds:
        train_mask = fam != fold
        test_mask = fam == fold
        if rank > 0:
            _, B, _, _ = factorial_vectors(Xl, rows, train_mask)
            Q = orth_basis(B, rank)
        else:
            Q = None
        Xt = remove(Xl, Q) if Q is not None else Xl

        s_purpose[test_mask] = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
        b2c_train, b2c_test = train_mask & is_bench, test_mask & is_casual
        if b2c_test.any():
            s_b2c[b2c_test] = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        c2b_train, c2b_test = train_mask & is_casual, test_mask & is_bench
        if c2b_test.any():
            s_c2b[c2b_test] = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])
    return s_purpose, s_b2c, s_c2b


def _job(layer, rank, Xl, rows, y, fam, is_bench, is_casual):
    t0 = time.time()
    s_p, s_b, s_c = oof_scores_layer_rank(Xl, rows, y, fam, is_bench, is_casual, rank)
    return layer, rank, s_p, s_b, s_c, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "results" / f"full_layer_gap_erasure_scores_{args.model}.npz"
    npz_path = MODELS[args.model]

    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr, flush=True)

    z = zipfile.ZipFile(npz_path)
    raw_ids = np.load(z.open("item_ids.npy"))
    keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
    assert keep_mask.sum() == len(rows_by_id)

    print(f"loading full tensor for {args.model} ...", file=sys.stderr, flush=True)
    t0 = time.time()
    X = np.load(npz_path)["X"][keep_mask]  # (n_kept, n_layers, hidden) float32
    n_layers = X.shape[1]
    print(f"  loaded {X.shape} in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
    y = np.array([int(r["label"]) for r in kept_rows])
    fam = np.array([r["purpose_family_id"] for r in kept_rows])
    fmt = np.array([r["format"] for r in kept_rows])
    blk = np.array([r["payload_block_id"] for r in kept_rows])
    is_bench, is_casual = fmt == "benchmark", fmt == "casual"
    n_items = len(y)

    # Materialize one contiguous slice per layer up front (same pattern as
    # full320_v3_erasure_full_layer_sweep.py) so joblib can memmap each large
    # array once and share it across the 5 rank-jobs for that layer.
    layer_slices = [np.ascontiguousarray(X[:, layer, :]) for layer in range(n_layers)]
    del X
    gc.collect()

    jobs = [(layer, rank) for layer in range(n_layers) for rank in RANKS]
    print(f"dispatching {len(jobs)} (layer,rank) jobs, n_jobs={N_JOBS}", file=sys.stderr, flush=True)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_job)(layer, rank, layer_slices[layer], kept_rows, y, fam, is_bench, is_casual)
        for layer, rank in jobs
    )

    n_ranks = len(RANKS)
    rank_pos = {r: i for i, r in enumerate(RANKS)}
    purpose_scores = np.full((n_ranks, n_layers, n_items), np.nan, dtype=np.float32)
    b2c_scores = np.full((n_ranks, n_layers, n_items), np.nan, dtype=np.float32)
    c2b_scores = np.full((n_ranks, n_layers, n_items), np.nan, dtype=np.float32)
    for layer, rank, s_p, s_b, s_c, dt in results:
        ri = rank_pos[rank]
        purpose_scores[ri, layer] = s_p
        b2c_scores[ri, layer] = s_b
        c2b_scores[ri, layer] = s_c
        print(f"  layer={layer:3d} rank={rank:3d} done in {dt:.1f}s", file=sys.stderr, flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        model=args.model,
        n_layers=n_layers,
        ranks=np.array(RANKS),
        y=y,
        payload_block_id=blk,
        purpose_family_id=fam,
        format=fmt,
        purpose_scores=purpose_scores,
        b2c_scores=b2c_scores,
        c2b_scores=c2b_scores,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
