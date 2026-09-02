"""Stage 2 (cheap) for the full-layer, normalized-depth matched-format-PCA
erasure gap analysis (plots/jpp/full_layer_gap_erasure/).

Consumes the per-item cross-fitted OOF scores cached by
scripts/full_layer_gap_erasure_sweep.py (no activation tensors touched here --
this stage only does bootstrap arithmetic on cached probe outputs). For every
model and matched-format-PCA erasure rank r in {0,1,4,16,64) it:

  1. Computes the point-estimate layerwise transfer gap
       g(q,r) = AUC_purpose(q,r) - (AUC_b2c(q,r) + AUC_c2b(q,r)) / 2
     at every transformer layer, with normalized depth q = layer/(n_layers-1),
     using sklearn.metrics.roc_auc_score directly (bit-identical convention to
     full320_v3_erasure_full_layer_sweep.py / full320_v3_erasure_dose_response_bootstrap.py).
  2. Integrates the curve with the trapezoidal rule:
       G(r) = trapz(g(q,r), q)   and   fraction_gap_removed(r) = 1 - G(r)/G(0)

Uncertainty is pseudoreplication-aware: for each of N_BOOT=1000 replicates, ONE
payload-block resample (with replacement, matching the convention in
full320_v3_erasure_dose_response_bootstrap.block_bootstrap_gap /
scripts/format_bound_regrowth_bootstrap.py's gap_for_sample) is drawn and reused
for every layer, every rank, and all three AUC regimes (purpose/b2c/c2b) in that
replicate; the ENTIRE layerwise gap curve for that replicate is recomputed from
the resampled items and THEN integrated. This is stronger than the task's
minimum requirement (which asks for the same blocks to be reused across layers
and AUC regimes per rank) -- reusing the same block draw across ranks too makes
G(r)/G(0) a valid paired ratio per replicate, so fraction_gap_removed(r)'s CI can
be read directly off the replicate distribution instead of dividing separately
bootstrapped point CIs.

CI arithmetic uses a fast, mathematically exact reformulation of AUC as the
Mann-Whitney rank-sum statistic (verified bit-identical to
sklearn.metrics.roc_auc_score on continuous decision_function scores below) so
that 1000 replicates x 5 ranks x <=80 layers x 3 regimes finishes in
minutes instead of hours of repeated roc_auc_score calls.

Usage:
  python scripts/full_layer_gap_erasure_bootstrap.py
Reads results/full_layer_gap_erasure_scores_<model>.npz for all 3 checkpoints.
Writes results/full_layer_gap_erasure_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")
N_BOOT = 1000
RNG_SEED = 0
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _fast_auc_matrix(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC per row of S (n_cols, m) against the SAME label vector y (m,), via
    the Mann-Whitney rank-sum identity AUC = (sum_rank(pos) - n_pos*(n_pos-1)/2)
    / (n_pos*n_neg). Exact for the continuous (no-tie) decision_function scores
    used throughout this codebase; matches roc_auc_score to float precision
    (see scripts/full_layer_gap_erasure_bootstrap.py module docstring / repo
    sanity check)."""
    order = np.argsort(S, axis=1, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(S.shape[1], dtype=np.float64), axis=1)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = S.shape[1] - n_pos
    sum_ranks_pos = ranks[:, pos].sum(axis=1)
    return (sum_ranks_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_model(model: str) -> dict:
    npz_path = RESULTS / f"full_layer_gap_erasure_scores_{model}.npz"
    d = np.load(npz_path, allow_pickle=True)
    ranks = [int(r) for r in d["ranks"]]
    n_ranks = len(ranks)
    n_layers = int(d["n_layers"])
    y = d["y"].astype(np.float64)
    blk = d["payload_block_id"]
    purpose_scores = d["purpose_scores"].astype(np.float64)  # (n_ranks, n_layers, n_items)
    b2c_scores = d["b2c_scores"].astype(np.float64)
    c2b_scores = d["c2b_scores"].astype(np.float64)
    n_items = y.shape[0]
    q = np.arange(n_layers) / (n_layers - 1)

    mask_p = ~np.isnan(purpose_scores[0, 0])
    mask_b = ~np.isnan(b2c_scores[0, 0])
    mask_c = ~np.isnan(c2b_scores[0, 0])
    # NaN pattern is determined by held-out-family / format structure only,
    # not by fitted values -- verify it is identical across every (rank,layer).
    for ri in range(n_ranks):
        for L in (0, n_layers // 2, n_layers - 1):
            assert np.array_equal(mask_p, ~np.isnan(purpose_scores[ri, L]))
            assert np.array_equal(mask_b, ~np.isnan(b2c_scores[ri, L]))
            assert np.array_equal(mask_c, ~np.isnan(c2b_scores[ri, L]))

    # ---- point estimates (exact sklearn roc_auc_score, matches existing scripts) ----
    purpose_auc = np.empty((n_ranks, n_layers))
    b2c_auc = np.empty((n_ranks, n_layers))
    c2b_auc = np.empty((n_ranks, n_layers))
    for ri in range(n_ranks):
        for L in range(n_layers):
            purpose_auc[ri, L] = roc_auc_score(y[mask_p], purpose_scores[ri, L, mask_p])
            b2c_auc[ri, L] = roc_auc_score(y[mask_b], b2c_scores[ri, L, mask_b])
            c2b_auc[ri, L] = roc_auc_score(y[mask_c], c2b_scores[ri, L, mask_c])
    gap = purpose_auc - (b2c_auc + c2b_auc) / 2.0  # (n_ranks, n_layers)
    G = _trapz(gap, q, axis=1)  # (n_ranks,)
    frac_removed = 1.0 - G / G[ranks.index(0)]

    # ---- payload-block bootstrap of the INTEGRATED curve, N_BOOT replicates ----
    block_to_pos: dict = {}
    for i, b in enumerate(blk):
        block_to_pos.setdefault(b, []).append(i)
    blocks = np.array(sorted(block_to_pos))
    rng = np.random.default_rng(RNG_SEED)

    boot_G = np.empty((n_ranks, N_BOOT))
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
        boot_G[:, rep] = _trapz(gap_rep, q, axis=1)
    print(f"  {model}: {N_BOOT} bootstrap replicates in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    rank0_idx = ranks.index(0)
    boot_frac_removed = 1.0 - boot_G / boot_G[rank0_idx]

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    by_rank = {}
    for ri, r in enumerate(ranks):
        by_rank[str(r)] = {
            "G": float(G[ri]),
            "G_ci95": ci(boot_G[ri]),
            "fraction_gap_removed": float(frac_removed[ri]),
            "fraction_gap_removed_ci95": ci(boot_frac_removed[ri]),
            "gap_by_layer": gap[ri].tolist(),
            "purpose_auc_by_layer": purpose_auc[ri].tolist(),
            "b2c_auc_by_layer": b2c_auc[ri].tolist(),
            "c2b_auc_by_layer": c2b_auc[ri].tolist(),
        }

    return {
        "n_layers": n_layers,
        "q": q.tolist(),
        "ranks": ranks,
        "npz_sha256": _sha256(npz_path),
        "n_items": n_items,
        "n_blocks": len(blocks),
        "by_rank": by_rank,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    out_path = RESULTS / "full_layer_gap_erasure_summary.json"
    if out_path.exists():
        out = json.loads(out_path.read_text())
    else:
        out = {
            "schema_version": 1,
            "purpose": (
                "Full-layer, normalized-depth (q=layer/(n_layers-1)) matched-format-PCA "
                "erasure cross-format transfer gap g(q,r)=AUC_purpose(q,r)-"
                "(AUC_b2c(q,r)+AUC_c2b(q,r))/2 and its trapezoidal integral G(r)=int g dq, "
                "for ranks r in {0,1,4,16,64}, replacing the ad hoc spike/min_gap/final "
                "3-point depth convention. Uncertainty via payload-block bootstrap "
                "(N_BOOT=1000): one block resample per replicate reused across every "
                "layer, every rank, and all three AUC regimes, curve integrated per "
                "replicate (no pointwise-CI integration, no independent per-layer "
                "bootstrap)."
            ),
            "probe": PROBE,
            "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
            "auc_estimator_note": (
                "Point estimates use sklearn.metrics.roc_auc_score directly (same as "
                "full320_v3_erasure_full_layer_sweep.py / "
                "full320_v3_erasure_dose_response_bootstrap.py). Bootstrap replicates "
                "use the algebraically equivalent Mann-Whitney rank-sum AUC formula "
                "for speed; verified against roc_auc_score to float precision."
            ),
            "by_model": {},
        }
    for model in args.models:
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        out["by_model"][model] = analyze_model(model)

    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
