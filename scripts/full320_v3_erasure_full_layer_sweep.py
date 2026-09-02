"""Full-layer-depth matched-format-PCA erasure sweep (single rank, all layers),
replacing the ad hoc "spike/min_gap/final" 3-point convention with the actual
continuous depth trajectory. All 3 checkpoints, every layer, rank=64
(the strongest erasure point already validated at 3 depths).

Raw (no-erasure) baseline per layer is NOT recomputed here -- it is already
pinned per-layer in results/layer_sweep_full320-v3_<model>.json and merged in
at plotting time. This script only computes the post-erasure quantities.

Designed for the uahpc cluster (32 cores, ~150G mem): loads the full
(n_items, n_layers, hidden) float32 tensor per model directly (no streaming
row-by-row extraction, since every layer is needed anyway) and keeps
computation in float32 to bound memory (70B: ~30GB for the full tensor).

Usage: python scripts/full320_v3_erasure_full_layer_sweep.py [out_json]
"""
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, fit_probe, orth_basis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
    "llama31_70b": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
    "llama33_70b": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
}
RANK = 64
N_JOBS = int(os.environ.get("N_JOBS", "28"))


def remove(X, Q):
    return X if Q is None or Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def score_layer(Xl, rows, y, fam, is_bench, is_casual, rank):
    """5-fold cross-fitted matched_pca erasure + retrain at one layer. Returns
    pooled purpose/b2c/c2b AUC and the sanity-check format AUC."""
    folds = sorted(set(fam))
    s_purpose = np.full(len(y), np.nan)
    s_format = np.full(len(y), np.nan)
    s_b2c = np.full(len(y), np.nan)
    s_c2b = np.full(len(y), np.nan)
    fmt_bin = is_bench.astype(int)
    for fold in folds:
        train_mask = fam != fold
        test_mask = fam == fold
        _, B, _, _ = factorial_vectors(Xl, rows, train_mask)
        Q = orth_basis(B, rank)
        Xt = remove(Xl, Q)

        s_purpose[test_mask] = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
        s_format[test_mask] = fit_probe(Xt[train_mask], fmt_bin[train_mask], Xt[test_mask])
        b2c_train, b2c_test = train_mask & is_bench, test_mask & is_casual
        if b2c_test.any():
            s_b2c[b2c_test] = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        c2b_train, c2b_test = train_mask & is_casual, test_mask & is_bench
        if c2b_test.any():
            s_c2b[c2b_test] = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])

    m_p, m_f, m_b, m_c = ~np.isnan(s_purpose), ~np.isnan(s_format), ~np.isnan(s_b2c), ~np.isnan(s_c2b)
    purpose_auc = float(roc_auc_score(y[m_p], s_purpose[m_p]))
    format_auc = float(roc_auc_score(fmt_bin[m_f], s_format[m_f]))
    b2c_auc = float(roc_auc_score(y[m_b], s_b2c[m_b]))
    c2b_auc = float(roc_auc_score(y[m_c], s_c2b[m_c]))
    gap = purpose_auc - (b2c_auc + c2b_auc) / 2
    return {"purpose_auc_after_erasure": purpose_auc, "format_auc_after_erasure": format_auc,
            "b2c_auc_after_erasure": b2c_auc, "c2b_auc_after_erasure": c2b_auc, "gap_after_erasure": gap}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "full320_v3_erasure_full_layer_sweep.json")
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    result = {"schema_version": 1, "mode": "development",
              "purpose": "Full-layer-depth matched-format-PCA erasure sweep at rank=64, "
                         "replacing the 3-point spike/min_gap/final convention with the "
                         "continuous depth trajectory. Raw baseline is NOT recomputed here -- "
                         "merge with results/layer_sweep_full320-v3_<model>.json at plot time.",
              "rank": RANK, "by_model": {}}

    for model, npz_path in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        z = zipfile.ZipFile(npz_path)
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        print("  loading full tensor...", file=sys.stderr, flush=True)
        X = np.load(npz_path)["X"][keep_mask]  # (n_kept, n_layers, hidden) float32
        n_layers = X.shape[1]
        print(f"  loaded {X.shape}", file=sys.stderr, flush=True)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"

        outs = Parallel(n_jobs=N_JOBS)(
            delayed(score_layer)(np.ascontiguousarray(X[:, L, :]), kept_rows, y, fam, is_bench, is_casual, RANK)
            for L in range(n_layers)
        )
        by_layer = {str(L): outs[L] for L in range(n_layers)}
        result["by_model"][model] = {"n_layers": n_layers, "by_layer": by_layer}
        for L in range(n_layers):
            r = outs[L]
            print(f"  layer {L}: purpose={r['purpose_auc_after_erasure']:.4f} "
                  f"gap={r['gap_after_erasure']:.4f} format(sanity)={r['format_auc_after_erasure']:.4f}",
                  file=sys.stderr, flush=True)
        del X

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
