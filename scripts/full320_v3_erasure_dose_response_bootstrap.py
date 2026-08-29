"""Block-bootstrap CIs for the matched-format-PCA erasure dose-response
(purpose AUC and cross-format transfer gap vs. rank), all 3 checkpoints, all 3
pinned depths, ranks {0 (raw, no erasure), 1, 4, 16, 64}.

Designed to run on the uahpc cluster (32 cores, 125GB RAM -- no memory-driven
n_jobs throttling needed, unlike the original 16GB workstation runs).

Stage 1: for each (model, depth, rank), 5-fold held-out-purpose-family
cross-fitted OOF scoring (StandardScaler + LogisticRegression(C=0.1), matching
scripts/sweep_full.py / scripts/full320_v3_format_erasure_retrain.py exactly),
caching per-item (payload_block_id, y_true, score) for the purpose/b2c/c2b
regimes. rank=0 means no erasure (this is the "raw" baseline point, computed
via the same probe/fold code path for full internal consistency, rather than
re-using the separate results/full320_v3_c_robustness.json numbers).

Stage 2: payload-block bootstrap (resample the ~320 blocks with replacement,
N_BOOT=1000, matching scripts/format_bound_regrowth_bootstrap.py's convention)
of the cached OOF scores -- no refitting, so this stage is cheap once stage 1
is cached.

Point estimates from stage 1 are printed for a manual sanity cross-check
against results/full320_v3_format_erasure_retrain.json / c_robustness.json
(should match closely; small differences possible since the "raw" point here
is computed by this script's own fold code, not re-used from the other file).

Usage: python scripts/full320_v3_erasure_dose_response_bootstrap.py [out_json]
"""
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import numpy.lib.format as npfmt
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, fit_probe, orth_basis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": {"npz": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
                    "min_layer": 14, "final_layer": 31, "spike_layer": 3},
    "llama31_70b": {"npz": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
                     "min_layer": 19, "final_layer": 79, "spike_layer": 4},
    "llama33_70b": {"npz": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
                     "min_layer": 21, "final_layer": 79, "spike_layer": 4},
}
RANKS = (0, 1, 4, 16, 64)
N_BOOT = 1000
N_JOBS = int(os.environ.get("N_JOBS", "20"))
RNG_SEED = 0


def remove(X, Q):
    return X if Q is None or Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def stream_extract_layers(npz_path, layer_indices):
    layer_indices = sorted(layer_indices)
    z = zipfile.ZipFile(npz_path)
    f = z.open("X.npy")
    ver = npfmt.read_magic(f)
    if ver == (1, 0):
        shape, fortran, dtype = npfmt.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = npfmt.read_array_header_2_0(f)
    assert not fortran and dtype == np.dtype("float32")
    n_items, n_layers, hidden = shape
    row_bytes = n_layers * hidden * dtype.itemsize
    out = np.empty((n_items, len(layer_indices), hidden), dtype=np.float32)
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        for j, L in enumerate(layer_indices):
            out[i, j] = row[L]
        if (i + 1) % 4000 == 0:
            print(f"  ...{i + 1}/{n_items}", file=sys.stderr, flush=True)
    f.close()
    return out, layer_indices


def score_fold(Xl, rows, y, fam, is_bench, is_casual, fold, rank):
    train_mask = fam != fold
    test_mask = fam == fold
    if rank > 0:
        _, B, _, _ = factorial_vectors(Xl, rows, train_mask)
        Q = orth_basis(B, rank)
    else:
        Q = None
    Xt = remove(Xl, Q)

    out = {}
    s = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
    out["purpose"] = (test_mask.nonzero()[0], s)

    b2c_train, b2c_test = train_mask & is_bench, test_mask & is_casual
    if b2c_test.any():
        s = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        out["b2c"] = (b2c_test.nonzero()[0], s)
    c2b_train, c2b_test = train_mask & is_casual, test_mask & is_bench
    if c2b_test.any():
        s = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])
        out["c2b"] = (c2b_test.nonzero()[0], s)
    return fold, rank, out


def block_bootstrap_auc(y, s, blk, blocks, n_boot, rng, idx_mask=None):
    if idx_mask is not None:
        keep = idx_mask
        y, s, blk = y[keep], s[keep], blk[keep]
    block_to_pos = {}
    for i, b in enumerate(blk):
        block_to_pos.setdefault(b, []).append(i)
    out = np.empty(n_boot)
    for r in range(n_boot):
        sample_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([block_to_pos[b] for b in sample_blocks if b in block_to_pos])
        out[r] = roc_auc_score(y[idx], s[idx])
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "full320_v3_erasure_dose_response_bootstrap.json")
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    result = {"schema_version": 1, "mode": "development",
              "purpose": "Payload-block-bootstrapped (N_BOOT=1000) CIs for purpose AUC and "
                         "cross-format transfer gap vs matched-format-PCA erasure rank, all 3 "
                         "checkpoints, 3 pinned depths, ranks 0(raw)/1/4/16/64. Computed on uahpc.",
              "n_boot": N_BOOT, "ranks": list(RANKS), "probe": PROBE, "by_model": {}}

    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        depths = {"spike": cfg["spike_layer"], "min_gap": cfg["min_layer"], "final": cfg["final_layer"]}
        X, layer_list = stream_extract_layers(cfg["npz"], sorted(set(depths.values())))
        layer_pos = {L: j for j, L in enumerate(layer_list)}
        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        X = X[keep_mask].astype(np.float64)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        blk = np.array([r["payload_block_id"] for r in kept_rows])
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"
        folds = sorted(set(fam))
        blocks = np.array(sorted(set(blk)))

        model_out = {"depth_role": {str(v): k for k, v in depths.items()}, "by_depth": {}}
        for depth, L in depths.items():
            Xl = np.ascontiguousarray(X[:, layer_pos[L], :])
            print(f"  --- {depth} (layer {L}) ---", file=sys.stderr, flush=True)

            jobs = [(rank, fold) for rank in RANKS for fold in folds]
            outs = Parallel(n_jobs=N_JOBS)(
                delayed(score_fold)(Xl, kept_rows, y, fam, is_bench, is_casual, fold, rank)
                for rank, fold in jobs
            )
            by_rank = {r: {"purpose": [np.full(len(y), np.nan)],
                           "b2c": [np.full(len(y), np.nan)],
                           "c2b": [np.full(len(y), np.nan)]} for r in RANKS}
            for (fold, rank, fold_out) in outs:
                for regime in ("purpose", "b2c", "c2b"):
                    if regime in fold_out:
                        idx, s = fold_out[regime]
                        by_rank[rank][regime][0][idx] = s

            depth_out = {}
            rng = np.random.default_rng(RNG_SEED)
            for rank in RANKS:
                s_purpose = by_rank[rank]["purpose"][0]
                s_b2c = by_rank[rank]["b2c"][0]
                s_c2b = by_rank[rank]["c2b"][0]
                m_p, m_b, m_c = ~np.isnan(s_purpose), ~np.isnan(s_b2c), ~np.isnan(s_c2b)
                purpose_auc = float(roc_auc_score(y[m_p], s_purpose[m_p]))
                b2c_auc = float(roc_auc_score(y[m_b], s_b2c[m_b]))
                c2b_auc = float(roc_auc_score(y[m_c], s_c2b[m_c]))
                gap = purpose_auc - (b2c_auc + c2b_auc) / 2

                boot_purpose = block_bootstrap_auc(y, s_purpose, blk, blocks, N_BOOT, rng, idx_mask=m_p)
                boot_b2c = block_bootstrap_auc(y, s_b2c, blk, blocks, N_BOOT, rng, idx_mask=m_b)
                boot_c2b = block_bootstrap_auc(y, s_c2b, blk, blocks, N_BOOT, rng, idx_mask=m_c)
                boot_gap = boot_purpose - (boot_b2c + boot_c2b) / 2

                def ci(a):
                    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

                depth_out[str(rank)] = {
                    "purpose_auc": purpose_auc, "purpose_auc_ci95": ci(boot_purpose),
                    "b2c_auc": b2c_auc, "c2b_auc": c2b_auc, "gap": gap, "gap_ci95": ci(boot_gap),
                }
                print(f"    rank={rank}: purpose={purpose_auc:.4f} CI={ci(boot_purpose)} "
                      f"gap={gap:.4f} CI={ci(boot_gap)}", file=sys.stderr, flush=True)
            model_out["by_depth"][depth] = depth_out
        result["by_model"][model] = model_out
        del X

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
