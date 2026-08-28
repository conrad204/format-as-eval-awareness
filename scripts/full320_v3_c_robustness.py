"""Regularization-robustness sweep for the pinned full320-v3 balanced 2x2 probe.

Reuses the exact OOF probe procedure from scripts/sweep_full.py (StandardScaler +
LogisticRegression, 5-fold cue-family-held-out CV, decorrelated/b2c/c2b regimes)
on cached full320-v3 activations, but sweeps the logistic C penalty instead of
pinning it at 0.1. Restricted to the same three depths used throughout this repo
for depth-wise reporting (min-gap layer, final layer, spike layer; see
scripts/format_bound_regrowth_bootstrap.py) to keep memory bounded on a 16GB
workstation with a >15GB (decompressed) per-model activation tensor.

Purpose: verify that the depth-wise purpose AUC (decorrelated regime) and the
cross-format transfer gap (decorrelated AUC minus mean of b2c/c2b AUC) are not
artifacts of the pinned C=0.1 regularization strength. If the sign and rough
magnitude of both quantities hold across three orders of magnitude in C, they
are not regularization artifacts of that one operating point.

Usage: python scripts/full320_v3_c_robustness.py [out_path]
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import numpy.lib.format as npfmt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
# Same per-model depths as scripts/format_bound_regrowth_bootstrap.py, so this
# sweep is directly comparable to the pinned depth-wise results.
MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "min_layer": 14, "final_layer": 31, "spike_layer": 3},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "min_layer": 19, "final_layer": 79, "spike_layer": 4},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "min_layer": 21, "final_layer": 79, "spike_layer": 4},
}
C_GRID = (0.01, 0.1, 1.0, 10.0)  # 0.1 is the pinned scripts/sweep_full.py value
N_JOBS = 6
# The system temp drive is nearly full; point joblib's memmapping cache at the
# repo drive, which has room for the (n_items, hidden) float64 layer slices.
JOBLIB_TEMP_FOLDER = str(Path(__file__).resolve().parent.parent / "tmp_joblib")
Path(JOBLIB_TEMP_FOLDER).mkdir(parents=True, exist_ok=True)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_extract_layers(npz_path: str, layer_indices: list[int]):
    """Stream-decompress X.npy from the npz zip, keeping only requested layers.

    Never materializes the full (n_items, n_layers, hidden) array in memory.
    Copied from scripts/format_bound_regrowth_bootstrap.py (identical contract).
    """
    layer_indices = sorted(layer_indices)
    z = zipfile.ZipFile(npz_path)
    f = z.open("X.npy")
    ver = npfmt.read_magic(f)
    if ver == (1, 0):
        shape, fortran, dtype = npfmt.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = npfmt.read_array_header_2_0(f)
    assert not fortran
    n_items, n_layers, hidden = shape
    assert dtype == np.dtype("float32")
    row_bytes = n_layers * hidden * dtype.itemsize
    out = np.empty((n_items, len(layer_indices), hidden), dtype=np.float32)
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        for j, L in enumerate(layer_indices):
            out[i, j] = row[L]
        if (i + 1) % 2000 == 0:
            print(f"  ...extracted {i + 1}/{n_items} items", file=sys.stderr, flush=True)
    f.close()
    return out, layer_indices


def oof_scores(X, y, fam, tr_mask, te_mask, C):
    s = np.full(len(y), np.nan)
    for fold in sorted(set(fam)):
        trm = (fam != fold) & tr_mask
        tem = (fam == fold) & te_mask
        if not tem.any():
            continue
        sc = StandardScaler()
        m = LogisticRegression(C=C, max_iter=3000, random_state=0)
        m.fit(sc.fit_transform(X[trm]), y[trm])
        s[tem] = m.decision_function(sc.transform(X[tem]))
    return s


def one_layer_C(Xl, y, fam, is_bench, is_casual, ALL, C):
    s_decorr = oof_scores(Xl, y, fam, ALL, ALL, C)
    s_b2c = oof_scores(Xl, y, fam, is_bench, is_casual, C)
    s_c2b = oof_scores(Xl, y, fam, is_casual, is_bench, C)
    auc_decorr = float(roc_auc_score(y, s_decorr))
    m_b2c, m_c2b = ~np.isnan(s_b2c), ~np.isnan(s_c2b)
    auc_b2c = float(roc_auc_score(y[m_b2c], s_b2c[m_b2c]))
    auc_c2b = float(roc_auc_score(y[m_c2b], s_c2b[m_c2b]))
    gap = auc_decorr - (auc_b2c + auc_c2b) / 2
    return {"auc_decorr": auc_decorr, "auc_b2c": auc_b2c, "auc_c2b": auc_c2b, "gap_point": gap}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/full320_v3_c_robustness.json"
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    n_total = len(rows_all)
    # Exclude the frozen no-cue control arm (intended_purpose == "none",
    # label == null): it has no purpose label and cannot enter this AUC.
    # Matches scripts/sweep_full.py and scripts/format_bound_regrowth_bootstrap.py.
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    n_excluded = n_total - len(rows_by_id)
    print(f"stated rows: {len(rows_by_id)} / {n_total} ({n_excluded} no-cue rows excluded)", file=sys.stderr)

    results = {
        "schema_version": 1,
        "purpose": "C-regularization robustness check for the pinned balanced 2x2 purpose probe "
                   "(scripts/sweep_full.py): depth-wise purpose AUC (decorrelated regime) and "
                   "cross-format transfer gap, swept over C, at the pinned min-gap/final/spike depths.",
        "mode": "development",
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": {"path": ITEMS_PATH, "sha256": sha256_file(ITEMS_PATH)},
        "c_grid": list(C_GRID),
        "pinned_c": 0.1,
        "n_total_rows": n_total,
        "n_stated_rows": len(rows_by_id),
        "n_excluded_no_cue_rows": n_excluded,
    }
    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        layer_list = sorted({cfg["min_layer"], cfg["final_layer"], cfg["spike_layer"]})
        X, layer_list = stream_extract_layers(cfg["npz"], layer_list)
        layer_pos = {L: j for j, L in enumerate(layer_list)}

        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id), (model, keep_mask.sum(), len(rows_by_id))
        X = X[keep_mask].astype(np.float64)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        ALL = np.ones(len(kept_rows), bool)
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"

        depths = [cfg["min_layer"], cfg["final_layer"], cfg["spike_layer"]]
        Xl_by_layer = {L: np.ascontiguousarray(X[:, layer_pos[L], :]) for L in depths}
        hidden = Xl_by_layer[depths[0]].shape[1]
        n_jobs = N_JOBS if hidden < 8192 else 4  # cap concurrency: 8192-dim float64
        del X  # the (n_items, 3, hidden) float32 source is no longer needed
        jobs = [(L, C) for L in depths for C in C_GRID]
        outs = Parallel(n_jobs=n_jobs, temp_folder=JOBLIB_TEMP_FOLDER)(
            delayed(one_layer_C)(Xl_by_layer[L], y, fam, is_bench, is_casual, ALL, C)
            for L, C in jobs
        )
        per_layer = {}
        for (L, C), r in zip(jobs, outs):
            per_layer.setdefault(str(L), {})[str(C)] = r
            print(f"  layer {L} C={C}: decorr={r['auc_decorr']:.4f} b2c={r['auc_b2c']:.4f} "
                  f"c2b={r['auc_c2b']:.4f} gap={r['gap_point']:.4f}", file=sys.stderr, flush=True)

        depth_names = {str(cfg["min_layer"]): "min_gap", str(cfg["final_layer"]): "final", str(cfg["spike_layer"]): "spike"}
        results[model] = {"depth_role": depth_names, "by_layer": per_layer}
        del Xl_by_layer

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
