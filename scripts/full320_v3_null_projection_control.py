"""Null-projection control for the matched-format low-rank erasure result.

Single condition (by explicit scope decision): llama31_8b, spike layer (L3),
rank 64 -- the setting with the largest observed transfer-gap rescue in
results/full320_v3_format_erasure_retrain.json. Question: is the gap rescue
specific to the matched-format direction, or does removing ANY rank-64
subspace rescue the gap about as well?

Two null families, same cross-fitting protocol as scripts/sweep_full.py /
scripts/full320_v3_format_erasure_retrain.py (5 held-out purpose-family
folds, StandardScaler + LogisticRegression(C=0.1, max_iter=3000)):

  random_orthogonal: a fresh random rank-64 orthonormal subspace per
    repetition (QR of a Gaussian matrix). Data-independent, so the same
    draw is reused across all 5 folds within a repetition -- no leakage
    risk since it never touches labels or activations to construct.

  top_pca: the top-64 principal components of the TRAINING fold's
    activations (no label information), recomputed separately inside each
    fold (this one *is* data-dependent, so it is fit train-only per fold,
    matching the leakage discipline used for the matched-format basis).

For each repetition/null-type we recompute the OOF purpose AUC, b2c AUC,
c2b AUC, and gap = purpose_auc - mean(b2c_auc, c2b_auc), exactly as for the
matched condition, then report the empirical null distribution of
Delta_G = G_raw - G_projected, its percentile placement of the observed
matched Delta_G, an empirical one-sided p-value (fraction of null
Delta_G >= observed Delta_G), and the purpose-AUC change distribution.

Usage: python scripts/full320_v3_null_projection_control.py [n_reps] [out_path]
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
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, fit_probe  # noqa: E402

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
NPZ_PATH = "v3-activations/activations/llama31_8b/activations_all_layers.npz"
LAYER = 3   # 8B spike layer, largest observed rescue
RANK = 64   # 8B's largest tested matched_pca rank
N_JOBS = 6
JOBLIB_TEMP_FOLDER = str(Path(__file__).resolve().parent.parent / "tmp_joblib")
Path(JOBLIB_TEMP_FOLDER).mkdir(parents=True, exist_ok=True)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_extract_one_layer(npz_path: str, layer: int):
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
    out = np.empty((n_items, hidden), dtype=np.float32)
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        out[i] = row[layer]
        if (i + 1) % 2000 == 0:
            print(f"  ...extracted {i + 1}/{n_items} items", file=sys.stderr, flush=True)
    f.close()
    return out


def remove(X, Q):
    return X if Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def random_orthogonal(d: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(G)
    return Q


def top_pca_basis(Xtr: np.ndarray, rank: int, seed: int) -> np.ndarray:
    mean = Xtr.mean(axis=0)
    _, _, vt = randomized_svd(Xtr - mean, n_components=rank, random_state=seed)
    return vt.T  # (d, rank), orthonormal columns


def cross_fitted_metrics(Xl, y, fam, is_bench, is_casual, Q_or_none, per_fold_pca_rank, seed):
    """Q_or_none: fixed (d, rank) basis shared across folds (random-orthogonal case),
    or None to trigger per-fold PCA fit on that fold's training rows."""
    folds = sorted(set(fam))
    s_purpose = np.full(len(y), np.nan)
    s_b2c = np.full(len(y), np.nan)
    s_c2b = np.full(len(y), np.nan)
    for fi, fold in enumerate(folds):
        train_mask = fam != fold
        test_mask = fam == fold
        if Q_or_none is not None:
            Q = Q_or_none
        else:
            Q = top_pca_basis(Xl[train_mask], per_fold_pca_rank, seed=seed * 1000 + fi)
        Xt = remove(Xl, Q)
        s_purpose[test_mask] = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
        b2c_train, b2c_test = train_mask & is_bench, test_mask & is_casual
        if b2c_test.any():
            s_b2c[b2c_test] = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        c2b_train, c2b_test = train_mask & is_casual, test_mask & is_bench
        if c2b_test.any():
            s_c2b[c2b_test] = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])
    purpose_auc = float(roc_auc_score(y, s_purpose))
    m_b2c, m_c2b = ~np.isnan(s_b2c), ~np.isnan(s_c2b)
    b2c_auc = float(roc_auc_score(y[m_b2c], s_b2c[m_b2c]))
    c2b_auc = float(roc_auc_score(y[m_c2b], s_c2b[m_c2b]))
    gap = purpose_auc - (b2c_auc + c2b_auc) / 2
    return {"purpose_auc": purpose_auc, "b2c_auc": b2c_auc, "c2b_auc": c2b_auc, "gap": gap}


def one_rep(null_type, rep, Xl, y, fam, is_bench, is_casual, hidden, rank):
    rng = np.random.default_rng(10_000 + rep)
    if null_type == "random_orthogonal":
        Q = random_orthogonal(hidden, rank, rng)
        return cross_fitted_metrics(Xl, y, fam, is_bench, is_casual, Q, None, rep)
    else:
        return cross_fitted_metrics(Xl, y, fam, is_bench, is_casual, None, rank, rep)


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/full320_v3_null_projection_control.json"

    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    X = stream_extract_one_layer(NPZ_PATH, LAYER)
    z = zipfile.ZipFile(NPZ_PATH)
    raw_ids = np.load(z.open("item_ids.npy"))
    keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
    assert keep_mask.sum() == len(rows_by_id)
    Xl = X[keep_mask].astype(np.float64)
    del X
    kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
    y = np.array([int(r["label"]) for r in kept_rows])
    fam = np.array([r["purpose_family_id"] for r in kept_rows])
    fmt = np.array([r["format"] for r in kept_rows])
    is_bench, is_casual = fmt == "benchmark", fmt == "casual"
    hidden = Xl.shape[1]

    baseline = json.loads(Path("results/full320_v3_c_robustness.json").read_text())
    raw = baseline["llama31_8b"]["by_layer"][str(LAYER)]["0.1"]
    G_raw = raw["gap_point"]
    purpose_raw = raw["auc_decorr"]

    erasure = json.loads(Path("results/full320_v3_format_erasure_retrain.json").read_text())
    matched = erasure["llama31_8b"]["by_layer"][str(LAYER)]["matched_pca"][str(RANK)]
    G_matched = matched["gap_after_erasure"]
    purpose_matched = matched["purpose_auc_after_erasure"]
    delta_g_matched = G_raw - G_matched
    delta_purpose_matched = purpose_matched - purpose_raw

    print(f"observed: G_raw={G_raw:.4f} purpose_raw={purpose_raw:.4f} "
          f"G_matched={G_matched:.4f} purpose_matched={purpose_matched:.4f} "
          f"Delta_G_matched={delta_g_matched:.4f}", file=sys.stderr, flush=True)

    jobs = [(nt, r) for nt in ("random_orthogonal", "top_pca") for r in range(n_reps)]
    outs = Parallel(n_jobs=N_JOBS, temp_folder=JOBLIB_TEMP_FOLDER)(
        delayed(one_rep)(nt, r, Xl, y, fam, is_bench, is_casual, hidden, RANK)
        for nt, r in jobs
    )

    by_type = {"random_orthogonal": [], "top_pca": []}
    for (nt, r), res in zip(jobs, outs):
        res["delta_g"] = G_raw - res["gap"]
        res["delta_purpose"] = res["purpose_auc"] - purpose_raw
        by_type[nt].append(res)
        print(f"  {nt} rep {r}: gap={res['gap']:.4f} delta_g={res['delta_g']:.4f} "
              f"purpose_auc={res['purpose_auc']:.4f}", file=sys.stderr, flush=True)

    summary = {}
    for nt, reps in by_type.items():
        dgs = np.array([x["delta_g"] for x in reps])
        dps = np.array([x["delta_purpose"] for x in reps])
        percentile_of_observed = float(np.mean(dgs <= delta_g_matched) * 100)
        p_value_one_sided = float(np.mean(dgs >= delta_g_matched))  # P(null rescues >= matched)
        summary[nt] = {
            "n_reps": len(reps),
            "delta_g_null_mean": float(dgs.mean()),
            "delta_g_null_std": float(dgs.std()),
            "delta_g_null_min_max": [float(dgs.min()), float(dgs.max())],
            "delta_purpose_null_mean": float(dps.mean()),
            "delta_purpose_null_std": float(dps.std()),
            "observed_delta_g_percentile_within_null": percentile_of_observed,
            "empirical_p_null_delta_g_geq_observed": p_value_one_sided,
        }

    result = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "Null-projection control: does removing a random or top-variance rank-64 "
                   "subspace rescue the b2c/c2b transfer gap as well as the matched-format "
                   "basis does, at llama31_8b spike layer (L3)?",
        "condition": {"model": "llama31_8b", "layer": LAYER, "rank": RANK, "n_reps": n_reps},
        "protocol": "same 5-fold held-out purpose-family cross-fitting and probe "
                    "(StandardScaler + LogisticRegression) as scripts/sweep_full.py / "
                    "scripts/full320_v3_format_erasure_retrain.py; random_orthogonal is one "
                    "shared data-independent draw per repetition, top_pca is refit per fold "
                    "on that fold's training rows only (no leakage).",
        "probe": PROBE,
        "observed": {
            "G_raw": G_raw, "purpose_auc_raw": purpose_raw,
            "G_matched_pca_rank64": G_matched, "purpose_auc_matched_pca_rank64": purpose_matched,
            "delta_g_matched": delta_g_matched, "delta_purpose_matched": delta_purpose_matched,
        },
        "null_summary": summary,
        "reps": {nt: reps for nt, reps in by_type.items()},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": {"path": ITEMS_PATH, "sha256": sha256_file(ITEMS_PATH)},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    for nt, s in summary.items():
        print(f"{nt}: observed Delta_G={delta_g_matched:.4f} vs null mean={s['delta_g_null_mean']:.4f} "
              f"std={s['delta_g_null_std']:.4f} percentile={s['observed_delta_g_percentile_within_null']:.1f} "
              f"p={s['empirical_p_null_delta_g_geq_observed']:.3f}")


if __name__ == "__main__":
    main()
