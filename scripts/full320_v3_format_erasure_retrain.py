"""Linear format-erasure + purpose-probe retrain on the pinned full320-v3 activations.

Answers: is the depth-wise purpose signal and cross-format transfer gap from the
balanced 2x2 probe (scripts/sweep_full.py) carried by a linearly-decodable format
subspace, or does it survive once that subspace is removed?

Method (matched_mean, reused unmodified from scripts/format_erasure_development.py,
which is the method that empirically erases format to chance AUC on pilot80 while
leaving purpose AUC near-unchanged -- see results/format_erasure_8b_v2.json):
  For each held-out purpose-family fold f (5 folds, fg1..fg5):
    1. Build the crossed evaluation/deployment x benchmark/casual factorial groups
       from TRAINING rows only (fam != f). The B vector is the format main-effect
       contrast averaged over those groups.
    2. Q = top `rank` left-singular directions of mean(B) (an orthonormal basis for
       the linear format subspace, fit on training data only).
    3. Project Q out of both the training and the held-out fold's activations:
       X' = X - (X @ Q) @ Q.T.
    4. Refit the SAME balanced-2x2 purpose probe (StandardScaler + LogisticRegression
       C=0.1, matching scripts/sweep_full.py) on the erased training activations;
       score the erased held-out fold. Also refit a format probe on the erased
       activations as a sanity check that the transform actually erased format
       (expect AUC ~0.5), and refit the b2c/c2b cross-format transfer regimes on
       the erased activations using the SAME per-fold Q.

The erasure basis Q is fit separately inside each held-out fold using only that
fold's training rows -- fitting one Q on all data (or on data including the held
-out fold) would leak the held-out fold's format labels into the transform and
bias the post-erasure test AUCs upward. This script enforces that by construction
(Q is a function of X[train_mask], rows, train_mask only).

Usage: python scripts/full320_v3_format_erasure_retrain.py [out_path]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, fit_probe, orth_basis, remove

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "min_layer": 14, "final_layer": 31, "spike_layer": 3},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "min_layer": 19, "final_layer": 79, "spike_layer": 4},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "min_layer": 21, "final_layer": 79, "spike_layer": 4},
}
# matched_mean collapses to a single direction regardless of rank (see
# erase_and_score_fold), so its rank grid is just for parity with the
# existing scripts/format_erasure_development.py reporting convention.
# matched_pca does not average first, so rank genuinely controls how many
# format directions are removed; the grid is chosen a priori (doublings from
# 1 to 64, matching the RANKS tuple used elsewhere in this repo) to show
# format AUC's dose-response as more of the format subspace is removed --
# not chosen by looking at held-out purpose AUC.
METHODS_FULL = {"matched_mean": (1, 4), "matched_pca": (1, 4, 16, 64)}
# 70B (8192-dim) fits are ~2x the cost of 8B per call and this workstation is
# memory-constrained (16GB) at n_jobs>2 for that width; use a thinned rank
# grid there (still spans raw vs partial vs heavier erasure) to keep runtime
# bounded. Chosen for cost, not by inspecting held-out AUC.
METHODS_THIN = {"matched_mean": (1,), "matched_pca": (1, 16)}
N_JOBS = 6
JOBLIB_TEMP_FOLDER = str(Path(__file__).resolve().parent.parent / "tmp_joblib")
Path(JOBLIB_TEMP_FOLDER).mkdir(parents=True, exist_ok=True)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_extract_layers(npz_path: str, layer_indices: list[int]):
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


def erase_and_score_fold(Xl, rows, y, fam, fmt_bin, is_bench, is_casual, fold, rank, method):
    train_mask = fam != fold
    test_mask = fam == fold
    # Q is a function of TRAIN rows only -- no held-out-fold information enters
    # the basis. This is the per-fold cross-fitting the task requires.
    _, B, _, _ = factorial_vectors(Xl, rows, train_mask)
    if method == "matched_mean":
        # Average the per-group format contrast first, then take its direction:
        # collapses to a single direction regardless of `rank` (matches the
        # existing scripts/format_erasure_development.py behavior/results).
        Q = orth_basis(B.mean(axis=0), rank)
    elif method == "matched_pca":
        # No averaging: an SVD basis over all per-group format contrasts, so
        # `rank` genuinely controls how many format directions are removed.
        Q = orth_basis(B, rank)
    else:
        raise ValueError(method)
    Xt = remove(Xl, Q)

    out = {}
    s_purpose = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
    out["purpose"] = (y[test_mask], s_purpose)
    s_format = fit_probe(Xt[train_mask], fmt_bin[train_mask], Xt[test_mask])
    out["format"] = (fmt_bin[test_mask], s_format)

    b2c_train = train_mask & is_bench
    b2c_test = test_mask & is_casual
    if b2c_test.any():
        s = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        out["b2c"] = (y[b2c_test], s)
    c2b_train = train_mask & is_casual
    c2b_test = test_mask & is_bench
    if c2b_test.any():
        s = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])
        out["c2b"] = (y[c2b_test], s)
    return fold, rank, out


def aggregate(fold_outs, key):
    ys, ss = [], []
    for out in fold_outs:
        if key in out:
            yy, sc = out[key]
            ys.append(yy)
            ss.append(sc)
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    return float(roc_auc_score(y, s))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/full320_v3_format_erasure_retrain.json"
    only_models = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set(MODELS)
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    n_total = len(rows_all)
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    n_excluded = n_total - len(rows_by_id)
    print(f"stated rows: {len(rows_by_id)} / {n_total} ({n_excluded} no-cue rows excluded)", file=sys.stderr)

    baseline = {}
    baseline_path = Path("results/full320_v3_c_robustness.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())

    existing = json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    results = {
        "schema_version": 2,
        "purpose": "Per-fold cross-fitted linear format-erasure (reused from "
                   "scripts/format_erasure_development.py) then retrain of the pinned balanced "
                   "2x2 purpose probe (scripts/sweep_full.py) on erased activations, at the pinned "
                   "min-gap/final/spike depths. Two erasure bases are compared: matched_mean "
                   "(single averaged format-contrast direction) and matched_pca (un-averaged "
                   "per-group format-contrast basis, rank genuinely varies the amount removed).",
        "mode": "development",
        "methods": {"note": "grid is thinner for 8192-dim models; see per-model 'method_grid' field"},
        "probe": PROBE,
        "leakage_note": "the erasure basis Q is fit per held-out purpose-family fold using only "
                         "that fold's training rows; it is never fit on, or using labels from, "
                         "the held-out fold it is then applied to.",
        "baseline_source": "results/full320_v3_c_robustness.json (C=0.1, the pinned scripts/sweep_full.py "
                            "value), reused rather than recomputed since it is identical fold/layer setup.",
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": {"path": ITEMS_PATH, "sha256": sha256_file(ITEMS_PATH)},
        "n_total_rows": n_total,
        "n_stated_rows": len(rows_by_id),
        "n_excluded_no_cue_rows": n_excluded,
    }
    for model in existing:
        if model not in MODELS or model not in ("llama31_8b", "llama31_70b", "llama33_70b"):
            continue
        if model not in only_models:
            results[model] = existing[model]  # carry over already-computed models unchanged
    for model, cfg in MODELS.items():
        if model not in only_models:
            continue
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
        fmt_bin = (fmt == "benchmark").astype(int)
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"
        folds = sorted(set(fam))

        depths = [cfg["min_layer"], cfg["final_layer"], cfg["spike_layer"]]
        Xl_by_layer = {L: np.ascontiguousarray(X[:, layer_pos[L], :]) for L in depths}
        hidden = Xl_by_layer[depths[0]].shape[1]
        methods = METHODS_FULL if hidden < 8192 else METHODS_THIN
        n_jobs = N_JOBS if hidden < 8192 else 2  # remove() allocates a full-size copy per call
        del X  # the (n_items, 3, hidden) float32 source is no longer needed
        jobs = [(L, method, rank, fold) for L in depths for method, ranks in methods.items()
                for rank in ranks for fold in folds]
        outs = Parallel(n_jobs=n_jobs, temp_folder=JOBLIB_TEMP_FOLDER)(
            delayed(erase_and_score_fold)(
                Xl_by_layer[L], kept_rows, y, fam, fmt_bin, is_bench, is_casual, fold, rank, method)
            for L, method, rank, fold in jobs
        )
        by_layer_method_rank = {}
        for (L, method, rank, fold), (fold_out, rank_out, fold_result) in zip(jobs, outs):
            by_layer_method_rank.setdefault(str(L), {}).setdefault(method, {}).setdefault(
                str(rank), []).append(fold_result)

        depth_names = {str(cfg["min_layer"]): "min_gap", str(cfg["final_layer"]): "final", str(cfg["spike_layer"]): "spike"}
        model_out = {"depth_role": depth_names, "method_grid": {m: list(rs) for m, rs in methods.items()}, "by_layer": {}}
        model_baseline = baseline.get(model, {}).get("by_layer", {})
        for L in depths:
            layer_out = {}
            b = model_baseline.get(str(L), {}).get("0.1")
            if b:
                layer_out["raw_baseline"] = {
                    "purpose_auc": b["auc_decorr"], "b2c_auc": b["auc_b2c"],
                    "c2b_auc": b["auc_c2b"], "gap": b["gap_point"],
                }
            for method, ranks in methods.items():
                layer_out[method] = {}
                for rank in ranks:
                    fold_results = by_layer_method_rank[str(L)][method][str(rank)]
                    purpose_auc = aggregate(fold_results, "purpose")
                    format_auc = aggregate(fold_results, "format")
                    b2c_auc = aggregate(fold_results, "b2c")
                    c2b_auc = aggregate(fold_results, "c2b")
                    gap = purpose_auc - (b2c_auc + c2b_auc) / 2
                    layer_out[method][str(rank)] = {
                        "purpose_auc_after_erasure": purpose_auc,
                        "format_auc_after_erasure": format_auc,
                        "b2c_auc_after_erasure": b2c_auc,
                        "c2b_auc_after_erasure": c2b_auc,
                        "gap_after_erasure": gap,
                    }
                    print(f"  layer {L} {method} rank {rank}: purpose={purpose_auc:.4f} "
                          f"format(sanity)={format_auc:.4f} b2c={b2c_auc:.4f} c2b={c2b_auc:.4f} "
                          f"gap={gap:.4f}", file=sys.stderr, flush=True)
            model_out["by_layer"][str(L)] = layer_out
        results[model] = model_out
        del Xl_by_layer

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
