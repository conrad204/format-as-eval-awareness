"""Descriptive frozen-activation geometry: does the matched-format PCA subspace that
rescues cross-format transfer (scripts/full320_v3_format_erasure_retrain.py,
scripts/full320_v3_null_projection_control.py) contain little purpose-decoding
energy, while itself sitting mostly inside the dominant high-variance activation
subspace? Run across all three checkpoints and the three pinned depths per model.

Everything here is a single closed-form vector/subspace computation from the full
stated-row set -- no cross-validation, no repetitions, no forward pass through any
model. This is pure linear algebra on cached activations. It is NOT a causal,
steering, or mechanistic-circuit claim, and near-zero cosines between two
essentially random high-dimensional directions are NOT by themselves surprising
(two independent random unit vectors in R^4096 have cosine ~ N(0, 1/4096), i.e.
std ~0.016) -- the substantive quantities here are the subspace ENERGY fractions
(rank-r squared-projection norms), not the pairwise single-direction cosines.

Matched-format PCA basis Q_format(r): orth_basis(B, r) where B is the per-group
evaluation/deployment x benchmark/casual contrast (scripts/format_erasure_development.py
factorial_vectors), exactly the "matched_pca" method used by the erasure retrain
script -- but fit on ALL stated rows here (descriptive geometry), not per held-out
fold, since no probe is scored from it.

For each model x depth x rank r in {1,4,16,64}:
  E_purpose_in_format(r)  = ||Q_format(r)^T w_purpose||^2 / ||w_purpose||^2
      (w_purpose is unit norm, so this is just ||Q_format(r)^T w_purpose||^2)
  frac_format_subspace_from_purpose(r) = E_purpose_in_format(r) / r
      ("reverse" view: average, over the r orthonormal basis directions, of how
      much each one points along w_purpose)
  frac_format_subspace_in_top64(r) = ||V64 @ Q_format(r)||_F^2 / r
      (average energy of the r format-subspace directions inside the top-64 PCs)

Per model x depth (rank-independent):
  purpose_energy_in_top64 = ||V64 @ w_purpose||^2
  pairwise cosines among: matched_mean rank-1 direction, top-1 PC, purpose probe
  direction, format probe direction (kept from the prior single-model pass, for
  continuity -- reported but not treated as remarkable on their own).

Usage: python scripts/full320_v3_direction_geometry.py [out_json] [out_csv]
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import numpy.lib.format as npfmt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, orth_basis  # noqa: E402

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
# Same per-model depths used throughout this repo's depth-wise reporting
# (scripts/format_bound_regrowth_bootstrap.py, scripts/full320_v3_c_robustness.py).
MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz",
                    "depths": {"spike": 3, "min_gap": 14, "final": 31}},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz",
                     "depths": {"spike": 4, "min_gap": 19, "final": 79}},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz",
                     "depths": {"spike": 4, "min_gap": 21, "final": 79}},
}
RANKS = (1, 4, 16, 64)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def plain_direction(X, target):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(**PROBE).fit(sc.transform(X), target)
    q = m.coef_[0] / sc.scale_  # map back to raw activation space
    return unit(q)


def cos(a, b):
    return float(np.dot(a, b))


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else "results/full320_v3_direction_geometry.json"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "results/full320_v3_direction_geometry.csv"

    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}

    result = {
        "schema_version": 2,
        "mode": "development",
        "purpose": "Descriptive frozen-activation geometry: fraction of the purpose-probe "
                   "direction's energy inside the exact matched-format-PCA subspace used by "
                   "the erasure experiment, at ranks 1/4/16/64, across all 3 checkpoints and "
                   "the 3 pinned depths per model; plus how much of that same subspace sits "
                   "inside the dominant top-64-PC activation subspace. No causal or "
                   "interventional claim -- closed-form linear algebra on cached activations.",
        "ranks": list(RANKS),
        "by_model": {},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": {"path": ITEMS_PATH, "sha256": sha256_file(ITEMS_PATH)},
    }
    csv_rows = []

    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        depths = cfg["depths"]
        X, layer_list = stream_extract_layers(cfg["npz"], sorted(set(depths.values())))
        layer_pos = {L: j for j, L in enumerate(layer_list)}
        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        X = X[keep_mask].astype(np.float64)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        fmt_bin = (fmt == "benchmark").astype(int)
        ALL = np.ones(len(kept_rows), bool)

        model_out = {"by_depth": {}}
        for name, L in depths.items():
            Xl = X[:, layer_pos[L], :]
            print(f"  --- {name} (layer {L}) ---", file=sys.stderr, flush=True)

            _, B, _, _ = factorial_vectors(Xl, kept_rows, ALL)
            matched_mean_dir = unit(B.mean(axis=0))  # rank-1 "matched_mean" direction, for continuity

            mean = Xl.mean(axis=0)
            _, _, vt64 = randomized_svd(Xl - mean, n_components=64, random_state=0)
            top_pc_dir = unit(vt64[0])

            purpose_dir = plain_direction(Xl, y)
            format_dir = plain_direction(Xl, fmt_bin)
            purpose_energy_in_top64 = float(np.sum((vt64 @ purpose_dir) ** 2))

            depth_out = {
                "layer": L,
                # kept for continuity; NOT treated as remarkable on their own (see module docstring)
                "cos_matched_mean_vs_top_pc": cos(matched_mean_dir, top_pc_dir),
                "cos_matched_mean_vs_purpose": cos(matched_mean_dir, purpose_dir),
                "cos_top_pc_vs_purpose": cos(top_pc_dir, purpose_dir),
                "cos_purpose_vs_format": cos(purpose_dir, format_dir),
                "purpose_energy_in_top64_pcs": purpose_energy_in_top64,
                "by_rank": {},
            }
            for r in RANKS:
                Qr = orth_basis(B, r)  # exact matched_pca basis, rank r (orthonormal, d x r)
                rank_eff = Qr.shape[1]  # orth_basis may return fewer columns if B has lower rank
                e_purpose_in_format = float(np.sum((Qr.T @ purpose_dir) ** 2))
                frac_format_from_purpose = e_purpose_in_format / rank_eff if rank_eff else float("nan")
                frac_format_in_top64 = float(np.sum((vt64 @ Qr) ** 2)) / rank_eff if rank_eff else float("nan")
                row = {
                    "E_purpose_in_format_subspace": e_purpose_in_format,
                    "frac_format_subspace_energy_from_purpose": frac_format_from_purpose,
                    "frac_format_subspace_in_top64_pcs": frac_format_in_top64,
                    "effective_rank": rank_eff,
                }
                depth_out["by_rank"][str(r)] = row
                csv_rows.append({
                    "model": model, "depth": name, "layer": L, "rank": r, "effective_rank": rank_eff,
                    "E_purpose_in_format_subspace": e_purpose_in_format,
                    "frac_format_subspace_energy_from_purpose": frac_format_from_purpose,
                    "frac_format_subspace_in_top64_pcs": frac_format_in_top64,
                    "purpose_energy_in_top64_pcs": purpose_energy_in_top64,
                })
                print(f"    rank {r}: E_purpose_in_format={e_purpose_in_format:.4f} "
                      f"frac_format_in_top64={frac_format_in_top64:.4f}", file=sys.stderr, flush=True)

            model_out["by_depth"][name] = depth_out
        result["by_model"][model] = model_out
        del X

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
