"""Apples-to-apples version of the high-variance-subspace conditional null.

The matched-format PCA basis Q_format(r) is only approximately contained in the
empirical top-64 PC subspace (92-100% energy at ranks <=16, per
results/full320_v3_direction_geometry.json), whereas the conditional-null draws
in scripts/full320_v3_conditional_hv_null.py are subspaces drawn entirely INSIDE
that top-64 subspace. This script makes the comparison apples-to-apples: project
Q_format(r) into the top-64 PC subspace, re-orthonormalize, and compare the
PROJECTED matched-format subspace's captured purpose energy against the same
conditional null.

This requires the actual purpose-direction coordinates within the top-64 PC
subspace (not just their norm, unlike the previous conditional-null script,
which only needed ||c||^2 thanks to rotational invariance against a RANDOM
comparison subspace -- here the comparison subspace is a SPECIFIC, data-derived
one, so the real alignment matters). That vector was not persisted, so this one
extraction pass is unavoidable; it reproduces the identical deterministic
purpose_dir / top-64-PC pipeline already used in
scripts/full320_v3_direction_geometry.py (same code, same random_state=0).

For each model x depth x rank r in {1,4,16}:
  1. Q_format(r) = orth_basis(B, r)              (exact matched_pca basis, d x r)
  2. coords = V64 @ Q_format(r)                   (64 x r, format basis in PC coords)
  3. Q_proj = re-orthonormalized column space of coords (64 x keep, keep <= r)
  4. observed = ||Q_proj^T c||^2  where c = V64 @ w_purpose (real 64-dim vector)
  5. conditional null: 10k draws of random rank-keep orthonormal subspaces of R^64,
     using the SAME real c (not a norm-matched stand-in -- we have the real vector
     now, so use it directly; this also matches results within rounding of the
     rotational-invariance shortcut used previously, which was mathematically
     exact regardless).
"""
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
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, orth_basis  # noqa: E402

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz",
                    "depths": {"spike": 3, "min_gap": 14, "final": 31}},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz",
                     "depths": {"spike": 4, "min_gap": 19, "final": 79}},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz",
                     "depths": {"spike": 4, "min_gap": 21, "final": 79}},
}
RANKS = (1, 4, 16)
N_DRAWS = 10_000
SEED = 0


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
    q = m.coef_[0] / sc.scale_
    return unit(q)


def reorthonormalize(coords_64_by_r):
    """Column space of a 64 x r matrix -> orthonormal 64 x keep basis (keep <= r),
    dropping near-zero singular directions exactly as orth_basis() does."""
    U, s, _ = np.linalg.svd(coords_64_by_r, full_matrices=False)
    if len(s) == 0 or not np.any(np.isfinite(s)) or s[0] <= 0:
        return U[:, :0]
    keep = int(np.sum(s > max(s[0], 1.0) * 1e-10))
    return U[:, :keep]


def conditional_draws(c, rank, n_draws, rng):
    d = len(c)
    out = np.empty(n_draws)
    for i in range(n_draws):
        G = rng.standard_normal((d, rank))
        R, _ = np.linalg.qr(G)
        out[i] = np.sum((R.T @ c) ** 2)
    return out


def main():
    out_path = "results/full320_v3_projected_basis_conditional_null.json"
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    rng = np.random.default_rng(SEED)

    cells = []
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
        ALL = np.ones(len(kept_rows), bool)

        for depth, L in depths.items():
            Xl = X[:, layer_pos[L], :]
            print(f"  --- {depth} (layer {L}) ---", file=sys.stderr, flush=True)

            _, B, _, _ = factorial_vectors(Xl, kept_rows, ALL)
            mean = Xl.mean(axis=0)
            _, _, vt64 = randomized_svd(Xl - mean, n_components=64, random_state=0)
            purpose_dir = plain_direction(Xl, y)
            c = vt64 @ purpose_dir  # real 64-dim coordinate vector, not a norm-matched stand-in
            P_check = float(np.sum(c ** 2))  # should match previously saved purpose_energy_in_top64_pcs

            for r in RANKS:
                Qr = orth_basis(B, r)
                coords = vt64 @ Qr  # (64, r): format basis in the PC coordinate system
                Q_proj = reorthonormalize(coords)
                keep = Q_proj.shape[1]
                observed = float(np.sum((Q_proj.T @ c) ** 2)) if keep else 0.0

                draws = conditional_draws(c, keep, N_DRAWS, rng) if keep else np.zeros(N_DRAWS)
                k = int(np.sum(draws <= observed))
                p_empirical = (k + 1) / (N_DRAWS + 1)
                p_analytic = float(beta.cdf(observed / P_check, keep / 2, (64 - keep) / 2)) if (P_check > 0 and 0 < keep < 64) else float("nan")

                cell = {
                    "model": model, "depth": depth, "rank_requested": r, "rank_effective_after_projection": keep,
                    "purpose_energy_in_top64_pcs_P": P_check,
                    "raw_format_subspace_energy_captured_by_purpose": float(np.sum((vt64 @ unit(Qr[:, 0])) ** 2)) if r == 1 else None,
                    "projected_basis_energy_fraction_of_full_rank_r": float(np.sum(coords ** 2) / r),
                    "observed_energy_projected_basis": observed,
                    "conditional_null_mean": P_check * keep / 64,
                    "conditional_null_median_empirical": float(np.median(draws)),
                    "exceedances_leq_observed": f"{k}/{N_DRAWS}",
                    "p_empirical": p_empirical,
                    "p_analytic": p_analytic,
                }
                cells.append(cell)
                print(f"    r={r} (eff={keep}): obs={observed:.5f} cond_null_mean={cell['conditional_null_mean']:.5f} "
                      f"p_emp={p_empirical:.4f} p_analytic={p_analytic:.4f}", file=sys.stderr, flush=True)
        del X

    n_sig = sum(c["p_empirical"] < 0.05 for c in cells)
    by_rank_sig = {r: sum(1 for c in cells if c["rank_requested"] == r and c["p_empirical"] < 0.05) for r in RANKS}
    result = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "Apples-to-apples high-variance-subspace conditional null: matched-format "
                   "PCA basis projected into and re-orthonormalized within the empirical top-64 "
                   "PC subspace, then compared to the same random-subspace-within-top-64 "
                   "conditional null used in scripts/full320_v3_conditional_hv_null.py. Uses the "
                   "real purpose-direction coordinate vector (not a norm-matched stand-in), "
                   "requiring one deterministic re-extraction pass (same pipeline as "
                   "scripts/full320_v3_direction_geometry.py).",
        "n_draws": N_DRAWS, "ranks_tested": list(RANKS),
        "cells": cells,
        "summary": {"n_cells": len(cells), "n_significant_p<0.05": n_sig, "by_rank_significant": by_rank_sig},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": {"path": ITEMS_PATH, "sha256": sha256_file(ITEMS_PATH)},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\n{n_sig}/{len(cells)} cells significant at p<0.05; by rank: {by_rank_sig}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
