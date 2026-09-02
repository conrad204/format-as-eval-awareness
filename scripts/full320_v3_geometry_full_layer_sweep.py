"""Full-layer-resolution, apples-to-apples matched-format-subspace geometry test.

This is the SAME correctness-critical logic as scripts/full320_v3_projected_basis_conditional_null.py
(no shortened/shortcut version): the matched-format PCA basis Q_format(r) is projected
into the empirical top-64 PC subspace and RE-ORTHONORMALIZED there, then compared
against a conditional null of random rank-r subspaces drawn uniformly from within that
SAME top-64 subspace, using the REAL purpose-direction coordinate vector (not a
norm-matched stand-in). Applied at every layer of every checkpoint, not just 3
hand-picked depths.

Also reports the full-space isotropic Beta(r/2,(d-r)/2) null at every layer, for
direct comparison with scripts/full320_v3_direction_geometry_significance.json's
3-depth version.

Designed for the uahpc cluster: bulk-loads the full (n_items, n_layers, hidden)
float32 tensor per model (like scripts/full320_v3_erasure_full_layer_sweep.py) and
parallelizes the per-layer computation (1 non-CV logistic fit + cheap linear algebra
per layer -- no cross-validation, no probe retraining) across layers with joblib.

Usage: python scripts/full320_v3_geometry_full_layer_sweep.py [out_json]
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
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, orth_basis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
    "llama31_70b": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
    "llama33_70b": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
}
RANKS = (1, 4, 16, 64)
COND_RANKS = (1, 4, 16)  # rank 64 excluded: a random rank-64 subspace of R^64 is the whole space
N_DRAWS = 10_000
SEED = 0
N_JOBS = int(os.environ.get("N_JOBS", "28"))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def plain_direction(X, target):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(**PROBE).fit(sc.transform(X), target)
    return unit(m.coef_[0] / sc.scale_)


def reorthonormalize(coords_64_by_r):
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


def score_layer(Xl, rows, y, seed):
    """Returns per-rank full-space Beta-null and apples-to-apples conditional-null
    results for one layer. No cross-validation -- descriptive geometry only."""
    rng = np.random.default_rng(seed)
    d = Xl.shape[1]
    ALL = np.ones(len(y), bool)
    _, B, _, _ = factorial_vectors(Xl, rows, ALL)
    purpose_dir = plain_direction(Xl, y)
    mean = Xl.mean(axis=0)
    _, _, vt64 = randomized_svd(Xl - mean, n_components=64, random_state=0)
    c = vt64 @ purpose_dir  # real 64-dim coordinate vector
    P = float(np.sum(c ** 2))

    out = {"purpose_energy_in_top64_pcs": P, "by_rank": {}}
    for r in RANKS:
        Qr = orth_basis(B, r)
        e_full = float(np.sum((Qr.T @ purpose_dir) ** 2))
        p_full_space = float(beta.cdf(e_full, r / 2, (d - r) / 2))
        row = {"E_purpose_in_format_subspace": e_full, "p_full_space_isotropic": p_full_space}

        if r in COND_RANKS:
            coords = vt64 @ Qr
            Q_proj = reorthonormalize(coords)
            keep = Q_proj.shape[1]
            observed_proj = float(np.sum((Q_proj.T @ c) ** 2)) if keep else 0.0
            draws = conditional_draws(c, keep, N_DRAWS, rng) if keep else np.zeros(N_DRAWS)
            k = int(np.sum(draws <= observed_proj))
            p_cond = (k + 1) / (N_DRAWS + 1)
            row.update({
                "rank_effective_after_projection": keep,
                "observed_energy_projected_basis": observed_proj,
                "p_conditional_apples_to_apples": p_cond,
            })
        out["by_rank"][str(r)] = row
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "full320_v3_geometry_full_layer_sweep.json")
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    result = {"schema_version": 1, "mode": "development",
              "purpose": "Full-layer-resolution apples-to-apples matched-format subspace "
                         "geometry: matched-format PCA basis projected into and "
                         "re-orthonormalized within the empirical top-64 PC subspace, "
                         "compared against a conditional null of random subspaces drawn "
                         "from within that same top-64 subspace, using the real "
                         "purpose-direction coordinate vector (not a norm-matched "
                         "stand-in). No shortcuts. Also reports the full-space isotropic "
                         "Beta(r/2,(d-r)/2) null at every layer.",
              "ranks": list(RANKS), "conditional_ranks": list(COND_RANKS), "n_draws": N_DRAWS,
              "by_model": {}}

    for model, npz_path in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        z = zipfile.ZipFile(npz_path)
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        print("  loading full tensor...", file=sys.stderr, flush=True)
        X = np.load(npz_path)["X"][keep_mask].astype(np.float64)
        n_layers = X.shape[1]
        print(f"  loaded {X.shape}", file=sys.stderr, flush=True)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])

        outs = Parallel(n_jobs=N_JOBS)(
            delayed(score_layer)(np.ascontiguousarray(X[:, L, :]), kept_rows, y, SEED * 10_000 + L)
            for L in range(n_layers)
        )
        by_layer = {str(L): outs[L] for L in range(n_layers)}
        result["by_model"][model] = {"n_layers": n_layers, "by_layer": by_layer}
        for L in range(n_layers):
            r16 = outs[L]["by_rank"]["16"]
            print(f"  layer {L}: P={outs[L]['purpose_energy_in_top64_pcs']:.5f} "
                  f"r=16 E={r16['E_purpose_in_format_subspace']:.5f} "
                  f"p_full={r16['p_full_space_isotropic']:.4f} "
                  f"p_cond={r16.get('p_conditional_apples_to_apples', float('nan')):.4f}",
                  file=sys.stderr, flush=True)
        del X

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
