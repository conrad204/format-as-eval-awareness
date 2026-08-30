#!/usr/bin/env python3
"""Emergency minimal rank-64 residual sweep: is C_perp_B specifically load-bearing?

Answers exactly one question per model, at full depth, from frozen activations:
after removing the rank-64 B-derived subspace Q_B, does removing the residual
interaction subspace Q_Cperp reduce the cross-format purpose-transfer gap more
than an equal-rank random Q_B-orthogonal subspace?

Scope reductions relative to ``full320_v3_factorial_residual_sweep.py``:

* probe retraining only at rank 64, only for
  ``raw``, ``minus_b``, ``minus_b_cperp``, ``minus_b_rperp``;
* held-out C capture still reported at r = 1, 4, 16, 64 as *prefixes* of the one
  rank-64 Q_B (no extra probe fits);
* effect-matrix bases may use an exact Gram-eigendecomposition path, gated by a
  numerical equivalence check against the repository ``orth_basis``.

Everything else is the frozen contract, imported rather than re-implemented:
matched factorial effects, leave-one-purpose-family-out folds, StandardScaler +
LogisticRegression(C=0.1) probes, the three purpose-probe regimes, and the
seeded rank-matched B-orthogonal random control.

Activations are read from local disk only: the compressed bank is expanded once
into per-layer float32 shards, and each layer worker memory-maps its own shard.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import orth_basis, remove  # noqa: E402
from full320_v3_factorial_residual_sweep import (  # noqa: E402
    B64_REFERENCE,
    ITEMS_PATH,
    MODELS,
    RAW_REFERENCES,
    _atomic_json,
    _atomic_npz,
    captured_energy,
    factorial_effects,
    fit_regimes,
    item_id_digest,
    random_basis_orthogonal_to,
    sha256_file,
)

CONDITIONS = ("raw", "minus_b", "minus_b_cperp", "minus_b_rperp")
REGIMES = ("joint", "b2c", "c2b")
RANK = 64
CAPTURE_RANKS = (1, 4, 16, 64)
PARTIAL_SCHEMA_VERSION = 1
DEFAULT_TOL = 1e-6
BASIS_TOL = 1e-8

_SHARED: dict = {}


# --------------------------------------------------------------------------- #
# exact low-rank basis: m x m eigendecomposition instead of m x d full SVD
# --------------------------------------------------------------------------- #
def gram_orth_basis(V: np.ndarray, rank: int = RANK) -> np.ndarray:
    """Top-`rank` right-singular directions of V via its m x m Gram matrix.

    Same uncentered convention and truncation rule as ``orth_basis``; cheap when
    the effect matrix has far fewer rows than feature dimensions.
    """
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[None, :]
    if not np.any(np.isfinite(V)):
        return np.zeros((V.shape[1], 0))
    eigenvalues, vectors = np.linalg.eigh(V @ V.T)
    order = np.argsort(eigenvalues)[::-1]
    singular = np.sqrt(np.clip(eigenvalues[order], 0.0, None))
    vectors = vectors[:, order]
    if not len(singular) or singular[0] <= 0.0:
        return np.zeros((V.shape[1], 0))
    keep = min(
        int(rank),
        int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-10)),
        min(V.shape),
    )
    if keep == 0:
        return np.zeros((V.shape[1], 0))
    return (V.T @ vectors[:, :keep]) / singular[:keep]


def subspace_deviation(q_fast: np.ndarray, q_reference: np.ndarray, sample: np.ndarray) -> dict:
    """Rank, principal-cosine, projector, and transformed-activation deviations."""
    if q_fast.shape != q_reference.shape:
        return {
            "rank_mismatch": True,
            "fast_rank": int(q_fast.shape[1]),
            "reference_rank": int(q_reference.shape[1]),
            "min_principal_cosine": 0.0,
            "max_projector_error": float("inf"),
            "max_transformed_error": float("inf"),
        }
    cosines = np.linalg.svd(q_fast.T @ q_reference, compute_uv=False) if q_fast.shape[1] else np.ones(1)
    projector_error = (
        float(np.max(np.abs(q_fast @ q_fast.T - q_reference @ q_reference.T))) if q_fast.shape[1] else 0.0
    )
    transformed_error = float(np.max(np.abs(remove(sample, q_fast) - remove(sample, q_reference))))
    return {
        "rank_mismatch": False,
        "fast_rank": int(q_fast.shape[1]),
        "reference_rank": int(q_reference.shape[1]),
        "min_principal_cosine": float(np.min(cosines)),
        "max_projector_error": projector_error,
        "max_transformed_error": transformed_error,
    }


def validate_basis_backend(
    X: np.ndarray,
    rows: list[dict],
    train_mask: np.ndarray,
    rank: int = RANK,
    tol: float = BASIS_TOL,
) -> dict:
    """Gate the Gram path against ``orth_basis`` on real B and C_perp_B matrices."""
    B, C, _, _, _, _ = factorial_effects(X, rows, train_mask)
    sample = X[: min(64, len(X))]
    checks = {}
    q_b_reference = orth_basis(B, rank)
    q_b_fast = gram_orth_basis(B, rank)
    checks["B"] = subspace_deviation(q_b_fast, q_b_reference, sample)
    c_perp = remove(C, q_b_reference)
    checks["C_perp_B"] = subspace_deviation(
        gram_orth_basis(c_perp, rank), orth_basis(c_perp, rank), sample
    )
    checks["tolerance"] = tol
    checks["passed"] = bool(
        all(
            (not v["rank_mismatch"])
            and v["min_principal_cosine"] > 1.0 - tol
            and v["max_projector_error"] < tol
            and v["max_transformed_error"] < tol
            for key, v in checks.items()
            if key in ("B", "C_perp_B")
        )
    )
    return checks


# --------------------------------------------------------------------------- #
# local per-layer shards
# --------------------------------------------------------------------------- #
def shard_path(shard_dir: Path, layer: int) -> Path:
    return shard_dir / f"layer_{layer:03d}.f32.npy"


def prepare_shards(npz_path: Path, keep_mask: np.ndarray, shard_dir: Path) -> tuple[int, int, float]:
    """Expand the compressed bank into local float32 per-layer shards, once."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with np.load(npz_path) as payload:
        shape = payload["X"].shape
    n_layers, hidden = int(shape[1]), int(shape[2])
    if all(shard_path(shard_dir, layer).exists() for layer in range(n_layers)):
        return n_layers, hidden, 0.0
    print(f"expanding {npz_path.name} -> {shard_dir} ({n_layers} layers)", file=sys.stderr, flush=True)
    X = np.load(npz_path)["X"][keep_mask]
    for layer in range(n_layers):
        target = shard_path(shard_dir, layer)
        if target.exists():
            continue
        tmp = target.with_name(f".{target.name}.tmp")
        # np.save appends ".npy" to a path argument; write through a handle so the
        # temp name stays exactly what os.replace expects.
        with tmp.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(X[:, layer, :], dtype=np.float32))
        os.replace(tmp, target)
    del X
    gc.collect()
    return n_layers, hidden, time.time() - started


# --------------------------------------------------------------------------- #
# per-layer work
# --------------------------------------------------------------------------- #
def init_worker(shared: dict) -> None:
    _SHARED.clear()
    _SHARED.update(shared)


def project_out_with_energy(X: np.ndarray, Q: np.ndarray, test_mask: np.ndarray):
    """Remove span(Q) and report the activation energy that removal took out.

    Identical arithmetic to ``remove`` (``X - (X @ Q) Qᵀ``); the projection is
    already materialized, so the energy fractions are free. Answers whether a
    C-specific gain is partly just "this subspace removes more variance".
    """
    total_all = float(np.einsum("ij,ij->", X, X, dtype=np.float64))
    total_heldout = float(np.einsum("ij,ij->", X[test_mask], X[test_mask], dtype=np.float64))
    if Q.shape[1] == 0:
        return X, {
            "rank": 0,
            "removed_all": 0.0,
            "total_all": total_all,
            "fraction_all": 0.0,
            "removed_heldout": 0.0,
            "total_heldout": total_heldout,
            "fraction_heldout": 0.0,
        }
    projection = X @ Q
    residual = X - projection @ Q.T
    removed_all = float(np.einsum("ij,ij->", projection, projection, dtype=np.float64))
    held = projection[test_mask]
    removed_heldout = float(np.einsum("ij,ij->", held, held, dtype=np.float64))
    return residual, {
        "rank": int(Q.shape[1]),
        "removed_all": removed_all,
        "total_all": total_all,
        "fraction_all": removed_all / total_all,
        "removed_heldout": removed_heldout,
        "total_heldout": total_heldout,
        "fraction_heldout": removed_heldout / total_heldout,
    }


def layer_job(layer: int):
    rows = _SHARED["rows"]
    y = _SHARED["y"]
    fam = _SHARED["fam"]
    is_bench = _SHARED["is_bench"]
    is_casual = _SHARED["is_casual"]
    basis = gram_orth_basis if _SHARED["basis_backend"] == "gram" else orth_basis

    started = time.time()
    # float32 exactly as the frozen sweep feeds probes; `remove` promotes the
    # projected conditions to float64 on the same code path. Gates 5-6 need this.
    X = np.asarray(np.load(shard_path(_SHARED["shard_dir"], layer), mmap_mode="r"), dtype=np.float32)
    ci = {name: i for i, name in enumerate(CONDITIONS)}
    scores = np.full((len(CONDITIONS), len(REGIMES), len(y)), np.nan, dtype=np.float32)

    max_identity_error = 0.0
    max_qb_qcperp = 0.0
    max_qb_qrperp = 0.0
    held_family_violations = 0
    min_basis_train_groups = None
    capture_folds: dict[str, list[dict]] = {str(r): [] for r in CAPTURE_RANKS}
    random_perp_seeds: list[dict] = []
    energy_folds: dict[str, list[dict]] = {"b": [], "cperp": [], "rperp": []}

    for fold_pos, fold in enumerate(sorted(set(fam))):
        train_mask = fam != fold
        test_mask = fam == fold

        for regime_i, values in enumerate(fit_regimes(X, y, train_mask, test_mask, is_bench, is_casual)):
            scores[ci["raw"], regime_i, test_mask] = values[test_mask]

        B, C, _, _, basis_families, identity_error = factorial_effects(X, rows, train_mask)
        max_identity_error = max(max_identity_error, identity_error)
        held_family_violations += int(np.sum(basis_families == fold))
        min_basis_train_groups = len(B) if min_basis_train_groups is None else min(min_basis_train_groups, len(B))
        _, C_holdout, _, _, heldout_families, _ = factorial_effects(X, rows, test_mask)
        if set(heldout_families) != {fold}:
            raise AssertionError("held-out C matrix does not contain exactly the held family")

        q_b = basis(B, RANK)
        q_cperp = basis(remove(C, q_b), RANK)
        random_perp_seed = 30_260_829 + layer * 10_007 + fold_pos * 101 + RANK
        q_rperp = random_basis_orthogonal_to(X.shape[1], q_cperp.shape[1], q_b, random_perp_seed)
        random_perp_seeds.append(
            {"fold": str(fold), "seed": random_perp_seed, "rank": int(q_rperp.shape[1])}
        )
        if q_b.shape[1] and q_cperp.shape[1]:
            max_qb_qcperp = max(max_qb_qcperp, float(np.max(np.abs(q_b.T @ q_cperp))))
        if q_b.shape[1] and q_rperp.shape[1]:
            max_qb_qrperp = max(max_qb_qrperp, float(np.max(np.abs(q_b.T @ q_rperp))))

        for prefix in CAPTURE_RANKS:
            numerator, denominator, fraction = captured_energy(C_holdout, q_b[:, : min(prefix, q_b.shape[1])])
            capture_folds[str(prefix)].append(
                {
                    "fold": str(fold),
                    "numerator": numerator,
                    "denominator": denominator,
                    "fraction": fraction,
                    "prefix_rank": int(min(prefix, q_b.shape[1])),
                }
            )

        # One projection against Q_B, reused by all three removal conditions; each
        # projection also yields its removed-energy fraction at no extra cost.
        Xb, energy_b = project_out_with_energy(X, q_b, test_mask)
        energy_b["fold"] = str(fold)
        energy_folds["b"].append(energy_b)
        Xc, energy_c = project_out_with_energy(Xb, q_cperp, test_mask)
        energy_c["fold"] = str(fold)
        energy_folds["cperp"].append(energy_c)
        Xr, energy_r = project_out_with_energy(Xb, q_rperp, test_mask)
        energy_r["fold"] = str(fold)
        energy_folds["rperp"].append(energy_r)
        for condition, Xt in (("minus_b", Xb), ("minus_b_cperp", Xc), ("minus_b_rperp", Xr)):
            for regime_i, values in enumerate(fit_regimes(Xt, y, train_mask, test_mask, is_bench, is_casual)):
                scores[ci[condition], regime_i, test_mask] = values[test_mask]
        del Xb, Xc, Xr
        gc.collect()

    capture_summary = {}
    for prefix, folds in capture_folds.items():
        numerator = sum(v["numerator"] for v in folds)
        denominator = sum(v["denominator"] for v in folds)
        capture_summary[prefix] = {
            "cross_fitted_fraction": numerator / denominator,
            "numerator": numerator,
            "denominator": denominator,
            "nominal_isotropic_chance": int(prefix) / X.shape[1],
            "chance_multiple": (numerator / denominator) / (int(prefix) / X.shape[1]),
            "by_fold": folds,
        }
    energy_summary = {}
    for name, folds in energy_folds.items():
        energy_summary[name] = {
            "pooled_fraction_all": sum(v["removed_all"] for v in folds) / sum(v["total_all"] for v in folds),
            "pooled_fraction_heldout": sum(v["removed_heldout"] for v in folds)
            / sum(v["total_heldout"] for v in folds),
            "mean_rank": float(np.mean([v["rank"] for v in folds])),
            "by_fold": folds,
        }
    # >1 means Q_C|B strips more residual activation energy than the rank-matched
    # random control, i.e. part of any C-specific gap change could be variance, not C.
    for scope in ("all", "heldout"):
        random_fraction = energy_summary["rperp"][f"pooled_fraction_{scope}"]
        energy_summary[f"cperp_over_rperp_{scope}"] = (
            energy_summary["cperp"][f"pooled_fraction_{scope}"] / random_fraction
            if random_fraction > 0
            else None
        )

    sanity = {
        # float32 activations (the frozen probe dtype) make the factorial identity
        # exact only to float32 rounding, so the gate below is scale-relative.
        "max_factorial_identity_error": max_identity_error,
        "max_abs_activation": float(np.max(np.abs(X))),
        "max_qb_t_qcperp": max_qb_qcperp,
        "max_qb_t_rperp": max_qb_qrperp,
        "held_family_basis_violations": held_family_violations,
        "min_training_groups_per_fold": int(min_basis_train_groups or 0),
        "c_capture_by_rank": capture_summary,
        "residual_energy_removed": energy_summary,
        "random_perp_seeds": random_perp_seeds,
        "basis_backend": _SHARED["basis_backend"],
    }
    return layer, scores, sanity, time.time() - started


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def partial_layer_path(partial_dir: Path, layer: int) -> Path:
    return partial_dir / f"layer_{layer:03d}.npz"


def write_partial_layer(
    partial_dir: Path,
    model: str,
    layer: int,
    layer_scores: np.ndarray,
    sanity: dict,
    elapsed: float,
    ids: np.ndarray,
    completed_layers: list[int],
    n_layers: int,
) -> None:
    _atomic_npz(
        partial_layer_path(partial_dir, layer),
        schema_version=np.asarray(PARTIAL_SCHEMA_VERSION),
        model=np.asarray(model),
        layer=np.asarray(layer),
        item_id_digest=np.asarray(item_id_digest(ids)),
        scores=layer_scores,
        sanity_json=np.asarray(json.dumps(sanity, sort_keys=True)),
        elapsed=np.asarray(elapsed),
    )
    _atomic_json(
        partial_dir / "manifest.json",
        {
            "schema_version": PARTIAL_SCHEMA_VERSION,
            "status": "complete" if len(completed_layers) == n_layers else "running",
            "model": model,
            "n_layers": n_layers,
            "n_items": int(len(ids)),
            "item_id_digest": item_id_digest(ids),
            "conditions": list(CONDITIONS),
            "ranks": [RANK],
            "regimes": list(REGIMES),
            "completed_layers": sorted(completed_layers),
        },
    )


def load_partial_layer(partial_dir: Path, layer: int, model: str, ids: np.ndarray):
    with np.load(partial_layer_path(partial_dir, layer), allow_pickle=False) as payload:
        if str(payload["model"].item()) != model or int(payload["layer"].item()) != layer:
            raise ValueError(f"partial checkpoint identity mismatch at layer {layer}")
        if str(payload["item_id_digest"].item()) != item_id_digest(ids):
            raise ValueError(f"partial checkpoint item identity mismatch at layer {layer}")
        return (
            np.asarray(payload["scores"], dtype=np.float32),
            json.loads(str(payload["sanity_json"].item())),
            float(payload["elapsed"].item()),
        )


# --------------------------------------------------------------------------- #
# reference gates
# --------------------------------------------------------------------------- #
def auc_arrays(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    out = np.full(scores.shape[:-1], np.nan, dtype=np.float64)
    for index in np.ndindex(scores.shape[:-1]):
        values = scores[index]
        mask = np.isfinite(values)
        if mask.any():
            out[index] = roc_auc_score(y[mask], values[mask])
    return out


def validate_references(model: str, aucs: np.ndarray, tolerance: float, mode: str = "strict") -> dict:
    """Compare raw and rank-64 ``minus_b`` AUCs to the checked-in legacy sweeps.

    ``aucs``: (condition, regime, layer). Both references were produced by
    different pipelines than this one, so the full per-layer, per-regime deviation
    is always recorded. ``mode="strict"`` aborts when a deviation exceeds
    ``tolerance``; ``mode="report"`` records the failure and continues, which is
    only legitimate because every condition in this run shares one probe code path
    and the decision statistic is an internal contrast between them.
    """
    ci = {name: i for i, name in enumerate(CONDITIONS)}
    raw_ref = json.loads(RAW_REFERENCES[model].read_text())
    b64 = json.loads(B64_REFERENCE.read_text())["by_model"][model]["by_layer"]
    deviations = {"raw": {}, "minus_b": {}}
    for regime_i, regime in enumerate(REGIMES):
        raw_key = {"joint": "decorrelated", "b2c": "b2c", "c2b": "c2b"}[regime]
        b64_key = {
            "joint": "purpose_auc_after_erasure",
            "b2c": "b2c_auc_after_erasure",
            "c2b": "c2b_auc_after_erasure",
        }[regime]
        raw_expected = np.asarray(
            [float(raw_ref[layer][raw_key]["auc"]) for layer in range(aucs.shape[2])]
        )
        b64_expected = np.asarray(
            [float(b64[str(layer)][b64_key]) for layer in range(aucs.shape[2])]
        )
        for name, expected in (("raw", raw_expected), ("minus_b", b64_expected)):
            observed = aucs[ci[name], regime_i]
            error = observed - expected
            deviations[name][regime] = {
                "max_abs_error": float(np.max(np.abs(error))),
                "mean_signed_error": float(np.mean(error)),
                "pearson_r": float(np.corrcoef(observed, expected)[0, 1]),
                "by_layer_abs_error": np.abs(error).tolist(),
            }
    max_raw = max(v["max_abs_error"] for v in deviations["raw"].values())
    max_b64 = max(v["max_abs_error"] for v in deviations["minus_b"].values())
    strict_pass = bool(max_raw <= tolerance and max_b64 <= tolerance)
    if mode == "strict" and not strict_pass:
        raise AssertionError(
            f"legacy reference reproduction failed: raw {max_raw}, rank64 minus_b {max_b64} > {tolerance}"
        )
    if not strict_pass:
        print(
            f"WARNING: legacy reference deviation raw={max_raw:.3e} minus_b={max_b64:.3e} "
            f"exceeds {tolerance:.0e}; recorded, not gated (mode=report)",
            file=sys.stderr,
            flush=True,
        )
    return {
        "mode": mode,
        "tolerance": tolerance,
        "strict_pass": strict_pass,
        "max_raw_auc_abs_error": max_raw,
        "max_rank64_minus_b_auc_abs_error": max_b64,
        "note": "raw and minus_b are compared to sweeps produced by different pipelines; a nonzero deviation is a cross-pipeline comparability limit, not an internal inconsistency",
        "by_condition": deviations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--npz", default=None, help="local activation bank; defaults to the repo path")
    ap.add_argument("--items", default=str(ITEMS_PATH))
    ap.add_argument("--shard-dir", default=None, help="local per-layer float32 shard cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sanity-out", default=None)
    ap.add_argument("--partial-dir", default=None)
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", "1")))
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="benchmark subset")
    ap.add_argument("--benchmark", action="store_true", help="run --layers only; skip final assembly")
    ap.add_argument("--basis-backend", choices=("gram", "svd"), default="gram")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOL)
    ap.add_argument("--basis-tolerance", type=float, default=BASIS_TOL)
    ap.add_argument(
        "--reference-mode",
        choices=("strict", "report"),
        default="strict",
        help="strict aborts on legacy-reference deviation; report records it and continues",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz) if args.npz else MODELS[args.model]
    items_path = Path(args.items)
    out_path = Path(args.out) if args.out else npz_path.parent / f"emergency_cperp_scores_{args.model}.npz"
    sanity_path = Path(args.sanity_out) if args.sanity_out else out_path.with_suffix(".sanity.json")
    partial_dir = Path(args.partial_dir) if args.partial_dir else out_path.with_name(f"{out_path.name}.partial")
    timing_path = Path(args.timing_out) if args.timing_out else out_path.with_suffix(".timing.json")
    shard_dir = Path(args.shard_dir) if args.shard_dir else out_path.with_name(f"shards_{args.model}")
    if not args.benchmark and (out_path.exists() or sanity_path.exists()):
        raise FileExistsError(f"immutable output exists: {out_path} or {sanity_path}")

    rows_all = [json.loads(line) for line in items_path.open(encoding="utf-8") if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    import zipfile

    with zipfile.ZipFile(npz_path) as archive:
        raw_ids = np.load(archive.open("item_ids.npy")).astype(str)
    keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
    if int(keep_mask.sum()) != len(rows_by_id):
        raise ValueError("activation/item membership mismatch")

    n_layers, hidden_dim, shard_seconds = prepare_shards(npz_path, keep_mask, shard_dir)
    rows = [rows_by_id[iid] for iid in raw_ids[keep_mask]]
    ids = np.asarray([r["item_id"] for r in rows])
    y = np.asarray([int(r["label"]) for r in rows])
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    blocks = np.asarray([r["payload_block_id"] for r in rows])
    is_bench, is_casual = fmt == "benchmark", fmt == "casual"

    # Gate 7: the fast basis must reproduce the frozen subspace before any use.
    calibration_layer = int(min(n_layers - 1, max(0, n_layers // 4)))
    calibration = np.asarray(
        np.load(shard_path(shard_dir, calibration_layer), mmap_mode="r"), dtype=np.float64
    )
    folds = sorted(set(fam))
    basis_check = validate_basis_backend(
        calibration, rows, fam != folds[0], RANK, args.basis_tolerance
    )
    basis_check["calibration_layer"] = calibration_layer
    basis_check["requested_backend"] = args.basis_backend
    backend = args.basis_backend
    if backend == "gram" and not basis_check["passed"]:
        backend = "svd"
        print("gram basis failed its equivalence gate; falling back to svd", file=sys.stderr, flush=True)
    basis_check["effective_backend"] = backend
    del calibration
    gc.collect()

    scores = np.full((len(CONDITIONS), len(REGIMES), n_layers, len(y)), np.nan, dtype=np.float32)
    layer_sanity: dict[str, dict] = {}
    layer_seconds: dict[str, float] = {}
    completed_layers: list[int] = []
    manifest_path = partial_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": PARTIAL_SCHEMA_VERSION,
            "model": args.model,
            "n_layers": n_layers,
            "n_items": len(ids),
            "item_id_digest": item_id_digest(ids),
            "conditions": list(CONDITIONS),
            "ranks": [RANK],
            "regimes": list(REGIMES),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"partial manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}")
        completed_layers = sorted(int(layer) for layer in manifest.get("completed_layers", []))
        for layer in completed_layers:
            layer_scores, sanity, elapsed = load_partial_layer(partial_dir, layer, args.model, ids)
            scores[:, :, layer, :] = layer_scores
            layer_sanity[str(layer)] = sanity
            layer_seconds[str(layer)] = elapsed
        print(f"resumed {len(completed_layers)}/{n_layers} layers", file=sys.stderr, flush=True)
    elif not args.resume:
        _atomic_json(
            manifest_path,
            {
                "schema_version": PARTIAL_SCHEMA_VERSION,
                "status": "running",
                "model": args.model,
                "n_layers": n_layers,
                "n_items": int(len(ids)),
                "item_id_digest": item_id_digest(ids),
                "conditions": list(CONDITIONS),
                "ranks": [RANK],
                "regimes": list(REGIMES),
                "completed_layers": [],
            },
        )

    requested = list(range(n_layers)) if args.layers is None else [int(v) for v in args.layers]
    if any(layer < 0 or layer >= n_layers for layer in requested):
        raise ValueError(f"requested layer outside 0..{n_layers - 1}")
    pending = [layer for layer in requested if layer not in set(completed_layers)]

    shared = {
        "rows": rows,
        "y": y,
        "fam": fam,
        "is_bench": is_bench,
        "is_casual": is_casual,
        "shard_dir": shard_dir,
        "basis_backend": backend,
    }
    run_started = time.time()
    if pending:
        workers = max(1, min(args.n_jobs, len(pending)))
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(shared,)) as pool:
            futures = {pool.submit(layer_job, layer): layer for layer in pending}
            for future in as_completed(futures):
                layer, layer_scores, sanity, elapsed = future.result()
                scores[:, :, layer, :] = layer_scores
                layer_sanity[str(layer)] = sanity
                layer_seconds[str(layer)] = elapsed
                completed_layers = sorted(int(key) for key in layer_sanity)
                write_partial_layer(
                    partial_dir, args.model, layer, layer_scores, sanity, elapsed, ids, completed_layers, n_layers
                )
                print(
                    f"layer={layer} done in {elapsed:.1f}s; partial={len(completed_layers)}/{n_layers}",
                    file=sys.stderr,
                    flush=True,
                )
    wall_seconds = time.time() - run_started

    finished = sorted(float(v) for v in layer_seconds.values())
    timing = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_items": int(len(ids)),
        "n_jobs": args.n_jobs,
        "basis_backend": backend,
        "shard_expand_seconds": shard_seconds,
        "layers_completed": len(layer_sanity),
        "wall_seconds_this_invocation": wall_seconds,
        "median_layer_seconds": float(np.median(finished)) if finished else None,
        "max_layer_seconds": float(max(finished)) if finished else None,
        "projected_remaining_seconds": (
            float(np.median(finished)) * (n_layers - len(layer_sanity)) / max(1, args.n_jobs)
            if finished
            else None
        ),
        "layer_seconds": layer_seconds,
        "host": {
            "node": platform.node(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in timing.items() if k != "layer_seconds"}, indent=2))

    if args.benchmark or len(layer_sanity) != n_layers:
        print(
            f"benchmark/partial run: {len(layer_sanity)}/{n_layers} layers; no final artifact written",
            file=sys.stderr,
            flush=True,
        )
        return

    aucs = auc_arrays(scores, y)
    reference = validate_references(args.model, aucs, args.tolerance, args.reference_mode)
    max_abs_activation = max(v["max_abs_activation"] for v in layer_sanity.values())
    global_checks = {
        "max_factorial_identity_error": max(v["max_factorial_identity_error"] for v in layer_sanity.values()),
        "max_abs_activation": max_abs_activation,
        "factorial_identity_relative_budget": args.tolerance * max(1.0, max_abs_activation),
        "max_qb_t_qcperp": max(v["max_qb_t_qcperp"] for v in layer_sanity.values()),
        "max_qb_t_rperp": max(v["max_qb_t_rperp"] for v in layer_sanity.values()),
        "held_family_basis_violations": sum(v["held_family_basis_violations"] for v in layer_sanity.values()),
    }
    # The factorial identity is exact in exact arithmetic; on float32 activations the
    # residual is pure rounding, so it is gated relative to activation magnitude.
    if global_checks["max_factorial_identity_error"] > global_checks["factorial_identity_relative_budget"]:
        raise AssertionError(global_checks)
    for key in ("max_qb_t_qcperp", "max_qb_t_rperp"):
        if global_checks[key] > args.tolerance:
            raise AssertionError(global_checks)
    if global_checks["held_family_basis_violations"] != 0:
        raise AssertionError(global_checks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.asarray(1),
        mode=np.asarray("development"),
        model=np.asarray(args.model),
        ranks=np.asarray([RANK], dtype=np.int16),
        conditions=np.asarray(CONDITIONS),
        regimes=np.asarray(REGIMES),
        n_layers=np.asarray(n_layers),
        item_id=ids,
        payload_block_id=blocks,
        purpose_family_id=fam,
        fold_id=fam,
        format=fmt,
        y=y,
        # Singleton rank axis keeps the frozen (condition, rank, regime, layer, item) layout.
        scores=scores[:, None, :, :, :].transpose(0, 1, 2, 3, 4),
    )
    sanity_payload = {
        "schema_version": 1,
        "mode": "development",
        "scope": "emergency minimal rank-64 C_perp_B specificity run",
        "model": args.model,
        "source_activation": str(npz_path),
        "source_activation_sha256": sha256_file(npz_path),
        "items_path": str(items_path),
        "items_sha256": sha256_file(items_path),
        "script_sha256": sha256_file(Path(__file__)),
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
        "conditions": list(CONDITIONS),
        "ranks": [RANK],
        "capture_ranks": list(CAPTURE_RANKS),
        "hidden_dim": int(hidden_dim),
        "basis": "leading singular directions of uncentered matched effect matrices; Q_B is B-derived and may capture C",
        "basis_backend_check": basis_check,
        "fold_semantics": "leave-one-purpose-family-out; held family excluded from every basis and probe fit",
        "c_capture": "prefixes of one rank-64 Q_B; cross-fitted held-family ||C_holdout Q_B(:r)||_F^2/||C_holdout||_F^2 pooled over folds",
        "random_perp_control": "seeded rank(Q_Cperp) random subspace orthogonal to Q_B, same seed formula as the canonical sweep",
        "reference_reproduction": reference,
        "global_checks": global_checks,
        "timing": timing,
        "identifier_fields": ["item_id", "payload_block_id", "purpose_family_id", "fold_id", "format", "y"],
        "layer_checks": layer_sanity,
        "interpretation_boundary": "Frozen-activation representational intervention plus probe retraining; not behavioral steering or model-output causality.",
    }
    sanity_path.write_text(json.dumps(sanity_payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} and {sanity_path}")


if __name__ == "__main__":
    main()
