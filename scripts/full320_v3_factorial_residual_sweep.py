#!/usr/bin/env python3
"""Cross-fitted frozen-activation decomposition of cross-format purpose portability.

For each matched full320-v3 2x2 group and held-purpose-family fold, this script
constructs the purpose-averaged format main effect B, the purpose-by-format
interaction C, and the natural purpose-conditioned format effects
F_eval=B+C and F_deploy=B-C.  Bases are leading singular directions of the
UNCENTERED matched effect matrices (the repository ``orth_basis`` convention),
not centered PCA directions.

No model forward passes are performed.  Every basis is fitted on training
purpose families only.  The exact existing joint, benchmark->casual, and
casual->benchmark purpose-probe regimes are refit after each projection.  The
output caches per-item OOF scores for correlation-aware integrated-gap
bootstrapping in ``full320_v3_factorial_residual_bootstrap.py``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import fit_probe, groups, orth_basis, remove  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
    "llama31_70b": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
    "llama33_70b": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
}
RAW_REFERENCES = {
    model: ROOT / "results" / f"layer_sweep_full320-v3_{model}.json" for model in MODELS
}
B64_REFERENCE = ROOT / "results" / "full320_v3_erasure_full_layer_sweep.json"
RANKS = (1, 4, 16, 64)
CONDITIONS = (
    "raw",
    "minus_b",
    "minus_c",
    "minus_b_cperp",
    "minus_b_rperp",
    "minus_bc_joint",
    "minus_feval",
    "minus_fdeploy",
    "random_rankmatched",
)
REGIMES = ("joint", "b2c", "c2b")
DEFAULT_TOL = 1e-6


PARTIAL_SCHEMA_VERSION = 1


def _atomic_npz(path: Path, **arrays) -> None:
    """Write a compressed NumPy archive without exposing a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp_path, path)


def _atomic_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so readers see complete manifests only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def item_id_digest(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


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
    """Publish one completed layer and then atomically advance its manifest."""
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
            "ranks": list(RANKS),
            "regimes": list(REGIMES),
            "completed_layers": sorted(completed_layers),
        },
    )


def load_partial_layer(partial_dir: Path, layer: int, model: str, ids: np.ndarray) -> tuple[np.ndarray, dict, float]:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def factorial_effects(X: np.ndarray, rows: list[dict], train_mask: np.ndarray):
    """Return uncentered matched B, C, F_eval and F_deploy effect matrices."""
    matched = groups(
        rows,
        np.asarray([r["render_slot"] == "stated" for r in rows]),
        train_mask,
    )
    b_values, c_values, fe_values, fd_values = [], [], [], []
    group_families = []
    max_identity_error = 0.0
    for _, eb, ec, db, dc in matched:
        cell_indices = (eb, ec, db, dc)
        cell_families = {rows[i]["purpose_family_id"] for i in cell_indices}
        if len(cell_families) != 1:
            raise ValueError("matched group spans purpose_family_id")
        if not all(train_mask[i] for i in cell_indices):
            raise ValueError("matched group includes a held-out row")
        b = ((X[eb] + X[db]) - (X[ec] + X[dc])) / 4.0
        c = (X[eb] - X[ec] - X[db] + X[dc]) / 4.0
        fe = (X[eb] - X[ec]) / 2.0
        fd = (X[db] - X[dc]) / 2.0
        max_identity_error = max(
            max_identity_error,
            float(np.max(np.abs(fe - (b + c)))),
            float(np.max(np.abs(fd - (b - c)))),
        )
        b_values.append(b)
        c_values.append(c)
        fe_values.append(fe)
        fd_values.append(fd)
        group_families.append(next(iter(cell_families)))
    if not b_values:
        raise ValueError("no complete matched training groups")
    return (
        np.asarray(b_values),
        np.asarray(c_values),
        np.asarray(fe_values),
        np.asarray(fd_values),
        np.asarray(group_families),
        max_identity_error,
    )


def combined_basis(q_b: np.ndarray, q_cperp: np.ndarray) -> np.ndarray:
    """Orthonormal basis of span(Q_B, Q_C|B), without centering."""
    if q_b.shape[1] == 0:
        return q_cperp
    if q_cperp.shape[1] == 0:
        return q_b
    q, r = np.linalg.qr(np.concatenate([q_b, q_cperp], axis=1), mode="reduced")
    keep = np.abs(np.diag(r)) > max(float(np.max(np.abs(np.diag(r)))), 1.0) * 1e-10
    return q[:, keep]


def random_basis(d: int, rank: int, seed: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros((d, 0), dtype=np.float64)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, rank)), mode="reduced")
    return q[:, :rank]


def random_basis_orthogonal_to(d: int, rank: int, q_exclude: np.ndarray, seed: int) -> np.ndarray:
    """Seeded random rank-r basis constrained to the complement of q_exclude."""
    if rank <= 0:
        return np.zeros((d, 0), dtype=np.float64)
    draws = np.random.default_rng(seed).standard_normal((rank, d))
    return orth_basis(remove(draws, q_exclude), rank)


def captured_energy(C: np.ndarray, q_b: np.ndarray) -> tuple[float, float, float]:
    """Return captured energy, total energy, and their ratio."""
    numerator = float(np.sum((C @ q_b) ** 2))
    denominator = float(np.sum(C ** 2))
    if denominator <= 0:
        raise ValueError("C has zero energy")
    return numerator, denominator, numerator / denominator


def transformed(X: np.ndarray, condition: str, bases: dict[str, np.ndarray]) -> np.ndarray:
    if condition == "raw":
        return X
    if condition == "minus_b":
        return remove(X, bases["b"])
    if condition == "minus_c":
        return remove(X, bases["c"])
    if condition == "minus_b_cperp":
        return remove(remove(X, bases["b"]), bases["cperp"])
    if condition == "minus_b_rperp":
        return remove(remove(X, bases["b"]), bases["rperp"])
    if condition == "minus_bc_joint":
        return remove(X, bases["joint"])
    if condition == "minus_feval":
        return remove(X, bases["feval"])
    if condition == "minus_fdeploy":
        return remove(X, bases["fdeploy"])
    if condition == "random_rankmatched":
        return remove(X, bases["random"])
    raise KeyError(condition)


def fit_regimes(
    Xt: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    is_bench: np.ndarray,
    is_casual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    joint = np.full(n, np.nan, dtype=np.float64)
    b2c = np.full(n, np.nan, dtype=np.float64)
    c2b = np.full(n, np.nan, dtype=np.float64)
    joint[test_mask] = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
    b_train, b_test = train_mask & is_bench, test_mask & is_casual
    c_train, c_test = train_mask & is_casual, test_mask & is_bench
    if b_test.any():
        b2c[b_test] = fit_probe(Xt[b_train], y[b_train], Xt[b_test])
    if c_test.any():
        c2b[c_test] = fit_probe(Xt[c_train], y[c_train], Xt[c_test])
    return joint, b2c, c2b


def layer_job(
    model: str,
    layer: int,
    Xl: np.ndarray,
    rows: list[dict],
    y: np.ndarray,
    fam: np.ndarray,
    is_bench: np.ndarray,
    is_casual: np.ndarray,
):
    started = time.time()
    folds = sorted(set(fam))
    n = len(y)
    scores = np.full(
        (len(CONDITIONS), len(RANKS), len(REGIMES), n),
        np.nan,
        dtype=np.float32,
    )
    ci = {name: i for i, name in enumerate(CONDITIONS)}
    max_identity_error = 0.0
    max_bc_orthogonality = 0.0
    max_sequential_joint_error = 0.0
    min_basis_train_groups = None
    held_family_violations = 0
    overlap_by_rank: dict[str, list[np.ndarray]] = {str(r): [] for r in RANKS}
    capture_by_rank: dict[str, list[dict]] = {str(r): [] for r in RANKS}
    random_perp_seeds: dict[str, list[dict]] = {str(r): [] for r in RANKS}
    max_br_orthogonality = 0.0

    # RAW is rank-invariant. Compute it once per fold and copy across rank slots.
    for fold_pos, fold in enumerate(folds):
        train_mask = fam != fold
        test_mask = fam == fold
        raw = fit_regimes(Xl, y, train_mask, test_mask, is_bench, is_casual)
        for ri in range(len(RANKS)):
            for regime_i, values in enumerate(raw):
                scores[ci["raw"], ri, regime_i, test_mask] = values[test_mask]

        B, C, Fe, Fd, basis_families, identity_error = factorial_effects(Xl, rows, train_mask)
        max_identity_error = max(max_identity_error, identity_error)
        held_family_violations += int(np.sum(basis_families == fold))
        min_basis_train_groups = len(B) if min_basis_train_groups is None else min(min_basis_train_groups, len(B))
        _, C_holdout, _, _, heldout_families, _ = factorial_effects(Xl, rows, test_mask)
        if set(heldout_families) != {fold}:
            raise AssertionError("held-out C matrix does not contain exactly the held family")

        for ri, rank in enumerate(RANKS):
            q_b = orth_basis(B, rank)
            q_c = orth_basis(C, rank)
            c_perp = remove(C, q_b)
            q_cperp = orth_basis(c_perp, rank)
            q_joint = combined_basis(q_b, q_cperp)
            q_fe = orth_basis(Fe, rank)
            q_fd = orth_basis(Fd, rank)
            random_perp_seed = 30_260_829 + layer * 10_007 + fold_pos * 101 + rank
            q_rperp = random_basis_orthogonal_to(
                Xl.shape[1],
                q_cperp.shape[1],
                q_b,
                random_perp_seed,
            )
            q_random = random_basis(
                Xl.shape[1],
                q_joint.shape[1],
                seed=20_260_829 + layer * 10_007 + fold_pos * 101 + rank,
            )
            c_capture_numerator, c_capture_denominator, c_capture_fraction = captured_energy(C_holdout, q_b)
            capture_by_rank[str(rank)].append(
                {
                    "fold": str(fold),
                    "numerator": c_capture_numerator,
                    "denominator": c_capture_denominator,
                    "fraction": c_capture_fraction,
                }
            )
            random_perp_seeds[str(rank)].append(
                {"fold": str(fold), "seed": random_perp_seed, "rank": int(q_rperp.shape[1])}
            )
            orth_error = float(np.max(np.abs(q_b.T @ q_cperp))) if q_b.shape[1] and q_cperp.shape[1] else 0.0
            max_bc_orthogonality = max(max_bc_orthogonality, orth_error)
            br_error = float(np.max(np.abs(q_b.T @ q_rperp))) if q_b.shape[1] and q_rperp.shape[1] else 0.0
            max_br_orthogonality = max(max_br_orthogonality, br_error)

            sample = Xl[test_mask][: min(64, int(test_mask.sum()))]
            seq_sample = remove(remove(sample, q_b), q_cperp)
            joint_sample = remove(sample, q_joint)
            seq_error = float(np.max(np.abs(seq_sample - joint_sample))) if len(sample) else 0.0
            max_sequential_joint_error = max(max_sequential_joint_error, seq_error)

            if q_fe.shape[1] and q_fd.shape[1]:
                overlap_by_rank[str(rank)].append(np.linalg.svd(q_fe.T @ q_fd, compute_uv=False))

            bases = {
                "b": q_b,
                "c": q_c,
                "cperp": q_cperp,
                "rperp": q_rperp,
                "joint": q_joint,
                "feval": q_fe,
                "fdeploy": q_fd,
                "random": q_random,
            }
            for condition in CONDITIONS[1:]:
                Xt = transformed(Xl, condition, bases)
                fitted = fit_regimes(Xt, y, train_mask, test_mask, is_bench, is_casual)
                for regime_i, values in enumerate(fitted):
                    scores[ci[condition], ri, regime_i, test_mask] = values[test_mask]
                del Xt

    overlap_summary = {}
    for rank, values in overlap_by_rank.items():
        if values:
            all_values = np.concatenate(values)
            overlap_summary[rank] = {
                "mean_principal_cosine": float(np.mean(all_values)),
                "min_principal_cosine": float(np.min(all_values)),
                "max_principal_cosine": float(np.max(all_values)),
            }


    capture_summary = {}
    for rank, values in capture_by_rank.items():
        numerator = sum(v["numerator"] for v in values)
        denominator = sum(v["denominator"] for v in values)
        capture_summary[rank] = {
            "cross_fitted_fraction": numerator / denominator,
            "numerator": numerator,
            "denominator": denominator,
            "nominal_isotropic_chance": int(rank) / Xl.shape[1],
            "chance_multiple": (numerator / denominator) / (int(rank) / Xl.shape[1]),
            "by_fold": values,
        }
    sanity = {
        "max_factorial_identity_error": max_identity_error,
        "max_qb_t_qcperp": max_bc_orthogonality,
        "max_qb_t_rperp": max_br_orthogonality,
        "max_sequential_joint_projection_error": max_sequential_joint_error,
        "held_family_basis_violations": held_family_violations,
        "min_training_groups_per_fold": int(min_basis_train_groups or 0),
        "natural_subspace_overlap": overlap_summary,
        "c_capture_by_rank": capture_summary,
        "random_perp_seeds": random_perp_seeds,
    }
    return layer, scores, sanity, time.time() - started


def auc_arrays(scores: np.ndarray, y: np.ndarray):
    out = np.full(scores.shape[:-1], np.nan, dtype=np.float64)
    for index in np.ndindex(scores.shape[:-1]):
        values = scores[index]
        mask = np.isfinite(values)
        if mask.any():
            out[index] = roc_auc_score(y[mask], values[mask])
    return out


def validate_references(model: str, aucs: np.ndarray, tolerance: float) -> dict:
    ci = {name: i for i, name in enumerate(CONDITIONS)}
    raw_ref = json.loads(RAW_REFERENCES[model].read_text())
    b64 = json.loads(B64_REFERENCE.read_text())["by_model"][model]["by_layer"]
    ri64 = RANKS.index(64)
    max_raw = 0.0
    max_b64 = 0.0
    for layer in range(aucs.shape[3]):
        raw_expected = (
            float(raw_ref[layer]["decorrelated"]["auc"]),
            float(raw_ref[layer]["b2c"]["auc"]),
            float(raw_ref[layer]["c2b"]["auc"]),
        )
        raw_actual = aucs[ci["raw"], 0, :, layer]
        max_raw = max(max_raw, float(np.max(np.abs(raw_actual - raw_expected))))
        b_expected = (
            float(b64[str(layer)]["purpose_auc_after_erasure"]),
            float(b64[str(layer)]["b2c_auc_after_erasure"]),
            float(b64[str(layer)]["c2b_auc_after_erasure"]),
        )
        b_actual = aucs[ci["minus_b"], ri64, :, layer]
        max_b64 = max(max_b64, float(np.max(np.abs(b_actual - b_expected))))
    if max_raw > tolerance:
        raise AssertionError(f"raw reference reproduction failed: {max_raw} > {tolerance}")
    if max_b64 > tolerance:
        raise AssertionError(f"rank64 B reference reproduction failed: {max_b64} > {tolerance}")
    return {
        "tolerance": tolerance,
        "max_raw_auc_abs_error": max_raw,
        "max_rank64_b_auc_abs_error": max_b64,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--sanity-out", default=None)
    ap.add_argument("--partial-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", "1")))
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOL)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "results" / f"full320_v3_factorial_residual_scores_{args.model}.npz"
    sanity_path = Path(args.sanity_out) if args.sanity_out else out_path.with_suffix(".sanity.json")
    partial_dir = Path(args.partial_dir) if args.partial_dir else out_path.with_name(f"{out_path.name}.partial")
    if out_path.exists() or sanity_path.exists():
        raise FileExistsError(f"immutable output exists: {out_path} or {sanity_path}")
    if partial_dir.exists() and not args.resume and any(partial_dir.iterdir()):
        raise FileExistsError(f"partial output exists; pass --resume: {partial_dir}")

    rows_all = [json.loads(line) for line in ITEMS_PATH.open(encoding="utf-8") if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    npz_path = MODELS[args.model]
    with zipfile.ZipFile(npz_path) as archive:
        raw_ids = np.load(archive.open("item_ids.npy")).astype(str)
    keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
    if int(keep_mask.sum()) != len(rows_by_id):
        raise ValueError("activation/item membership mismatch")

    print(f"loading {args.model} frozen activations", file=sys.stderr, flush=True)
    X = np.load(npz_path)["X"][keep_mask]
    rows = [rows_by_id[iid] for iid in raw_ids[keep_mask]]
    ids = np.asarray([r["item_id"] for r in rows])
    y = np.asarray([int(r["label"]) for r in rows])
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    blocks = np.asarray([r["payload_block_id"] for r in rows])
    is_bench, is_casual = fmt == "benchmark", fmt == "casual"
    n_layers = X.shape[1]
    hidden_dim = X.shape[2]
    scores = np.full(
        (len(CONDITIONS), len(RANKS), len(REGIMES), n_layers, len(y)),
        np.nan,
        dtype=np.float32,
    )
    layer_sanity = {}
    completed_layers: list[int] = []
    manifest_path = partial_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"resume manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": PARTIAL_SCHEMA_VERSION,
            "model": args.model,
            "n_layers": n_layers,
            "n_items": len(ids),
            "item_id_digest": item_id_digest(ids),
            "conditions": list(CONDITIONS),
            "ranks": list(RANKS),
            "regimes": list(REGIMES),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"partial manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}")
        completed_layers = sorted(int(layer) for layer in manifest.get("completed_layers", []))
        if any(layer < 0 or layer >= n_layers for layer in completed_layers):
            raise ValueError("partial manifest contains an invalid layer")
        for layer in completed_layers:
            layer_scores, sanity, elapsed = load_partial_layer(partial_dir, layer, args.model, ids)
            expected_shape = (len(CONDITIONS), len(RANKS), len(REGIMES), len(y))
            if layer_scores.shape != expected_shape:
                raise ValueError(f"partial checkpoint shape mismatch at layer {layer}")
            scores[:, :, :, layer, :] = layer_scores
            layer_sanity[str(layer)] = sanity
            print(f"resumed layer={layer} ({len(completed_layers)}/{n_layers}) in {elapsed:.1f}s", file=sys.stderr, flush=True)
    else:
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
                "ranks": list(RANKS),
                "regimes": list(REGIMES),
                "completed_layers": [],
            },
        )

    completed_set = set(completed_layers)
    pending_layers = [layer for layer in range(n_layers) if layer not in completed_set]
    if pending_layers:
        slices = {
            layer: np.ascontiguousarray(X[:, layer, :])
            for layer in pending_layers
        }
        del X
        gc.collect()
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(layer_job)(args.model, layer, slices[layer], rows, y, fam, is_bench, is_casual)
            for layer in pending_layers
        )
        for layer, layer_scores, sanity, elapsed in results:
            scores[:, :, :, layer, :] = layer_scores
            layer_sanity[str(layer)] = sanity
            completed_layers = sorted([int(key) for key in layer_sanity])
            write_partial_layer(
                partial_dir,
                args.model,
                layer,
                layer_scores,
                sanity,
                elapsed,
                ids,
                completed_layers,
                n_layers,
            )
            print(
                f"layer={layer} done in {elapsed:.1f}s; partial={len(completed_layers)}/{n_layers}",
                file=sys.stderr,
                flush=True,
            )
        del slices
        gc.collect()
    else:
        del X
        gc.collect()

    aucs = auc_arrays(scores, y)
    reference = validate_references(args.model, aucs, args.tolerance)
    global_checks = {
        "max_factorial_identity_error": max(v["max_factorial_identity_error"] for v in layer_sanity.values()),
        "max_qb_t_qcperp": max(v["max_qb_t_qcperp"] for v in layer_sanity.values()),
        "max_qb_t_rperp": max(v["max_qb_t_rperp"] for v in layer_sanity.values()),
        "max_sequential_joint_projection_error": max(v["max_sequential_joint_projection_error"] for v in layer_sanity.values()),
        "held_family_basis_violations": sum(v["held_family_basis_violations"] for v in layer_sanity.values()),
    }
    if global_checks["max_factorial_identity_error"] > args.tolerance:
        raise AssertionError(global_checks)
    if global_checks["max_qb_t_qcperp"] > args.tolerance:
        raise AssertionError(global_checks)
    if global_checks["max_qb_t_rperp"] > args.tolerance:
        raise AssertionError(global_checks)
    if global_checks["max_sequential_joint_projection_error"] > args.tolerance:
        raise AssertionError(global_checks)
    if global_checks["held_family_basis_violations"] != 0:
        raise AssertionError(global_checks)
    seq_i = CONDITIONS.index("minus_b_cperp")
    joint_i = CONDITIONS.index("minus_bc_joint")
    score_equivalence = float(np.nanmax(np.abs(scores[seq_i] - scores[joint_i])))
    if score_equivalence > args.tolerance:
        raise AssertionError(f"sequential/joint OOF score mismatch: {score_equivalence}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.asarray(1),
        mode=np.asarray("development"),
        model=np.asarray(args.model),
        ranks=np.asarray(RANKS, dtype=np.int16),
        conditions=np.asarray(CONDITIONS),
        regimes=np.asarray(REGIMES),
        n_layers=np.asarray(n_layers),
        item_id=ids,
        payload_block_id=blocks,
        purpose_family_id=fam,
        fold_id=fam,
        format=fmt,
        y=y,
        scores=scores,
    )
    sanity_payload = {
        "schema_version": 1,
        "mode": "development",
        "model": args.model,
        "source_activation": str(npz_path),
        "source_activation_sha256": sha256_file(npz_path),
        "items_path": str(ITEMS_PATH),
        "items_sha256": sha256_file(ITEMS_PATH),
        "script_sha256": sha256_file(Path(__file__)),
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
        "partial_dir": str(partial_dir),
        "partial_manifest": str(manifest_path),
        "partial_schema_version": PARTIAL_SCHEMA_VERSION,
        "resumed": bool(args.resume),
        "basis": "leading singular directions of uncentered matched effect matrices; Q_B is a B-derived subspace and is not assumed pure because it can capture C",
        "hidden_dim": int(hidden_dim),
        "fold_semantics": "leave-one-purpose-family-out; held family excluded from every basis and probe fit",
        "c_capture": "cross-fitted held-family ||C_holdout Q_B||_F^2 / ||C_holdout||_F^2; numerators and denominators pooled across folds",
        "random_perp_control": "seeded rank(Q_C_perp_B) random subspace projected orthogonal to Q_B within the same fold/layer/rank; seeds recorded in layer_checks",
        "reference_reproduction": reference,
        "global_checks": global_checks,
        "max_sequential_joint_oof_score_error": score_equivalence,
        "identifier_fields": ["item_id", "payload_block_id", "purpose_family_id", "fold_id", "format", "y"],
        "layer_checks": layer_sanity,
        "interpretation_boundary": "Frozen-activation representational intervention plus probe retraining; not behavioral steering or model-output causality.",
    }
    sanity_path.write_text(json.dumps(sanity_payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} and {sanity_path}")


if __name__ == "__main__":
    main()
