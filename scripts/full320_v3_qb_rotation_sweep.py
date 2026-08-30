#!/usr/bin/env python3
"""Is the portability rescue inside Q_B carried by its C-aligned directions?

The emergency run falsified the residual-C story (specific_C < 0 on 8B). This is
the narrower successor: rotate the *existing* rank-64 Q_B so its columns are
ordered by how strongly they capture the training interaction matrix C, then
compare three rank-16 subspaces that all live inside the same parent Q_B.

    M       = C_train @ Q_B
    M       = U S Vᵀ                       (descending singular values)
    Q_B_rot = Q_B @ V

    topC16    = Q_B_rot[:, :16]            most C-aligned directions in Q_B
    bottomC16 = Q_B_rot[:, -16:]           least C-aligned directions in Q_B
    randomB16 = Q_B @ R                    seeded random 16-D rotation in span(Q_B)

Equal rank, common parent subspace, so the contrast isolates C-alignment rather
than "being in Q_B". Each condition is removed from the raw activations, and the
rescue is scored against the raw baseline already computed by the emergency run --
raw, minus_b, minus_b_cperp and minus_b_rperp are NOT recomputed here.

Probe contract, folds, regimes, item set and no-leakage rules are unchanged:
45 probe fits per layer (3 conditions x 5 folds x 3 regimes).
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
from format_erasure_development import orth_basis  # noqa: E402
from full320_v3_emergency_cperp_sweep import (  # noqa: E402
    RANK,
    _peek_x_header,
    gram_orth_basis,
    load_partial_layer,
    partial_layer_path,
    prepare_shards,
    project_out_with_energy,
    shard_path,
)
from full320_v3_factorial_residual_sweep import (  # noqa: E402
    ITEMS_PATH,
    MODELS,
    _atomic_json,
    _atomic_npz,
    captured_energy,
    factorial_effects,
    fit_regimes,
    item_id_digest,
    sha256_file,
)

CONDITIONS = ("topC16", "bottomC16", "randomB16")
REGIMES = ("joint", "b2c", "c2b")
SUB_RANK = 16
PARTIAL_SCHEMA_VERSION = 1
DEFAULT_TOL = 1e-6
RANDOM_B16_SEED_BASE = 40_260_829

_SHARED: dict = {}


def rotate_qb_by_c_alignment(q_b: np.ndarray, C_train: np.ndarray):
    """Order the columns of Q_B by how strongly they capture C_train."""
    if q_b.shape[1] < 2 * SUB_RANK:
        raise ValueError(f"Q_B rank {q_b.shape[1]} too small for disjoint top/bottom {SUB_RANK}")
    M = np.asarray(C_train, dtype=np.float64) @ q_b
    _, singular, vh = np.linalg.svd(M, full_matrices=False)
    return q_b @ vh.T, singular


def random_subspace_inside(q_b: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Seeded random orthonormal rank-`rank` subspace of span(Q_B)."""
    draws = np.random.default_rng(seed).standard_normal((q_b.shape[1], rank))
    coefficients, _ = np.linalg.qr(draws)
    return q_b @ coefficients[:, :rank]


def containment_error(sub: np.ndarray, q_b: np.ndarray) -> float:
    """max |Q_sub - Q_B Q_Bᵀ Q_sub|; zero iff span(Q_sub) ⊆ span(Q_B)."""
    return float(np.max(np.abs(sub - q_b @ (q_b.T @ sub))))


def init_worker(shared: dict) -> None:
    _SHARED.clear()
    _SHARED.update(shared)


def layer_job(layer: int):
    rows = _SHARED["rows"]
    y = _SHARED["y"]
    fam = _SHARED["fam"]
    is_bench = _SHARED["is_bench"]
    is_casual = _SHARED["is_casual"]
    basis = gram_orth_basis if _SHARED["basis_backend"] == "gram" else orth_basis

    started = time.time()
    X = np.asarray(np.load(shard_path(_SHARED["shard_dir"], layer), mmap_mode="r"), dtype=np.float32)
    ci = {name: i for i, name in enumerate(CONDITIONS)}
    scores = np.full((len(CONDITIONS), len(REGIMES), len(y)), np.nan, dtype=np.float32)

    max_containment = 0.0
    max_orthonormality = 0.0
    held_family_violations = 0
    capture_folds: dict[str, list[dict]] = {name: [] for name in CONDITIONS}
    energy_folds: dict[str, list[dict]] = {name: [] for name in CONDITIONS}
    alignment_folds: list[dict] = []

    for fold_pos, fold in enumerate(sorted(set(fam))):
        train_mask = fam != fold
        test_mask = fam == fold

        B, C, _, _, basis_families, _ = factorial_effects(X, rows, train_mask)
        held_family_violations += int(np.sum(basis_families == fold))
        _, C_holdout, _, _, heldout_families, _ = factorial_effects(X, rows, test_mask)
        if set(heldout_families) != {fold}:
            raise AssertionError("held-out C matrix does not contain exactly the held family")

        q_b = basis(B, RANK)
        q_rot, singular = rotate_qb_by_c_alignment(q_b, C)
        subspaces = {
            "topC16": q_rot[:, :SUB_RANK],
            "bottomC16": q_rot[:, -SUB_RANK:],
            "randomB16": random_subspace_inside(
                q_b, SUB_RANK, RANDOM_B16_SEED_BASE + layer * 10_007 + fold_pos * 101 + SUB_RANK
            ),
        }
        total_alignment = float(np.sum(singular ** 2))
        alignment_folds.append(
            {
                "fold": str(fold),
                "qb_rank": int(q_b.shape[1]),
                "top16_alignment_energy_fraction": float(np.sum(singular[:SUB_RANK] ** 2) / total_alignment),
                "bottom16_alignment_energy_fraction": float(
                    np.sum(singular[-SUB_RANK:] ** 2) / total_alignment
                ),
                "singular_values": singular.tolist(),
            }
        )

        for name, sub in subspaces.items():
            max_containment = max(max_containment, containment_error(sub, q_b))
            max_orthonormality = max(
                max_orthonormality, float(np.max(np.abs(sub.T @ sub - np.eye(sub.shape[1]))))
            )
            numerator, denominator, fraction = captured_energy(C_holdout, sub)
            capture_folds[name].append(
                {"fold": str(fold), "numerator": numerator, "denominator": denominator, "fraction": fraction}
            )
            Xt, energy = project_out_with_energy(X, sub, test_mask)
            energy["fold"] = str(fold)
            energy_folds[name].append(energy)
            for regime_i, values in enumerate(
                fit_regimes(Xt, y, train_mask, test_mask, is_bench, is_casual)
            ):
                scores[ci[name], regime_i, test_mask] = values[test_mask]
            del Xt
        gc.collect()

    def pooled(folds: list[dict], numerator_key: str, denominator_key: str) -> float:
        return sum(v[numerator_key] for v in folds) / sum(v[denominator_key] for v in folds)

    sanity = {
        "max_subspace_containment_error": max_containment,
        "max_orthonormality_error": max_orthonormality,
        "held_family_basis_violations": held_family_violations,
        "max_abs_activation": float(np.max(np.abs(X))),
        "c_capture": {
            name: {
                "cross_fitted_fraction": pooled(folds, "numerator", "denominator"),
                "nominal_isotropic_chance": SUB_RANK / X.shape[1],
                "by_fold": folds,
            }
            for name, folds in capture_folds.items()
        },
        "energy_removed": {
            name: {
                "pooled_fraction_all": pooled(folds, "removed_all", "total_all"),
                "pooled_fraction_heldout": pooled(folds, "removed_heldout", "total_heldout"),
                "by_fold": folds,
            }
            for name, folds in energy_folds.items()
        },
        "c_alignment_spectrum": alignment_folds,
        "basis_backend": _SHARED["basis_backend"],
    }
    return layer, scores, sanity, time.time() - started


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
            "ranks": [SUB_RANK],
            "regimes": list(REGIMES),
            "completed_layers": sorted(completed_layers),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--npz", default=None)
    ap.add_argument("--items", default=str(ITEMS_PATH))
    ap.add_argument("--shard-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sanity-out", default=None)
    ap.add_argument("--partial-dir", default=None)
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", "1")))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--basis-backend", choices=("gram", "svd"), default="gram")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOL)
    args = ap.parse_args()

    npz_path = Path(args.npz) if args.npz else MODELS[args.model]
    items_path = Path(args.items)
    out_path = Path(args.out) if args.out else npz_path.parent / f"qb_rotation_scores_{args.model}.npz"
    sanity_path = Path(args.sanity_out) if args.sanity_out else out_path.with_suffix(".sanity.json")
    partial_dir = Path(args.partial_dir) if args.partial_dir else out_path.with_name(f"{out_path.name}.partial")
    timing_path = Path(args.timing_out) if args.timing_out else out_path.with_suffix(".timing.json")
    shard_dir = Path(args.shard_dir) if args.shard_dir else out_path.with_name(f"shards_{args.model}")
    if not args.benchmark and (out_path.exists() or sanity_path.exists()):
        raise FileExistsError(f"immutable output exists: {out_path} or {sanity_path}")

    import zipfile

    rows_all = [json.loads(line) for line in items_path.open(encoding="utf-8") if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    with zipfile.ZipFile(npz_path) as archive:
        raw_ids = np.load(archive.open("item_ids.npy")).astype(str)
    keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
    if int(keep_mask.sum()) != len(rows_by_id):
        raise ValueError("activation/item membership mismatch")

    # Read the header only; shards are expanded after we know what is still pending,
    # so an assembly-only rerun costs no expansion at all.
    shape, _ = _peek_x_header(npz_path)
    n_layers, hidden_dim = int(shape[1]), int(shape[2])
    shard_seconds = 0.0
    rows = [rows_by_id[iid] for iid in raw_ids[keep_mask]]
    ids = np.asarray([r["item_id"] for r in rows])
    y = np.asarray([int(r["label"]) for r in rows])
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    blocks = np.asarray([r["payload_block_id"] for r in rows])
    is_bench, is_casual = fmt == "benchmark", fmt == "casual"

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
            "ranks": [SUB_RANK],
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

    requested = list(range(n_layers)) if args.layers is None else [int(v) for v in args.layers]
    pending = [layer for layer in requested if layer not in set(completed_layers)]
    if pending:
        _, _, shard_seconds = prepare_shards(npz_path, keep_mask, shard_dir, layers=pending)
    shared = {
        "rows": rows,
        "y": y,
        "fam": fam,
        "is_bench": is_bench,
        "is_casual": is_casual,
        "shard_dir": shard_dir,
        "basis_backend": args.basis_backend,
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

    finished = sorted(float(v) for v in layer_seconds.values())
    timing = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_jobs": args.n_jobs,
        "shard_expand_seconds": shard_seconds,
        "layers_completed": len(layer_sanity),
        "wall_seconds_this_invocation": time.time() - run_started,
        "median_layer_seconds": float(np.median(finished)) if finished else None,
        "layer_seconds": layer_seconds,
        "host": {"node": platform.node(), "cpu_count": os.cpu_count(), "python": platform.python_version()},
    }
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")

    if args.benchmark or len(layer_sanity) != n_layers:
        print(f"partial run: {len(layer_sanity)}/{n_layers} layers; no final artifact", file=sys.stderr, flush=True)
        return

    global_checks = {
        "max_subspace_containment_error": max(v["max_subspace_containment_error"] for v in layer_sanity.values()),
        "max_orthonormality_error": max(v["max_orthonormality_error"] for v in layer_sanity.values()),
        "held_family_basis_violations": sum(v["held_family_basis_violations"] for v in layer_sanity.values()),
        "max_abs_activation": max(v["max_abs_activation"] for v in layer_sanity.values()),
    }
    budget = args.tolerance * max(1.0, global_checks["max_abs_activation"])
    if global_checks["max_subspace_containment_error"] > budget:
        raise AssertionError(global_checks)
    if global_checks["max_orthonormality_error"] > args.tolerance:
        raise AssertionError(global_checks)
    if global_checks["held_family_basis_violations"] != 0:
        raise AssertionError(global_checks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.asarray(1),
        mode=np.asarray("development"),
        model=np.asarray(args.model),
        ranks=np.asarray([SUB_RANK], dtype=np.int16),
        conditions=np.asarray(CONDITIONS),
        regimes=np.asarray(REGIMES),
        n_layers=np.asarray(n_layers),
        item_id=ids,
        payload_block_id=blocks,
        purpose_family_id=fam,
        fold_id=fam,
        format=fmt,
        y=y,
        scores=scores[:, None, :, :, :],
    )
    sanity_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "development",
                "scope": "within-Q_B C-alignment rank-16 test; raw baseline reused from the emergency run",
                "model": args.model,
                "conditions": list(CONDITIONS),
                "sub_rank": SUB_RANK,
                "parent_rank": RANK,
                "source_activation": str(npz_path),
                "source_activation_sha256": sha256_file(npz_path),
                "items_path": str(items_path),
                "items_sha256": sha256_file(items_path),
                "script_sha256": sha256_file(Path(__file__)),
                "output_path": str(out_path),
                "output_sha256": sha256_file(out_path),
                "rotation": "M = C_train @ Q_B; M = U S Vᵀ; Q_B_rot = Q_B @ V, columns ordered by C alignment",
                "random_control": f"seeded orthonormal 16-D rotation inside span(Q_B), seed {RANDOM_B16_SEED_BASE} + layer*10007 + fold*101 + 16",
                "fold_semantics": "leave-one-purpose-family-out; held family excluded from Q_B, from the C rotation, and from every probe fit",
                "global_checks": global_checks,
                "timing": timing,
                "layer_checks": layer_sanity,
                "interpretation_boundary": "Frozen-activation representational intervention plus probe retraining; not behavioral steering or model-output causality.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path} and {sanity_path}")


if __name__ == "__main__":
    main()
