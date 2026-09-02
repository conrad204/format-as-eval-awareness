"""Is the format basis SPECIAL, or does deleting variance help transfer generally?

The gap-removal result projects out a rank-r format-associated basis and reports
that cross-format transfer improves. At rank 64 that basis captures the large
majority of held-out centered variance, so rank alone is not a fair control: any
high-variance subspace might do the same work.

For each layer, rank and held-out relational family this script compares the
SAME erasure endpoint across four bases, all cross-fitted on training families
only:

  format    matched benchmark-minus-casual block contrasts (the paper's basis)
  top_pc    leading principal components of the training activations
  random_hv random rank-r subspace drawn inside the top-64 PC subspace
            (the repository's high-variance conditional-null convention)
  nuisance  a non-format surface contrast (payload-parity within each format),
            i.e. a real textual contrast that is orthogonal to the format axis

Captured held-out variance is recorded for every basis, so comparisons are
variance-transparent rather than rank-only. The empirical p-value is the
fraction of random high-variance draws that reduce the gap at least as much as
the format basis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import orth_basis, remove  # noqa: E402
from binding_v1_analyze import (EXPECTED_ITEMS_SHA256, REGIMES, fit_score,  # noqa: E402
                                probe_metrics, sha256_file)

DEFAULT_TOP_PC_POOL = 64


def block_contrast(Xl: np.ndarray, block: np.ndarray, mask: np.ndarray,
                   split: np.ndarray) -> np.ndarray:
    """Mean(split==True) - mean(split==False) within each block, stacked."""
    by: dict[str, list[int]] = {}
    for i in np.flatnonzero(mask):
        by.setdefault(block[i], []).append(i)
    rows = []
    for members in by.values():
        ii = np.asarray(members)
        a, b = ii[split[ii]], ii[~split[ii]]
        if len(a) == 0 or len(b) == 0:
            continue
        rows.append(Xl[a].astype(np.float64).mean(axis=0)
                    - Xl[b].astype(np.float64).mean(axis=0))
    return np.asarray(rows)


def top_pcs(Xl: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    Xc = Xl[mask].astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return np.ascontiguousarray(Vt[:k].T)


def captured_variance(Xl: np.ndarray, mask: np.ndarray, Q: np.ndarray) -> float:
    Xc = Xl[mask].astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    denom = float(np.sum(Xc * Xc))
    return float(np.sum((Xc @ Q) ** 2) / denom) if denom > 0 else float("nan")


def gap_for_basis(Xl: np.ndarray, y: np.ndarray, fmt: np.ndarray, family: np.ndarray,
                  held: str, Q: np.ndarray | None) -> tuple[float, dict[str, float]]:
    train, test = family != held, family == held
    Xp = Xl if Q is None else remove(Xl.astype(np.float32), Q.astype(np.float32))
    scores = {}
    for regime in REGIMES:
        _, values = fit_score(Xp, y, fmt, train, test, regime)
        full = np.full(len(y), np.nan, dtype=np.float32)
        full[test] = values
        scores[regime] = full
    m = probe_metrics(y, fmt, scores["joint"], scores["benchmark"], scores["casual"], test)
    return m["transfer_gap"], m


def one_cell(layer: int, Xl: np.ndarray, y: np.ndarray, fmt: np.ndarray, family: np.ndarray,
             block: np.ndarray, mapping: np.ndarray, selected: np.ndarray, held: str,
             rank: int, n_draws: int, seed: int, pool_k: int) -> dict:
    train = family != held
    is_bench = fmt == "benchmark"
    raw_gap, raw_metrics = gap_for_basis(Xl, y, fmt, family, held, None)

    bases: dict[str, np.ndarray] = {}
    bases["format"] = orth_basis(block_contrast(Xl, block, train, is_bench), rank)
    bases["top_pc"] = top_pcs(Xl, train, rank)
    # Non-format control basis built the SAME way as the format basis (within-block
    # contrasts of real textual variation), but along the route-assignment and
    # selected-route axes instead of the benchmark/casual axis. Neither axis alone
    # predicts the label, and neither is the format contrast.
    nuis_rows = [block_contrast(Xl, block, train, mapping == 1),
                 block_contrast(Xl, block, train, selected == 1)]
    nuis_rows = [r for r in nuis_rows if len(r)]
    if not nuis_rows:
        raise RuntimeError(f"empty nuisance contrast for held-out {held}")
    bases["nuisance"] = orth_basis(np.vstack(nuis_rows), rank)

    pool = top_pcs(Xl, train, pool_k)
    rng = np.random.default_rng(seed)
    out = {"layer": layer, "rank": rank, "held_out_family": held,
           "raw_transfer_gap": raw_gap, "raw_joint_auc": raw_metrics["joint_format_auc"],
           "by_basis": {}}
    for name, Q in bases.items():
        gap, met = gap_for_basis(Xl, y, fmt, family, held, Q)
        out["by_basis"][name] = {
            "projected_transfer_gap": gap, "delta_transfer_gap": gap - raw_gap,
            "projected_joint_auc": met["joint_format_auc"],
            "delta_joint_auc": met["joint_format_auc"] - raw_metrics["joint_format_auc"],
            "captured_heldout_variance": captured_variance(Xl, family == held, Q),
        }
    draws = []
    for _ in range(n_draws):
        R = rng.standard_normal((pool_k, rank))
        Qr, _ = np.linalg.qr(R)
        Q = pool @ Qr
        gap, met = gap_for_basis(Xl, y, fmt, family, held, Q)
        draws.append({"delta_transfer_gap": gap - raw_gap,
                      "delta_joint_auc": met["joint_format_auc"] - raw_metrics["joint_format_auc"],
                      "captured_heldout_variance": captured_variance(Xl, family == held, Q)})
    d_fmt = out["by_basis"]["format"]["delta_transfer_gap"]
    deltas = np.array([d["delta_transfer_gap"] for d in draws])
    out["random_hv"] = {
        "n_draws": n_draws,
        "delta_transfer_gap_mean": float(deltas.mean()),
        "delta_transfer_gap_q05": float(np.quantile(deltas, 0.05)),
        "delta_transfer_gap_min": float(deltas.min()),
        "captured_heldout_variance_mean": float(np.mean([d["captured_heldout_variance"] for d in draws])),
        "empirical_p_random_at_least_as_good": float(np.mean(deltas <= d_fmt)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--ranks", default="4,16,64")
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--top-pc-pool", type=int, default=DEFAULT_TOP_PC_POOL,
                    help="random draws are taken inside this many leading PCs; must exceed rank")
    ap.add_argument("--fixed-layer", type=int)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if sha256_file(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("frozen binding_v1 items hash mismatch")
    rows_raw = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    with np.load(args.activations, allow_pickle=False) as z:
        ids, X = z["item_ids"].astype(str), z["X"].astype(np.float32)
    by_id = {r["item_id"]: r for r in rows_raw}
    if len(ids) != len(rows_raw) or set(ids) != set(by_id):
        raise RuntimeError("activation IDs do not match frozen binding_v1 bank")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    family = np.array([r["binding_family"] for r in rows])
    block = np.array([r["binding_block_id"] for r in rows])
    mapping = np.array([r["binding_mapping"] for r in rows], dtype=np.int8)
    selected = np.array([r["binding_selected"] for r in rows], dtype=np.int8)
    families = sorted(set(family.tolist()))

    if X.ndim == 3:
        available = list(range(X.shape[1]))
    elif X.ndim == 2:
        if args.fixed_layer is None:
            raise RuntimeError("--fixed-layer required for 2-D activations")
        available = [args.fixed_layer]
    else:
        raise RuntimeError(f"unexpected activation shape {X.shape}")
    layers = [int(x) for x in args.layers.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    cells = []
    for L in layers:
        Xl = X[:, available.index(L), :] if X.ndim == 3 else X
        for rank in ranks:
            got = Parallel(n_jobs=min(args.jobs, len(families)), backend="threading")(
                delayed(one_cell)(L, Xl, y, fmt, family, block, mapping, selected, held,
                                  rank, args.n_draws, args.seed + i, args.top_pc_pool)
                for i, held in enumerate(families))
            cells.extend(got)
            fam_fmt = float(np.mean([c["by_basis"]["format"]["delta_transfer_gap"] for c in got]))
            fam_rand = float(np.mean([c["random_hv"]["delta_transfer_gap_mean"] for c in got]))
            fam_pc = float(np.mean([c["by_basis"]["top_pc"]["delta_transfer_gap"] for c in got]))
            fam_nui = float(np.mean([c["by_basis"]["nuisance"]["delta_transfer_gap"] for c in got]))
            print(f"L{L} r{rank}: dGap format={fam_fmt:+.4f} top_pc={fam_pc:+.4f} "
                  f"nuisance={fam_nui:+.4f} random_hv={fam_rand:+.4f}", flush=True)

    summary = {}
    for L in layers:
        for rank in ranks:
            sel = [c for c in cells if c["layer"] == L and c["rank"] == rank]
            if not sel:
                continue
            key = f"L{L}::r{rank}"
            fmt_d = np.array([c["by_basis"]["format"]["delta_transfer_gap"] for c in sel])
            summary[key] = {
                "n_families": len(sel),
                "family_mean_delta_gap": {
                    name: float(np.mean([c["by_basis"][name]["delta_transfer_gap"] for c in sel]))
                    for name in ("format", "top_pc", "nuisance")},
                "family_mean_delta_gap_random_hv": float(
                    np.mean([c["random_hv"]["delta_transfer_gap_mean"] for c in sel])),
                "family_mean_captured_variance": {
                    name: float(np.mean([c["by_basis"][name]["captured_heldout_variance"] for c in sel]))
                    for name in ("format", "top_pc", "nuisance")},
                "family_mean_captured_variance_random_hv": float(
                    np.mean([c["random_hv"]["captured_heldout_variance_mean"] for c in sel])),
                "families_format_beats_random_mean": int(np.sum(
                    [c["by_basis"]["format"]["delta_transfer_gap"]
                     < c["random_hv"]["delta_transfer_gap_mean"] for c in sel])),
                "mean_empirical_p_random_at_least_as_good": float(np.mean(
                    [c["random_hv"]["empirical_p_random_at_least_as_good"] for c in sel])),
                "family_mean_delta_joint_auc": {
                    name: float(np.mean([c["by_basis"][name]["delta_joint_auc"] for c in sel]))
                    for name in ("format", "top_pc", "nuisance")},
                "format_delta_gap_values": {c["held_out_family"]: float(v)
                                            for c, v in zip(sel, fmt_d)},
            }
    result = {
        "schema_version": 1,
        "experiment": "binding_v1_erasure_conditional_null",
        "model_key": args.model_key,
        "question": "does the format basis remove the transfer gap beyond a variance-matched high-variance subspace?",
        "top_pc_pool": args.top_pc_pool,
        "n_random_draws_per_cell": args.n_draws,
        "inference_unit": "binding_family",
        "items_sha256": sha256_file(args.items),
        "activations_sha256": sha256_file(args.activations),
        "summary": summary,
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
