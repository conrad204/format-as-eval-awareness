"""Does a format-associated subspace explain cross-format failure in binding_v1?

For each layer, rank, and held-out relational family, a format-associated basis
Q_format is estimated from matched benchmark-minus-casual contrasts using ONLY
the training families, projected out with I - QQ^T, and the family-held-out
binding probes are refit on the projected representation. Cross-fitting keeps
the held-out family out of both the basis and the probe.
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
                                probe_metrics, sha256_file, summarize_families)

RANKS = (1, 4, 16, 64)


def format_contrasts(Xl: np.ndarray, block: np.ndarray, fmt: np.ndarray,
                     rows_mask: np.ndarray) -> np.ndarray:
    by_block: dict[str, list[int]] = {}
    for i in np.flatnonzero(rows_mask):
        by_block.setdefault(block[i], []).append(i)
    rows = []
    for members in by_block.values():
        ii = np.asarray(members)
        b = ii[fmt[ii] == "benchmark"]
        c = ii[fmt[ii] == "casual"]
        if len(b) == 0 or len(c) == 0:
            continue
        rows.append(Xl[b].astype(np.float64).mean(axis=0)
                    - Xl[c].astype(np.float64).mean(axis=0))
    return np.asarray(rows)


def one_family(Xl: np.ndarray, y: np.ndarray, fmt: np.ndarray, family: np.ndarray,
               block: np.ndarray, held: str, rank: int) -> tuple[str, dict[str, np.ndarray]]:
    train, test = family != held, family == held
    D = format_contrasts(Xl, block, fmt, train)
    Q = orth_basis(D, rank)
    if Q.shape[1] != rank:
        raise RuntimeError(f"basis rank {Q.shape[1]} != {rank} for held-out {held}")
    Xp = remove(Xl.astype(np.float32), Q.astype(np.float32))
    out = {}
    for regime in REGIMES:
        _, scores = fit_score(Xp, y, fmt, train, test, regime)
        out[regime] = scores
    return held, out


def analyze_layer(layer: int, Xl: np.ndarray, y: np.ndarray, fmt: np.ndarray,
                  family: np.ndarray, block: np.ndarray, raw_per_family: dict,
                  ranks: tuple[int, ...], jobs: int) -> dict:
    families = sorted(set(family.tolist()))
    result = {"layer": layer, "by_rank": {}}
    for rank in ranks:
        scores = {r: np.full(len(y), np.nan, dtype=np.float32) for r in REGIMES}
        fitted = Parallel(n_jobs=min(jobs, len(families)), backend="threading")(
            delayed(one_family)(Xl, y, fmt, family, block, held, rank) for held in families)
        for held, per_regime in fitted:
            mask = family == held
            for regime, values in per_regime.items():
                scores[regime][mask] = values
        if any(np.isnan(v).any() for v in scores.values()):
            raise RuntimeError("incomplete projected predictions")
        projected = {n: probe_metrics(y, fmt, scores["joint"], scores["benchmark"],
                                      scores["casual"], family == n) for n in families}
        delta = {n: {"delta_joint_format_auc": projected[n]["joint_format_auc"] - raw_per_family[n]["joint_format_auc"],
                     "delta_cross_format_mean_auc": projected[n]["cross_format_mean_auc"] - raw_per_family[n]["cross_format_mean_auc"],
                     "delta_transfer_gap": projected[n]["transfer_gap"] - raw_per_family[n]["transfer_gap"]}
                 for n in families}
        result["by_rank"][str(rank)] = {
            "projected_per_family": projected,
            "projected_family_unit_summary": summarize_families(projected),
            "projected_minus_raw_per_family": delta,
            "projected_minus_raw_family_unit_summary": summarize_families(delta),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--raw-analysis", type=Path, required=True)
    ap.add_argument("--fixed-layer", type=int)
    ap.add_argument("--layers", help="comma-separated subset of layers")
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANKS))
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
    raw = json.loads(args.raw_analysis.read_text(encoding="utf-8"))
    raw_by_layer = {rec["layer"]: rec["binding_to_binding"]["per_family"] for rec in raw["layers"]}
    ranks = tuple(int(x) for x in args.ranks.split(","))

    if X.ndim == 3:
        available = list(range(X.shape[1]))
    elif X.ndim == 2:
        if args.fixed_layer is None:
            raise RuntimeError("--fixed-layer required for 2-D activations")
        available = [args.fixed_layer]
    else:
        raise RuntimeError(f"unexpected activation shape {X.shape}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else available

    out_layers = []
    for L in layers:
        Xl = X[:, available.index(L), :] if X.ndim == 3 else X
        out_layers.append(analyze_layer(L, Xl, y, fmt, family, block,
                                        raw_by_layer[L], ranks, args.jobs))
        print(f"format-subspace layer {L} complete", flush=True)
    result = {
        "schema_version": 1,
        "experiment": "binding_v1_format_subspace",
        "model_key": args.model_key,
        "basis": "matched benchmark-minus-casual block contrasts from training families only",
        "cross_fitting": "leave-one-entire-relational-family-out for both basis and probe",
        "ranks": list(ranks),
        "inference_unit": "binding_family",
        "items_sha256": sha256_file(args.items),
        "activations_sha256": sha256_file(args.activations),
        "layers": out_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "n_layers": len(out_layers)}, indent=2))


if __name__ == "__main__":
    main()
