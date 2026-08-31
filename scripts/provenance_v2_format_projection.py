"""Apply cross-fitted no-cue format bases to external provenance activations.

For each original purpose-family fold and layer:
  * estimate Q_nocue,r from only that fold's full320 training groups;
  * take the matching persisted stated-purpose probe from the same fold;
  * score external provenance rows raw and after h <- (I-QQ^T)h.
Fold scores are averaged exactly as in the frozen-probe scorer. No probe is
retrained, and no provenance row enters the basis. This is a representation-
score diagnostic, not a behavioral causal intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from format_erasure_development import orth_basis
from full320_v3_nocue_format_subspace import (
    nocue_payload_contrasts,
    training_group_keys,
)
from provenance_v1_score import (
    FOLDS,
    REGIMES,
    bootstrap_indices,
    point_metrics,
    sha256_file,
)


def affine_fold(bank, layer: int, regime: str, fold: str) -> tuple[np.ndarray, float]:
    p = f"L{layer:03d}::{regime}::fold_{fold}"
    coef = bank[f"{p}::coef"].astype(np.float64)
    mean = bank[f"{p}::scaler_mean"].astype(np.float64)
    scale = bank[f"{p}::scaler_scale"].astype(np.float64)
    weight = coef / scale
    offset = float(bank[f"{p}::intercept"]) - float(mean @ weight)
    return weight, offset


def delta_metrics(projected: dict, raw: dict) -> dict:
    scalar = [k for k, v in raw.items() if isinstance(v, (int, float))]
    return {f"delta_{k}": float(projected[k] - raw[k]) for k in scalar}


def add_bootstrap(record, raw_scores, projected_scores, y, fmt, source, pair, samples):
    keys = (
        "pooled_auc", "benchmark_auc", "casual_auc", "source_pair_auc_mean",
        "paired_benchmark_minus_casual_shift_eval",
        "paired_benchmark_minus_casual_shift_deploy",
        "provenance_x_format_score_interaction",
    )
    for ri, regime in enumerate(REGIMES):
        raw_vals = {k: [] for k in keys}
        proj_vals = {k: [] for k in keys}
        delta_vals = {k: [] for k in keys}
        for idx in samples:
            r = point_metrics(raw_scores[ri, idx], y[idx], fmt[idx], source[idx], pair[idx])
            p = point_metrics(projected_scores[ri, idx], y[idx], fmt[idx], source[idx], pair[idx])
            for key in keys:
                raw_vals[key].append(r[key])
                proj_vals[key].append(p[key])
                delta_vals[key].append(p[key] - r[key])
        record["regimes"][regime]["block_bootstrap_ci95"] = {
            "raw": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                    for k, v in raw_vals.items()},
            "projected": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                          for k, v in proj_vals.items()},
            "projected_minus_raw": {
                k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                for k, v in delta_vals.items()
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--reference-items", type=Path, required=True)
    ap.add_argument("--reference-activations", type=Path, required=True)
    ap.add_argument("--provenance-items", type=Path, required=True)
    ap.add_argument("--provenance-activations", type=Path, required=True)
    ap.add_argument("--probe-bank", type=Path, required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref_raw = [json.loads(x) for x in args.reference_items.read_text(encoding="utf-8").splitlines() if x]
    prov_raw = [json.loads(x) for x in args.provenance_items.read_text(encoding="utf-8").splitlines() if x]
    with np.load(args.reference_activations, allow_pickle=False) as z:
        ref_ids, Xref = z["item_ids"].astype(str), z["X"].astype(np.float32)
    with np.load(args.provenance_activations, allow_pickle=False) as z:
        prov_ids, Xprov = z["item_ids"].astype(str), z["X"].astype(np.float32)
    ref_by_id, prov_by_id = ({r["item_id"]: r for r in ref_raw},
                             {r["item_id"]: r for r in prov_raw})
    if set(ref_ids) != set(ref_by_id) or set(prov_ids) != set(prov_by_id):
        raise RuntimeError("activation/item ID mismatch")
    ref_rows = [ref_by_id[x] for x in ref_ids]
    prov_rows = [prov_by_id[x] for x in prov_ids]
    stated = np.array([r["render_slot"] == "stated" for r in ref_rows])
    fam = np.array([r.get("purpose_family_id", "") for r in ref_rows])
    y = np.array([r["label"] for r in prov_rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in prov_rows])
    source = np.array([r["source_dataset"] for r in prov_rows])
    pair = np.array([r["pair_id"] for r in prov_rows])
    layers = [int(x) for x in args.layers.split(",")]
    samples = bootstrap_indices(pair, source, args.n_bootstrap)
    out_layers = []

    with np.load(args.probe_bank, allow_pickle=False) as bank:
        for layer in layers:
            Xr = np.ascontiguousarray(Xref[:, layer, :])
            Xe = np.ascontiguousarray(Xprov[:, layer, :])
            nc = nocue_payload_contrasts(Xr, ref_rows)
            raw_scores = np.zeros((len(REGIMES), len(y)), dtype=np.float64)
            projected_scores = np.zeros_like(raw_scores)
            overlap = {r: [] for r in REGIMES}
            variance = []
            basis_rows = []
            centered = Xe.astype(np.float64) - Xe.mean(axis=0, dtype=np.float64)
            denom = float(np.sum(centered * centered))
            for fold in FOLDS:
                train = stated & (fam != fold)
                keys = training_group_keys(ref_rows, train)
                D = np.asarray([nc[block] for block, _ in keys])
                Q = orth_basis(D, args.rank)
                if Q.shape[1] != args.rank:
                    raise RuntimeError(f"{fold}: basis rank {Q.shape[1]}")
                variance.append(float(np.sum((centered @ Q) ** 2) / denom))
                basis_rows.append({"fold": fold, "n_training_groups": len(keys),
                                   "n_unique_payloads": len({k[0] for k in keys})})
                for ri, regime in enumerate(REGIMES):
                    w, offset = affine_fold(bank, layer, regime, fold)
                    projected_w = w - Q @ (Q.T @ w)
                    raw_scores[ri] += Xe @ w + offset
                    projected_scores[ri] += Xe @ projected_w + offset
                    overlap[regime].append(float(np.sum((Q.T @ w) ** 2) / np.sum(w ** 2)))
            raw_scores /= len(FOLDS)
            projected_scores /= len(FOLDS)
            regimes = {}
            for ri, regime in enumerate(REGIMES):
                raw_metric = point_metrics(raw_scores[ri], y, fmt, source, pair)
                projected_metric = point_metrics(projected_scores[ri], y, fmt, source, pair)
                regimes[regime] = {
                    "raw": raw_metric,
                    "projected": projected_metric,
                    "projected_minus_raw": delta_metrics(projected_metric, raw_metric),
                    "probe_weight_fraction_in_qnocue_by_fold": overlap[regime],
                    "probe_weight_fraction_in_qnocue_mean": float(np.mean(overlap[regime])),
                }
            record = {
                "layer": layer,
                "normalized_depth": layer / (Xprov.shape[1] - 1),
                "regimes": regimes,
                "external_centered_variance_captured_by_fold": variance,
                "external_centered_variance_captured_mean": float(np.mean(variance)),
                "basis_rows_by_fold": basis_rows,
            }
            add_bootstrap(record, raw_scores, projected_scores, y, fmt, source, pair, samples)
            out_layers.append(record)
            print(f"layer {layer} projected", flush=True)
            del Xr, Xe, centered

    manifest = json.loads(args.provenance_items.with_name("freeze_manifest.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "experiment": f"{manifest['experiment']}_external_qnocue_projection",
        "model_key": args.model_key,
        "rank": args.rank,
        "basis": "fold-specific uncentered SVD of no-cue matched benchmark-minus-casual contrasts",
        "basis_uses_provenance_rows": False,
        "probe_retrained_after_projection": False,
        "claim_boundary": "frozen activation-score diagnostic, not behavioral causality",
        "n_bootstrap": args.n_bootstrap,
        "same_block_sample_across_raw_projected_layers_regimes": True,
        "reference_items_sha256": sha256_file(args.reference_items),
        "reference_activations_sha256": sha256_file(args.reference_activations),
        "provenance_items_sha256": sha256_file(args.provenance_items),
        "provenance_activations_sha256": sha256_file(args.provenance_activations),
        "probe_bank_sha256": sha256_file(args.probe_bank),
        "layers": out_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "layers": layers}))


if __name__ == "__main__":
    main()
