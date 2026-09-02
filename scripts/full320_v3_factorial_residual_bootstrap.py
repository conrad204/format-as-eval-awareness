#!/usr/bin/env python3
"""Correlation-aware bootstrap for the factorial B/C residual sweep.

One payload-block draw is reused across every layer, rank, condition, and AUC
regime in each replicate.  Each replicate recomputes and then integrates the
entire normalized-depth transfer-gap curve; pointwise confidence intervals are
never integrated.  Bootstrap AUC uses block multiplicities as sample weights,
which avoids the duplicated-score tie bug induced by physically duplicating
resampled rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")
N_BOOT = 1000
RNG_SEED = 0
_trapz = getattr(np, "trapezoid", None) or np.trapz


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def prepare_auc_order(scores: np.ndarray, y: np.ndarray):
    """Pre-sort fixed OOF scores and record their exact tie groups.

    Float32 OOF scores do contain exact ties, so tie structure is carried
    explicitly and tied pairs get half credit -- exactly the Mann-Whitney
    statistic ``sklearn.metrics.roc_auc_score`` computes. Scores are fixed across
    bootstrap replicates, so groups are computed once and only weights change.
    """
    order = np.argsort(scores, axis=1, kind="mergesort")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    positive = y[order] == 1
    n = scores.shape[1]
    index = np.arange(n)
    is_new = np.empty_like(sorted_scores, dtype=bool)
    is_new[:, 0] = True
    is_new[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    group_start = np.maximum.accumulate(np.where(is_new, index, 0), axis=1)
    reversed_new = np.empty_like(is_new)
    reversed_new[:, -1] = True
    reversed_new[:, :-1] = is_new[:, 1:]
    group_end = np.minimum.accumulate(np.where(reversed_new, index, n - 1)[:, ::-1], axis=1)[:, ::-1]
    return order, positive, group_start.astype(np.int64), group_end.astype(np.int64)


def weighted_auc_from_order(prepared, weights: np.ndarray) -> np.ndarray:
    """Weighted Mann-Whitney AUC with exact half credit for tied scores."""
    order, positive, group_start, group_end = prepared
    w = weights[order]
    negative_weight = w * ~positive
    # cumulative[:, k] is the negative weight strictly below sorted position k.
    cumulative = np.zeros((w.shape[0], w.shape[1] + 1), dtype=np.float64)
    np.cumsum(negative_weight, axis=1, out=cumulative[:, 1:])
    below = np.take_along_axis(cumulative, group_start, axis=1)
    within = np.take_along_axis(cumulative, group_end + 1, axis=1) - below
    positive_weight = w * positive
    numerator = np.sum(positive_weight * (below + 0.5 * within), axis=1)
    total_positive = np.sum(positive_weight, axis=1)
    total_negative = cumulative[:, -1]
    if np.any(total_positive == 0) or np.any(total_negative == 0):
        raise ValueError("bootstrap draw omitted a class")
    return numerator / (total_positive * total_negative)


def specificity_components(i_b, i_bc, i_br):
    """Return extra_C, extra_random, and their paired difference."""
    extra_c = i_b - i_bc
    extra_random = i_b - i_br
    return extra_c, extra_random, extra_c - extra_random


def point_aucs(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    # scores: condition, rank, regime, layer, item
    out = np.full(scores.shape[:-1], np.nan, dtype=np.float64)
    for index in np.ndindex(scores.shape[:-1]):
        values = scores[index]
        mask = np.isfinite(values)
        out[index] = roc_auc_score(y[mask], values[mask])
    return out


def analyze_model(model: str, n_boot: int) -> tuple[dict, list[dict]]:
    path = RESULTS / f"full320_v3_factorial_residual_scores_{model}.npz"
    data = np.load(path, allow_pickle=True)
    conditions = [str(v) for v in data["conditions"]]
    ranks = [int(v) for v in data["ranks"]]
    regimes = [str(v) for v in data["regimes"]]
    scores = data["scores"].astype(np.float64)
    y = data["y"].astype(np.int8)
    blocks = data["payload_block_id"].astype(str)
    n_layers = int(data["n_layers"])
    q = np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
    cidx = {name: i for i, name in enumerate(conditions)}
    ridx = {name: i for i, name in enumerate(regimes)}

    expected_shape = (len(conditions), len(ranks), len(regimes), n_layers, len(y))
    if scores.shape != expected_shape:
        raise ValueError(f"unexpected score shape {scores.shape}, expected {expected_shape}")
    masks = {}
    for regime, regime_i in ridx.items():
        mask = np.isfinite(scores[0, 0, regime_i, 0])
        if not np.all(np.isfinite(scores[:, :, regime_i, :, mask])):
            raise ValueError(f"nonconstant finite-score mask for {regime}")
        if np.any(np.isfinite(scores[:, :, regime_i, :, ~mask])):
            raise ValueError(f"nonconstant NaN-score mask for {regime}")
        masks[regime] = mask

    auc = point_aucs(scores, y)
    gap = auc[:, :, ridx["joint"]] - (
        auc[:, :, ridx["b2c"]] + auc[:, :, ridx["c2b"]]
    ) / 2.0
    integrated_gap = _trapz(gap, q, axis=2)
    integrated_auc = _trapz(auc, q, axis=3)

    unique_blocks = np.asarray(sorted(set(blocks)))
    block_pos = {block: i for i, block in enumerate(unique_blocks)}
    item_block = np.asarray([block_pos[b] for b in blocks], dtype=np.int32)
    n_cells = len(conditions) * len(ranks) * n_layers
    prepared = {}
    for regime, regime_i in ridx.items():
        mask = masks[regime]
        # NOTE: scores[:, :, regime_i, :, mask] would move the masked axis to the
        # front (advanced indices separated by a slice), scrambling the reshape.
        matrix = scores[:, :, regime_i, :, :][:, :, :, mask].reshape(n_cells, int(mask.sum()))
        prepared[regime] = (prepare_auc_order(matrix, y[mask]), item_block[mask])

    boot_gap = np.empty((len(conditions), len(ranks), n_boot), dtype=np.float64)
    boot_auc = np.empty((len(conditions), len(ranks), len(regimes), n_boot), dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    started = time.time()
    for rep in range(n_boot):
        sampled = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        counts = np.bincount(sampled, minlength=len(unique_blocks)).astype(np.float64)
        rep_auc = np.empty((len(conditions), len(ranks), len(regimes), n_layers), dtype=np.float64)
        for regime, regime_i in ridx.items():
            regime_prepared, regime_item_block = prepared[regime]
            values = weighted_auc_from_order(regime_prepared, counts[regime_item_block])
            rep_auc[:, :, regime_i, :] = values.reshape(len(conditions), len(ranks), n_layers)
        rep_gap = rep_auc[:, :, ridx["joint"]] - (
            rep_auc[:, :, ridx["b2c"]] + rep_auc[:, :, ridx["c2b"]]
        ) / 2.0
        boot_gap[:, :, rep] = _trapz(rep_gap, q, axis=2)
        boot_auc[:, :, :, rep] = _trapz(rep_auc, q, axis=3)
        if (rep + 1) % 100 == 0:
            print(f"  {model}: bootstrap {rep + 1}/{n_boot}", file=sys.stderr, flush=True)
    elapsed = time.time() - started

    def paired(before: str, after: str, rank_i: int) -> dict:
        delta = integrated_gap[cidx[before], rank_i] - integrated_gap[cidx[after], rank_i]
        boot_delta = boot_gap[cidx[before], rank_i] - boot_gap[cidx[after], rank_i]
        raw = integrated_gap[cidx["raw"], rank_i]
        boot_raw = boot_gap[cidx["raw"], rank_i]
        fraction = delta / raw
        boot_fraction = boot_delta / boot_raw
        direction_deltas = {}
        for regime in ("b2c", "c2b"):
            regime_i = ridx[regime]
            # Higher transfer AUC after projection means improved portability.
            auc_delta = integrated_auc[cidx[after], rank_i, regime_i] - integrated_auc[cidx[before], rank_i, regime_i]
            boot_auc_delta = boot_auc[cidx[after], rank_i, regime_i] - boot_auc[cidx[before], rank_i, regime_i]
            direction_deltas[regime] = {
                "integrated_auc_improvement": float(auc_delta),
                "ci95": ci(boot_auc_delta),
            }
        return {
            "before": before,
            "after": after,
            "delta_integrated_gap": float(delta),
            "delta_integrated_gap_ci95": ci(boot_delta),
            "fraction_raw_gap_removed": float(fraction),
            "fraction_raw_gap_removed_ci95": ci(boot_fraction),
            "directional_transfer": direction_deltas,
        }

    def incremental_specificity(rank_i: int) -> dict:
        b = cidx["minus_b"]
        c = cidx["minus_b_cperp"]
        r = cidx["minus_b_rperp"]
        extra_c, extra_random, specific_c = specificity_components(
            integrated_gap[b, rank_i],
            integrated_gap[c, rank_i],
            integrated_gap[r, rank_i],
        )
        boot_extra_c, boot_extra_random, boot_specific_c = specificity_components(
            boot_gap[b, rank_i],
            boot_gap[c, rank_i],
            boot_gap[r, rank_i],
        )
        return {
            "extra_C": float(extra_c),
            "extra_C_ci95": ci(boot_extra_c),
            "extra_random": float(extra_random),
            "extra_random_ci95": ci(boot_extra_random),
            "specific_C": float(specific_c),
            "specific_C_ci95": ci(boot_specific_c),
            "definition": "(I_-B - I_-[B,C_perp_B]) - (I_-B - I_-[B,R_perp_B])",
        }

    by_rank = {}
    csv_rows = []
    for rank_i, rank in enumerate(ranks):
        by_condition = {}
        for condition_i, condition in enumerate(conditions):
            by_condition[condition] = {
                "integrated_gap": float(integrated_gap[condition_i, rank_i]),
                "integrated_gap_ci95": ci(boot_gap[condition_i, rank_i]),
                "gap_by_layer": gap[condition_i, rank_i].tolist(),
                "joint_auc_by_layer": auc[condition_i, rank_i, ridx["joint"]].tolist(),
                "b2c_auc_by_layer": auc[condition_i, rank_i, ridx["b2c"]].tolist(),
                "c2b_auc_by_layer": auc[condition_i, rank_i, ridx["c2b"]].tolist(),
            }
            csv_rows.append({
                "model": model,
                "rank": rank,
                "condition": condition,
                "integrated_gap": float(integrated_gap[condition_i, rank_i]),
                "ci95_low": ci(boot_gap[condition_i, rank_i])[0],
                "ci95_high": ci(boot_gap[condition_i, rank_i])[1],
            })
        by_rank[str(rank)] = {
            "by_condition": by_condition,
            "paired": {
                "delta_I_B": paired("raw", "minus_b", rank_i),
                "delta_I_C": paired("raw", "minus_c", rank_i),
                "delta_I_C_given_B": paired("minus_b", "minus_b_cperp", rank_i),
                "delta_I_random_given_B": paired("minus_b", "minus_b_rperp", rank_i),
                "specific_C_increment": incremental_specificity(rank_i),
                "delta_I_total": paired("raw", "minus_b_cperp", rank_i),
                "direct_joint_check": paired("minus_b_cperp", "minus_bc_joint", rank_i),
                "natural_F_eval": paired("raw", "minus_feval", rank_i),
                "natural_F_deploy": paired("raw", "minus_fdeploy", rank_i),
                "generic_rankmatched": paired("raw", "random_rankmatched", rank_i),
            },
        }

    sanity_path = path.with_suffix(".sanity.json")
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    capture_by_rank = {}
    integrated_capture = []
    b_gap_reduction = []
    for rank_i, rank in enumerate(ranks):
        by_layer_capture = [
            float(sanity["layer_checks"][str(layer)]["c_capture_by_rank"][str(rank)]["cross_fitted_fraction"])
            for layer in range(n_layers)
        ]
        capture_integral = float(_trapz(by_layer_capture, q))
        delta_b = float(integrated_gap[cidx["raw"], rank_i] - integrated_gap[cidx["minus_b"], rank_i])
        specific_c = by_rank[str(rank)]["paired"]["specific_C_increment"]["specific_C"]
        capture_by_rank[str(rank)] = {
            "c_capture_by_layer": by_layer_capture,
            "integrated_c_capture": capture_integral,
            "delta_integrated_gap_B": delta_b,
            "specific_C": specific_c,
        }
        integrated_capture.append(capture_integral)
        b_gap_reduction.append(delta_b)
        for row in csv_rows:
            if row["rank"] == rank:
                row["integrated_c_capture"] = capture_integral
                row["delta_integrated_gap_B"] = delta_b
                row["specific_C"] = specific_c
    capture_gap_correlation = float(np.corrcoef(integrated_capture, b_gap_reduction)[0, 1])

    return ({
        "n_layers": n_layers,
        "q": q.tolist(),
        "ranks": ranks,
        "conditions": conditions,
        "regimes": regimes,
        "n_items": len(y),
        "n_blocks": len(unique_blocks),
        "scores_path": str(path),
        "scores_sha256": sha256_file(path),
        "bootstrap_elapsed_seconds": elapsed,
        "sanity_path": str(sanity_path),
        "sanity_sha256": sha256_file(sanity_path),
        "c_capture_analysis": {
            "formula": "cross-fitted ||C_holdout Q_B(r)||_F^2 / ||C_holdout||_F^2, pooled by numerator and denominator over held-purpose-family folds",
            "by_rank": capture_by_rank,
            "rankwise_pearson_c_capture_vs_delta_I_B": capture_gap_correlation,
            "inference": "descriptive four-rank comparison; no layerwise or rankwise p-values",
        },
        "by_rank": by_rank,
    }, csv_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=str(RESULTS / "full320_v3_factorial_residual_summary.json"))
    ap.add_argument("--csv", default=str(RESULTS / "full320_v3_factorial_residual_summary.csv"))
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise ValueError("N_BOOT must be >=1000 for reportable output")
    out_path, csv_path = Path(args.out), Path(args.csv)
    if out_path.exists() or csv_path.exists():
        raise FileExistsError(f"immutable output exists: {out_path} or {csv_path}")

    output = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "Factorial residual decomposition of cross-format purpose-probe non-portability on frozen activations, including C captured by the B-derived subspace.",
        "basis_convention": "leading singular directions of uncentered matched effect matrices; Q_B is B-derived, not a pure B-main-effect subspace",
        "metric": "g(q)=AUC_joint(q)-(AUC_b2c(q)+AUC_c2b(q))/2; I=trapezoid(g,q), q=layer/(n_layers-1)",
        "bootstrap": "payload-block resampling; one draw reused across every layer, rank, condition, and AUC regime; curve integrated within replicate",
        "n_boot": args.n_boot,
        "rng_seed": RNG_SEED,
        "interpretation_boundary": "Representational intervention plus probe retraining on frozen activations. C-capture/gap associations are descriptive. No behavioral steering or model-output causal claim.",
        "by_model": {},
    }
    all_csv = []
    for model in args.models:
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        output["by_model"][model], rows = analyze_model(model, args.n_boot)
        all_csv.extend(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "rank", "condition", "integrated_gap", "ci95_low", "ci95_high", "integrated_c_capture", "delta_integrated_gap_B", "specific_C"])
        writer.writeheader()
        writer.writerows(all_csv)
    print(f"wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
