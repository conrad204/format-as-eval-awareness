#!/usr/bin/env python3
"""Block bootstrap for the emergency rank-64 C_perp_B specificity run.

Metric and resampling are the frozen ones, imported from
``full320_v3_factorial_residual_bootstrap``:

    g(q) = AUC_joint(q) - (AUC_b2c(q) + AUC_c2b(q)) / 2
    I    = trapezoid(g, q),  q = layer / (n_layers - 1)

One payload-block resample per replicate is reused across every layer,
condition, and regime, and the depth curve is integrated inside the replicate.

Reported decision quantity:

    extra_C     = I_-B - I_-[B,C_perp_B]
    extra_rand  = I_-B - I_-[B,R_perp_B]
    specific_C  = extra_C - extra_rand      (paired 95% block-bootstrap CI)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full320_v3_factorial_residual_bootstrap import (  # noqa: E402
    _trapz,
    ci,
    point_aucs,
    sha256_file,
    specificity_components,
)
# Single tie-aware implementation lives in the canonical bootstrap module; these
# aliases keep the emergency call sites and tests pointed at one code path.
from full320_v3_factorial_residual_bootstrap import (  # noqa: E402
    prepare_auc_order as prepare_tie_aware_order,
)
from full320_v3_factorial_residual_bootstrap import (  # noqa: E402
    weighted_auc_from_order as weighted_auc_with_ties,
)


RNG_SEED = 0
N_BOOT = 1000
CAPTURE_RANKS = (1, 4, 16, 64)


def analyze(scores_path: Path, n_boot: int) -> dict:
    data = np.load(scores_path, allow_pickle=True)
    conditions = [str(v) for v in data["conditions"]]
    ranks = [int(v) for v in data["ranks"]]
    regimes = [str(v) for v in data["regimes"]]
    scores = data["scores"].astype(np.float64)
    y = data["y"].astype(np.int8)
    blocks = data["payload_block_id"].astype(str)
    model = str(data["model"])
    n_layers = int(data["n_layers"])
    if len(ranks) != 1:
        raise ValueError(f"emergency run must carry exactly one rank, got {ranks}")
    for required in ("raw", "minus_b", "minus_b_cperp", "minus_b_rperp"):
        if required not in conditions:
            raise ValueError(f"missing required condition {required}")

    expected_shape = (len(conditions), 1, len(regimes), n_layers, len(y))
    if scores.shape != expected_shape:
        raise ValueError(f"unexpected score shape {scores.shape}, expected {expected_shape}")

    q = np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
    cidx = {name: i for i, name in enumerate(conditions)}
    ridx = {name: i for i, name in enumerate(regimes)}

    masks = {}
    for regime, regime_i in ridx.items():
        mask = np.isfinite(scores[0, 0, regime_i, 0])
        if not np.all(np.isfinite(scores[:, :, regime_i, :, mask])):
            raise ValueError(f"nonconstant finite-score mask for {regime}")
        if np.any(np.isfinite(scores[:, :, regime_i, :, ~mask])):
            raise ValueError(f"nonconstant NaN-score mask for {regime}")
        masks[regime] = mask

    auc = point_aucs(scores, y)
    gap = auc[:, :, ridx["joint"]] - (auc[:, :, ridx["b2c"]] + auc[:, :, ridx["c2b"]]) / 2.0
    integrated_gap = _trapz(gap, q, axis=2)[:, 0]
    integrated_auc = _trapz(auc, q, axis=3)[:, 0, :]

    unique_blocks = np.asarray(sorted(set(blocks)))
    block_pos = {block: i for i, block in enumerate(unique_blocks)}
    item_block = np.asarray([block_pos[b] for b in blocks], dtype=np.int32)
    n_cells = len(conditions) * n_layers
    prepared = {}
    for regime, regime_i in ridx.items():
        mask = masks[regime]
        # NOTE: scores[:, 0, regime_i, :, mask] would move the masked axis to the
        # front (advanced indices separated by a slice), scrambling the reshape.
        matrix = scores[:, 0, regime_i, :, :][:, :, mask].reshape(n_cells, int(mask.sum()))
        prepared[regime] = (prepare_tie_aware_order(matrix, y[mask]), item_block[mask])

    boot_gap = np.empty((len(conditions), n_boot), dtype=np.float64)
    boot_auc = np.empty((len(conditions), len(regimes), n_boot), dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    started = time.time()
    for rep in range(n_boot):
        sampled = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        counts = np.bincount(sampled, minlength=len(unique_blocks)).astype(np.float64)
        rep_auc = np.empty((len(conditions), len(regimes), n_layers), dtype=np.float64)
        for regime, regime_i in ridx.items():
            regime_prepared, regime_item_block = prepared[regime]
            values = weighted_auc_with_ties(regime_prepared, counts[regime_item_block])
            rep_auc[:, regime_i, :] = values.reshape(len(conditions), n_layers)
        rep_gap = rep_auc[:, ridx["joint"]] - (rep_auc[:, ridx["b2c"]] + rep_auc[:, ridx["c2b"]]) / 2.0
        boot_gap[:, rep] = _trapz(rep_gap, q, axis=1)
        boot_auc[:, :, rep] = _trapz(rep_auc, q, axis=2)
        if (rep + 1) % 200 == 0:
            print(f"  {model}: bootstrap {rep + 1}/{n_boot}", file=sys.stderr, flush=True)
    elapsed = time.time() - started

    def paired(before: str, after: str) -> dict:
        delta = integrated_gap[cidx[before]] - integrated_gap[cidx[after]]
        boot_delta = boot_gap[cidx[before]] - boot_gap[cidx[after]]
        raw = integrated_gap[cidx["raw"]]
        directional = {}
        for regime in ("b2c", "c2b"):
            regime_i = ridx[regime]
            auc_delta = integrated_auc[cidx[after], regime_i] - integrated_auc[cidx[before], regime_i]
            boot_auc_delta = boot_auc[cidx[after], regime_i] - boot_auc[cidx[before], regime_i]
            directional[regime] = {
                "integrated_auc_improvement": float(auc_delta),
                "ci95": ci(boot_auc_delta),
            }
        return {
            "before": before,
            "after": after,
            "delta_integrated_gap": float(delta),
            "delta_integrated_gap_ci95": ci(boot_delta),
            "fraction_raw_gap_removed": float(delta / raw),
            "fraction_raw_gap_removed_ci95": ci(boot_delta / boot_gap[cidx["raw"]]),
            "directional_transfer": directional,
        }

    extra_c, extra_random, specific_c = specificity_components(
        integrated_gap[cidx["minus_b"]],
        integrated_gap[cidx["minus_b_cperp"]],
        integrated_gap[cidx["minus_b_rperp"]],
    )
    boot_extra_c, boot_extra_random, boot_specific_c = specificity_components(
        boot_gap[cidx["minus_b"]],
        boot_gap[cidx["minus_b_cperp"]],
        boot_gap[cidx["minus_b_rperp"]],
    )
    specific_ci = ci(boot_specific_c)
    decision = {
        "extra_C": float(extra_c),
        "extra_C_ci95": ci(boot_extra_c),
        "extra_random": float(extra_random),
        "extra_random_ci95": ci(boot_extra_random),
        "specific_C": float(specific_c),
        "specific_C_ci95": specific_ci,
        "specific_C_excludes_zero": bool(specific_ci[0] > 0.0 or specific_ci[1] < 0.0),
        "specific_C_positive_and_excludes_zero": bool(specific_ci[0] > 0.0),
        "boot_fraction_specific_C_positive": float(np.mean(boot_specific_c > 0.0)),
        "definition": "(I_-B - I_-[B,C_perp_B]) - (I_-B - I_-[B,R_perp_B])",
    }

    sanity_path = scores_path.with_suffix(".sanity.json")
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    capture = {}
    for prefix in CAPTURE_RANKS:
        by_layer = [
            float(sanity["layer_checks"][str(layer)]["c_capture_by_rank"][str(prefix)]["cross_fitted_fraction"])
            for layer in range(n_layers)
        ]
        chance = float(
            sanity["layer_checks"]["0"]["c_capture_by_rank"][str(prefix)]["nominal_isotropic_chance"]
        )
        capture[str(prefix)] = {
            "c_capture_by_layer": by_layer,
            "integrated_c_capture": float(_trapz(by_layer, q)),
            "nominal_isotropic_chance": chance,
            "chance_multiple": float(_trapz(by_layer, q)) / chance,
        }

    return {
        "model": model,
        "n_layers": n_layers,
        "n_items": int(len(y)),
        "n_blocks": int(len(unique_blocks)),
        "rank": ranks[0],
        "conditions": conditions,
        "regimes": regimes,
        "scores_path": str(scores_path),
        "scores_sha256": sha256_file(scores_path),
        "sanity_path": str(sanity_path),
        "sanity_sha256": sha256_file(sanity_path),
        "n_boot": n_boot,
        "rng_seed": RNG_SEED,
        "bootstrap_elapsed_seconds": elapsed,
        "reference_reproduction": sanity.get("reference_reproduction"),
        "global_checks": sanity.get("global_checks"),
        "basis_backend_check": sanity.get("basis_backend_check"),
        "integrated_gap": {name: float(integrated_gap[cidx[name]]) for name in conditions},
        "integrated_gap_ci95": {name: ci(boot_gap[cidx[name]]) for name in conditions},
        "gap_by_layer": {name: gap[cidx[name], 0].tolist() for name in conditions},
        "auc_by_layer": {
            name: {regime: auc[cidx[name], 0, ridx[regime]].tolist() for regime in regimes}
            for name in conditions
        },
        "paired": {
            "delta_I_B": paired("raw", "minus_b"),
            "delta_I_C_given_B": paired("minus_b", "minus_b_cperp"),
            "delta_I_random_given_B": paired("minus_b", "minus_b_rperp"),
            "delta_I_total": paired("raw", "minus_b_cperp"),
        },
        "decision": decision,
        "c_capture": capture,
        "metric": "g(q)=AUC_joint(q)-(AUC_b2c(q)+AUC_c2b(q))/2; I=trapezoid(g,q), q=layer/(n_layers-1)",
        "bootstrap": "payload-block resampling; one draw reused across every layer, condition, and regime; curve integrated within replicate",
        "interpretation_boundary": "Representational intervention plus probe retraining on frozen activations. No behavioral steering or model-output causal claim.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="emergency_cperp_scores_<model>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise ValueError("n-boot must be >=1000 for reportable output")

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"immutable output exists: {out_path}")
    result = analyze(Path(args.scores), args.n_boot)
    result["schema_version"] = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "rank", "condition", "integrated_gap", "ci95_low", "ci95_high"],
            )
            writer.writeheader()
            for condition in result["conditions"]:
                low, high = result["integrated_gap_ci95"][condition]
                writer.writerow(
                    {
                        "model": result["model"],
                        "rank": result["rank"],
                        "condition": condition,
                        "integrated_gap": result["integrated_gap"][condition],
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
        print(f"wrote {out_path} and {csv_path}")
    else:
        print(f"wrote {out_path}")

    decision = result["decision"]
    print(
        json.dumps(
            {
                "model": result["model"],
                "I_raw": result["integrated_gap"]["raw"],
                "I_minus_B": result["integrated_gap"]["minus_b"],
                "I_minus_B_Cperp": result["integrated_gap"]["minus_b_cperp"],
                "I_minus_B_Rperp": result["integrated_gap"]["minus_b_rperp"],
                "extra_C": decision["extra_C"],
                "extra_random": decision["extra_random"],
                "specific_C": decision["specific_C"],
                "specific_C_ci95": decision["specific_C_ci95"],
                "specific_C_positive_and_excludes_zero": decision["specific_C_positive_and_excludes_zero"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
