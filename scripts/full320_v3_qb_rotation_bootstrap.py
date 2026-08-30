#!/usr/bin/env python3
"""Paired bootstrap for the within-Q_B C-alignment test.

Joins the new rank-16 conditions (topC16 / bottomC16 / randomB16) to the `raw`
baseline already produced by the emergency run, so no baseline is recomputed. One
payload-block resample per replicate is shared across every layer, condition and
regime, and the depth curve is integrated inside the replicate.

    G_c,l  = AUC_joint - (AUC_b2c + AUC_c2b)/2
    I_c    = trapezoid(G_c, q),  q = layer/(n_layers-1)
    rescue_c = I_raw - I_c

    C_enrichment          = rescue_topC16 - rescue_bottomC16
    C_vs_random_inside_B  = rescue_topC16 - rescue_randomB16
"""
from __future__ import annotations

import argparse
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
    prepare_auc_order,
    sha256_file,
    weighted_auc_from_order,
)

RNG_SEED = 0
N_BOOT = 1000
BASELINE = "raw"
NEW_CONDITIONS = ("topC16", "bottomC16", "randomB16")


def load_aligned(emergency_path: Path, rotation_path: Path):
    """Stack the reused raw baseline and the new conditions on identical items."""
    emergency = np.load(emergency_path, allow_pickle=True)
    rotation = np.load(rotation_path, allow_pickle=True)
    for key in ("item_id", "payload_block_id", "y", "format", "purpose_family_id"):
        if not np.array_equal(emergency[key].astype(str), rotation[key].astype(str)):
            raise ValueError(f"{key} differs between the emergency and rotation runs")
    if int(emergency["n_layers"]) != int(rotation["n_layers"]):
        raise ValueError("layer count differs between runs")
    if str(emergency["model"]) != str(rotation["model"]):
        raise ValueError("model differs between runs")

    emergency_conditions = [str(v) for v in emergency["conditions"]]
    rotation_conditions = [str(v) for v in rotation["conditions"]]
    regimes = [str(v) for v in emergency["regimes"]]
    if [str(v) for v in rotation["regimes"]] != regimes:
        raise ValueError("regime order differs between runs")
    for name in NEW_CONDITIONS:
        if name not in rotation_conditions:
            raise ValueError(f"rotation run is missing condition {name}")

    baseline = emergency["scores"][emergency_conditions.index(BASELINE), 0]
    stacked = [baseline] + [
        rotation["scores"][rotation_conditions.index(name), 0] for name in NEW_CONDITIONS
    ]
    conditions = [BASELINE, *NEW_CONDITIONS]
    scores = np.stack(stacked, axis=0).astype(np.float64)  # (condition, regime, layer, item)
    return {
        "model": str(emergency["model"]),
        "conditions": conditions,
        "regimes": regimes,
        "scores": scores,
        "y": emergency["y"].astype(np.int8),
        "blocks": emergency["payload_block_id"].astype(str),
        "n_layers": int(emergency["n_layers"]),
    }


def analyze(emergency_path: Path, rotation_path: Path, n_boot: int) -> dict:
    data = load_aligned(emergency_path, rotation_path)
    conditions, regimes = data["conditions"], data["regimes"]
    scores, y, blocks = data["scores"], data["y"], data["blocks"]
    n_layers = data["n_layers"]
    q = np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
    cidx = {name: i for i, name in enumerate(conditions)}
    ridx = {name: i for i, name in enumerate(regimes)}

    masks = {}
    for regime, regime_i in ridx.items():
        mask = np.isfinite(scores[0, regime_i, 0])
        if not np.all(np.isfinite(scores[:, regime_i, :, mask])):
            raise ValueError(f"nonconstant finite-score mask for {regime}")
        masks[regime] = mask

    auc = point_aucs(scores[:, None], y)[:, 0]
    gap = auc[:, ridx["joint"]] - (auc[:, ridx["b2c"]] + auc[:, ridx["c2b"]]) / 2.0
    integrated = _trapz(gap, q, axis=1)

    unique_blocks = np.asarray(sorted(set(blocks)))
    block_pos = {block: i for i, block in enumerate(unique_blocks)}
    item_block = np.asarray([block_pos[b] for b in blocks], dtype=np.int32)
    n_cells = len(conditions) * n_layers
    prepared = {}
    for regime, regime_i in ridx.items():
        mask = masks[regime]
        matrix = scores[:, regime_i, :, :][:, :, mask].reshape(n_cells, int(mask.sum()))
        prepared[regime] = (prepare_auc_order(matrix, y[mask]), item_block[mask])

    boot = np.empty((len(conditions), n_boot), dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    started = time.time()
    for rep in range(n_boot):
        sampled = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        counts = np.bincount(sampled, minlength=len(unique_blocks)).astype(np.float64)
        rep_auc = np.empty((len(conditions), len(regimes), n_layers), dtype=np.float64)
        for regime, regime_i in ridx.items():
            regime_prepared, regime_item_block = prepared[regime]
            rep_auc[:, regime_i, :] = weighted_auc_from_order(
                regime_prepared, counts[regime_item_block]
            ).reshape(len(conditions), n_layers)
        rep_gap = rep_auc[:, ridx["joint"]] - (rep_auc[:, ridx["b2c"]] + rep_auc[:, ridx["c2b"]]) / 2.0
        boot[:, rep] = _trapz(rep_gap, q, axis=1)
        if (rep + 1) % 200 == 0:
            print(f"  bootstrap {rep + 1}/{n_boot}", file=sys.stderr, flush=True)
    elapsed = time.time() - started

    rescue = {name: float(integrated[cidx[BASELINE]] - integrated[cidx[name]]) for name in NEW_CONDITIONS}
    boot_rescue = {name: boot[cidx[BASELINE]] - boot[cidx[name]] for name in NEW_CONDITIONS}
    contrasts = {}
    for label, (a, b) in {
        "C_enrichment": ("topC16", "bottomC16"),
        "C_vs_random_inside_B": ("topC16", "randomB16"),
        "bottom_vs_random_inside_B": ("bottomC16", "randomB16"),
    }.items():
        point = rescue[a] - rescue[b]
        draws = boot_rescue[a] - boot_rescue[b]
        interval = ci(draws)
        contrasts[label] = {
            "difference": point,
            "ci95": interval,
            "excludes_zero": bool(interval[0] > 0.0 or interval[1] < 0.0),
            "positive_and_excludes_zero": bool(interval[0] > 0.0),
            "boot_fraction_positive": float(np.mean(draws > 0.0)),
            "definition": f"rescue_{a} - rescue_{b}",
        }

    rotation_sanity = json.loads(rotation_path.with_suffix(".sanity.json").read_text(encoding="utf-8"))
    layer_checks = rotation_sanity["layer_checks"]
    layers = sorted(layer_checks, key=int)
    diagnostics = {}
    for name in NEW_CONDITIONS:
        capture = [layer_checks[key]["c_capture"][name]["cross_fitted_fraction"] for key in layers]
        energy = [layer_checks[key]["energy_removed"][name]["pooled_fraction_all"] for key in layers]
        diagnostics[name] = {
            "integrated_c_capture": float(_trapz(capture, q)),
            "integrated_energy_removed": float(_trapz(energy, q)),
            "c_capture_by_layer": capture,
            "energy_removed_by_layer": energy,
        }

    return {
        "schema_version": 1,
        "model": data["model"],
        "test": "within-Q_B C-alignment: rank-16 topC16 vs bottomC16 vs randomB16, all inside the same rank-64 Q_B",
        "n_layers": n_layers,
        "n_items": int(len(y)),
        "n_blocks": int(len(unique_blocks)),
        "n_boot": n_boot,
        "rng_seed": RNG_SEED,
        "bootstrap_elapsed_seconds": elapsed,
        "emergency_scores_path": str(emergency_path),
        "emergency_scores_sha256": sha256_file(emergency_path),
        "rotation_scores_path": str(rotation_path),
        "rotation_scores_sha256": sha256_file(rotation_path),
        "integrated_gap": {name: float(integrated[cidx[name]]) for name in conditions},
        "integrated_gap_ci95": {name: ci(boot[cidx[name]]) for name in conditions},
        "gap_by_layer": {name: gap[cidx[name]].tolist() for name in conditions},
        "rescue": rescue,
        "rescue_ci95": {name: ci(boot_rescue[name]) for name in NEW_CONDITIONS},
        "contrasts": contrasts,
        "subspace_diagnostics": diagnostics,
        "global_checks": rotation_sanity["global_checks"],
        "metric": "g(q)=AUC_joint(q)-(AUC_b2c(q)+AUC_c2b(q))/2; I=trapezoid(g,q); rescue=I_raw-I_condition",
        "bootstrap": "payload-block resampling shared across every layer, condition and regime; integrated within replicate",
        "interpretation_boundary": "Representational intervention plus probe retraining on frozen activations. No behavioral steering or model-output causal claim.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emergency-scores", required=True)
    ap.add_argument("--rotation-scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise ValueError("n-boot must be >=1000 for reportable output")
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"immutable output exists: {out_path}")

    result = analyze(Path(args.emergency_scores), Path(args.rotation_scores), args.n_boot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "model": result["model"],
                "I_raw": result["integrated_gap"]["raw"],
                "I_topC16": result["integrated_gap"]["topC16"],
                "I_bottomC16": result["integrated_gap"]["bottomC16"],
                "I_randomB16": result["integrated_gap"]["randomB16"],
                "rescue": result["rescue"],
                "C_enrichment": result["contrasts"]["C_enrichment"],
                "C_vs_random_inside_B": result["contrasts"]["C_vs_random_inside_B"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
