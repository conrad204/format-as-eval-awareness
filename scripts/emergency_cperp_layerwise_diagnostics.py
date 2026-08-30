#!/usr/bin/env python3
"""Two zero-compute diagnostics from a completed emergency C_perp_B run.

1. Layerwise trajectory of the incremental C_perp_B effect, instead of only the
   depth integral: per layer, the gap change from removing Q_C|B on top of Q_B,
   and the same for the rank-matched random control, plus their difference.

2. Association between held-out C captured *inside* Q_B and the amount of
   transfer rescue that removing Q_B buys, with depth controlled. Depth is a
   strong common cause of both, so the partial correlation given normalized depth
   q is reported alongside the raw correlation, with a payload-block-free
   layer-level permutation test that respects depth by permuting residuals.

Reads only the committed score archive and sanity manifest. No probe fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_trapz = getattr(np, "trapezoid", None) or np.trapz


def layer_aucs(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """scores: (condition, regime, layer, item) -> (condition, regime, layer)."""
    out = np.full(scores.shape[:-1], np.nan)
    for index in np.ndindex(scores.shape[:-1]):
        values = scores[index]
        mask = np.isfinite(values)
        out[index] = roc_auc_score(y[mask], values[mask])
    return out


def residualize(values: np.ndarray, covariate: np.ndarray, degree: int = 2) -> np.ndarray:
    """Residuals of `values` after least-squares removal of a polynomial in q."""
    design = np.vander(covariate, degree + 1, increasing=True)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def permutation_p_value(x: np.ndarray, z: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided p for corr(x, z) under permutation of one residual vector."""
    observed = abs(float(np.corrcoef(x, z)[0, 1]))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        if abs(float(np.corrcoef(rng.permutation(x), z)[0, 1])) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-rank", default="64")
    ap.add_argument("--n-perm", type=int, default=20000)
    args = ap.parse_args()

    scores_path = Path(args.scores)
    data = np.load(scores_path, allow_pickle=True)
    sanity = json.loads(scores_path.with_suffix(".sanity.json").read_text(encoding="utf-8"))
    conditions = [str(v) for v in data["conditions"]]
    regimes = [str(v) for v in data["regimes"]]
    y = data["y"].astype(np.int8)
    n_layers = int(data["n_layers"])
    q = np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
    cidx = {name: i for i, name in enumerate(conditions)}
    ridx = {name: i for i, name in enumerate(regimes)}

    auc = layer_aucs(data["scores"].astype(np.float64)[:, 0], y)
    gap = auc[:, ridx["joint"]] - (auc[:, ridx["b2c"]] + auc[:, ridx["c2b"]]) / 2.0

    # --- diagnostic 1: layerwise incremental C_perp_B effect -----------------
    incremental_c = gap[cidx["minus_b"]] - gap[cidx["minus_b_cperp"]]
    incremental_random = gap[cidx["minus_b"]] - gap[cidx["minus_b_rperp"]]
    specific_by_layer = incremental_c - incremental_random
    peak = int(np.argmax(np.abs(specific_by_layer)))
    trajectory = {
        "definition": "per layer: (g_-B - g_-[B,Cperp]) minus (g_-B - g_-[B,Rperp]); positive means C_perp helps beyond random",
        "q": q.tolist(),
        "gap_raw": gap[cidx["raw"]].tolist(),
        "gap_minus_b": gap[cidx["minus_b"]].tolist(),
        "incremental_C": incremental_c.tolist(),
        "incremental_random": incremental_random.tolist(),
        "specific_C_by_layer": specific_by_layer.tolist(),
        "n_layers_specific_C_positive": int(np.sum(specific_by_layer > 0)),
        "n_layers": n_layers,
        "max_abs_layer": peak,
        "max_abs_layer_q": float(q[peak]),
        "max_abs_value": float(specific_by_layer[peak]),
        "sign_is_constant": bool(np.all(specific_by_layer <= 0) or np.all(specific_by_layer >= 0)),
        "early_half_mean": float(np.mean(specific_by_layer[: n_layers // 2])),
        "late_half_mean": float(np.mean(specific_by_layer[n_layers // 2 :])),
    }

    # --- diagnostic 2: C capture inside Q_B vs minus-B rescue ----------------
    layer_checks = sanity["layer_checks"]
    capture = np.asarray(
        [
            float(layer_checks[str(layer)]["c_capture_by_rank"][args.capture_rank]["cross_fitted_fraction"])
            for layer in range(n_layers)
        ]
    )
    rescue_b = gap[cidx["raw"]] - gap[cidx["minus_b"]]
    capture_residual = residualize(capture, q)
    rescue_residual = residualize(rescue_b, q)
    association = {
        "definition": (
            f"layerwise held-out C capture by rank-{args.capture_rank} Q_B vs rescue "
            "(g_raw - g_-B); depth q removed from both sides by quadratic least squares"
        ),
        "capture_by_layer": capture.tolist(),
        "rescue_b_by_layer": rescue_b.tolist(),
        "integrated_capture": float(_trapz(capture, q)),
        "integrated_rescue_b": float(_trapz(rescue_b, q)),
        "pearson_raw": float(np.corrcoef(capture, rescue_b)[0, 1]),
        "spearman_raw": float(
            np.corrcoef(np.argsort(np.argsort(capture)), np.argsort(np.argsort(rescue_b)))[0, 1]
        ),
        "pearson_partial_given_depth": float(np.corrcoef(capture_residual, rescue_residual)[0, 1]),
        "permutation_p_partial": permutation_p_value(capture_residual, rescue_residual, args.n_perm, 0),
        "n_perm": args.n_perm,
        "depth_control": "quadratic polynomial in q",
        "caveat": "layers are not independent; this is a descriptive association, not a causal or preregistered test",
    }

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"immutable output exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": str(data["model"]),
        "scores_path": str(scores_path),
        "capture_rank": int(args.capture_rank),
        "layerwise_cperp_trajectory": trajectory,
        "capture_vs_rescue_association": association,
        "interpretation_boundary": "Descriptive layerwise diagnostics on frozen-activation results. No causal or behavioral claim.",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "model": payload["model"],
                "specific_C_layers_positive": trajectory["n_layers_specific_C_positive"],
                "specific_C_sign_constant": trajectory["sign_is_constant"],
                "specific_C_early_half_mean": trajectory["early_half_mean"],
                "specific_C_late_half_mean": trajectory["late_half_mean"],
                "capture_vs_rescue_pearson_raw": association["pearson_raw"],
                "capture_vs_rescue_partial_given_depth": association["pearson_partial_given_depth"],
                "permutation_p_partial": association["permutation_p_partial"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
