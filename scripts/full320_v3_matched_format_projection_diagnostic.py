"""Matched-format-projection transfer diagnostic (full-layer, all 3 models).

Merges the already-computed raw all-layer sweeps
(results/layer_sweep_full320-v3_<model>.json) with the already-computed
rank-64 matched-format-projected sweep
(results/full320_v3_erasure_full_layer_sweep.json, "erasure" = projecting out
the rank-64 matched-format subspace spanned by the format main-effect/
interaction basis B, cross-fitted per held-out cue-vocabulary fold, then
retraining the purpose probe on the residual) to test whether matched-format
projection removes the cross-format transfer penalty while leaving
joint-format purpose decodability intact.

No new probes are fit here. This script only reads the two existing sweep
JSONs, joins them by (model, layer), and derives:

  raw_gap   = raw_joint_auc   - mean(raw_b2c_auc,   raw_c2b_auc)
  proj_gap  = proj_joint_auc  - mean(proj_b2c_auc,  proj_c2b_auc)   (already
              stored as gap_after_erasure; recomputed here for a symmetric
              audit trail, and checked equal to the stored value)
  delta_gap            = raw_gap - proj_gap                 ("gap rescue")
  delta_joint_auc      = proj_joint_auc - raw_joint_auc
  delta_b2c_auc        = proj_b2c_auc   - raw_b2c_auc
  delta_c2b_auc        = proj_c2b_auc   - raw_c2b_auc
  rescue_fraction      = delta_gap / raw_gap   only where raw_gap >= RAW_GAP_EPS
                          (0.01 AUC), else null + unstable_ratio=true
  delta_format_auc     = proj_format_auc - raw_format_auc   (format
                          decodability before/after the same projection --
                          the "does format still exist" check; format_auc is
                          a sanity probe trained to predict format from the
                          (post-projection) activations, not the purpose probe)

"raw_joint_auc" is the "decorrelated" regime (probe trained on both formats
together) in results/layer_sweep_full320-v3_<model>.json; "proj_joint_auc" is
the same regime after the rank-64 matched-format projection
(purpose_auc_after_erasure). b2c/c2b are the one-format-trained,
other-format-tested cross-format transfer regimes, before and after the same
projection.

Usage: python scripts/full320_v3_matched_format_projection_diagnostic.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

RAW_PATHS = {
    "llama31_8b": RESULTS / "layer_sweep_full320-v3_llama31_8b.json",
    "llama31_70b": RESULTS / "layer_sweep_full320-v3_llama31_70b.json",
    "llama33_70b": RESULTS / "layer_sweep_full320-v3_llama33_70b.json",
}
PROJ_PATH = RESULTS / "full320_v3_erasure_full_layer_sweep.json"
MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "llama31_70b": "Llama-3.1-70B",
    "llama33_70b": "Llama-3.3-70B",
}

RAW_GAP_EPS = 0.01  # AUC points; below this, raw_gap is not "meaningfully positive"
PURPOSE_FORMED_AUC = 0.95  # post-purpose-formation region cutoff on raw joint AUC

OUT_JSON = RESULTS / "full320_v3_matched_format_projection_diagnostic.json"
OUT_CSV = RESULTS / "full320_v3_matched_format_projection_diagnostic.csv"


def load_raw(model):
    rows = json.loads(RAW_PATHS[model].read_text())
    rows = sorted(rows, key=lambda r: r["layer"])
    n_layers = len(rows)
    out = {}
    for r in rows:
        layer = r["layer"]
        joint = r["decorrelated"]["auc"]
        b2c = r["b2c"]["auc"]
        c2b = r["c2b"]["auc"]
        out[layer] = {
            "n_layers": n_layers,
            "raw_joint_auc": joint,
            "raw_b2c_auc": b2c,
            "raw_c2b_auc": c2b,
            "raw_gap": joint - 0.5 * (b2c + c2b),
            "raw_format_auc": r["format"]["auc"],
        }
    return out


def load_proj(model):
    d = json.loads(PROJ_PATH.read_text())["by_model"][model]
    out = {}
    for layer_str, v in d["by_layer"].items():
        layer = int(layer_str)
        joint = v["purpose_auc_after_erasure"]
        b2c = v["b2c_auc_after_erasure"]
        c2b = v["c2b_auc_after_erasure"]
        gap = joint - 0.5 * (b2c + c2b)
        assert abs(gap - v["gap_after_erasure"]) < 1e-9, (model, layer, gap, v["gap_after_erasure"])
        out[layer] = {
            "proj_joint_auc": joint,
            "proj_b2c_auc": b2c,
            "proj_c2b_auc": c2b,
            "proj_gap": gap,
            "proj_format_auc": v["format_auc_after_erasure"],
        }
    return out


def build_table():
    rows = []
    for model in RAW_PATHS:
        raw = load_raw(model)
        proj = load_proj(model)
        assert set(raw) == set(proj), (model, sorted(set(raw) ^ set(proj)))
        for layer in sorted(raw):
            r, p = raw[layer], proj[layer]
            n_layers = r["n_layers"]
            raw_gap, proj_gap = r["raw_gap"], p["proj_gap"]
            delta_gap = raw_gap - proj_gap
            unstable = abs(raw_gap) < RAW_GAP_EPS
            rescue_fraction = None if unstable else delta_gap / raw_gap
            rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "layer": layer,
                "n_layers": n_layers,
                "relative_depth": layer / (n_layers - 1),
                "raw_joint_auc": r["raw_joint_auc"],
                "raw_b2c_auc": r["raw_b2c_auc"],
                "raw_c2b_auc": r["raw_c2b_auc"],
                "raw_gap": raw_gap,
                "proj_joint_auc": p["proj_joint_auc"],
                "proj_b2c_auc": p["proj_b2c_auc"],
                "proj_c2b_auc": p["proj_c2b_auc"],
                "proj_gap": proj_gap,
                "proj_format_auc": p["proj_format_auc"],
                "raw_format_auc": r["raw_format_auc"],
                "delta_format_auc": p["proj_format_auc"] - r["raw_format_auc"],
                "delta_gap": delta_gap,
                "delta_joint_auc": p["proj_joint_auc"] - r["raw_joint_auc"],
                "delta_b2c_auc": p["proj_b2c_auc"] - r["raw_b2c_auc"],
                "delta_c2b_auc": p["proj_c2b_auc"] - r["raw_c2b_auc"],
                "rescue_fraction": rescue_fraction,
                "rescue_fraction_unstable": unstable,
            })
    return rows


def summarize(rows):
    summaries = {}
    for model in RAW_PATHS:
        mrows = [r for r in rows if r["model"] == model]
        eligible = [r for r in mrows if r["raw_joint_auc"] >= PURPOSE_FORMED_AUC]
        n_total = len(mrows)
        n_eligible = len(eligible)
        if n_eligible == 0:
            summaries[model] = {
                "model_label": MODEL_LABELS[model],
                "n_layers_total": n_total,
                "n_layers_eligible": 0,
                "note": f"no layer reached raw joint AUC >= {PURPOSE_FORMED_AUC}",
            }
            continue

        def arr(key):
            return np.array([r[key] for r in eligible], dtype=float)

        raw_gap = arr("raw_gap")
        proj_gap = arr("proj_gap")
        delta_gap = arr("delta_gap")
        delta_joint = arr("delta_joint_auc")
        delta_b2c = arr("delta_b2c_auc")
        delta_c2b = arr("delta_c2b_auc")
        stable = [r for r in eligible if not r["rescue_fraction_unstable"]]
        rescue_fracs = np.array([r["rescue_fraction"] for r in stable], dtype=float)

        summaries[model] = {
            "model_label": MODEL_LABELS[model],
            "n_layers_total": n_total,
            "n_layers_eligible": n_eligible,
            "eligible_layer_range": [min(r["layer"] for r in eligible), max(r["layer"] for r in eligible)],
            "eligible_relative_depth_range": [
                min(r["relative_depth"] for r in eligible),
                max(r["relative_depth"] for r in eligible),
            ],
            "raw_gap_mean": float(raw_gap.mean()),
            "raw_gap_median": float(np.median(raw_gap)),
            "proj_gap_mean": float(proj_gap.mean()),
            "proj_gap_median": float(np.median(proj_gap)),
            "delta_gap_mean": float(delta_gap.mean()),
            "delta_gap_median": float(np.median(delta_gap)),
            "n_layers_stable_denominator": int(len(stable)),
            "rescue_fraction_median_excl_near_zero": (
                float(np.median(rescue_fracs)) if len(rescue_fracs) else None
            ),
            "mean_abs_delta_joint_auc": float(np.abs(delta_joint).mean()),
            "mean_delta_b2c_auc": float(delta_b2c.mean()),
            "mean_delta_c2b_auc": float(delta_c2b.mean()),
            "frac_layers_gap_reduced": float(np.mean(delta_gap > 0)),
            "frac_layers_both_directions_improved": float(
                np.mean((delta_b2c > 0) & (delta_c2b > 0))
            ),
            "format_persistence_all_layers": {
                "n_layers": n_total,
                "raw_format_auc_median": float(np.median([r["raw_format_auc"] for r in mrows])),
                "raw_format_auc_min": float(np.min([r["raw_format_auc"] for r in mrows])),
                "proj_format_auc_median": float(np.median([r["proj_format_auc"] for r in mrows])),
                "proj_format_auc_min": float(np.min([r["proj_format_auc"] for r in mrows])),
                "proj_format_auc_max": float(np.max([r["proj_format_auc"] for r in mrows])),
                "note": (
                    "Format decodability before/after the rank-64 matched-format projection, "
                    "computed over ALL layers (not gated by purpose formation) since format "
                    "decodability is a separate question from purpose decodability."
                ),
            },
        }
    return summaries


def main():
    rows = build_table()
    summaries = summarize(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    payload = {
        "schema_version": 1,
        "purpose": (
            "Per-layer raw vs rank-64 matched-format-projected purpose/transfer "
            "AUCs and gap-rescue diagnostics for all layers of Llama-3.1-8B, "
            "Llama-3.1-70B, Llama-3.3-70B, joined from "
            "results/layer_sweep_full320-v3_<model>.json (raw) and "
            "results/full320_v3_erasure_full_layer_sweep.json (rank-64 "
            "matched-format projection). No new probes fit; pure derived join."
        ),
        "definitions": {
            "raw_gap": "raw_joint_auc - mean(raw_b2c_auc, raw_c2b_auc)",
            "proj_gap": "proj_joint_auc - mean(proj_b2c_auc, proj_c2b_auc)",
            "delta_gap": "raw_gap - proj_gap (positive = projection shrinks the gap)",
            "delta_joint_auc": "proj_joint_auc - raw_joint_auc",
            "delta_b2c_auc": "proj_b2c_auc - raw_b2c_auc",
            "delta_c2b_auc": "proj_c2b_auc - raw_c2b_auc",
            "rescue_fraction": "delta_gap / raw_gap, only defined where |raw_gap| >= "
                                f"{RAW_GAP_EPS} AUC; otherwise null and flagged unstable",
            "purpose_formed_region": f"layers with raw_joint_auc >= {PURPOSE_FORMED_AUC}, "
                                      "defined from the raw trajectory only",
        },
        "raw_gap_eps": RAW_GAP_EPS,
        "purpose_formed_auc_cutoff": PURPOSE_FORMED_AUC,
        "inputs": {
            "raw": {k: str(v.relative_to(ROOT)) for k, v in RAW_PATHS.items()},
            "projected": str(PROJ_PATH.relative_to(ROOT)),
        },
        "model_summaries": summaries,
        "per_layer": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
