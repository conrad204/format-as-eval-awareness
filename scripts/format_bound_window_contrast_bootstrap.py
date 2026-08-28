"""Predeclared fixed-window bootstrap for the cross-format generalization gap.

Fixes a selection-bias trap in the earlier regrowth-bootstrap script: bootstrapping
gap_final - gap_at_argmin(gap) is biased positive by construction, because the
comparison layer is chosen from the same curve being tested (argmin of a noisy
curve is <= its true minimum in expectation, so "final - selected minimum" is
mechanically non-negative-biased even under a flat null).

This script instead uses two DEPTH WINDOWS fixed independently of the observed
curve shape: mid = relative depth in [0.35, 0.55], late = relative depth in
[0.75, 1.00]. For each model, sample the (up to) 11 layers nearest the grid
{0.35, 0.40, ..., 0.55} u {0.75, 0.80, ..., 1.00}, refit OOF probe scores once
per layer/regime (same procedure as scripts/sweep_full.py), then payload-block
bootstrap (1000 reps) the WINDOW-MEAN gap contrast

    contrast = mean(gap over late-window layers) - mean(gap over mid-window layers)

Excludes the 3,840 no-cue control rows (label is null; see
scripts/format_bound_regrowth_bootstrap.py for the same documented exclusion).
"""
import json
import sys
import zipfile

import numpy as np
from sklearn.metrics import roc_auc_score

from format_bound_regrowth_bootstrap import ITEMS_PATH, oof_scores, stream_extract_layers

MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "n_layers": 32},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "n_layers": 80},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "n_layers": 80},
}
MID_WINDOW_DEPTHS = [0.35, 0.40, 0.45, 0.50, 0.55]
LATE_WINDOW_DEPTHS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
N_BOOT = 1000
RNG_SEED = 0


def depths_to_layers(depths, n_layers):
    return sorted({round(d * (n_layers - 1)) for d in depths})


def main():
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    n_total = len(rows_all)
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    n_excluded_no_cue = n_total - len(rows_by_id)
    print(f"stated rows: {len(rows_by_id)} / {n_total} ({n_excluded_no_cue} no-cue rows excluded)", file=sys.stderr)

    results = {"n_total_rows": n_total, "n_stated_rows": len(rows_by_id), "n_excluded_no_cue_rows": n_excluded_no_cue,
               "mid_window_depths": MID_WINDOW_DEPTHS, "late_window_depths": LATE_WINDOW_DEPTHS}

    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        n_layers = cfg["n_layers"]
        mid_layers = depths_to_layers(MID_WINDOW_DEPTHS, n_layers)
        late_layers = depths_to_layers(LATE_WINDOW_DEPTHS, n_layers)
        all_layers = sorted(set(mid_layers) | set(late_layers))
        print(f"  mid-window layers {mid_layers}, late-window layers {late_layers}", file=sys.stderr)

        X, layer_list = stream_extract_layers(cfg["npz"], all_layers)
        layer_pos = {L: j for j, L in enumerate(layer_list)}

        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        X = X[keep_mask]
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        blk = np.array([r["payload_block_id"] for r in kept_rows])
        ALL = np.ones(len(kept_rows), bool)
        is_bench = fmt == "benchmark"
        is_casual = fmt == "casual"

        per_layer = {}
        for L in all_layers:
            Xl = X[:, layer_pos[L], :]
            s_decorr = oof_scores(Xl, y, fam, ALL, ALL)
            s_b2c = oof_scores(Xl, y, fam, is_bench, is_casual)
            s_c2b = oof_scores(Xl, y, fam, is_casual, is_bench)
            auc_decorr = roc_auc_score(y, s_decorr)
            m_b2c = ~np.isnan(s_b2c)
            m_c2b = ~np.isnan(s_c2b)
            auc_b2c = roc_auc_score(y[m_b2c], s_b2c[m_b2c])
            auc_c2b = roc_auc_score(y[m_c2b], s_c2b[m_c2b])
            gap_point = auc_decorr - (auc_b2c + auc_c2b) / 2
            per_layer[L] = dict(s_decorr=s_decorr, s_b2c=s_b2c, s_c2b=s_c2b, gap_point=gap_point)
            print(f"  layer {L} (depth {L/(n_layers-1):.3f}): gap={gap_point:.4f}", file=sys.stderr, flush=True)

        point_mid = float(np.mean([per_layer[L]["gap_point"] for L in mid_layers]))
        point_late = float(np.mean([per_layer[L]["gap_point"] for L in late_layers]))
        point_contrast = point_late - point_mid

        blocks = np.array(sorted(set(blk)))
        n_blocks = len(blocks)
        block_to_idx = {b: np.where(blk == b)[0] for b in blocks}
        rng = np.random.default_rng(RNG_SEED)

        def gap_for_layer_sample(L, idx):
            d = per_layer[L]
            auc_decorr = roc_auc_score(y[idx], d["s_decorr"][idx])
            m_b2c = idx[~np.isnan(d["s_b2c"][idx])]
            m_c2b = idx[~np.isnan(d["s_c2b"][idx])]
            auc_b2c = roc_auc_score(y[m_b2c], d["s_b2c"][m_b2c])
            auc_c2b = roc_auc_score(y[m_c2b], d["s_c2b"][m_c2b])
            return auc_decorr - (auc_b2c + auc_c2b) / 2

        contrast_boot = np.empty(N_BOOT)
        mid_boot = np.empty(N_BOOT)
        late_boot = np.empty(N_BOOT)
        for r in range(N_BOOT):
            sample_blocks = rng.choice(blocks, size=n_blocks, replace=True)
            idx = np.concatenate([block_to_idx[b] for b in sample_blocks])
            mid_val = float(np.mean([gap_for_layer_sample(L, idx) for L in mid_layers]))
            late_val = float(np.mean([gap_for_layer_sample(L, idx) for L in late_layers]))
            mid_boot[r] = mid_val
            late_boot[r] = late_val
            contrast_boot[r] = late_val - mid_val

        def ci(a):
            return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

        results[model] = dict(
            mid_layers=mid_layers,
            late_layers=late_layers,
            mid_window_gap_point=point_mid,
            late_window_gap_point=point_late,
            contrast_point=point_contrast,
            mid_window_gap_ci95=ci(mid_boot),
            late_window_gap_ci95=ci(late_boot),
            contrast_ci95=ci(contrast_boot),
            frac_boot_contrast_positive=float(np.mean(contrast_boot > 0)),
        )
        print(
            f"  window contrast (late-mean - mid-mean) point={point_contrast:.4f} "
            f"95% CI={results[model]['contrast_ci95']} frac>0={results[model]['frac_boot_contrast_positive']:.3f}",
            file=sys.stderr, flush=True,
        )
        del X

    out_path = "artifacts/format_bound_window_contrast_bootstrap.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
