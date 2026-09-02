"""Targeted intervention: does erasing linear format information rescue the
two failure modes identified in format_probe_signal_noise_decomposition.py?

Not an open-ended search -- this tests one specific, pre-registered prediction
from that decomposition: if the casual->benchmark signal-loss failure mode
and the benchmark->casual noise-growth failure mode are mediated by LINEARLY
decodable format information, then surgically erasing that information
(LEACE: LEAst-squares Concept Erasure, Belrose et al. 2023) should
selectively shrink both, while leaving held-out purpose decodability intact.
If it does not, that is a real, reportable negative result: the failure
modes are not explained by a simple linear format subspace.

Procedure, matching the OOF fold structure used throughout this project:
  For each of the 5 cue-family folds f:
    - Fit a LEACE eraser for format (benchmark=0, casual=1) on the TRAINING
      rows only (all families except f, both purposes, both formats).
    - Apply that eraser to BOTH the training rows and the held-out family f's
      rows (a fixed linear map fit only on non-test data -- no leakage).
  This produces one OOF-erased representation per item, exactly analogous to
  the OOF probe scores used elsewhere.

Then, on the erased representations:
  1. Verify erasure: fit a format probe (same 5-fold OOF logistic regression)
     and confirm its AUC has collapsed toward chance.
  2. Recompute the full signal/noise decomposition for both transfer
     directions (bb_self/c2b_transfer, cc_self/b2c_transfer), exactly as in
     format_probe_signal_noise_decomposition.py.
  3. Recompute held-out purpose AUC (decorrelated: trained+tested on both
     formats) to check purpose decodability survives the erasure.

Same 3 fixed layers as the parent decomposition (27% depth anchor, 50% mid,
final layer) for direct pre/post comparison, same 3,840-row no-cue exclusion.
"""
import json
import sys
import time
import zipfile

import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from format_bound_regrowth_bootstrap import ITEMS_PATH, oof_scores, stream_extract_layers
from format_probe_signal_noise_decomposition import depth_to_layer, paired_diff

MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "n_layers": 32},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "n_layers": 80},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "n_layers": 80},
}
FIXED_ANCHOR_DEPTH = 0.27
MID_DEPTH = 0.5


def leace_erase_oof(Xl: np.ndarray, fmt: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """Fold-respecting LEACE erasure of format info: fit on train folds only,
    apply to held-out family, matching the project's OOF convention."""
    z_all = torch.tensor((fmt == "casual").astype(np.float32))
    x_all = torch.tensor(Xl.astype(np.float32))
    X_erased = np.empty_like(Xl)
    for f in sorted(set(fam)):
        trm = fam != f
        tem = fam == f
        t0 = time.time()
        eraser = LeaceEraser.fit(x_all[trm], z_all[trm])
        with torch.no_grad():
            X_erased[tem] = eraser(x_all[tem]).numpy()
        print(f"    fold {f}: LEACE fit+apply in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    return X_erased


def decompose_direction(layer_scores: dict, self_name: str, transfer_name: str, purpose, blk, y) -> dict:
    out = {}
    for name in (self_name, transfer_name):
        scores, test_mask = layer_scores[name]
        valid = test_mask & ~np.isnan(scores)
        e_scores = scores[valid & (purpose == "evaluation")]
        d_scores = scores[valid & (purpose == "deployment")]
        mean_e, var_e = float(np.mean(e_scores)), float(np.var(e_scores, ddof=1))
        mean_d, var_d = float(np.mean(d_scores)), float(np.var(d_scores, ddof=1))
        signal = mean_e - mean_d
        pooled_sd = float(np.sqrt((var_e + var_d) / 2))
        real_auc = float(roc_auc_score(y[valid], scores[valid]))
        diffs = paired_diff(scores, purpose, blk, test_mask)
        out[name] = {
            "unpaired_signal_mean_diff": signal,
            "unpaired_pooled_sd": pooled_sd,
            "unpaired_d_prime": signal / pooled_sd,
            "real_auc": real_auc,
            "paired_signal_mean_diff": float(np.mean(diffs)),
            "paired_noise_sd": float(np.std(diffs, ddof=1)),
        }
    return out


def main():
    torch.set_num_threads(12)
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    results = {}
    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        n_layers = cfg["n_layers"]
        anchor_L = depth_to_layer(FIXED_ANCHOR_DEPTH, n_layers)
        mid_L = depth_to_layer(MID_DEPTH, n_layers)
        final_L = n_layers - 1
        layer_list = sorted({anchor_L, mid_L, final_L})

        X, layer_list = stream_extract_layers(cfg["npz"], layer_list)
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
        purpose = np.array([r["intended_purpose"] for r in kept_rows])
        is_bench = fmt == "benchmark"
        is_casual = fmt == "casual"
        ALL = np.ones(len(kept_rows), bool)

        model_result = {"anchor_layer": int(anchor_L), "mid_layer": int(mid_L), "final_layer": int(final_L), "layers": {}}

        for L in [anchor_L, mid_L, final_L]:
            print(f"  --- layer {L} ---", file=sys.stderr, flush=True)
            Xl = X[:, layer_pos[L], :]
            X_erased = leace_erase_oof(Xl, fmt, fam)

            # 1. verify erasure: format decodability should collapse
            s_format = oof_scores(X_erased, (fmt == "casual").astype(int), fam, ALL, ALL)
            format_auc_post = float(roc_auc_score((fmt == "casual").astype(int), s_format))

            # 2. purpose decodability preserved?
            s_decorr_pre = oof_scores(Xl, y, fam, ALL, ALL)
            s_decorr_post = oof_scores(X_erased, y, fam, ALL, ALL)
            purpose_auc_pre = float(roc_auc_score(y, s_decorr_pre))
            purpose_auc_post = float(roc_auc_score(y, s_decorr_post))

            # 3. re-run the 4 transfer regimes on erased data
            s_bb = oof_scores(X_erased, y, fam, is_bench, is_bench)
            s_cc = oof_scores(X_erased, y, fam, is_casual, is_casual)
            s_b2c = oof_scores(X_erased, y, fam, is_bench, is_casual)
            s_c2b = oof_scores(X_erased, y, fam, is_casual, is_bench)
            layer_scores = {
                "bb_self": (s_bb, is_bench), "cc_self": (s_cc, is_casual),
                "b2c_transfer": (s_b2c, is_casual), "c2b_transfer": (s_c2b, is_bench),
            }
            bench_direction = decompose_direction(layer_scores, "bb_self", "c2b_transfer", purpose, blk, y)
            casual_direction = decompose_direction(layer_scores, "cc_self", "b2c_transfer", purpose, blk, y)

            model_result["layers"][str(L)] = {
                "format_auc_post_erasure": format_auc_post,
                "purpose_auc_pre_erasure": purpose_auc_pre,
                "purpose_auc_post_erasure": purpose_auc_post,
                "bench_direction_post_erasure": bench_direction,
                "casual_direction_post_erasure": casual_direction,
            }
            print(
                f"  layer {L}: format_auc_post={format_auc_post:.3f} (chance=0.5) | "
                f"purpose_auc pre={purpose_auc_pre:.3f} post={purpose_auc_post:.3f} | "
                f"bench: self_d'={bench_direction['bb_self']['unpaired_d_prime']:.3f} "
                f"transfer_d'={bench_direction['c2b_transfer']['unpaired_d_prime']:.3f} | "
                f"casual: self_d'={casual_direction['cc_self']['unpaired_d_prime']:.3f} "
                f"transfer_d'={casual_direction['b2c_transfer']['unpaired_d_prime']:.3f}",
                file=sys.stderr, flush=True,
            )

        results[model] = model_result
        del X

    out_path = "artifacts/format_leace_erasure_decomposition.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
