"""Exact decomposition: does transferred-probe AUC collapse from signal loss or noise growth?

For each model and layer, fit four OOF regimes with the SAME 5-fold
cue-family CV as scripts/sweep_full.py:
  bb = train benchmark, test benchmark (held-out family)   -- "self" baseline
  cc = train casual,    test casual                        -- "self" baseline
  b2c = train benchmark, test casual                       -- transfer
  c2b = train casual,    test benchmark                    -- transfer

Critically, bb and c2b are BOTH scored on the SAME held-out benchmark-format
test items (different probe: benchmark-self vs casual-trained-transfer).
Symmetrically cc and b2c are both scored on the same held-out casual-format
items. That gives an apples-to-apples comparison: same test set, same
class-separation task, only the probe direction differs (self vs
cross-trained).

Payload blocks appear in every purpose x format cell with the same content,
so pairing (E score - D score) within (payload_block_id, test_format) cancels
content variance -- the "matched eval/deploy twins" decomposition requested.
For each regime's paired differences delta_b:
  signal = mean(delta_b)          -- how much the probe separates E from D
  noise  = std(delta_b)           -- how much that separation varies block-to-block
  d_prime = signal / noise        -- approximates AUC via Phi(d_prime / sqrt(2))

Comparing (signal, noise) for the self regime against the transfer regime run
on the SAME test items isolates whether the transfer AUC drop comes from the
probe's mean-difference direction losing alignment with the true separation
in the other format (signal drop) or from increased scatter along that
direction in the other format (noise growth), or both.

Depth anchor is FIXED (not selected from the observed peak) at relative
depth 0.27, matching the user's fixed-anchor robustness requirement -- this
avoids the "the anchor itself is a chosen extremum" critique that applied to
earlier analyses in this project.
"""
import json
import sys
import zipfile

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from format_bound_regrowth_bootstrap import ITEMS_PATH, oof_scores, stream_extract_layers

MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "n_layers": 32},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "n_layers": 80},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "n_layers": 80},
}
FIXED_ANCHOR_DEPTH = 0.27
MID_DEPTH = 0.5
FINAL_DEPTH = 1.0


def depth_to_layer(depth, n_layers):
    return round(depth * (n_layers - 1))


def paired_diff(scores, purpose, blk, test_mask):
    """Group scores by payload_block among rows with test_mask True; return
    array of (E score - D score) per block that has both purposes present."""
    d = {}
    idx = np.where(test_mask)[0]
    for i in idx:
        if np.isnan(scores[i]):
            continue
        d.setdefault(blk[i], {})[purpose[i]] = scores[i]
    diffs = [v["evaluation"] - v["deployment"] for v in d.values() if len(v) == 2]
    return np.array(diffs)


def main():
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
        print(f"  layers: anchor(0.27)={anchor_L} mid(0.5)={mid_L} final={final_L}", file=sys.stderr)

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

        model_result = {"anchor_layer": int(anchor_L), "mid_layer": int(mid_L), "final_layer": int(final_L),
                         "anchor_depth": FIXED_ANCHOR_DEPTH, "layers": {}}

        for L in [anchor_L, mid_L, final_L]:
            Xl = X[:, layer_pos[L], :]
            s_bb = oof_scores(Xl, y, fam, is_bench, is_bench)
            s_cc = oof_scores(Xl, y, fam, is_casual, is_casual)
            s_b2c = oof_scores(Xl, y, fam, is_bench, is_casual)
            s_c2b = oof_scores(Xl, y, fam, is_casual, is_bench)

            regimes = {}
            for name, scores, test_mask in [
                ("bb_self", s_bb, is_bench),
                ("cc_self", s_cc, is_casual),
                ("b2c_transfer", s_b2c, is_casual),
                ("c2b_transfer", s_c2b, is_bench),
            ]:
                valid = test_mask & ~np.isnan(scores)
                e_scores = scores[valid & (purpose == "evaluation")]
                d_scores = scores[valid & (purpose == "deployment")]
                mean_e, var_e = float(np.mean(e_scores)), float(np.var(e_scores, ddof=1))
                mean_d, var_d = float(np.mean(d_scores)), float(np.var(d_scores, ddof=1))
                signal = mean_e - mean_d
                pooled_sd = float(np.sqrt((var_e + var_d) / 2))
                d_prime = signal / pooled_sd
                approx_auc = float(norm.cdf(d_prime / np.sqrt(2)))
                real_auc = float(roc_auc_score(y[valid], scores[valid]))

                diffs = paired_diff(scores, purpose, blk, test_mask)
                paired_signal = float(np.mean(diffs))
                paired_noise = float(np.std(diffs, ddof=1))

                regimes[name] = {
                    "n_e": int(len(e_scores)), "n_d": int(len(d_scores)),
                    "unpaired_signal_mean_diff": signal,
                    "unpaired_pooled_sd": pooled_sd,
                    "unpaired_d_prime": d_prime,
                    "approx_auc_from_dprime": approx_auc,
                    "real_auc": real_auc,
                    "n_pairs": int(len(diffs)),
                    "paired_signal_mean_diff": paired_signal,
                    "paired_noise_sd": paired_noise,
                    "paired_d_prime": paired_signal / paired_noise,
                }

            model_result["layers"][str(L)] = regimes

            # payload-block bootstrap CI on the signal/noise decomposition and its
            # log-share split, for both transfer directions, at this layer
            all_scores = {"bb_self": s_bb, "cc_self": s_cc, "b2c_transfer": s_b2c, "c2b_transfer": s_c2b}
            all_masks = {"bb_self": is_bench, "cc_self": is_casual, "b2c_transfer": is_casual, "c2b_transfer": is_bench}
            blocks = np.array(sorted(set(blk)))
            n_blocks = len(blocks)
            block_to_idx = {b: np.where(blk == b)[0] for b in blocks}
            rng = np.random.default_rng(20260828)
            n_boot = 1000

            def unpaired_signal_noise_dprime(scores, test_mask, idx):
                m = idx[test_mask[idx] & ~np.isnan(scores[idx])]
                e = scores[m[purpose[m] == "evaluation"]]
                d = scores[m[purpose[m] == "deployment"]]
                if len(e) < 2 or len(d) < 2:
                    return np.nan, np.nan, np.nan
                sig = float(np.mean(e) - np.mean(d))
                sd = float(np.sqrt((np.var(e, ddof=1) + np.var(d, ddof=1)) / 2))
                return sig, sd, sig / sd if sd > 0 else np.nan

            bootstrap_ci = {}
            for pair_name, self_name, transfer_name in [("bench", "bb_self", "c2b_transfer"), ("casual", "cc_self", "b2c_transfer")]:
                signal_share_boot = np.empty(n_boot)
                noise_share_boot = np.empty(n_boot)
                dprime_self_boot = np.empty(n_boot)
                dprime_transfer_boot = np.empty(n_boot)
                for r in range(n_boot):
                    sample_blocks = rng.choice(blocks, size=n_blocks, replace=True)
                    idx = np.concatenate([block_to_idx[b] for b in sample_blocks])
                    sig_s, sd_s, dp_s = unpaired_signal_noise_dprime(all_scores[self_name], all_masks[self_name], idx)
                    sig_t, sd_t, dp_t = unpaired_signal_noise_dprime(all_scores[transfer_name], all_masks[transfer_name], idx)
                    dprime_self_boot[r] = dp_s
                    dprime_transfer_boot[r] = dp_t
                    if sig_s > 0 and sig_t > 0 and sd_s > 0 and sd_t > 0:
                        log_signal_drop = np.log(sig_s) - np.log(sig_t)
                        log_noise_grow = np.log(sd_t) - np.log(sd_s)
                        total = log_signal_drop + log_noise_grow
                        signal_share_boot[r] = 100 * log_signal_drop / total if total else np.nan
                        noise_share_boot[r] = 100 * log_noise_grow / total if total else np.nan
                    else:
                        signal_share_boot[r] = np.nan
                        noise_share_boot[r] = np.nan

                def ci(a):
                    a = a[~np.isnan(a)]
                    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]

                bootstrap_ci[pair_name] = {
                    "n_boot": n_boot,
                    "n_boot_valid": int(np.sum(~np.isnan(signal_share_boot))),
                    "signal_share_ci95": ci(signal_share_boot),
                    "noise_share_ci95": ci(noise_share_boot),
                    "dprime_self_ci95": ci(dprime_self_boot),
                    "dprime_transfer_ci95": ci(dprime_transfer_boot),
                }
            model_result["layers"][str(L)]["bootstrap_ci"] = bootstrap_ci
            for pair_name, ci_d in bootstrap_ci.items():
                print(
                    f"    [{pair_name}] bootstrap 95% CI: signal-share={ci_d['signal_share_ci95']} "
                    f"noise-share={ci_d['noise_share_ci95']} (n_valid={ci_d['n_boot_valid']}/{ci_d['n_boot']})",
                    file=sys.stderr, flush=True,
                )

        results[model] = model_result
        del X

    out_path = "artifacts/format_probe_signal_noise_decomposition.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
