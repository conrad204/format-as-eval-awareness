"""Observational purpose x format effect-coded decomposition, per full320 layer.

Fits the same decomposition as Eq. 1 of the paper (h = mu + p*A + f*B + p*f*C + eps)
directly on full320 activations at EVERY layer, for all three models -- something
currently done only at a handful of frozen depths in the separate pilot80 causal
dataset. This is cell-mean arithmetic, not probe fitting: no logistic regression,
no CV folds, one streaming pass per model.

For each layer, compute the four cell means (mu_EB, mu_EC, mu_DB, mu_DC) over the
7,680 stated-purpose rows (no-cue control rows excluded; see
scripts/format_bound_regrowth_bootstrap.py for the same documented exclusion), then

  A = ((mu_EB - mu_DB) + (mu_EC - mu_DC)) / 2   # format-shared purpose effect
  B = ((mu_EB - mu_EC) + (mu_DB - mu_DC)) / 2   # purpose-shared format effect
  C = (mu_EB - mu_EC) - (mu_DB - mu_DC)         # purpose x format interaction

  rho(layer) = ||C|| / ||A||

Then compares rho(layer) against the already-computed cross-format transfer gap
G(layer) from the pinned full320 sweeps, per model, as a purely observational,
non-causal association (Spearman rank correlation across layers, reported with
the explicit caveat that layers are repeated/autocorrelated measurements within
a model, not independent samples -- the paper's own stated limitation elsewhere).
"""
import json
import sys
import zipfile

import numpy as np
from scipy.stats import spearmanr

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
PINNED_DIR = "artifacts/pinned_full320"
MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz",
                    "sweep": f"{PINNED_DIR}/layer_sweep_full320-v3_llama31_8b.json"},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz",
                     "sweep": f"{PINNED_DIR}/layer_sweep_full320-v3_llama31_70b.json"},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz",
                     "sweep": f"{PINNED_DIR}/layer_sweep_full320-v3_llama33_70b.json"},
}


def cell_means_all_layers(npz_path, cell_masks, n_items_expected):
    """Stream X.npy once; accumulate per-cell, per-layer running sums.

    cell_masks: dict[cell_name] -> boolean array over the FULL raw item order
    (aligned to item_ids in the npz), so we can accumulate on the fly without
    ever holding the full (n_items, n_layers, hidden) tensor in memory.
    """
    z = zipfile.ZipFile(npz_path)
    f = z.open("X.npy")
    import numpy.lib.format as npfmt
    ver = npfmt.read_magic(f)
    if ver == (1, 0):
        shape, fortran, dtype = npfmt.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = npfmt.read_array_header_2_0(f)
    assert not fortran and dtype == np.dtype("float32")
    n_items, n_layers, hidden = shape
    assert n_items == n_items_expected

    sums = {cell: np.zeros((n_layers, hidden), dtype=np.float64) for cell in cell_masks}
    counts = {cell: 0 for cell in cell_masks}
    row_bytes = n_layers * hidden * dtype.itemsize
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        for cell, mask in cell_masks.items():
            if mask[i]:
                sums[cell] += row
                counts[cell] += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{n_items}", file=sys.stderr, flush=True)
    f.close()
    means = {cell: sums[cell] / counts[cell] for cell in cell_masks}
    return means, counts, n_layers


def main():
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    results = {}
    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        cell_masks = {
            "EB": np.array([rows_by_id.get(i, {}).get("intended_purpose") == "evaluation" and rows_by_id.get(i, {}).get("format") == "benchmark" for i in raw_ids]),
            "EC": np.array([rows_by_id.get(i, {}).get("intended_purpose") == "evaluation" and rows_by_id.get(i, {}).get("format") == "casual" for i in raw_ids]),
            "DB": np.array([rows_by_id.get(i, {}).get("intended_purpose") == "deployment" and rows_by_id.get(i, {}).get("format") == "benchmark" for i in raw_ids]),
            "DC": np.array([rows_by_id.get(i, {}).get("intended_purpose") == "deployment" and rows_by_id.get(i, {}).get("format") == "casual" for i in raw_ids]),
        }
        means, counts, n_layers = cell_means_all_layers(cfg["npz"], cell_masks, len(raw_ids))
        print(f"  cell counts: {counts}", file=sys.stderr)

        A = ((means["EB"] - means["DB"]) + (means["EC"] - means["DC"])) / 2  # (n_layers, hidden)
        B = ((means["EB"] - means["EC"]) + (means["DB"] - means["DC"])) / 2
        C = (means["EB"] - means["EC"]) - (means["DB"] - means["DC"])
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
        norm_C = np.linalg.norm(C, axis=1)
        rho = norm_C / norm_A

        sweep = json.load(open(cfg["sweep"]))
        assert len(sweep) == n_layers
        depth = np.array([r["layer"] / (n_layers - 1) for r in sweep])
        dual = np.array([r["decorrelated"]["auc"] for r in sweep])
        cross_mean = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in sweep])
        gap = dual - cross_mean

        rho_spear, rho_p = spearmanr(rho, gap)
        # also restrict to post-spike layers (depth >= 0.15) to avoid the
        # early-layer noise floor dominating the rank correlation
        post_spike = depth >= 0.15
        rho_spear_post, rho_p_post = spearmanr(rho[post_spike], gap[post_spike])

        results[model] = {
            "n_layers": int(n_layers),
            "depth": depth.tolist(),
            "norm_A": norm_A.tolist(),
            "norm_B": norm_B.tolist(),
            "norm_C": norm_C.tolist(),
            "rho_C_over_A": rho.tolist(),
            "gap": gap.tolist(),
            "spearman_rho_vs_gap_all_layers": {"r": float(rho_spear), "p": float(rho_p)},
            "spearman_rho_vs_gap_post_spike": {"r": float(rho_spear_post), "p": float(rho_p_post), "n_layers_used": int(post_spike.sum())},
        }
        print(f"  spearman(rho, gap) all-layers r={rho_spear:.3f} p={rho_p:.4f}; "
              f"post-spike (depth>=0.15) r={rho_spear_post:.3f} p={rho_p_post:.4f}",
              file=sys.stderr, flush=True)

    out_path = "artifacts/format_interaction_geometry_full320.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
