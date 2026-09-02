"""Does the purpose *direction* rotate apart between formats as depth grows?

Motivation: the earlier rho(l) = ||C(l)|| / ||A(l)|| analysis
(format_interaction_geometry_full320.py) failed a depth-confound check --
both ||A|| and ||C|| track relative depth at r~0.97-1.00, almost certainly
because residual-stream activation norms are known to grow with depth in
transformers for purely architectural reasons (residual accumulation), not
because of anything specific to purpose/format structure. A magnitude ratio
inherits that confound even after taking a ratio, if numerator and
denominator both scale with the same ambient norm-growth factor.

This script instead computes a SCALE-INVARIANT, DIRECTIONAL quantity that
sidesteps that confound by construction and has a much more direct
mechanistic link to cross-format transfer failure:

  dir_benchmark(l) = mean_EB(l) - mean_DB(l)   # purpose direction within benchmark format
  dir_casual(l)    = mean_EC(l) - mean_DC(l)   # purpose direction within casual format
  cos_align(l)     = cosine_similarity(dir_benchmark(l), dir_casual(l))

If the two format-specific purpose directions are well aligned (cos_align
near 1), a probe trained on one format's direction naturally transfers to the
other. If they rotate apart (cos_align falls), transfer degrades almost by
definition -- this is a much tighter mechanistic story than "some norm ratio
happens to move together with something else."

Also records the ambient per-layer activation norm (mean ||activation|| over
all stated rows) to directly show whether ||A||/||C|| growth in the earlier
script is largely just riding the known residual-stream norm-growth curve.

Same exclusions and CV-fold-free cell-mean approach as
format_interaction_geometry_full320.py: 3,840 no-cue control rows excluded
(no purpose label), one streaming pass per model, no probe fitting.
"""
import json
import sys
import zipfile

import numpy as np
from scipy.stats import rankdata, spearmanr

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


def stream_stats(npz_path, cell_masks, n_items_expected):
    """One streaming pass: per-cell running sums (for means) AND running sum of
    per-item activation norms (ambient scale), per layer. Never materializes the
    full (n_items, n_layers, hidden) tensor.
    """
    import numpy.lib.format as npfmt

    z = zipfile.ZipFile(npz_path)
    f = z.open("X.npy")
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
    norm_sum = np.zeros(n_layers, dtype=np.float64)
    norm_count = 0
    row_bytes = n_layers * hidden * dtype.itemsize
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        any_cell = False
        for cell, mask in cell_masks.items():
            if mask[i]:
                sums[cell] += row
                counts[cell] += 1
                any_cell = True
        if any_cell:
            norm_sum += np.linalg.norm(row, axis=1)
            norm_count += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{n_items}", file=sys.stderr, flush=True)
    f.close()
    means = {cell: sums[cell] / counts[cell] for cell in cell_masks}
    ambient_norm = norm_sum / norm_count
    return means, counts, ambient_norm, n_layers


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        m, c = np.polyfit(b, a, 1)
        return a - (m * b + c)

    return spearmanr(resid(rx, rz), resid(ry, rz))


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
        means, counts, ambient_norm, n_layers = stream_stats(cfg["npz"], cell_masks, len(raw_ids))
        print(f"  cell counts: {counts}", file=sys.stderr)

        dir_benchmark = means["EB"] - means["DB"]  # (n_layers, hidden)
        dir_casual = means["EC"] - means["DC"]
        num = np.sum(dir_benchmark * dir_casual, axis=1)
        denom = np.linalg.norm(dir_benchmark, axis=1) * np.linalg.norm(dir_casual, axis=1)
        cos_align = num / denom
        norm_benchmark = np.linalg.norm(dir_benchmark, axis=1)
        norm_casual = np.linalg.norm(dir_casual, axis=1)

        sweep = json.load(open(cfg["sweep"]))
        assert len(sweep) == n_layers
        depth = np.array([r["layer"] / (n_layers - 1) for r in sweep])
        dual = np.array([r["decorrelated"]["auc"] for r in sweep])
        cross_mean_auc = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in sweep])
        gap = dual - cross_mean_auc

        post_spike = depth >= 0.15
        cos_p, gap_p, depth_p = cos_align[post_spike], gap[post_spike], depth[post_spike]

        r_raw, p_raw = spearmanr(cos_p, gap_p)
        r_partial, p_partial = partial_spearman(cos_p, gap_p, depth_p)
        d_cos, d_gap = np.diff(cos_p), np.diff(gap_p)
        r_delta, p_delta = spearmanr(d_cos, d_gap)
        r_cos_depth, _ = spearmanr(cos_p, depth_p)
        r_ambientnorm_depth, _ = spearmanr(ambient_norm[post_spike], depth_p)

        results[model] = {
            "n_layers": int(n_layers),
            "depth": depth.tolist(),
            "cos_align_benchmark_casual": cos_align.tolist(),
            "norm_dir_benchmark": norm_benchmark.tolist(),
            "norm_dir_casual": norm_casual.tolist(),
            "ambient_activation_norm": ambient_norm.tolist(),
            "gap": gap.tolist(),
            "post_spike": {
                "n": int(post_spike.sum()),
                "spearman_cos_vs_gap_raw": {"r": float(r_raw), "p": float(p_raw)},
                "spearman_cos_vs_gap_partial_depth": {"r": float(r_partial), "p": float(p_partial)},
                "spearman_delta_cos_vs_delta_gap": {"r": float(r_delta), "p": float(p_delta)},
                "spearman_cos_vs_depth": float(r_cos_depth),
                "spearman_ambient_norm_vs_depth": float(r_ambientnorm_depth),
            },
        }
        print(f"  cos_align vs gap: raw r={r_raw:.3f} p={p_raw:.4f} | partial(depth) r={r_partial:.3f} p={p_partial:.4f} | "
              f"delta r={r_delta:.3f} p={p_delta:.4f} | cos~depth r={r_cos_depth:.3f} | ambient_norm~depth r={r_ambientnorm_depth:.3f}",
              file=sys.stderr, flush=True)

    out_path = "artifacts/format_direction_alignment_full320.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
