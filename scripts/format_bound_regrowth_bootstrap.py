"""Block-bootstrap CI for the cross-format generalization gap's late-layer re-widening.

Reuses the exact OOF probe procedure from scripts/sweep_full.py (StandardScaler +
LogisticRegression(C=0.1), 5-fold cue-family-held-out CV) on cached full320-v3
activations, restricted to a handful of layers (min-gap layer, final layer, spike
layer) to keep memory bounded on a 16GB workstation with a 30GB (decompressed)
per-model activation tensor.

For each model: fit OOF decision scores once per layer/regime (decorrelated, b2c,
c2b), then payload-block bootstrap (resample the 320 blocks with replacement,
1000 reps) the *already-fit* scores to get a CI on gap_final - gap_min, i.e. the
late-layer re-widening magnitude, and its sign.
"""
import json
import sys
import zipfile

import numpy as np
import numpy.lib.format as npfmt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ITEMS_PATH = "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "min_layer": 14, "final_layer": 31, "spike_layer": 3},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "min_layer": 19, "final_layer": 79, "spike_layer": 4},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "min_layer": 21, "final_layer": 79, "spike_layer": 4},
}
N_BOOT = 1000
RNG_SEED = 0


def stream_extract_layers(npz_path, layer_indices):
    """Stream-decompress X.npy from the npz zip, keeping only requested layers.

    Never materializes the full (n_items, n_layers, hidden) array in memory.
    """
    layer_indices = sorted(layer_indices)
    z = zipfile.ZipFile(npz_path)
    f = z.open("X.npy")
    ver = npfmt.read_magic(f)
    if ver == (1, 0):
        shape, fortran, dtype = npfmt.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = npfmt.read_array_header_2_0(f)
    assert not fortran
    n_items, n_layers, hidden = shape
    assert dtype == np.dtype("float32")
    row_bytes = n_layers * hidden * dtype.itemsize
    out = np.empty((n_items, len(layer_indices), hidden), dtype=np.float32)
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        for j, L in enumerate(layer_indices):
            out[i, j] = row[L]
        if (i + 1) % 2000 == 0:
            print(f"  ...extracted {i + 1}/{n_items} items", file=sys.stderr, flush=True)
    f.close()
    return out, layer_indices


def oof_scores(X, y, fam, tr_mask, te_mask):
    s = np.full(len(y), np.nan)
    for fold in sorted(set(fam)):
        trm = (fam != fold) & tr_mask
        tem = (fam == fold) & te_mask
        if not tem.any():
            continue
        sc = StandardScaler()
        m = LogisticRegression(C=0.1, max_iter=3000, random_state=0)
        m.fit(sc.fit_transform(X[trm]), y[trm])
        s[tem] = m.decision_function(sc.transform(X[tem]))
    return s


def main():
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    n_total = len(rows_all)
    # The frozen full320-v3 item set includes a fifth "no-cue" control arm
    # (intended_purpose == "none", label == null) outside the crossed
    # purpose x format 2x2 design; it is scored elsewhere (cue-excision
    # surface battery, Jacobian-lens no-cue format diagnostic) and cannot
    # enter an eval-vs-deploy AUC because it has no purpose label. We
    # exclude it here for that structural reason, matching the pinned
    # scripts/sweep_full.py pipeline exactly (this is why the pinned
    # artifacts and main_02's caption report "7,680 stated rows", not
    # 11,520). Record the exclusion count rather than dropping it silently.
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    n_excluded_no_cue = n_total - len(rows_by_id)
    print(f"stated rows: {len(rows_by_id)} / {n_total} "
          f"({n_excluded_no_cue} no-cue control rows excluded)", file=sys.stderr)

    results = {"n_total_rows": n_total, "n_stated_rows": len(rows_by_id),
               "n_excluded_no_cue_rows": n_excluded_no_cue}
    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        layer_list = sorted({cfg["min_layer"], cfg["final_layer"], cfg["spike_layer"]})
        X, layer_list = stream_extract_layers(cfg["npz"], layer_list)
        layer_pos = {L: j for j, L in enumerate(layer_list)}

        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id), (model, keep_mask.sum(), len(rows_by_id))
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
        for L in [cfg["min_layer"], cfg["final_layer"], cfg["spike_layer"]]:
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
            per_layer[L] = dict(
                s_decorr=s_decorr, s_b2c=s_b2c, s_c2b=s_c2b,
                auc_decorr=auc_decorr, auc_b2c=auc_b2c, auc_c2b=auc_c2b,
                gap_point=gap_point,
            )
            print(f"  layer {L}: decorr={auc_decorr:.4f} b2c={auc_b2c:.4f} c2b={auc_c2b:.4f} gap={gap_point:.4f}",
                  file=sys.stderr, flush=True)

        blocks = np.array(sorted(set(blk)))
        n_blocks = len(blocks)
        block_to_idx = {b: np.where(blk == b)[0] for b in blocks}
        rng = np.random.default_rng(RNG_SEED)

        def gap_for_sample(L, idx):
            d = per_layer[L]
            auc_decorr = roc_auc_score(y[idx], d["s_decorr"][idx])
            m_b2c = idx[~np.isnan(d["s_b2c"][idx])]
            m_c2b = idx[~np.isnan(d["s_c2b"][idx])]
            auc_b2c = roc_auc_score(y[m_b2c], d["s_b2c"][m_b2c])
            auc_c2b = roc_auc_score(y[m_c2b], d["s_c2b"][m_c2b])
            return auc_decorr - (auc_b2c + auc_c2b) / 2

        widening = np.empty(N_BOOT)
        gap_min_boot = np.empty(N_BOOT)
        gap_final_boot = np.empty(N_BOOT)
        for r in range(N_BOOT):
            sample_blocks = rng.choice(blocks, size=n_blocks, replace=True)
            idx = np.concatenate([block_to_idx[b] for b in sample_blocks])
            g_min = gap_for_sample(cfg["min_layer"], idx)
            g_final = gap_for_sample(cfg["final_layer"], idx)
            gap_min_boot[r] = g_min
            gap_final_boot[r] = g_final
            widening[r] = g_final - g_min

        def ci(a):
            return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

        results[model] = dict(
            min_layer=cfg["min_layer"],
            final_layer=cfg["final_layer"],
            gap_min_point=per_layer[cfg["min_layer"]]["gap_point"],
            gap_final_point=per_layer[cfg["final_layer"]]["gap_point"],
            widening_point=per_layer[cfg["final_layer"]]["gap_point"] - per_layer[cfg["min_layer"]]["gap_point"],
            gap_min_ci95=ci(gap_min_boot),
            gap_final_ci95=ci(gap_final_boot),
            widening_ci95=ci(widening),
            frac_boot_widening_positive=float(np.mean(widening > 0)),
        )
        print(
            f"  widening (final-min) point={results[model]['widening_point']:.4f} "
            f"95% CI={results[model]['widening_ci95']} "
            f"frac>0={results[model]['frac_boot_widening_positive']:.3f}",
            file=sys.stderr, flush=True,
        )
        del X

    out_path = "artifacts/format_bound_regrowth_bootstrap.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
