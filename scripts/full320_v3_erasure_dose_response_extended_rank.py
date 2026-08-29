"""Extended-rank dose-response for the matched-format-PCA projection: does
pushing rank past 64 continue to shrink the cross-format transfer gap
without damaging joint purpose AUC, and how close does format decodability
get to chance (0.5 AUC)?

Same methodology, probe, folds, and 3 pinned depths as
scripts/full320_v3_erasure_dose_response_bootstrap.py (ranks 0/1/4/16/64),
extended with ranks 128/256/384, plus a format-readout AUC at every rank
(the sanity probe trained to predict benchmark-vs-casual format from the
post-projection residual -- same convention as
scripts/full320_v3_erasure_full_layer_sweep.py's format_auc_after_erasure).

The matched-format-PCA basis is the SVD of the per-training-fold matrix of
per-(payload_block_id, purpose_pair_id) format-contrast vectors ("B" in
scripts/format_erasure_development.factorial_vectors); its rank is capped by
the number of complete matched groups in the training split (~420-434 for
this dataset's 5-fold split, not by the ambient hidden dimension), so ranks
above ~400 are not meaningful for this specific estimator -- 384 is close to
that ceiling without exceeding it for any fold.

N_BOOT reduced from 1000 to 300 (still enough for a stable 95% CI at this
sample size) since this run adds 3 extra rank points across only 3 layers
per model; this keeps the added compute proportionate to the question being
asked (does the trend continue / where does it bend), not a new
confirmation-grade CI study.

Designed to run on the uahpc cluster (32 cores, 125GB RAM).

Usage: python scripts/full320_v3_erasure_dose_response_extended_rank.py [out_json]
"""
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import numpy.lib.format as npfmt
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, fit_probe, orth_basis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": {"npz": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
                    "min_layer": 14, "final_layer": 31, "spike_layer": 3},
    "llama31_70b": {"npz": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
                     "min_layer": 19, "final_layer": 79, "spike_layer": 4},
    "llama33_70b": {"npz": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
                     "min_layer": 21, "final_layer": 79, "spike_layer": 4},
}
RANKS = (0, 1, 4, 16, 64, 128, 256, 384)
N_BOOT = 300
N_JOBS = int(os.environ.get("N_JOBS", "20"))
RNG_SEED = 0


def remove(X, Q):
    return X if Q is None or Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def stream_extract_layers(npz_path, layer_indices):
    layer_indices = sorted(layer_indices)
    z = zipfile.ZipFile(npz_path)
    f = z.open("X.npy")
    ver = npfmt.read_magic(f)
    if ver == (1, 0):
        shape, fortran, dtype = npfmt.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = npfmt.read_array_header_2_0(f)
    assert not fortran and dtype == np.dtype("float32")
    n_items, n_layers, hidden = shape
    row_bytes = n_layers * hidden * dtype.itemsize
    out = np.empty((n_items, len(layer_indices), hidden), dtype=np.float32)
    for i in range(n_items):
        buf = f.read(row_bytes)
        row = np.frombuffer(buf, dtype=np.float32).reshape(n_layers, hidden)
        for j, L in enumerate(layer_indices):
            out[i, j] = row[L]
        if (i + 1) % 4000 == 0:
            print(f"  ...{i + 1}/{n_items}", file=sys.stderr, flush=True)
    f.close()
    return out, layer_indices


def score_fold(Xl, rows, y, fam, fmt_bin, is_bench, is_casual, fold, rank):
    train_mask = fam != fold
    test_mask = fam == fold
    if rank > 0:
        _, B, _, _ = factorial_vectors(Xl, rows, train_mask)
        Q = orth_basis(B, rank)
    else:
        Q = None
    Xt = remove(Xl, Q)
    eff_rank = 0 if Q is None else Q.shape[1]

    out = {"eff_rank": eff_rank}
    s = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
    out["purpose"] = (test_mask.nonzero()[0], s)

    s = fit_probe(Xt[train_mask], fmt_bin[train_mask], Xt[test_mask])
    out["format"] = (test_mask.nonzero()[0], s)

    b2c_train, b2c_test = train_mask & is_bench, test_mask & is_casual
    if b2c_test.any():
        s = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        out["b2c"] = (b2c_test.nonzero()[0], s)
    c2b_train, c2b_test = train_mask & is_casual, test_mask & is_bench
    if c2b_test.any():
        s = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])
        out["c2b"] = (c2b_test.nonzero()[0], s)
    return fold, rank, out


def block_bootstrap_gap(y, s_purpose, s_b2c, s_c2b, blk, blocks, n_boot, rng):
    """Paired block bootstrap; see full320_v3_erasure_dose_response_bootstrap.py
    for the rationale (one sample_blocks draw per replicate feeds all three
    AUCs so the gap statistic's covariance structure is preserved)."""
    block_to_pos = {}
    for i, b in enumerate(blk):
        block_to_pos.setdefault(b, []).append(i)
    purpose_out = np.empty(n_boot)
    b2c_out = np.empty(n_boot)
    c2b_out = np.empty(n_boot)
    m_p, m_b, m_c = ~np.isnan(s_purpose), ~np.isnan(s_b2c), ~np.isnan(s_c2b)
    for r in range(n_boot):
        sample_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([block_to_pos[b] for b in sample_blocks if b in block_to_pos])
        idx_p = idx[m_p[idx]]
        idx_b = idx[m_b[idx]]
        idx_c = idx[m_c[idx]]
        purpose_out[r] = roc_auc_score(y[idx_p], s_purpose[idx_p])
        b2c_out[r] = roc_auc_score(y[idx_b], s_b2c[idx_b])
        c2b_out[r] = roc_auc_score(y[idx_c], s_c2b[idx_c])
    return purpose_out, b2c_out, c2b_out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "full320_v3_erasure_dose_response_extended_rank.json")
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    result = {
        "schema_version": 1, "mode": "development",
        "purpose": (
            "Extended-rank (0/1/4/16/64/128/256/384) block-bootstrapped (N_BOOT=300) "
            "dose-response for purpose AUC, cross-format transfer gap, AND format-readout "
            "AUC vs. matched-format-PCA projection rank, all 3 checkpoints, 3 pinned depths. "
            "Answers: does pushing rank past 64 keep shrinking the gap without damaging "
            "purpose AUC, and how close does format get to chance? Companion to "
            "results/full320_v3_erasure_dose_response_bootstrap.json (ranks 0-64 only, no format AUC)."
        ),
        "n_boot": N_BOOT, "ranks": list(RANKS), "probe": PROBE, "by_model": {},
    }

    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        depths = {"spike": cfg["spike_layer"], "min_gap": cfg["min_layer"], "final": cfg["final_layer"]}
        X, layer_list = stream_extract_layers(cfg["npz"], sorted(set(depths.values())))
        layer_pos = {L: j for j, L in enumerate(layer_list)}
        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        X = X[keep_mask].astype(np.float64)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        fmt_bin = (fmt == "benchmark").astype(int)
        blk = np.array([r["payload_block_id"] for r in kept_rows])
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"
        folds = sorted(set(fam))
        blocks = np.array(sorted(set(blk)))

        model_out = {"depth_role": {str(v): k for k, v in depths.items()}, "by_depth": {}}
        for depth, L in depths.items():
            Xl = np.ascontiguousarray(X[:, layer_pos[L], :])
            print(f"  --- {depth} (layer {L}) ---", file=sys.stderr, flush=True)

            jobs = [(rank, fold) for rank in RANKS for fold in folds]
            outs = Parallel(n_jobs=N_JOBS)(
                delayed(score_fold)(Xl, kept_rows, y, fam, fmt_bin, is_bench, is_casual, fold, rank)
                for rank, fold in jobs
            )
            by_rank = {r: {"purpose": [np.full(len(y), np.nan)],
                           "format": [np.full(len(y), np.nan)],
                           "b2c": [np.full(len(y), np.nan)],
                           "c2b": [np.full(len(y), np.nan)],
                           "eff_rank": []} for r in RANKS}
            for (fold, rank, fold_out) in outs:
                by_rank[rank]["eff_rank"].append(fold_out["eff_rank"])
                for regime in ("purpose", "format", "b2c", "c2b"):
                    if regime in fold_out:
                        idx, s = fold_out[regime]
                        by_rank[rank][regime][0][idx] = s

            depth_out = {}
            rng = np.random.default_rng(RNG_SEED)
            for rank in RANKS:
                s_purpose = by_rank[rank]["purpose"][0]
                s_format = by_rank[rank]["format"][0]
                s_b2c = by_rank[rank]["b2c"][0]
                s_c2b = by_rank[rank]["c2b"][0]
                m_p, m_f, m_b, m_c = ~np.isnan(s_purpose), ~np.isnan(s_format), ~np.isnan(s_b2c), ~np.isnan(s_c2b)
                purpose_auc = float(roc_auc_score(y[m_p], s_purpose[m_p]))
                format_auc = float(roc_auc_score(fmt_bin[m_f], s_format[m_f]))
                b2c_auc = float(roc_auc_score(y[m_b], s_b2c[m_b]))
                c2b_auc = float(roc_auc_score(y[m_c], s_c2b[m_c]))
                gap = purpose_auc - (b2c_auc + c2b_auc) / 2

                boot_purpose, boot_b2c, boot_c2b = block_bootstrap_gap(
                    y, s_purpose, s_b2c, s_c2b, blk, blocks, N_BOOT, rng)
                boot_gap = boot_purpose - (boot_b2c + boot_c2b) / 2

                def ci(a):
                    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

                depth_out[str(rank)] = {
                    "purpose_auc": purpose_auc, "purpose_auc_ci95": ci(boot_purpose),
                    "format_auc": format_auc,
                    "b2c_auc": b2c_auc, "c2b_auc": c2b_auc, "gap": gap, "gap_ci95": ci(boot_gap),
                    "mean_eff_rank_across_folds": float(np.mean(by_rank[rank]["eff_rank"])),
                }
                print(f"    rank={rank} (eff={depth_out[str(rank)]['mean_eff_rank_across_folds']:.0f}): "
                      f"purpose={purpose_auc:.4f} CI={ci(boot_purpose)} format={format_auc:.4f} "
                      f"gap={gap:.4f} CI={ci(boot_gap)}", file=sys.stderr, flush=True)
            model_out["by_depth"][depth] = depth_out
        result["by_model"][model] = model_out
        del X

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
