"""Full-layer-depth matched-format-PCA reliability sweep: same computation as
scripts/full320_v3_erasure_full_layer_sweep.py (rank=64, 5-fold cross-fitted,
matched-format subspace removed, purpose probe retrained on the residual),
extended to answer "does removing the format subspace make the purpose probe
more reliable, not just higher-AUC" (RULES.md: never treat AUC and
balanced/threshold accuracy as interchangeable).

Adds, for the joint (purpose), b2c, and c2b regimes, at every layer, on the
post-erasure residual:

  - bal_acc:        balanced_accuracy_score(y, s >= 0) -- threshold reliability,
                     not just ranking.
  - paired_pos:      fraction of matched (payload_block_id, format) eval/deploy
                      pairs correctly ranked eval > deploy -- same convention as
                      scripts/sweep_full.py's paired() used for the raw sweep.
  - paired_mean:      mean signed score difference for those pairs.
  - cells:            per (intended_purpose, format) predicted-positive rate.

This is the exact same probe/threshold/pairing convention already used for
the RAW sweep (scripts/sweep_full.py), so raw-vs-projected balanced accuracy
and paired-consistency are apples to apples. Does not overwrite the original
AUC-only sweep; writes a new, separate output file.

Usage: python scripts/full320_v3_erasure_full_layer_sweep_reliability.py [out_json]
"""
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import factorial_vectors, fit_probe, orth_basis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
    "llama31_70b": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
    "llama33_70b": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
}
RANK = 64
N_JOBS = int(os.environ.get("N_JOBS", "28"))


def remove(X, Q):
    return X if Q is None or Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def paired(scores, blk, fmt, pur):
    d = {}
    for i in range(len(scores)):
        if np.isnan(scores[i]):
            continue
        d.setdefault((blk[i], fmt[i]), {})[pur[i]] = scores[i]
    v = [x["evaluation"] - x["deployment"] for x in d.values() if len(x) == 2]
    return (float(np.mean(np.array(v) > 0)), float(np.mean(v))) if v else (float("nan"), float("nan"))


def cellrates(pred, mask, pur, fmt):
    out = {}
    for pu in ("evaluation", "deployment"):
        for ft in ("benchmark", "casual"):
            c = mask & (pur == pu) & (fmt == ft)
            if c.sum():
                out[f"{pu}::{ft}"] = float(pred[c].mean() * 100)
    return out


def bundle(s, y, blk, fmt, pur, valid):
    m = valid & ~np.isnan(s)
    p = (s >= 0).astype(int)
    pp, pm = paired(np.where(valid, s, np.nan), blk, fmt, pur)
    return {
        "auc": float(roc_auc_score(y[m], s[m])),
        "bal_acc": float(balanced_accuracy_score(y[m], p[m])),
        "paired_pos": pp,
        "paired_mean": pm,
        "cells": cellrates(p, m, pur, fmt),
    }


def score_layer(Xl, rows, y, fam, blk, fmt, pur, is_bench, is_casual, rank):
    """5-fold cross-fitted matched_pca erasure + retrain at one layer. Returns
    the joint/b2c/c2b reliability bundle (auc, bal_acc, paired stats, cells)
    on the post-erasure residual, plus the sanity-check format AUC."""
    folds = sorted(set(fam))
    s_purpose = np.full(len(y), np.nan)
    s_format = np.full(len(y), np.nan)
    s_b2c = np.full(len(y), np.nan)
    s_c2b = np.full(len(y), np.nan)
    fmt_bin = is_bench.astype(int)
    for fold in folds:
        train_mask = fam != fold
        test_mask = fam == fold
        _, B, _, _ = factorial_vectors(Xl, rows, train_mask)
        Q = orth_basis(B, rank)
        Xt = remove(Xl, Q)

        s_purpose[test_mask] = fit_probe(Xt[train_mask], y[train_mask], Xt[test_mask])
        s_format[test_mask] = fit_probe(Xt[train_mask], fmt_bin[train_mask], Xt[test_mask])
        b2c_train, b2c_test = train_mask & is_bench, test_mask & is_casual
        if b2c_test.any():
            s_b2c[b2c_test] = fit_probe(Xt[b2c_train], y[b2c_train], Xt[b2c_test])
        c2b_train, c2b_test = train_mask & is_casual, test_mask & is_bench
        if c2b_test.any():
            s_c2b[c2b_test] = fit_probe(Xt[c2b_train], y[c2b_train], Xt[c2b_test])

    m_f = ~np.isnan(s_format)
    valid_all = np.ones(len(y), dtype=bool)
    purpose_bundle = bundle(s_purpose, y, blk, fmt, pur, valid_all)
    b2c_bundle = bundle(s_b2c, y, blk, fmt, pur, valid_all)
    c2b_bundle = bundle(s_c2b, y, blk, fmt, pur, valid_all)
    format_auc = float(roc_auc_score(fmt_bin[m_f], s_format[m_f]))
    gap = purpose_bundle["auc"] - (b2c_bundle["auc"] + c2b_bundle["auc"]) / 2
    return {
        "purpose_after_erasure": purpose_bundle,
        "b2c_after_erasure": b2c_bundle,
        "c2b_after_erasure": c2b_bundle,
        "format_auc_after_erasure": format_auc,
        "gap_after_erasure": gap,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "full320_v3_erasure_full_layer_sweep_reliability.json")
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    print(f"stated rows: {len(rows_by_id)} / {len(rows_all)}", file=sys.stderr)

    result = {
        "schema_version": 1, "mode": "development",
        "purpose": (
            "Reliability-extended companion to results/full320_v3_erasure_full_layer_sweep.json: "
            "same rank-64 matched-format-PCA erasure, same 5-fold cross-fit, same probe, but also "
            "reports balanced accuracy and matched-pair (payload_block_id x format, eval-vs-deploy) "
            "consistency on the post-erasure residual, using the exact convention of scripts/sweep_full.py "
            "(the source of the raw results/layer_sweep_full320-v3_<model>.json bal_acc/paired_pos/paired_mean "
            "fields), so raw-vs-projected reliability is apples to apples. Raw baseline is NOT recomputed "
            "here; merge with results/layer_sweep_full320-v3_<model>.json at analysis time."
        ),
        "rank": RANK, "by_model": {},
    }

    for model, npz_path in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        z = zipfile.ZipFile(npz_path)
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        print("  loading full tensor...", file=sys.stderr, flush=True)
        X = np.load(npz_path)["X"][keep_mask]  # (n_kept, n_layers, hidden) float32
        n_layers = X.shape[1]
        print(f"  loaded {X.shape}", file=sys.stderr, flush=True)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        pur = np.array([r["intended_purpose"] for r in kept_rows])
        blk = np.array([r["payload_block_id"] for r in kept_rows])
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"

        outs = Parallel(n_jobs=N_JOBS)(
            delayed(score_layer)(np.ascontiguousarray(X[:, L, :]), kept_rows, y, fam, blk, fmt, pur,
                                  is_bench, is_casual, RANK)
            for L in range(n_layers)
        )
        by_layer = {str(L): outs[L] for L in range(n_layers)}
        result["by_model"][model] = {"n_layers": n_layers, "by_layer": by_layer}
        for L in range(n_layers):
            r = outs[L]
            print(f"  layer {L}: purpose_auc={r['purpose_after_erasure']['auc']:.4f} "
                  f"purpose_bal_acc={r['purpose_after_erasure']['bal_acc']:.4f} "
                  f"gap={r['gap_after_erasure']:.4f} format(sanity)={r['format_auc_after_erasure']:.4f}",
                  file=sys.stderr, flush=True)
        del X

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
