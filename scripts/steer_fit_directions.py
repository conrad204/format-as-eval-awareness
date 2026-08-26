"""Fit steering directions at one layer from the checked-in v3 activations.

The directions are the published probes. For each regime the five held-fold
probes (StandardScaler + LogisticRegression(C=0.1, max_iter=3000,
random_state=0) on folds fg1-fg5, regime train/test masks copied from
scripts/nocue_sweep.py) are refit at the requested layer, and a regression
check compares their out-of-fold decision scores with the checked-in
results/nocue_scores/<model>.scores.npz rows for that layer. The format probe
reproduces to ~4e-6; the near-separable purpose probes do not reproduce
bit-for-bit across machines (lbfgs on a flat objective; OpenBLAS kernels
differ by CPU), so the check requires Pearson r >= 0.999 between refit and
checked-in scores and |AUC difference| <= 1e-3, and records the deviations.
Only then is each probe un-standardized into raw residual-stream space:

    decision(x) = w . ((x - mu) / sigma) + b = (w / sigma) . x + const

so w_raw = w / sigma is the direction the probe actually reads off the residual
stream. Each fold's w_raw is unit-normalized, the five are averaged and the
mean re-normalized; fold-to-fold cosines are recorded. Sign convention: the
positive class is benchmark (format) / evaluation (purpose), so +dose pushes
toward benchmark / evaluation.

Regimes fitted here:
  format        benchmark vs casual, stated stratum        (direction "fmt")
  decorrelated  evaluation vs deployment, both formats     ("pur_dec")
  confounded    evaluation::benchmark vs deployment::casual ("pur_conf")
Extra directions for comparison and control:
  fmt_dm, pur_dm   difference-of-class-means on the stated stratum
  rand0..rand2     seeded isotropic Gaussian unit vectors (equal-norm controls)

For every direction the script records projections x . v of all 11,520 items:
class means on the stated stratum and on the no-cue stratum, the training-
contrast gap Delta (the natural dose unit), the no-cue mean (the mean-ablation
target) and sd, and the mean residual norm at this layer for scale.

Sources (item order is asserted against the frozen jsonl):
  all-layers npz  {X (n, L, d) fp32, item_ids}       + --layer
  layer slice     {layer, item_ids, X_fp32}          (scripts/npz_stream_slice.py)
  405B shard      {layer, item_ids, X_bf16_u16}      (Conradf/v3-llama-3.1-405B-instruct)

Usage:
  steer_fit_directions.py <items.jsonl> <source.npz> <layer> <reference.scores.npz> <out_prefix>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Same single-threaded BLAS as scripts/nocue_sweep.py: the near-separable
# purpose probes are sensitive to summation order, and the regression check
# below demands the checked-in decision scores be reproduced.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
N_RANDOM = 5
R_MIN = 0.999
AUC_TOL = 1e-3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_layer(source: Path, layer: int, ids_expected: list[str]) -> np.ndarray:
    d = np.load(source)
    assert [str(x) for x in d["item_ids"]] == ids_expected, "item_ids do not match items jsonl order"
    if "X" in d.files:
        X = d["X"][:, layer, :].astype(np.float32)
    elif "X_fp32" in d.files:
        assert int(d["layer"]) == layer, f"slice is layer {int(d['layer'])}, wanted {layer}"
        X = d["X_fp32"].astype(np.float32)
    elif "X_bf16_u16" in d.files:
        assert int(d["layer"]) == layer, f"shard is layer {int(d['layer'])}, wanted {layer}"
        X = (d["X_bf16_u16"].astype(np.uint32) << 16).view(np.float32)
    else:
        sys.exit(f"unrecognized activation file keys: {d.files}")
    assert X.ndim == 2 and X.shape[0] == len(ids_expected)
    return np.ascontiguousarray(X)


def fit_fold(X, train, test, y, extra):
    """Same fit as nocue_sweep._fit_scores, additionally returning the model."""
    sc = StandardScaler().fit(X[train])
    model = LogisticRegression(**PROBE).fit(sc.transform(X[train]), y[train])
    return (
        model.decision_function(sc.transform(X[test])),
        model.decision_function(sc.transform(X[extra])),
        model,
        sc,
    )


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    items_path, source, layer, ref_path, out_prefix = (
        Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    ids = [r["item_id"] for r in rows]
    stated = np.array([r["render_slot"] == "stated" for r in rows])
    nocue = np.array([r["render_slot"] == "no_cue" for r in rows])
    label = np.array([-1 if r["label"] is None else int(r["label"]) for r in rows])
    fam = np.array([r["purpose_family_id"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    is_bench = fmt == "benchmark"
    fmt_label = is_bench.astype(int)
    assert stated.sum() + nocue.sum() == len(rows)

    X = load_layer(source, layer, ids)
    n, d = X.shape
    print(f"layer {layer}: X {X.shape} from {source.name}", flush=True)

    ref = np.load(ref_path)
    assert [str(x) for x in ref["item_ids"]] == ids
    nocue_ids_ref = [str(x) for x in ref["nocue_item_ids"]]
    assert nocue_ids_ref == [i for i, m in zip(ids, nocue) if m]

    train_of = {
        "format": stated,
        "decorrelated": stated,
        "confounded": stated & ((label == 1) & is_bench | (label == 0) & ~is_bench),
    }
    test_of = {"format": stated, "decorrelated": stated, "confounded": stated}
    y_of = {"format": fmt_label, "decorrelated": label, "confounded": label}
    names = {"format": "fmt", "decorrelated": "pur_dec", "confounded": "pur_conf"}

    directions: dict[str, np.ndarray] = {}
    info: dict = {"layer": layer, "hidden": int(d), "n_items": int(n),
                  "source": str(source), "source_sha256": sha256_file(source),
                  "items_sha256": sha256_file(items_path),
                  "reference_scores": str(ref_path), "probe": PROBE, "folds": FOLDS,
                  "regression_check": {}, "fold_cosines": {}, "fold_probe_auc": {}}
    for reg, name in names.items():
        oof = np.full(n, np.nan)
        nc_folds, w_folds, fold_auc = [], [], []
        for fold in FOLDS:
            train = train_of[reg] & (fam != fold)
            test = test_of[reg] & (fam == fold)
            s_test, s_nc, model, sc = fit_fold(X, train, test, y_of[reg], nocue)
            oof[test] = s_test
            nc_folds.append(s_nc)
            w_raw = model.coef_[0] / sc.scale_
            w_folds.append(unit(w_raw))
            fold_auc.append(float(roc_auc_score(y_of[reg][test], s_test)))
        nc = np.mean(nc_folds, axis=0)
        m = ~np.isnan(oof)
        ref_oof = ref[f"stated_oof::{reg}"][layer].astype(np.float64)
        ref_nc = ref[f"nocue::{reg}"][layer].astype(np.float64)
        dev_oof = float(np.nanmax(np.abs(oof[m] - ref_oof[m])))
        dev_nc = float(np.max(np.abs(nc - ref_nc)))
        assert np.isnan(ref_oof[~m]).all()
        auc = float(roc_auc_score(y_of[reg][m], oof[m]))
        auc_ref = float(roc_auc_score(y_of[reg][m], ref_oof[m]))
        r_oof = float(np.corrcoef(oof[m], ref_oof[m])[0, 1])
        r_nc = float(np.corrcoef(nc, ref_nc)[0, 1])
        info["regression_check"][reg] = {
            "max_abs_dev_stated_oof": dev_oof, "max_abs_dev_nocue": dev_nc,
            "sd_stated_oof_ref": float(np.std(ref_oof[m])), "pearson_r_stated_oof": r_oof,
            "pearson_r_nocue": r_nc, "oof_auc": auc, "oof_auc_checked_in": auc_ref}
        print(f"  {reg:12s} OOF auc {auc:.4f} (checked-in {auc_ref:.4f})  r={r_oof:.6f}/{r_nc:.6f}  "
              f"max|dev| stated {dev_oof:.2e} nocue {dev_nc:.2e} (score sd {np.std(ref_oof[m]):.2f})", flush=True)
        if min(r_oof, r_nc) < R_MIN or abs(auc - auc_ref) > AUC_TOL:
            sys.exit(f"REGRESSION CHECK FAILED for {reg}: r={r_oof:.5f}/{r_nc:.5f}, dAUC={auc - auc_ref:.2e}")
        W = np.stack(w_folds)
        C = W @ W.T
        info["fold_cosines"][name] = {"min": float(C[np.triu_indices(5, 1)].min()),
                                      "mean": float(C[np.triu_indices(5, 1)].mean())}
        info["fold_probe_auc"][name] = fold_auc
        directions[name] = unit(W.mean(axis=0))

    # difference-of-means directions on the stated stratum
    directions["fmt_dm"] = unit(X[stated & is_bench].mean(0) - X[stated & ~is_bench].mean(0))
    directions["pur_dm"] = unit(X[stated & (label == 1)].mean(0) - X[stated & (label == 0)].mean(0))
    for k in range(N_RANDOM):
        directions[f"rand{k}"] = unit(np.random.default_rng(1000 + k).standard_normal(d))

    keys = list(directions)
    info["cosines"] = {a: {b: cos(directions[a], directions[b]) for b in keys} for a in keys}
    norms = np.linalg.norm(X, axis=1)
    info["residual_norm"] = {"mean_all": float(norms.mean()), "mean_nocue": float(norms[nocue].mean()),
                             "sd_nocue": float(norms[nocue].std(ddof=1))}

    cells = {
        "stated_eval_bench": stated & (label == 1) & is_bench,
        "stated_eval_casual": stated & (label == 1) & ~is_bench,
        "stated_deploy_bench": stated & (label == 0) & is_bench,
        "stated_deploy_casual": stated & (label == 0) & ~is_bench,
        "stated_bench": stated & is_bench, "stated_casual": stated & ~is_bench,
        "stated_eval": stated & (label == 1), "stated_deploy": stated & (label == 0),
        "nocue_bench": nocue & is_bench, "nocue_casual": nocue & ~is_bench, "nocue": nocue,
    }
    proj_all = {}
    info["directions"] = {}
    for name, v in directions.items():
        p = X.astype(np.float64) @ v.astype(np.float64)
        proj_all[name] = p.astype(np.float32)
        means = {c: float(p[mk].mean()) for c, mk in cells.items()}
        st = {
            "cell_means": means,
            "sd_stated": float(p[stated].std(ddof=1)),
            "sd_nocue": float(p[nocue].std(ddof=1)),
            "gap_format_stated": means["stated_bench"] - means["stated_casual"],
            "gap_format_nocue": means["nocue_bench"] - means["nocue_casual"],
            "gap_purpose_stated": means["stated_eval"] - means["stated_deploy"],
            "gap_confounded_stated": means["stated_eval_bench"] - means["stated_deploy_casual"],
            "nocue_mean": means["nocue"],
            "auc_format_stated": float(roc_auc_score(fmt_label[stated], p[stated])),
            "auc_purpose_stated": float(roc_auc_score(label[stated], p[stated])),
            "auc_format_nocue": float(roc_auc_score(fmt_label[nocue], p[nocue])),
        }
        contrast = {"fmt": "gap_format_stated", "fmt_dm": "gap_format_stated",
                    "pur_dec": "gap_purpose_stated", "pur_dm": "gap_purpose_stated",
                    "pur_conf": "gap_confounded_stated"}.get(name)
        st["delta_contrast"] = contrast
        st["delta"] = st[contrast] if contrast else None
        info["directions"][name] = st
        print(f"  {name:9s} delta={st['delta'] if st['delta'] is None else round(st['delta'], 3)} "
              f"fmt_gap_nocue={st['gap_format_nocue']:+.3f} sd_nocue={st['sd_nocue']:.3f} "
              f"auc_fmt={st['auc_format_stated']:.3f} auc_pur={st['auc_purpose_stated']:.3f}", flush=True)

    out = Path(out_prefix)
    np.savez(out.with_suffix(".npz"), item_ids=np.array(ids), layer=np.int16(layer),
             **{f"v::{k}": v for k, v in directions.items()},
             **{f"proj::{k}": p for k, p in proj_all.items()})
    out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.npz')} and {out.with_suffix('.json')}")
    print("FIT DONE")


if __name__ == "__main__":
    main()
