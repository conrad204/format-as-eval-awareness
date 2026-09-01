"""Selection-free estimate of latent-status decodability.

Two earlier controls were wrong and are superseded here:

* v1's "scrambled null" labelled rows by channel letter, which is trivially
  decodable (0.967), so it could only return ~1.0.
* v2's cross-mapping transfer trained on a single mapping. Within one mapping the
  status and the channel letter are perfectly confounded, so such a probe can
  only learn the letter and must invert on the reversed mapping. Its 0.316 is a
  guaranteed artifact and says nothing about whether the pooled probe reads
  status.

What survives as a valid design: pool both mappings, so the letter is orthogonal
to the label and a letter-only feature scores 0.5; cross-fit by episode family;
and null out with a permutation that shuffles status inside each
(family, mapping) block, preserving letter and mapping balance exactly.

The remaining freedom was layer choice. This script removes it with nested
selection: for each held-out family fold the layer is chosen using only the
training folds, then applied once to the held-out fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROBE = dict(C=0.1, max_iter=3000, random_state=0)
N_FOLDS = 4
COHERENT = ("map0_K", "map0_M", "map1_K", "map1_M")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_score(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(**PROBE).fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xte))


def nested_oof(X, y, fam, folds, layers_to_try):
    """Return out-of-fold scores where the layer is chosen inside the training folds."""
    scores = np.full(len(y), np.nan)
    chosen = {}
    assign = np.array([folds[f] for f in fam])
    for fold in range(N_FOLDS):
        outer_tr, outer_te = assign != fold, assign == fold
        inner_fams = sorted(set(fam[outer_tr]))
        inner_folds = {f: i % (N_FOLDS - 1) for i, f in enumerate(inner_fams)}
        inner_assign = np.array([inner_folds.get(f, -1) for f in fam])
        best_layer, best_auc = layers_to_try[0], -1.0
        for layer in layers_to_try:
            inner_scores = np.full(len(y), np.nan)
            for inner in range(N_FOLDS - 1):
                itr = outer_tr & (inner_assign != inner)
                ite = outer_tr & (inner_assign == inner)
                if ite.sum() == 0 or len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
                    continue
                inner_scores[ite] = fit_score(X[itr, layer, :], y[itr], X[ite, layer, :])
            ok = ~np.isnan(inner_scores)
            if ok.sum() and len(np.unique(y[ok])) == 2:
                auc = roc_auc_score(y[ok], inner_scores[ok])
                if auc > best_auc:
                    best_auc, best_layer = auc, layer
        chosen[fold] = {"layer": int(best_layer), "inner_auc": float(best_auc)}
        scores[outer_te] = fit_score(X[outer_tr, best_layer, :], y[outer_tr], X[outer_te, best_layer, :])
    return scores, chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if sha256(args.activations) != meta["activations_sha256"]:
        raise RuntimeError("activation hash mismatch")
    with np.load(args.activations, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        fam = z["episode_family"].astype(str)
        cond = z["condition"].astype(str)
        token = z["query_token"].astype(str)
        status = z["query_status"].astype(str)

    coh = np.isin(cond, COHERENT)
    Xc, famc, condc = X[coh], fam[coh], cond[coh]
    y = (status[coh] == "closed").astype(int)
    y_letter = (token[coh] == "K").astype(int)
    families = sorted(set(famc))
    folds = {f: i % N_FOLDS for i, f in enumerate(families)}
    rng = np.random.default_rng(args.seed)

    map0 = np.char.startswith(condc, "map0")
    y_perm = y.copy()
    for f in families:
        for sel in (map0, ~map0):
            idx = np.where((famc == f) & sel)[0]
            y_perm[idx] = rng.permutation(y[idx])

    layers_to_try = list(range(0, X.shape[1], args.layer_stride))
    out = {}
    for name, labels in (("status", y), ("permutation_null", y_perm), ("letter_control", y_letter)):
        scores, chosen = nested_oof(Xc, labels, famc, folds, layers_to_try)
        ok = ~np.isnan(scores)
        auc = float(roc_auc_score(labels[ok], scores[ok]))
        per_family = {}
        for f in families:
            sel = ok & (famc == f)
            if len(np.unique(labels[sel])) == 2:
                per_family[f] = float(roc_auc_score(labels[sel], scores[sel]))
        vals = np.array([per_family[k] for k in sorted(per_family)])
        draws = vals[rng.integers(0, len(vals), (args.n_bootstrap, len(vals)))].mean(axis=1)
        out[name] = {
            "pooled_auc": auc,
            "family_mean_auc": float(vals.mean()),
            "family_ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "families_above_chance": int((vals > 0.5).sum()),
            "n_families": len(vals),
            "layer_chosen_per_fold": chosen,
        }
        print(f"{name}: pooled {auc:.4f} family-mean {vals.mean():.4f} "
              f"CI {[round(float(x), 4) for x in np.quantile(draws, [0.025, 0.975])]} "
              f"families>0.5 {int((vals > 0.5).sum())}/{len(vals)}", flush=True)

    result = {
        "schema_version": 3, "model_key": meta["model_key"], "precision": meta.get("precision"),
        "activations_sha256": meta["activations_sha256"],
        "design": ("pooled over both mappings so the channel letter is orthogonal to status; "
                   "leave-episode-family-out cross-fitting; layer chosen inside training folds only"),
        "supersedes": ["probe_hidden_channel_status.py (invalid scrambled control)",
                       "probe_hidden_channel_status_v2.py (cross-mapping transfer is confounded by design)"],
        "layer_stride": args.layer_stride, "n_bootstrap": args.n_bootstrap,
        "readouts": out,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
