"""Cross-mapping probe test for latent status, with a valid null.

The first probe pass reported a broken control: its "scrambled null" labelled rows
by which channel letter the query used, and letter identity is trivially
decodable (AUC 0.967), so that control could only ever return ~1.0. It carried no
information. This version replaces it with the two controls that actually
discriminate.

* status_pooled - closed vs active over both mappings. Letter identity is
  orthogonal to the label by construction, so a letter-only feature scores 0.5.
* cross_mapping - train on one mapping, test on the reversed mapping. A direction
  encoding status transfers (AUC > 0.5). A direction encoding the letter inverts
  (AUC < 0.5). This is the decisive test.
* permutation_null - status labels permuted within episode family, preserving
  letter and mapping balance. Expected 0.5.
* letter_control - decode the letter itself, as a positive control that the
  extraction and probe pipeline work at all.
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


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(**PROBE).fit(scaler.transform(Xtr), ytr)
    return model.decision_function(scaler.transform(Xte))


def crossfit_auc(X: np.ndarray, y: np.ndarray, fam: np.ndarray, folds: dict[str, int]) -> float:
    scores = np.full(len(y), np.nan)
    assign = np.array([folds[f] for f in fam])
    for fold in range(N_FOLDS):
        tr, te = assign != fold, assign == fold
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scores[te] = fit_predict(X[tr], y[tr], X[te])
    ok = ~np.isnan(scores)
    return float(roc_auc_score(y[ok], scores[ok]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260909)
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

    families = sorted(set(fam))
    folds = {f: i % N_FOLDS for i, f in enumerate(families)}
    rng = np.random.default_rng(args.seed)

    coh = np.isin(cond, COHERENT)
    y_status = (status[coh] == "closed").astype(int)
    y_token = (token[coh] == "K").astype(int)
    fam_coh, cond_coh = fam[coh], cond[coh]
    map0 = np.char.startswith(cond_coh, "map0")
    map1 = np.char.startswith(cond_coh, "map1")

    # permutation null: shuffle status within each (family, mapping) block, which
    # preserves letter and mapping balance exactly
    y_perm = y_status.copy()
    for f in families:
        for sel in (map0, map1):
            idx = np.where((fam_coh == f) & sel)[0]
            y_perm[idx] = rng.permutation(y_status[idx])

    layers = []
    for layer in range(X.shape[1]):
        Xl = X[coh, layer, :]
        record = {
            "layer": layer,
            "status_pooled_auc": crossfit_auc(Xl, y_status, fam_coh, folds),
            "letter_control_auc": crossfit_auc(Xl, y_token, fam_coh, folds),
            "permutation_null_auc": crossfit_auc(Xl, y_perm, fam_coh, folds),
        }
        # decisive: train one mapping, test the reversed mapping (held-out families too)
        transfers = []
        for tr_mask, te_mask, name in ((map0, map1, "map0_to_map1"), (map1, map0, "map1_to_map0")):
            scores = np.full(te_mask.sum(), np.nan)
            te_fam = fam_coh[te_mask]
            for fold in range(N_FOLDS):
                tr = tr_mask & np.array([folds[f] != fold for f in fam_coh])
                te = np.array([folds[f] == fold for f in te_fam])
                if len(np.unique(y_status[tr])) < 2 or te.sum() == 0:
                    continue
                scores[te] = fit_predict(Xl[tr], y_status[tr], Xl[te_mask][te])
            ok = ~np.isnan(scores)
            auc = float(roc_auc_score(y_status[te_mask][ok], scores[ok]))
            record[f"cross_mapping_{name}_auc"] = auc
            transfers.append(auc)
        record["cross_mapping_mean_auc"] = float(np.mean(transfers))
        layers.append(record)
        print(f"layer {layer:2d} status {record['status_pooled_auc']:.3f} "
              f"cross-mapping {record['cross_mapping_mean_auc']:.3f} "
              f"letter {record['letter_control_auc']:.3f} "
              f"perm-null {record['permutation_null_auc']:.3f}", flush=True)

    best = max(layers, key=lambda r: r["cross_mapping_mean_auc"])
    result = {
        "schema_version": 2, "model_key": meta["model_key"], "precision": meta.get("precision"),
        "activations_sha256": meta["activations_sha256"], "n_layers": X.shape[1],
        "cross_fitting": f"leave-episode-family-out, {N_FOLDS} folds", "probe": PROBE,
        "supersedes": "probe_hidden_channel_status.py, whose scrambled control labelled rows by channel letter and was therefore uninformative",
        "layers": layers,
        "best_cross_mapping_layer": best["layer"],
        "best_cross_mapping_auc": best["cross_mapping_mean_auc"],
        "best_cross_mapping_detail": {k: v for k, v in best.items() if k != "layer"},
        "max_status_pooled_auc": max(r["status_pooled_auc"] for r in layers),
        "max_letter_control_auc": max(r["letter_control_auc"] for r in layers),
        "max_permutation_null_auc": max(r["permutation_null_auc"] for r in layers),
        "mean_permutation_null_auc": float(np.mean([r["permutation_null_auc"] for r in layers])),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
