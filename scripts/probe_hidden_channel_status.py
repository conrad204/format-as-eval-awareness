"""Cross-fitted probes on hidden-channel activations.

Three read-outs per layer, all with leave-episode-family-out cross-fitting and the
repository's standard probe (StandardScaler + LogisticRegression(C=0.1)):

* status_latent  - closed vs active, pooled over both mappings, so the visible
                   channel letter is orthogonal to the label by construction;
* token_control  - K vs M letter identity (positive control, should be high);
* status_scrambled - the same status contrast on scrambled prompts, where the
                   letter carries no status (null control, should be chance).

Uncertainty resamples held-out episode families.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_of(families: list[str]) -> dict[str, int]:
    ordered = sorted(families)
    return {f: i % N_FOLDS for i, f in enumerate(ordered)}


def oof_scores(X: np.ndarray, y: np.ndarray, fam: np.ndarray, folds: dict[str, int]) -> np.ndarray:
    scores = np.full(len(y), np.nan)
    assign = np.array([folds[f] for f in fam])
    for fold in range(N_FOLDS):
        train, test = assign != fold, assign == fold
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        scaler = StandardScaler().fit(X[train])
        model = LogisticRegression(**PROBE).fit(scaler.transform(X[train]), y[train])
        scores[test] = model.decision_function(scaler.transform(X[test]))
    return scores


def family_bootstrap(auc_by_family: dict[str, float], rng: np.random.Generator, n: int) -> list[float]:
    keys = sorted(auc_by_family)
    values = np.array([auc_by_family[k] for k in keys])
    draws = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260907)
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
    folds = fold_of(sorted(set(fam)))
    rng = np.random.default_rng(args.seed)

    coherent = np.isin(cond, ["map0_K", "map0_M", "map1_K", "map1_M"])
    scrambled = np.isin(cond, ["scrambled_K", "scrambled_M"])
    tasks = {
        "status_latent": (coherent, (status[coherent] == "closed").astype(int)),
        "token_control": (coherent, (token[coherent] == "K").astype(int)),
        "status_scrambled": (scrambled, np.array([1 if c == "scrambled_K" else 0 for c in cond[scrambled]])),
    }
    layers = []
    for layer in range(X.shape[1]):
        record = {"layer": layer}
        for name, (mask, y) in tasks.items():
            scores = oof_scores(X[mask, layer, :], y, fam[mask], folds)
            valid = ~np.isnan(scores)
            record[f"{name}_auc"] = float(roc_auc_score(y[valid], scores[valid]))
            per_family = {}
            for family in sorted(set(fam[mask])):
                sel = valid & (fam[mask] == family)
                if len(np.unique(y[sel])) == 2:
                    per_family[family] = float(roc_auc_score(y[sel], scores[sel]))
            record[f"{name}_family_mean_auc"] = float(np.mean(list(per_family.values())))
            record[f"{name}_family_ci95"] = family_bootstrap(per_family, rng, args.n_bootstrap)
        layers.append(record)
        print(f"layer {layer}: latent {record['status_latent_auc']:.3f} "
              f"token {record['token_control_auc']:.3f} "
              f"scrambled {record['status_scrambled_auc']:.3f}", flush=True)

    best = max(layers, key=lambda r: r["status_latent_family_mean_auc"])
    result = {
        "schema_version": 1, "model_key": meta["model_key"], "precision": meta.get("precision"),
        "activations_sha256": meta["activations_sha256"], "n_layers": X.shape[1],
        "cross_fitting": f"leave-episode-family-out, {N_FOLDS} folds",
        "probe": PROBE, "n_bootstrap": args.n_bootstrap,
        "note": ("status label is orthogonal to the visible channel letter because both mappings are pooled; "
                 "a letter-only probe scores 0.5"),
        "layers": layers,
        "best_latent_layer": best["layer"],
        "best_latent_family_mean_auc": best["status_latent_family_mean_auc"],
        "best_latent_family_ci95": best["status_latent_family_ci95"],
        "max_token_control_auc": max(r["token_control_auc"] for r in layers),
        "max_scrambled_family_mean_auc": max(r["status_scrambled_family_mean_auc"] for r in layers),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
