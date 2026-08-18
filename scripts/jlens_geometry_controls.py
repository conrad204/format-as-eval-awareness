"""Development-only J-lens energy, random-subspace, and gain controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
ENERGY_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995)
RANDOM_THRESHOLDS = (0.70, 0.80, 0.90, 0.95, 0.99)
RANDOM_SEEDS = tuple(range(32))
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_direction(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
    return model.coef_[0] / scaler.scale_, scaler


def auc(X, y, direction, scaler):
    if not np.linalg.norm(direction):
        return None
    return float(roc_auc_score(y, (X - scaler.mean_) @ direction))


def projection(direction, V, k):
    q = V[:, :k]
    return q @ (q.T @ direction)


def fwht(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    h = 1
    while h < y.size:
        for start in range(0, y.size, h * 2):
            a = y[start : start + h].copy()
            b = y[start + h : start + 2 * h].copy()
            y[start : start + h] = a + b
            y[start + h : start + 2 * h] = a - b
        h *= 2
    return y


def random_projection(direction, k, seed):
    """Structured Haar-like random subspace via signed/permuted Hadamard."""
    d = direction.shape[0]
    if d & (d - 1):
        raise ValueError("fast random-subspace control requires power-of-two dimension")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    permutation = rng.permutation(d)
    coeff = fwht(direction[permutation] * signs) / np.sqrt(d)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(d)
    m = min(k, d - k)
    if m == 0:
        return direction.copy(), np.zeros_like(direction)
    kept = np.zeros_like(coeff)
    kept[:m] = coeff[:m]
    transformed = fwht(kept) / np.sqrt(d)
    low = np.empty_like(direction)
    low[permutation] = transformed * signs
    if k <= d - k:
        return low, direction - low
    return direction - low, low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("lens", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 9, 12, 15, 20, 24])
    args = parser.parse_args()
    import torch

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    z = np.load(args.activations)
    if [str(x) for x in z["item_ids"]] != [r["item_id"] for r in rows]:
        raise ValueError("activation item_ids do not match metadata")
    X_all = np.asarray(z["X"], dtype=np.float64)
    if np.any(~np.isfinite(X_all)):
        raise ValueError("nonfinite activations")
    stated = np.asarray([r["render_slot"] == "stated" for r in rows])
    no_cue = np.asarray([r["render_slot"] == "no_cue" for r in rows])
    purpose_y = np.asarray([int(r["intended_purpose"] == "evaluation") for r in rows])
    format_y = np.asarray([int(r["format"] == "benchmark") for r in rows])
    family = np.asarray([r["purpose_family_id"] for r in rows])
    lens_data = torch.load(args.lens, map_location="cpu", weights_only=False)
    J = {int(k): np.asarray(v, dtype=np.float64) for k, v in lens_data["J"].items()}
    report_layers = []
    for layer in args.layers:
        X = X_all[:, layer, :]
        U, s, Vh = np.linalg.svd(J[layer], full_matrices=False)
        del U
        V = Vh.T
        energy = np.cumsum(s * s) / np.sum(s * s)
        d = X.shape[1]
        thresholds = {}
        for threshold in ENERGY_THRESHOLDS:
            k = int(np.searchsorted(energy, threshold) + 1)
            thresholds[str(threshold)] = {"rank": k, "dimension_fraction": k / d, "captured_frobenius_energy": float(energy[k - 1]), "folds": {}}
        folds = {}
        for fold in FOLDS:
            train = stated & (family != fold)
            test = stated & (family == fold)
            direction, scaler = fit_direction(X[train], purpose_y[train])
            norm = np.linalg.norm(direction)
            fold_out = {"n_train": int(train.sum()), "n_test": int(test.sum()), "full_auc": auc(X[test], purpose_y[test], direction, scaler), "purpose_gain": float(np.linalg.norm(J[layer] @ direction) / norm), "thresholds": {}}
            for threshold in ENERGY_THRESHOLDS:
                k = thresholds[str(threshold)]["rank"]
                top = projection(direction, V, k)
                low = direction - top
                fold_out["thresholds"][str(threshold)] = {"rank": k, "top_norm_fraction": float(np.linalg.norm(top) / norm), "low_norm_fraction": float(np.linalg.norm(low) / norm), "top_auc": auc(X[test], purpose_y[test], top, scaler), "low_auc": auc(X[test], purpose_y[test], low, scaler), "top_gain": float(np.linalg.norm(J[layer] @ top) / norm), "low_gain": float(np.linalg.norm(J[layer] @ low) / norm)}
            for threshold in RANDOM_THRESHOLDS:
                k = thresholds[str(threshold)]["rank"]
                values = []
                for seed in RANDOM_SEEDS:
                    top, low = random_projection(direction, k, seed)
                    values.append({"seed": seed, "top_norm_fraction": float(np.linalg.norm(top) / norm), "low_norm_fraction": float(np.linalg.norm(low) / norm), "top_auc": auc(X[test], purpose_y[test], top, scaler), "low_auc": auc(X[test], purpose_y[test], low, scaler)})
                fold_out.setdefault("random", {})[str(threshold)] = values
            folds[fold] = fold_out
        format_train = no_cue
        format_test = no_cue
        format_direction, format_scaler = fit_direction(X[format_train], format_y[format_train])
        random_gains = []
        rng = np.random.default_rng(20260817 + layer)
        for _ in range(32):
            r = rng.standard_normal(d); r /= np.linalg.norm(r)
            random_gains.append(float(np.linalg.norm(J[layer] @ r)))
        report_layers.append({"layer": layer, "dimension": d, "numerical_rank": int(np.linalg.matrix_rank(J[layer])), "singular_max": float(s[0]), "singular_min": float(s[-1]), "condition_number": float(s[0] / s[-1]), "thresholds": thresholds, "folds": folds, "format_full_auc": auc(X[format_test], format_y[format_test], format_direction, format_scaler), "format_gain": float(np.linalg.norm(J[layer] @ format_direction) / np.linalg.norm(format_direction)), "random_gain_mean": float(np.mean(random_gains)), "random_gain_sd": float(np.std(random_gains, ddof=1)), "random_gain_values": random_gains})
    result = {"schema_version": 1, "status": "development_only", "energy_thresholds": list(ENERGY_THRESHOLDS), "random_thresholds": list(RANDOM_THRESHOLDS), "random_seeds": list(RANDOM_SEEDS), "j_space_definition": "top right-singular-vector subspace by retained Jacobian Frobenius energy; exact row-space rank is reported separately", "items": {"path": str(args.items), "sha256": sha256(args.items)}, "activations": {"path": str(args.activations), "sha256": sha256(args.activations)}, "lens": {"path": str(args.lens), "sha256": sha256(args.lens), "n_prompts": lens_data["n_prompts"], "source_layers": lens_data["source_layers"]}, "layers": report_layers}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"layers": args.layers, "out": str(args.out)}))


if __name__ == "__main__":
    main()
