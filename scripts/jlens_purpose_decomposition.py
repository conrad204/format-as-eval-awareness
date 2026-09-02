"""Development-only decomposition of purpose probes in fitted J-lens input space."""
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
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
ENERGY_THRESHOLDS = (0.90, 0.95, 0.99)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_direction(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
    return model.coef_[0] / scaler.scale_, scaler, model


def auc_for_direction(X: np.ndarray, y: np.ndarray, direction: np.ndarray, scaler: StandardScaler) -> float:
    scores = (X - scaler.mean_) @ direction
    return float(roc_auc_score(y, scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("lens", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[12])
    args = parser.parse_args()

    import torch

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    npz = np.load(args.activations)
    item_ids = [str(value) for value in npz["item_ids"]]
    if item_ids != [row["item_id"] for row in rows]:
        raise ValueError("activation item_ids do not match metadata row order")
    X_all = np.asarray(npz["X"])
    labels = np.asarray([-1 if row["label"] is None else int(row["label"]) for row in rows])
    stated = np.asarray([row["render_slot"] == "stated" for row in rows])
    family = np.asarray([row["purpose_family_id"] for row in rows])
    if np.any(~np.isfinite(X_all)):
        raise ValueError("activation matrix contains non-finite values")

    lens_data = torch.load(args.lens, map_location="cpu", weights_only=False)
    jacobians = {int(k): np.asarray(v, dtype=np.float64) for k, v in lens_data["J"].items()}
    d = X_all.shape[-1]
    if lens_data["d_model"] != d:
        raise ValueError("lens and activation dimensions differ")

    report_layers = []
    for layer in args.layers:
        X = X_all[:, layer, :].astype(np.float64, copy=False)
        J = jacobians[layer]
        U, singular, Vh = np.linalg.svd(J, full_matrices=False)
        del U
        V = Vh.T
        energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        numerical_rank = int(np.linalg.matrix_rank(J))
        layer_report = {
            "layer": layer,
            "jacobian_shape": list(J.shape),
            "numerical_rank": numerical_rank,
            "singular_max": float(singular[0]),
            "singular_min": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]),
            "thresholds": {},
        }
        for threshold in ENERGY_THRESHOLDS:
            k = int(np.searchsorted(energy, threshold) + 1)
            basis = V[:, :k]
            layer_report["thresholds"][str(threshold)] = {
                "rank": k,
                "captured_frobenius_energy": float(energy[k - 1]),
                "folds": {},
            }
        exact_folds = {}
        for fold in FOLDS:
            train = stated & (family != fold)
            test = stated & (family == fold)
            direction, scaler, model = fit_direction(X[train], labels[train])
            norm = float(np.linalg.norm(direction))
            fold_result = {
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "original_auc": auc_for_direction(X[test], labels[test], direction, scaler),
                "original_norm": norm,
                "thresholds": {},
            }
            # The J-space is the right-singular-vector input subspace of J.
            for threshold in ENERGY_THRESHOLDS:
                k = int(np.searchsorted(energy, threshold) + 1)
                basis = V[:, :k]
                j_component = basis @ (basis.T @ direction)
                orth_component = direction - j_component
                fold_result["thresholds"][str(threshold)] = {
                    "rank": k,
                    "j_norm_fraction": float(np.linalg.norm(j_component) / norm),
                    "orth_norm_fraction": float(np.linalg.norm(orth_component) / norm),
                    "j_cosine": float(np.dot(direction, j_component) / (norm * np.linalg.norm(j_component))) if np.linalg.norm(j_component) else None,
                    "j_auc": auc_for_direction(X[test], labels[test], j_component, scaler) if np.linalg.norm(j_component) else None,
                    "orth_auc": auc_for_direction(X[test], labels[test], orth_component, scaler) if np.linalg.norm(orth_component) else None,
                }
            # Exact numerical row-space decomposition; full-rank J should make orth≈0.
            if numerical_rank == d:
                exact_j = direction.copy()
            else:
                exact_j = V[:, :numerical_rank] @ (V[:, :numerical_rank].T @ direction)
            exact_orth = direction - exact_j
            fold_result["exact_row_space"] = {
                "j_norm_fraction": float(np.linalg.norm(exact_j) / norm),
                "orth_norm_fraction": float(np.linalg.norm(exact_orth) / norm),
                "j_auc": auc_for_direction(X[test], labels[test], exact_j, scaler) if np.linalg.norm(exact_j) else None,
                "orth_auc": auc_for_direction(X[test], labels[test], exact_orth, scaler) if np.linalg.norm(exact_orth) else None,
            }
            exact_folds[fold] = fold_result
        layer_report["folds"] = exact_folds
        report_layers.append(layer_report)

    result = {
        "schema_version": 1,
        "status": "development_only",
        "purpose_direction": "held-vocabulary-fold logistic probe raw coefficient, averaged only for reporting; decomposition uses each held-out fold direction",
        "j_space_definition": "right-singular-vector input subspace of the fitted source-to-target Jacobian",
        "energy_thresholds": list(ENERGY_THRESHOLDS),
        "items": {"path": str(args.items), "sha256": sha256(args.items)},
        "activations": {"path": str(args.activations), "sha256": sha256(args.activations)},
        "lens": {"path": str(args.lens), "sha256": sha256(args.lens), "n_prompts": lens_data["n_prompts"], "source_layers": lens_data["source_layers"]},
        "layers": report_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"layers": args.layers, "out": str(args.out)}))


if __name__ == "__main__":
    main()
