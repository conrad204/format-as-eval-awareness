"""Development-only factorial geometry analysis on imported pilot activations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
    return model, scaler


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else float("nan")


def geometry_vectors(X: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    index = {row["item_id"]: i for i, row in enumerate(rows)}
    groups: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for row in rows:
        if row["render_slot"] != "stated":
            continue
        key = (row["payload_block_id"], row["purpose_pair_id"])
        groups.setdefault(key, {})[(row["intended_purpose"], row["format"])] = index[row["item_id"]]
    A, B, C = [], [], []
    for cells in groups.values():
        required = {("evaluation", "benchmark"), ("evaluation", "casual"), ("deployment", "benchmark"), ("deployment", "casual")}
        if set(cells) != required:
            continue
        h_eb = X[cells[("evaluation", "benchmark")]]
        h_ec = X[cells[("evaluation", "casual")]]
        h_db = X[cells[("deployment", "benchmark")]]
        h_dc = X[cells[("deployment", "casual")]]
        delta_b = h_eb - h_db
        delta_c = h_ec - h_dc
        A.append((delta_b + delta_c) / 4.0)
        C.append((delta_b - delta_c) / 4.0)
        B.append(((h_eb + h_db) - (h_ec + h_dc)) / 4.0)
    if not A:
        raise ValueError("no complete stated 2x2 payload groups")
    return np.asarray(A), np.asarray(B), np.asarray(C), len(A)


def held_scores(X: np.ndarray, rows: list[dict], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fam = np.asarray([row["purpose_family_id"] for row in rows])
    stated = np.asarray([row["render_slot"] == "stated" for row in rows])
    scores = np.full(len(rows), np.nan)
    directions = []
    for fold in FOLDS:
        train = stated & (fam != fold)
        test = stated & (fam == fold)
        model, scaler = fit(X[train], labels[train])
        scores[test] = model.decision_function(scaler.transform(X[test]))
        directions.append(model.coef_[0] / scaler.scale_)
    return scores, np.mean(np.asarray(directions), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    npz = np.load(args.activations)
    item_ids = [str(value) for value in npz["item_ids"]]
    if item_ids != [row["item_id"] for row in rows]:
        raise ValueError("activation item_ids do not match metadata row order")
    X_all = np.asarray(npz["X"])
    if X_all.ndim != 3:
        raise ValueError(f"expected all-layer X with ndim 3, got {X_all.shape}")
    labels = np.asarray([-1 if row["label"] is None else int(row["label"]) for row in rows])
    if np.any(~np.isfinite(X_all)):
        raise ValueError("activation matrix contains non-finite values")
    layers = list(range(X_all.shape[1])) if args.layers is None else args.layers
    if any(layer < 0 or layer >= X_all.shape[1] for layer in layers):
        raise ValueError(f"requested layer outside activation range: {layers}")
    table = []
    for layer in layers:
        X = X_all[:, layer, :].astype(np.float64, copy=False)
        A, B, C, n_groups = geometry_vectors(X, rows)
        scores, direction = held_scores(X, rows, labels)
        stated = np.asarray([row["render_slot"] == "stated" for row in rows])
        y = labels[stated]
        s = scores[stated]
        norm_a = float(np.mean(np.linalg.norm(A, axis=1)))
        norm_b = float(np.mean(np.linalg.norm(B, axis=1)))
        norm_c = float(np.mean(np.linalg.norm(C, axis=1)))
        d_b = 2 * (A + C)
        d_c = 2 * (A - C)
        flat_a, flat_b, flat_c = A.mean(axis=0), B.mean(axis=0), C.mean(axis=0)
        q_b, _ = np.linalg.qr(d_b.T)
        q_c, _ = np.linalg.qr(d_c.T)
        principal_cos = float(np.linalg.svd(q_b.T @ q_c, compute_uv=False)[0])
        table.append({
            "layer": layer,
            "n_complete_2x2_groups": n_groups,
            "held_stated_auc": float(roc_auc_score(y, s)),
            "held_stated_balanced_accuracy": float(balanced_accuracy_score(y, s > 0)),
            "norm_A": norm_a,
            "norm_B": norm_b,
            "norm_C": norm_c,
            "interaction_ratio_C_over_A": norm_c / norm_a if norm_a else float("nan"),
            "cosine_delta_b_delta_c": cosine(d_b.mean(axis=0), d_c.mean(axis=0)),
            "cosine_A_C": cosine(flat_a, flat_c),
            "principal_cosine_delta_subspaces": principal_cos,
            "purpose_direction_projection_on_B": cosine(direction, flat_b),
        })
    table_sorted = sorted(table, key=lambda row: row["held_stated_auc"], reverse=True)
    eligible = [row for row in table if row["held_stated_auc"] >= 0.8]
    report = {
        "schema_version": 1,
        "mode": "development",
        "items": {"path": str(args.items), "sha256": sha256(args.items)},
        "activations": {"path": str(args.activations), "sha256": sha256(args.activations)},
        "script_sha256": sha256(Path(__file__)),
        "probe": PROBE,
        "rows": len(rows),
        "layers": table_sorted,
        "selection_summary": {
            "max_held_auc": table_sorted[0],
            "min_interaction_eligible": min(eligible, key=lambda row: row["interaction_ratio_C_over_A"]) if eligible else None,
            "max_interaction_eligible": max(eligible, key=lambda row: row["interaction_ratio_C_over_A"]) if eligible else None,
        },
    }
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"layers": len(table), "max_auc_layer": table_sorted[0]["layer"], "max_auc": table_sorted[0]["held_stated_auc"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
