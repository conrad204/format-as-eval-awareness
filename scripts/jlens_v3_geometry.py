"""Offline jLens geometry controls for frozen full320-v3 activations."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ENERGY_THRESHOLDS = (0.90, 0.95, 0.99)
RANDOM_SEEDS = tuple(range(32))
FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fit_direction(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
    return model.coef_[0] / scaler.scale_, scaler


def auc(X: np.ndarray, y: np.ndarray, direction: np.ndarray, scaler: StandardScaler) -> float | None:
    if not np.linalg.norm(direction) or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, (X - scaler.mean_) @ direction))


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


def random_projection(direction: np.ndarray, k: int, seed: int):
    d = direction.shape[0]
    if d & (d - 1):
        raise ValueError("random control requires power-of-two hidden dimension")
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


def project(direction: np.ndarray, basis: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    top = basis[:, :k] @ (basis[:, :k].T @ direction)
    return top, direction - top


def fold_ids(rows: list[dict], key: str) -> dict[str, int]:
    values = sorted({str(row[key]) for row in rows})
    return {value: index % len(FOLDS) for index, value in enumerate(values)}


def summarize_direction(X, y, groups, J, V, energy, direction_kind: str) -> dict:
    group_values = sorted(set(groups))
    if len(group_values) < 2:
        raise ValueError(f"{direction_kind} requires at least two groups")
    reports = {}
    for fold_index, fold in enumerate(group_values):
        train = groups != fold
        test = groups == fold
        direction, scaler = fit_direction(X[train], y[train])
        norm = float(np.linalg.norm(direction))
        report = {
            "held_out_group": str(fold),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "n_train_groups": int(len(set(groups[train]))),
            "n_test_groups": int(len(set(groups[test]))),
            "full_auc": auc(X[test], y[test], direction, scaler),
            "jacobian_gain": float(np.linalg.norm(J @ direction) / norm),
            "thresholds": {},
        }
        for threshold in ENERGY_THRESHOLDS:
            k = int(np.searchsorted(energy, threshold) + 1)
            top, orth = project(direction, V, k)
            top_random = []
            for seed in RANDOM_SEEDS:
                random_top, _ = random_projection(direction, k, seed + 1000 * fold_index)
                top_random.append(auc(X[test], y[test], random_top, scaler))
            report["thresholds"][str(threshold)] = {
                "rank": k,
                "dimension_fraction": k / X.shape[1],
                "top_norm_fraction": float(np.linalg.norm(top) / norm),
                "orth_norm_fraction": float(np.linalg.norm(orth) / norm),
                "top_auc": auc(X[test], y[test], top, scaler),
                "orth_auc": auc(X[test], y[test], orth, scaler),
                "top_gain": float(np.linalg.norm(J @ top) / norm),
                "orth_gain": float(np.linalg.norm(J @ orth) / norm),
                "random_top_auc_mean": float(np.nanmean(top_random)),
                "random_top_auc_sd": float(np.nanstd(top_random, ddof=1)),
                "random_top_auc_values": top_random,
            }
        reports[str(fold)] = report
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--layers", nargs="+", type=int)
    args = parser.parse_args()

    import torch

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    z = np.load(args.activations)
    item_ids = [str(value) for value in z["item_ids"]]
    row_ids = [row["item_id"] for row in rows]
    if item_ids != row_ids:
        raise ValueError("activation item_ids do not match full320 item ordering")
    X_all = np.asarray(z["X"], dtype=np.float64)
    if X_all.shape[0] != len(rows) or not np.isfinite(X_all).all():
        raise ValueError("activation matrix shape/non-finite check failed")

    lens_data = torch.load(args.lens, map_location="cpu", weights_only=False)
    jacobians = {int(k): np.asarray(v, dtype=np.float64) for k, v in lens_data["J"].items()}
    layers = args.layers or sorted(jacobians)
    if any(layer not in jacobians for layer in layers):
        raise ValueError("requested layer is missing from the lens")
    purpose_mask = np.asarray([
        not row.get("legacy", False)
        and row.get("render_slot") == "stated"
        and row.get("intended_purpose") in {"evaluation", "deployment"}
        for row in rows
    ])
    format_mask = np.asarray([
        not row.get("legacy", False)
        and row.get("render_slot") == "no_cue"
        and row.get("format") in {"benchmark", "casual"}
        for row in rows
    ])
    purpose_rows = [row for row, keep in zip(rows, purpose_mask) if keep]
    format_rows = [row for row, keep in zip(rows, format_mask) if keep]
    if not purpose_rows or not format_rows:
        raise ValueError("full320 purpose or no-cue format rows are missing")
    purpose_y = np.asarray([int(row["intended_purpose"] == "evaluation") for row in purpose_rows])
    format_y = np.asarray([int(row["format"] == "benchmark") for row in format_rows])
    purpose_groups = np.asarray([row["purpose_family_id"] for row in purpose_rows])
    block_ids = sorted({row["payload_block_id"] for row in format_rows})
    block_fold = {block_id: f"block_fold_{index % len(FOLDS)}" for index, block_id in enumerate(block_ids)}
    format_groups = np.asarray([block_fold[row["payload_block_id"]] for row in format_rows])
    purpose_indices = np.flatnonzero(purpose_mask)
    format_indices = np.flatnonzero(format_mask)

    reports = []
    for layer in layers:
        J = jacobians[layer]
        if J.shape[1] != X_all.shape[2]:
            raise ValueError(f"layer {layer} Jacobian/input dimension mismatch")
        _, singular, Vh = np.linalg.svd(J, full_matrices=False)
        V = Vh.T
        energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        purpose_X = X_all[purpose_indices, layer, :]
        format_X = X_all[format_indices, layer, :]
        reports.append({
            "layer": layer,
            "dimension": int(X_all.shape[2]),
            "numerical_rank": int(np.linalg.matrix_rank(J)),
            "singular_max": float(singular[0]),
            "singular_min": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]),
            "purpose": summarize_direction(purpose_X, purpose_y, purpose_groups, J, V, energy, "purpose_family"),
            "format": summarize_direction(format_X, format_y, format_groups, J, V, energy, "payload_block"),
            "energy_thresholds": {
                str(threshold): {
                    "rank": int(np.searchsorted(energy, threshold) + 1),
                    "captured_frobenius_energy": float(energy[np.searchsorted(energy, threshold)]),
                }
                for threshold in ENERGY_THRESHOLDS
            },
        })
        print(f"analyzed layer {layer}", flush=True)

    result = {
        "schema_version": 1,
        "status": "development_only",
        "items": {"path": str(args.items), "sha256": sha256(args.items)},
        "activations": {"path": str(args.activations), "sha256": sha256(args.activations)},
        "lens": {"path": str(args.lens), "sha256": sha256(args.lens), "source_layers": lens_data["source_layers"]},
        "git_commit": git_commit(args.repo_root),
        "n_rows": len(rows),
        "n_purpose_rows": len(purpose_rows),
        "n_format_rows": len(format_rows),
        "n_payload_blocks": len({row["payload_block_id"] for row in rows}),
        "purpose_folds": sorted(set(purpose_groups)),
        "random_seeds": list(RANDOM_SEEDS),
        "energy_thresholds": list(ENERGY_THRESHOLDS),
        "layers": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"layers": layers, "out": str(args.out)}))


if __name__ == "__main__":
    main()
