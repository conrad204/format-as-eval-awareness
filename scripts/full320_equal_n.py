"""Development-only equal-N mixed-format control for full320-v3."""
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
ITEM_SHA256 = "5528819d74e696d1a595062047a829836dfde1d2a45287b06a5ac1ba5d1b6c67"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def block_bootstrap(blocks: np.ndarray, labels: np.ndarray, scores: np.ndarray, seed: int, n: int) -> dict:
    unique = np.asarray(sorted(set(blocks.tolist())))
    rng = np.random.default_rng(seed)
    point_auc = roc_auc_score(labels, scores)
    point_ba = balanced_accuracy_score(labels, scores > 0)
    by_block = {b: np.flatnonzero(blocks == b) for b in unique}
    draws = []
    for _ in range(n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_block[b] for b in chosen])
        if len(np.unique(labels[idx])) == 2:
            draws.append((roc_auc_score(labels[idx], scores[idx]), balanced_accuracy_score(labels[idx], scores[idx] > 0)))
    arr = np.asarray(draws)
    return {
        "auc": float(point_auc),
        "balanced_accuracy": float(point_ba),
        "bootstrap_blocks": int(len(unique)),
        "bootstrap_resamples": int(n),
        "auc_ci_2.5": float(np.quantile(arr[:, 0], 0.025)),
        "auc_ci_97.5": float(np.quantile(arr[:, 0], 0.975)),
        "balanced_accuracy_ci_2.5": float(np.quantile(arr[:, 1], 0.025)),
        "balanced_accuracy_ci_97.5": float(np.quantile(arr[:, 1], 0.975)),
    }


def run_layer(X: np.ndarray, rows: list[dict], layer: int, seed: int, bootstrap: int) -> dict:
    purpose = np.asarray([r["intended_purpose"] for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    family = np.asarray([r["purpose_family_id"] for r in rows])
    stated = np.asarray([r["render_slot"] == "stated" for r in rows])
    block = np.asarray([r["payload_block_id"] for r in rows])
    y = np.asarray([1 if p == "evaluation" else 0 for p in purpose], dtype=np.int8)
    layer_x = np.asarray(X[:, layer, :], dtype=np.float32)
    results = {}
    rng = np.random.default_rng(seed)
    for regime in ("benchmark_only", "casual_only", "equal_n_mixed"):
        scores = np.full(len(rows), np.nan, dtype=np.float64)
        for fold in FOLDS:
            train_blocks = np.unique(block[stated & (family != fold)])
            if regime == "benchmark_only":
                chosen = {b: "benchmark" for b in train_blocks}
            elif regime == "casual_only":
                chosen = {b: "casual" for b in train_blocks}
            else:
                order = rng.permutation(train_blocks)
                half = len(order) // 2
                chosen = {b: ("benchmark" if i < half else "casual") for i, b in enumerate(order)}
            train = stated & (family != fold) & np.asarray([chosen.get(b) == f for b, f in zip(block, fmt)])
            test = stated & (family == fold)
            scaler = StandardScaler().fit(layer_x[train])
            model = LogisticRegression(C=0.1, max_iter=3000, random_state=0).fit(scaler.transform(layer_x[train]), y[train])
            scores[test] = model.decision_function(scaler.transform(layer_x[test]))
        mask = ~np.isnan(scores)
        results[regime] = block_bootstrap(block[mask], y[mask], scores[mask], seed + 1000, bootstrap)
    results["layer"] = layer
    results["seed"] = seed
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260822, 20260823, 20260824, 20260825])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sha256(args.items) != ITEM_SHA256:
        raise ValueError("items hash does not match full320-v3")
    data = np.load(args.activations, allow_pickle=False)
    ids = [str(x) for x in data["item_ids"]]
    if ids != [r["item_id"] for r in rows]:
        raise ValueError("activation item_ids do not match rendered item order")
    if data["X"].shape[0] != len(rows):
        raise ValueError("activation row count mismatch")
    out = {
        "schema_version": 1,
        "status": "development_only",
        "analysis": "full320_equal_n_mixed_format_control",
        "items": {"path": str(args.items), "sha256": sha256(args.items), "n_rows": len(rows)},
        "activations": {"path": str(args.activations), "sha256": sha256(args.activations), "shape": list(data["X"].shape)},
        "model": args.model,
        "model_revision": args.model_revision,
        "layers": args.layers,
        "seeds": args.seeds,
        "bootstrap_unit": "payload_block",
        "bootstrap_resamples": args.bootstrap,
        "results": [run_layer(data["X"], rows, layer, seed, args.bootstrap) for layer in args.layers for seed in args.seeds],
        "claim_limitations": ["development-only", "observational probe endpoint", "not confirmation", "not behavioral causal evidence"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layers": args.layers, "seeds": args.seeds, "out": str(args.out)}))


if __name__ == "__main__":
    main()
