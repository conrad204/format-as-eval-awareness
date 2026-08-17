"""Assemble independently recomputed pilot endpoints, sweeps, and twin checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
REVISIONS = {
    "llama31-8b": "0e9e39f249a16976918f6564b8830bc894c89659",
    "llama31-70b": "1605565b47bb9346c5515c34102e054115b4f98b",
    "llama33-70b": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def held_scores(rows: list[dict], X: np.ndarray) -> np.ndarray:
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    stated = np.asarray([r["render_slot"] == "stated" for r in rows])
    y = np.asarray([-1 if r["label"] is None else int(r["label"]) for r in rows])
    scores = np.full(len(rows), np.nan)
    for fold in FOLDS:
        train = stated & (fam != fold)
        test = stated & (fam == fold)
        sc = StandardScaler().fit(X[train])
        m = LogisticRegression(**PROBE).fit(sc.transform(X[train]), y[train])
        scores[test] = m.decision_function(sc.transform(X[test]))
    return scores


def twin_check(rows: list[dict], scores: np.ndarray) -> dict:
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        if r["render_slot"] == "stated":
            groups.setdefault((r["payload_block_id"], r["purpose_pair_id"], r["format"]), []).append(i)
    valid, invalid, wins = 0, [], 0
    for key, idx in sorted(groups.items()):
        labels = [rows[i]["label"] for i in idx]
        if len(idx) != 2 or set(labels) != {0, 1} or any(np.isnan(scores[i]) for i in idx):
            invalid.append({"key": key, "n": len(idx), "labels": labels})
            continue
        valid += 1
        eval_score = scores[idx[labels.index(1)]]
        deploy_score = scores[idx[labels.index(0)]]
        wins += int(eval_score > deploy_score)
    return {"n_pairs": len(groups), "n_valid": valid, "n_invalid": len(invalid),
            "n_eval_gt_deploy": wins, "fraction": wins / valid if valid else None,
            "invalid_examples": invalid[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("results/pilot_reproduction_v3.json"))
    args = ap.parse_args()
    specs = {
        "llama31-8b": ("activations_dataset/activations/llama31_8b_pilot80", 12, "pilot80_analysis_8b_v3.json", "pilot80_sweep_full_8b_v3.json"),
        "llama31-70b": ("activations_dataset/activations/llama31_70b_pilot80", 31, "pilot80_analysis_31_70b_v3.json", "pilot80_sweep_full_31_70b_v3.json"),
        "llama33-70b": ("activations_dataset/activations/llama33_70b_pilot80", 17, "pilot80_analysis_33_70b_v3.json", "pilot80_sweep_full_33_70b_v3.json"),
    }
    report = {"schema_version": 3, "mode": "development", "probe": PROBE, "models": {}}
    for name, (rel, layer, endpoint_name, sweep_name) in specs.items():
        p = args.root / rel
        items, activations = p / "meta.jsonl", p / "activations_all_layers.npz"
        rows = [json.loads(x) for x in items.read_text(encoding="utf-8").splitlines() if x.strip()]
        z = np.load(activations)
        if [str(x) for x in z["item_ids"]] != [r["item_id"] for r in rows]:
            raise ValueError(f"item ordering mismatch: {name}")
        X = np.asarray(z["X"][:, layer, :], dtype=np.float64)
        endpoint_path, sweep_path = args.root / "results" / endpoint_name, args.root / "results" / sweep_name
        scores = held_scores(rows, X)
        model = {"model_revision": REVISIONS[name], "layer": layer,
                 "items_sha256": sha256_file(items), "activations_sha256": sha256_file(activations),
                 "n_rows": len(rows), "n_blocks": len(set(r["payload_block_id"] for r in rows)),
                 "endpoint": json.loads(endpoint_path.read_text()) if endpoint_path.exists() else None,
                 "all_layer_sweep": json.loads(sweep_path.read_text()) if sweep_path.exists() else None,
                 "paired_stated_twins": twin_check(rows, scores)}
        report["models"][name] = model
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, model in report["models"].items():
        t = model["paired_stated_twins"]
        sweep = model["all_layer_sweep"]
        best = max(sweep, key=lambda x: x["decorrelated"]["auc"]) if sweep else None
        print(name, "twins", t["n_eval_gt_deploy"], "/", t["n_valid"], "argmax", best["layer"] if best else None)


if __name__ == "__main__":
    main()
