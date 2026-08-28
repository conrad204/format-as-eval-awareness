"""Development-only cross-fitted format/purpose erasure on pilot activations."""
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
RANKS = (1, 2, 4, 8, 16, 32)
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit_probe(X: np.ndarray, y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    sc = StandardScaler().fit(X)
    m = LogisticRegression(**PROBE).fit(sc.transform(X), y)
    return m.decision_function(sc.transform(Z))


def auc(y: np.ndarray, s: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, s)), "bal_acc": float(balanced_accuracy_score(y, s > 0))}


def groups(rows: list[dict], stated_mask: np.ndarray, train_mask: np.ndarray):
    by = {}
    for i in np.flatnonzero(stated_mask & train_mask):
        key = (rows[i]["payload_block_id"], rows[i]["purpose_pair_id"])
        by.setdefault(key, {})[(rows[i]["intended_purpose"], rows[i]["format"])] = i
    out = []
    for key, cells in sorted(by.items()):
        needed = {(p, f) for p in ("evaluation", "deployment") for f in ("benchmark", "casual")}
        if needed <= set(cells):
            eb, ec = cells[("evaluation", "benchmark")], cells[("evaluation", "casual")]
            db, dc = cells[("deployment", "benchmark")], cells[("deployment", "casual")]
            out.append((key, eb, ec, db, dc))
    return out


def factorial_vectors(X: np.ndarray, rows: list[dict], mask: np.ndarray):
    gs = groups(rows, np.asarray([r["render_slot"] == "stated" for r in rows]), mask)
    A, B, C = [], [], []
    for _, eb, ec, db, dc in gs:
        A.append((X[eb] - X[db] + X[ec] - X[dc]) / 4.0)
        B.append(((X[eb] + X[db]) - (X[ec] + X[dc])) / 4.0)
        C.append((X[eb] - X[db] - X[ec] + X[dc]) / 4.0)
    if not A:
        raise ValueError("no complete train groups")
    return np.asarray(A), np.asarray(B), np.asarray(C), gs


def orth_basis(V: np.ndarray, rank: int) -> np.ndarray:
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[None, :]
    if not np.any(np.isfinite(V)):
        return np.zeros((V.shape[1], 0))
    # Rows are effects; SVD yields an orthonormal feature-space basis.
    _, s, vh = np.linalg.svd(V, full_matrices=False)
    keep = min(rank, int(np.sum(s > max(s[0], 1.0) * 1e-10)), vh.shape[0]) if len(s) else 0
    return vh[:keep].T


def remove(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return X if Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def format_basis(X: np.ndarray, yfmt: np.ndarray, method: str, rank: int) -> np.ndarray:
    if method == "matched_mean":
        return orth_basis(X, rank)
    if method == "matched_pca":
        return orth_basis(X, rank)
    sc = StandardScaler().fit(X)
    residual = X.copy()
    qs = []
    for _ in range(rank):
        m = LogisticRegression(**PROBE).fit(sc.transform(residual), yfmt)
        q = m.coef_[0] / sc.scale_
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n == 0:
            break
        q = q / n
        qs.append(q)
        residual = remove(residual, q[:, None])
    return orth_basis(np.asarray(qs), len(qs))


def metric_bundle(rows, X, train_fold, Q, *, transfer: bool):
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    slot = np.asarray([r["render_slot"] for r in rows])
    label = np.asarray([-1 if r["label"] is None else int(r["label"]) for r in rows])
    fmt = np.asarray([r["format"] for r in rows])
    train = (fam != train_fold) & (slot == "stated")
    test_stated = (fam == train_fold) & (slot == "stated")
    test_ind = (fam == train_fold) & (slot == "indirect")
    Xt = remove(X, Q)
    ytrain = label[train]
    stated_scores = fit_probe(Xt[train], ytrain, Xt[test_stated])
    format_scores = fit_probe(Xt[train], (fmt[train] == "benchmark").astype(int), Xt[test_stated])
    out = {"purpose": auc(label[test_stated], stated_scores),
           "format": auc((fmt[test_stated] == "benchmark").astype(int), format_scores)}
    if transfer:
        ind_scores = fit_probe(Xt[train], ytrain, Xt[test_ind])
        out["indirect"] = auc(label[test_ind], ind_scores)
    return out


def synthetic_checks() -> dict:
    rng = np.random.default_rng(0)
    n, d = 10000, 32
    p = rng.choice([-1, 1], n)
    f = rng.choice([-1, 1], n)
    X = p[:, None] * np.eye(1, d, 0) + f[:, None] * np.eye(1, d, 1) + rng.normal(0, .05, (n, d))
    raw_p = auc((p > 0).astype(int), fit_probe(X, (p > 0).astype(int), X))["auc"]
    q = np.eye(d)[:, 1:2]
    after = remove(X, q)
    fmt_after = auc((f > 0).astype(int), fit_probe(after, (f > 0).astype(int), after))["auc"]
    return {"seed": 0, "A_plus_B": {"raw_purpose_auc": raw_p, "format_after_removing_B": fmt_after},
            "pass": bool(raw_p > .95 and abs(fmt_after - .5) < .05)}


def run_model(items: Path, acts: Path, layer: int) -> dict:
    rows = [json.loads(x) for x in items.read_text(encoding="utf-8").splitlines() if x.strip()]
    z = np.load(acts)
    ids = [str(x) for x in z["item_ids"]]
    if ids != [r["item_id"] for r in rows]:
        raise ValueError("item IDs are not aligned")
    X = np.asarray(z["X"][:, layer, :], dtype=np.float64)
    fam = np.asarray([r["purpose_family_id"] for r in rows])
    stated = np.asarray([r["render_slot"] == "stated" for r in rows])
    out = {"layer": layer, "methods": {}, "reverse_purpose": {}}
    for method in ("matched_mean", "matched_pca", "logistic", "inlp"):
        out["methods"][method] = {}
        for rank in RANKS:
            fold_results = []
            for fold in FOLDS:
                train_mask = fam != fold
                A, B, C, _ = factorial_vectors(X, rows, train_mask)
                if method == "matched_mean":
                    Q = orth_basis(B.mean(axis=0), rank)
                elif method == "matched_pca":
                    Q = orth_basis(B, rank)
                elif method == "logistic":
                    train = stated & train_mask
                    Q = format_basis(X[train], (np.asarray([r["format"] for r in rows])[train] == "benchmark").astype(int), "logistic", 1)
                else:
                    train = stated & train_mask
                    Q = format_basis(X[train], (np.asarray([r["format"] for r in rows])[train] == "benchmark").astype(int), "inlp", rank)
                fold_results.append(metric_bundle(rows, X, fold, Q, transfer=True))
            out["methods"][method][str(rank)] = {"folds": fold_results,
                "purpose_auc": float(np.mean([x["purpose"]["auc"] for x in fold_results])),
                "format_auc": float(np.mean([x["format"]["auc"] for x in fold_results])),
                "indirect_auc": float(np.mean([x["indirect"]["auc"] for x in fold_results]))}
    # Reverse purpose removal uses train-only [A,C] basis; fresh format probe is retained.
    for rank in RANKS:
        vals = []
        for fold in FOLDS:
            A, B, C, _ = factorial_vectors(X, rows, fam != fold)
            Q = orth_basis(np.concatenate([A, C], axis=0), rank)
            vals.append(metric_bundle(rows, X, fold, Q, transfer=True))
        out["reverse_purpose"][str(rank)] = {"folds": vals,
            "purpose_auc": float(np.mean([x["purpose"]["auc"] for x in vals])),
            "format_auc": float(np.mean([x["format"]["auc"] for x in vals])),
            "indirect_auc": float(np.mean([x["indirect"]["auc"] for x in vals]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"schema_version": 1, "mode": "development", "model": args.model,
              "items": {"path": str(args.items), "sha256": sha256_file(args.items)},
              "activations": {"path": str(args.activations), "sha256": sha256_file(args.activations)},
              "layer": args.layer, "probe": PROBE, "ranks": list(RANKS),
              "synthetic_checks": synthetic_checks(), "result": run_model(args.items, args.activations, args.layer)}
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "layer": args.layer, "out": str(args.out)}))


if __name__ == "__main__":
    main()
