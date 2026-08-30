"""Refit and freeze the original explicit-purpose probes at every 8B layer.

Only the pre-existing full320 explicit stated-cue bank is used. The implicit-v2
bank and activations are never read. This produces fixed joint-format,
benchmark-only, and casual-only five-fold models for later zero-retraining
projection onto implicit-v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
REGIMES = ("joint", "benchmark", "casual")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
EXPECTED_ITEMS_SHA256 = "5528819d74e696d1a595062047a829836dfde1d2a45287b06a5ac1ba5d1b6c67"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_one(X: np.ndarray, y: np.ndarray, fam: np.ndarray, fmt: np.ndarray,
            layer: int, regime: str, fold: str) -> tuple[str, dict[str, np.ndarray]]:
    mask = fam != fold
    if regime != "joint":
        mask &= fmt == regime
    scaler = StandardScaler().fit(X[mask, layer, :])
    model = LogisticRegression(**PROBE).fit(scaler.transform(X[mask, layer, :]), y[mask])
    prefix = f"L{layer:03d}::{regime}::fold_{fold}"
    return prefix, {
        f"{prefix}::coef": model.coef_[0].astype(np.float32),
        f"{prefix}::intercept": np.asarray(model.intercept_[0], dtype=np.float32),
        f"{prefix}::scaler_mean": scaler.mean_.astype(np.float32),
        f"{prefix}::scaler_scale": scaler.scale_.astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    if sha256(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("explicit item bank hash mismatch")
    rows_all = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(args.activations, allow_pickle=False) as z:
        X_all = z["X"]
        ids = z["item_ids"].astype(str)
        by_id = {r["item_id"]: r for r in rows_all}
        rows = [by_id[x] for x in ids]
        stated = np.array([r.get("cue_style") == "stated" for r in rows])
        X = X_all[stated].astype(np.float32)
    selected = [r for r, keep in zip(rows, stated) if keep]
    y = np.array([r["label"] for r in selected], dtype=np.int8)
    fam = np.array([r["purpose_family_id"] for r in selected])
    fmt = np.array([r["format"] for r in selected])
    if set(fam) != set(FOLDS) or set(fmt) != {"benchmark", "casual"}:
        raise RuntimeError("unexpected explicit training strata")
    n_layers = X.shape[1]
    jobs = [(layer, regime, fold) for layer in range(n_layers)
            for regime in REGIMES for fold in FOLDS]
    fitted = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(fit_one)(X, y, fam, fmt, layer, regime, fold)
        for layer, regime, fold in jobs)
    arrays: dict[str, np.ndarray] = {}
    for _, values in fitted:
        arrays.update(values)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite frozen bank: {args.out}")
    np.savez_compressed(args.out, **arrays)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_IMPLICIT_SCORING",
        "source": "original full320 stated explicit-purpose rows only",
        "implicit_data_used": False,
        "items": str(args.items),
        "items_sha256": sha256(args.items),
        "activations": str(args.activations),
        "activations_sha256": sha256(args.activations),
        "probe": PROBE,
        "folds": list(FOLDS),
        "regimes": list(REGIMES),
        "n_layers": n_layers,
        "n_explicit_rows": len(selected),
        "bank_sha256": sha256(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
