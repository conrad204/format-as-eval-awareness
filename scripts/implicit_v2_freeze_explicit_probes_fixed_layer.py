"""Freeze original-explicit probe models at one pre-specified model layer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_bound_regrowth_bootstrap import stream_extract_layers  # noqa: E402

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
            layer: int, regime: str, fold: str) -> dict[str, np.ndarray]:
    mask = fam != fold
    if regime != "joint":
        mask &= fmt == regime
    scaler = StandardScaler().fit(X[mask])
    model = LogisticRegression(**PROBE).fit(scaler.transform(X[mask]), y[mask])
    p = f"L{layer:03d}::{regime}::fold_{fold}"
    return {f"{p}::coef": model.coef_[0].astype(np.float32),
            f"{p}::intercept": np.asarray(model.intercept_[0], dtype=np.float32),
            f"{p}::scaler_mean": scaler.mean_.astype(np.float32),
            f"{p}::scaler_scale": scaler.scale_.astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=15)
    args = ap.parse_args()
    if sha256(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("explicit item hash mismatch")
    rows_all = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    Xall, _ = stream_extract_layers(args.activations, [args.layer])
    with zipfile.ZipFile(args.activations) as z:
        ids = np.load(z.open("item_ids.npy")).astype(str)
    keep = np.array([x in by_id for x in ids])
    X = Xall[keep, 0, :]
    rows = [by_id[x] for x in ids[keep]]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fam = np.array([r["purpose_family_id"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    tasks = [(regime, fold) for regime in REGIMES for fold in FOLDS]
    fitted = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(fit_one)(X, y, fam, fmt, args.layer, regime, fold)
        for regime, fold in tasks)
    arrays = {key: value for result in fitted for key, value in result.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite {args.out}")
    np.savez_compressed(args.out, **arrays)
    manifest = {"schema_version": 1, "status": "FROZEN_BEFORE_IMPLICIT_SCORING",
                "model_key": args.model_key, "layer": args.layer,
                "implicit_data_used": False, "source": "original full320 stated explicit rows",
                "items_sha256": sha256(args.items), "activations_sha256": sha256(args.activations),
                "probe": PROBE, "folds": list(FOLDS), "regimes": list(REGIMES),
                "n_explicit_rows": len(rows), "bank_sha256": sha256(args.out)}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
