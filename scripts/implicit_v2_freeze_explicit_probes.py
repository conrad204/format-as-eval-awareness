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
            layer: int, regime: str, fold: str) -> tuple[str, dict[str, np.ndarray]]:
    mask = fam != fold
    if regime != "joint":
        mask &= fmt == regime
    scaler = StandardScaler().fit(X[mask])
    model = LogisticRegression(**PROBE).fit(scaler.transform(X[mask]), y[mask])
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
    ap.add_argument("--layer-chunk", type=int, default=8,
                    help="layers held in memory per streaming pass")
    args = ap.parse_args()
    if sha256(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("explicit item bank hash mismatch")
    rows_all = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    with zipfile.ZipFile(args.activations) as z:
        ids = np.load(z.open("item_ids.npy")).astype(str)
        with z.open("X.npy") as f:
            version = np.lib.format.read_magic(f)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, _ = reader(f)
    n_layers = int(shape[1])
    by_id = {r["item_id"]: r for r in rows_all}
    rows = [by_id[x] for x in ids]
    stated = np.array([r.get("cue_style") == "stated" for r in rows])
    selected = [r for r, keep in zip(rows, stated) if keep]
    y = np.array([r["label"] for r in selected], dtype=np.int8)
    fam = np.array([r["purpose_family_id"] for r in selected])
    fmt = np.array([r["format"] for r in selected])
    if set(fam) != set(FOLDS) or set(fmt) != {"benchmark", "casual"}:
        raise RuntimeError("unexpected explicit training strata")
    arrays: dict[str, np.ndarray] = {}
    layer = 0
    while True:
        chunk = [L for L in range(layer, min(layer + args.layer_chunk, n_layers))]
        if not chunk:
            break
        Xc, _ = stream_extract_layers(args.activations, chunk)
        Xs = Xc[stated].astype(np.float32)
        del Xc
        for j, L in enumerate(chunk):
            Xl = np.ascontiguousarray(Xs[:, j, :])
            fitted = Parallel(n_jobs=args.jobs, backend="threading")(
                delayed(fit_one)(Xl, y, fam, fmt, L, regime, fold)
                for regime in REGIMES for fold in FOLDS)
            for _, values in fitted:
                arrays.update(values)
            del Xl
            print(f"layer {L}/{n_layers - 1} fitted", flush=True)
        del Xs
        layer += args.layer_chunk
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
