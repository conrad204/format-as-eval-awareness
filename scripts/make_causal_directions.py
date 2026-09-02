"""Create predeclared development causal vectors from pilot activations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
LAYERS = (4, 12, 15, 24)
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def direction(X, y):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
    return model.coef_[0] / scaler.scale_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("lens", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    import torch
    rows = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    z = np.load(args.activations)
    if [str(x) for x in z["item_ids"]] != [r["item_id"] for r in rows]:
        raise ValueError("activation IDs mismatch")
    X = np.asarray(z["X"], dtype=np.float64)
    stated = np.asarray([r["render_slot"] == "stated" for r in rows])
    no_cue = np.asarray([r["render_slot"] == "no_cue" for r in rows])
    family = np.asarray([r["purpose_family_id"] for r in rows])
    purpose_y = np.asarray([r["intended_purpose"] == "evaluation" for r in rows], dtype=int)
    format_y = np.asarray([r["format"] == "benchmark" for r in rows], dtype=int)
    # Purpose vectors use stated rows only; no-cue rows are a held-out causal readout.
    purpose = {}
    fmt = {}
    for layer in LAYERS:
        per_fold = [direction(X[stated & (family != fold), layer], purpose_y[stated & (family != fold)]) for fold in FOLDS]
        purpose[layer] = np.mean(per_fold, axis=0)
        # Format vectors use no-cue rows, with leave-topic-out folds; causal readout uses held blocks.
        topics = sorted({r["topic_id"] for r in rows if r["render_slot"] == "no_cue"})
        fmt[layer] = np.mean([direction(X[no_cue & (np.asarray([r["topic_id"] for r in rows]) != topic), layer], format_y[no_cue & (np.asarray([r["topic_id"] for r in rows]) != topic)]) for topic in topics], axis=0)
    lens_data = torch.load(args.lens, map_location="cpu", weights_only=False)
    J = {int(k): np.asarray(v, dtype=np.float64) for k, v in lens_data["J"].items()}
    vectors = {}
    for layer in LAYERS:
        d = purpose[layer]
        _, s, Vh = np.linalg.svd(J[layer], full_matrices=False)
        energy = np.cumsum(s * s) / np.sum(s * s)
        k = int(np.searchsorted(energy, 0.99) + 1)
        V = Vh.T
        top = V[:, :k] @ (V[:, :k].T @ d)
        low = d - top
        rng = np.random.default_rng(20260817 + layer)
        random = rng.standard_normal(d.shape[0]); random = random / np.linalg.norm(random) * np.linalg.norm(d)
        vectors[layer] = np.stack([d, top, low, fmt[layer], random])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, layers=np.asarray(LAYERS), names=np.asarray(["full_purpose", "top_j_99", "low_gain_99", "format", "random"]), vectors=np.asarray([vectors[x] for x in LAYERS], dtype=np.float32), alpha_values=np.asarray([0.25, 0.5, 1.0], dtype=np.float32))
    print(json.dumps({"layers": LAYERS, "vectors": ["full_purpose", "top_j_99", "low_gain_99", "format", "random"], "alpha_values": [0.25, 0.5, 1.0]}))


if __name__ == "__main__":
    main()
