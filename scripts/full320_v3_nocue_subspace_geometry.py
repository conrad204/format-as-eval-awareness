"""Principal-angle geometry between the stated-purpose-derived matched-format
basis Q_B and the no-cue-derived format-associated basis Q_nocue.

No probe fitting: per layer and per held-purpose-family fold, both rank-r bases
are rebuilt exactly as in the erasure pipeline (uncentered SVD, training groups
only) and compared via principal cosines (SVD of Q_B^T Q_nocue), plus each
basis's share of held-out stated-row centered variance.

Usage: python scripts/full320_v3_nocue_subspace_geometry.py --model llama31_8b --rank 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import factorial_vectors, orth_basis  # noqa: E402
from full320_v3_nocue_format_subspace import (  # noqa: E402
    ITEMS, MODELS, nocue_payload_contrasts, training_group_keys,
)

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=sorted(MODELS))
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(x) for x in ITEMS.open(encoding="utf-8") if x.strip()]
    by_id = {r["item_id"]: r for r in rows}
    import zipfile
    with zipfile.ZipFile(MODELS[args.model]) as z:
        raw_ids = np.load(z.open("item_ids.npy"))
    ordered = [by_id[x] for x in raw_ids]
    stated_idx = np.flatnonzero([r["label"] is not None for r in ordered])
    stated_rows = [ordered[i] for i in stated_idx]
    fam = np.array([r["purpose_family_id"] for r in stated_rows])

    Xall = np.load(MODELS[args.model])["X"]
    n_layers = Xall.shape[1]
    out = {"model": args.model, "rank": args.rank, "n_layers": n_layers, "by_layer": {}}
    for L in range(n_layers):
        Xl = np.ascontiguousarray(Xall[:, L, :])
        Xs = Xl[stated_idx]
        nc = nocue_payload_contrasts(Xl, ordered)
        cosines, var_qb, var_qn = [], [], []
        for fold in sorted(set(fam)):
            train, test = fam != fold, fam == fold
            keys = training_group_keys(stated_rows, train)
            Qn = orth_basis(np.asarray([nc[b] for b, _ in keys]), args.rank)
            _, B, _, _ = factorial_vectors(Xs, stated_rows, train)
            Qb = orth_basis(B, args.rank)
            s = np.linalg.svd(Qb.T @ Qn, compute_uv=False)
            cosines.append(np.clip(s, 0.0, 1.0))
            Xh = Xs[test].astype(np.float64)
            Xh -= Xh.mean(axis=0, keepdims=True)
            denom = float(np.sum(Xh * Xh))
            var_qb.append(float(np.sum((Xh @ Qb) ** 2) / denom))
            var_qn.append(float(np.sum((Xh @ Qn) ** 2) / denom))
        Cm = np.mean(np.asarray(cosines), axis=0)
        out["by_layer"][str(L)] = {
            "normalized_depth": L / (n_layers - 1),
            "principal_cosine_max": float(Cm[0]),
            "principal_cosine_median": float(np.median(Cm)),
            "principal_cosine_min": float(Cm[-1]),
            "principal_angle_min_deg": float(np.degrees(np.arccos(Cm[0]))),
            "principal_angle_median_deg": float(np.degrees(np.arccos(np.median(Cm)))),
            "trace_overlap_over_rank": float(np.mean(Cm ** 2)),
            "n_principal_cosines_above_0p9": int(np.sum(Cm > 0.9)),
            "heldout_variance_qb": float(np.mean(var_qb)),
            "heldout_variance_qnocue": float(np.mean(var_qn)),
        }
        print(f"layer {L}: max_cos={Cm[0]:.3f} med_cos={np.median(Cm):.3f} "
              f"trace={np.mean(Cm ** 2):.3f}", file=sys.stderr, flush=True)
        del Xl, Xs

    path = ROOT / "results" / f"nocue_subspace_geometry_{args.model}_r{args.rank}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
