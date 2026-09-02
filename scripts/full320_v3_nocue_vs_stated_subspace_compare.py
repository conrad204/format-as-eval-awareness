"""Paired comparison: stated-purpose-derived Q_B vs no-cue-derived Q_nocue.

Bootstrap arithmetic only -- consumes the cached OOF probe scores from
scripts/full_layer_gap_erasure_sweep.py (raw rank 0 and stated-derived rank r)
and scripts/full320_v3_nocue_format_subspace.py (no-cue-derived rank r). Each
replicate draws ONE payload-block resample and reuses it across every layer and
every regime, so the fractions of integrated gap removed are paired.

Usage: python scripts/full320_v3_nocue_vs_stated_subspace_compare.py --model llama31_8b --rank 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def fast_auc(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(S, axis=1, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(S.shape[1], dtype=np.float64), axis=1)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return (ranks[:, pos].sum(axis=1) - npos * (npos - 1) / 2) / (npos * nneg)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stated = np.load(RESULTS / f"full_layer_gap_erasure_scores_{args.model}.npz", allow_pickle=True)
    nocue = np.load(RESULTS / f"nocue_format_subspace_scores_{args.model}_r{args.rank}.npz", allow_pickle=True)
    ranks = [int(r) for r in stated["ranks"]]
    y = stated["y"].astype(int)
    blk = stated["payload_block_id"]
    if not np.array_equal(y, nocue["y"].astype(int)) or not np.array_equal(blk, nocue["payload_block_id"]):
        raise ValueError("stated/no-cue artifacts are not row-aligned")
    n_layers = int(stated["n_layers"])
    q = np.arange(n_layers) / (n_layers - 1)

    regimes = {}
    for name, key in (("purpose", "purpose_scores"), ("b2c", "b2c_scores"), ("c2b", "c2b_scores")):
        regimes[name] = {
            "raw": stated[key][ranks.index(0)].astype(float),
            "qb": stated[key][ranks.index(args.rank)].astype(float),
            "qnocue": nocue[key].astype(float),
        }
        masks = [~np.isnan(v[0]) for v in regimes[name].values()]
        for m in masks[1:]:
            if not np.array_equal(masks[0], m):
                raise ValueError(f"{name}: held-out row pattern differs across regimes")
        regimes[name]["mask"] = masks[0]

    def gap_curves(auc_fn):
        out = {}
        for cond in ("raw", "qb", "qnocue"):
            p, b, c = (auc_fn(regimes[r][cond], regimes[r]["mask"]) for r in ("purpose", "b2c", "c2b"))
            out[cond] = p - (b + c) / 2
        return out

    point = gap_curves(lambda S, m: np.array([roc_auc_score(y[m], S[L, m]) for L in range(n_layers)]))
    G = {k: float(TRAPZ(v, q)) for k, v in point.items()}

    by_block = {}
    for i, b in enumerate(blk):
        by_block.setdefault(b, []).append(i)
    blocks = np.array(sorted(by_block))
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.n_bootstrap, 3))
    for rep in range(args.n_bootstrap):
        idx = np.concatenate([by_block[x] for x in rng.choice(blocks, len(blocks), replace=True)])

        def auc_fn(S, mask):
            ii = idx[mask[idx]]
            return fast_auc(S[:, ii], y[ii])
        rep_gaps = gap_curves(auc_fn)
        g = {k: float(TRAPZ(v, q)) for k, v in rep_gaps.items()}
        boot[rep] = [1 - g["qb"] / g["raw"], 1 - g["qnocue"] / g["raw"],
                     (1 - g["qnocue"] / g["raw"]) - (1 - g["qb"] / g["raw"])]

    out = {
        "model": args.model, "rank": args.rank, "n_layers": n_layers,
        "n_payload_blocks": len(blocks), "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "raw_integrated_gap": G["raw"],
        "stated_qb_integrated_gap": G["qb"],
        "nocue_integrated_gap": G["qnocue"],
        "stated_qb_fraction_removed": 1 - G["qb"] / G["raw"],
        "stated_qb_fraction_removed_ci95": ci(boot[:, 0]),
        "nocue_fraction_removed": 1 - G["qnocue"] / G["raw"],
        "nocue_fraction_removed_ci95": ci(boot[:, 1]),
        "nocue_minus_stated_fraction_removed": (1 - G["qnocue"] / G["raw"]) - (1 - G["qb"] / G["raw"]),
        "nocue_minus_stated_fraction_removed_ci95": ci(boot[:, 2]),
        "nocue_relative_to_stated": (1 - G["qnocue"] / G["raw"]) / (1 - G["qb"] / G["raw"]),
    }
    path = RESULTS / f"nocue_vs_stated_subspace_compare_{args.model}_r{args.rank}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
