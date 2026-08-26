"""Is the template ladder specific to *uncued* prompts, or a template offset?

The headline quantity pi(t) places each template's no-cue rows relative to the
global deployment centroid, so it is vulnerable to an obvious alternative: the
axis may simply carry a per-template offset that shifts *every* row rendered
through that template, cued rows included. If so, the ladder describes template
geometry rather than anything about how an uncued prompt is read.

This module separates the two. Within each stratum, scores are demeaned by
payload block, then averaged inside each template:

    offset_cued(t)  from the 7,680 explicitly cued rows
    offset_none(t)  from the 3,840 no-cue rows

Both are in the same axis units and both are centred at zero by construction,
so they are directly comparable. The quantity of interest is the *difference*:

    differential(t) = offset_none(t) - offset_cued(t)

A template effect that is common to cued and uncued rows cancels here. What
survives is the part of the ladder that exists only when no purpose is stated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis import _block_means  # noqa: E402
from axis import PROBE_CHOICES  # noqa: E402
import checkpoint  # noqa: E402
from design import load_design  # noqa: E402
from stats_utils import rank_exact_p  # noqa: E402

FREEZE = "activations/full320-v3"


def _block_demean(scores, blocks, n_blocks):
    means = _block_means(scores, blocks, np.ones(len(scores), dtype=bool), n_blocks)
    return scores - means[blocks]


def layer_row(oof, none, d, n_boot, seed):
    nb = d.n_blocks
    scale = (np.nanmean(_block_means(oof, d.cued_block, d.cued_y == 1, nb))
             - np.nanmean(_block_means(oof, d.cued_block, d.cued_y == 0, nb)))

    dc = _block_demean(oof, d.cued_block, nb)
    dn = _block_demean(none, d.none_block, nb)

    feats = d.tpl_feature_table
    groups = {
        "markup_minus_plain":
            np.array([feats[t]["serialization"] != "plain" for t in d.tpl_names]),
        "benchmark_minus_casual":
            np.array([feats[t]["format"] == "benchmark" for t in d.tpl_names]),
    }

    # Per-template offsets, kept at block resolution so the bootstrap can
    # resample blocks and recompute everything from the same draw.
    cued_bt, none_bt = {}, {}
    for i, t in enumerate(d.tpl_names):
        cued_bt[t] = _block_means(dc, d.cued_block, d.cued_tpl == i, nb)
        none_bt[t] = _block_means(dn, d.none_block, d.none_tpl == i, nb)

    def group_mean(tbl, mask):
        sel = [t for t, m in zip(d.tpl_names, mask) if m]
        return np.nanmean(np.stack([tbl[t] for t in sel]), axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    E = _block_means(oof, d.cued_block, d.cued_y == 1, nb)
    D = _block_means(oof, d.cued_block, d.cued_y == 0, nb)

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {}
    with np.errstate(invalid="ignore"):
        den = np.nanmean(E[idx], axis=1) - np.nanmean(D[idx], axis=1)
        for name, mask in groups.items():
            d_cued = group_mean(cued_bt, mask) - group_mean(cued_bt, ~mask)
            d_none = group_mean(none_bt, mask) - group_mean(none_bt, ~mask)
            b_cued = np.nanmean(d_cued[idx], axis=1) / den
            b_none = np.nanmean(d_none[idx], axis=1) / den
            out[name] = {
                "cued": {"point": float(np.nanmean(d_cued) / scale), "ci95": ci(b_cued)},
                "no_cue": {"point": float(np.nanmean(d_none) / scale), "ci95": ci(b_none)},
                "differential": {"point": float(np.nanmean(d_none - d_cued) / scale),
                                 "ci95": ci(b_none - b_cued)},
            }

    offs_c = {t: float(np.nanmean(v) / scale) for t, v in cued_bt.items()}
    offs_n = {t: float(np.nanmean(v) / scale) for t, v in none_bt.items()}
    r = float(np.corrcoef([offs_c[t] for t in d.tpl_names],
                          [offs_n[t] for t in d.tpl_names])[0, 1])

    # Exact rank test on the differential itself, which is the quantity the
    # headline claim rests on; the same test on raw pi lives in
    # template_analysis.py.
    diff_vec = [offs_n[t] - offs_c[t] for t in d.tpl_names]
    rank = rank_exact_p(diff_vec, groups["markup_minus_plain"])

    return out | {
        "differential_by_template": {t: offs_n[t] - offs_c[t] for t in d.tpl_names},
        "markup_rank_test_on_differential": rank,
        "offset_cued_by_template": offs_c,
        "offset_none_by_template": offs_n,
        "corr_cued_vs_none_offsets": r,
        "scale_stated": float(scale),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="*", type=int)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the output already exists")
    ap.add_argument("--probe", choices=list(PROBE_CHOICES), default="logistic",
                    help="estimator; non-default variants write to their own namespace")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    vkey = checkpoint.variant_key(args.model, args.probe)
    out_path = rd / f"{vkey}_offsets.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; pass --force to recompute")
        return 0
    z = np.load(rd / f"{vkey}_scores.npz")
    smeta = json.loads((rd / f"{vkey}_scores_meta.json").read_text())
    d = load_design(Path(FREEZE) / args.model / "meta.jsonl")

    layers = args.layers if args.layers else list(range(z["oof_cued"].shape[0]))
    rows = []
    for L in layers:
        r = layer_row(z["oof_cued"][L].astype(np.float64),
                      z["none"][L].astype(np.float64), d, args.n_boot, args.seed)
        r["layer"] = int(L)
        rows.append(r)

    path = out_path
    path.write_text(json.dumps({"header": smeta | {
        "analysis": "per-template axis offset, cued vs no-cue strata",
        "n_boot": args.n_boot, "seed": args.seed}, "layers": rows}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
