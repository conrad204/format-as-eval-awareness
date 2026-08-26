"""Does the wrapper change how strongly an *explicit* purpose cue registers?

The no-cue analysis asks where an uncued prompt sits on the purpose axis. This
asks a separate question of the same scores: within each template, how far
apart are the byte-identical evaluation and deployment twins?

    delta(t) = mean_block[ score(eval | block, t) - score(deployment | block, t) ]

Each block contributes exactly one evaluation row and one deployment row per
template, so delta(t) is a paired one-sentence intervention measured inside a
single wrapper. Reported relative to the pooled stated-cue effect, it says
whether a wrapper amplifies or damps the model's response to being told what
the response is for.

This is an independent readout of the same hypothesis as the no-cue ladder: if
markup serialization genuinely suppresses purpose representation, it should
show up here too, on rows that were part of the probe's own training
distribution.
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

FREEZE = "activations/full320-v3"


def layer_row(oof, d, n_boot, seed):
    nb = d.n_blocks
    E_all = _block_means(oof, d.cued_block, d.cued_y == 1, nb)
    D_all = _block_means(oof, d.cued_block, d.cued_y == 0, nb)
    scale = np.nanmean(E_all) - np.nanmean(D_all)

    per_tpl = {}
    per_tpl_d = {}
    diffs = {}
    sds = {}
    for i, t in enumerate(d.tpl_names):
        m = d.cued_tpl == i
        e = _block_means(oof, d.cued_block, m & (d.cued_y == 1), nb)
        p = _block_means(oof, d.cued_block, m & (d.cued_y == 0), nb)
        diffs[t] = e - p
        per_tpl[t] = float(np.nanmean(e - p) / scale)
        # Templates differ in how far apart they spread rows along the axis, and
        # a wider spread inflates the raw difference without meaning the cue
        # registered more strongly. The pooled within-class SD inside the
        # template removes that, turning the readout into a Cohen's d.
        sds[t] = float(np.sqrt(0.5 * (oof[m & (d.cued_y == 1)].var(ddof=1)
                                      + oof[m & (d.cued_y == 0)].var(ddof=1))))
        per_tpl_d[t] = float(np.nanmean(e - p) / sds[t])

    feats = d.tpl_feature_table
    markup = [t for t in d.tpl_names if feats[t]["serialization"] != "plain"]
    plain = [t for t in d.tpl_names if feats[t]["serialization"] == "plain"]
    bench = [t for t in d.tpl_names if feats[t]["format"] == "benchmark"]
    casual = [t for t in d.tpl_names if feats[t]["format"] == "casual"]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))

    def contrast(a, b):
        A = np.nanmean(np.stack([diffs[t] for t in a]), axis=0)
        B = np.nanmean(np.stack([diffs[t] for t in b]), axis=0)
        dd = A - B
        # Scale-normalized version: each template's difference is divided by its
        # own within-class SD before the groups are compared.
        An = np.nanmean(np.stack([diffs[t] / sds[t] for t in a]), axis=0)
        Bn = np.nanmean(np.stack([diffs[t] / sds[t] for t in b]), axis=0)
        ddn = An - Bn
        with np.errstate(invalid="ignore"):
            den = np.nanmean(E_all[idx], axis=1) - np.nanmean(D_all[idx], axis=1)
            r = np.nanmean(dd[idx], axis=1) / den
            rn = np.nanmean(ddn[idx], axis=1)
        r = r[np.isfinite(r)]
        rn = rn[np.isfinite(rn)]
        return {"ratio_stated_units": float(np.nanmean(dd) / scale),
                "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                "cohens_d_difference": float(np.nanmean(ddn)),
                "cohens_d_ci95": [float(np.percentile(rn, 2.5)), float(np.percentile(rn, 97.5))],
                "block_win_rate": float((dd[~np.isnan(dd)] > 0).mean())}

    return {
        "relative_cue_response_by_template": per_tpl,
        "cue_response_cohens_d_by_template": per_tpl_d,
        "markup_vs_plain": contrast(markup, plain),
        "benchmark_vs_casual": contrast(bench, casual),
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
    out_path = rd / f"{vkey}_cue_response.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; pass --force to recompute")
        return 0
    z = np.load(rd / f"{vkey}_scores.npz")
    smeta = json.loads((rd / f"{vkey}_scores_meta.json").read_text())
    d = load_design(Path(FREEZE) / args.model / "meta.jsonl")

    layers = args.layers if args.layers else list(range(z["oof_cued"].shape[0]))
    rows = []
    for L in layers:
        r = layer_row(z["oof_cued"][L].astype(np.float64), d, args.n_boot, args.seed)
        r["layer"] = int(L)
        rows.append(r)

    path = out_path
    path.write_text(json.dumps({"header": smeta | {"n_boot": args.n_boot, "seed": args.seed},
                                "layers": rows}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
