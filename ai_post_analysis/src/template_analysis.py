"""Per-template dose-response on the purpose axis.

The binary benchmark-vs-casual contrast used by prior work treats format as a
single visual style. The v3 catalog renders every payload block through 13
templates whose surface features vary independently, so the same block-level
scores that produce the binary contrast also support a per-template ordering.

This module reuses the block tables written by `run_analysis.py --stage main`
(no refitting) and computes, per layer:

  * pi(t): each template's no-cue placement in stated-cue units
  * the block-paired contrast between arbitrary template groups, expressed in
    the same units as the headline benchmark-minus-casual contrast
  * a variance decomposition of pi(t) into between- and within-format-family
    components, which asks whether the benchmark/casual label is the right
    description of what drives the shift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402
from axis import PROBE_CHOICES  # noqa: E402
from design import load_design  # noqa: E402
from stats_utils import rank_exact_p  # noqa: E402

FREEZE = "activations/full320-v3"


def group_contrast(blocks, tpls_a, tpls_b, n_boot=10000, seed=0):
    """Block-paired mean(group A) - mean(group B), in stated-cue units.

    Both groups are averaged within each payload block first, so the contrast
    is internal to a block and the bootstrap resamples blocks.
    """
    A = np.nanmean(np.stack([blocks[f"none_tpl::{t}"] for t in tpls_a]), axis=0)
    B = np.nanmean(np.stack([blocks[f"none_tpl::{t}"] for t in tpls_b]), axis=0)
    diff = A - B
    E, D = blocks["eval"], blocks["deploy"]

    nb = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    with np.errstate(invalid="ignore"):
        d_stated = np.nanmean(E[idx], axis=1) - np.nanmean(D[idx], axis=1)
        ratio = np.nanmean(diff[idx], axis=1) / d_stated
        smd = np.nanmean(diff[idx], axis=1) / np.nanstd(diff[idx], axis=1, ddof=1)
    ratio = ratio[np.isfinite(ratio)]
    smd = smd[np.isfinite(smd)]
    finite = diff[~np.isnan(diff)]
    return {
        "ratio_stated_units": float(np.nanmean(diff) / (np.nanmean(E) - np.nanmean(D))),
        "ratio_ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
        "smd_paired": float(np.nanmean(diff) / np.nanstd(finite, ddof=1)),
        "smd_ci95": [float(np.percentile(smd, 2.5)), float(np.percentile(smd, 97.5))],
        "block_win_rate": float((finite > 0).mean()),
        "n_blocks": int(len(finite)),
    }


def variance_decomposition(pi, tpl_format):
    """Split the spread of pi(t) into between-family and within-family parts."""
    pi = np.asarray(pi, dtype=float)
    fam = np.asarray(tpl_format)
    grand = pi.mean()
    between = 0.0
    within = 0.0
    for f in (0, 1):
        m = fam == f
        between += m.sum() * (pi[m].mean() - grand) ** 2
        within += ((pi[m] - pi[m].mean()) ** 2).sum()
    total = between + within
    return {"between_family": float(between), "within_family": float(within),
            "total": float(total),
            "within_share": float(within / total) if total > 0 else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the output already exists")
    ap.add_argument("--probe", choices=list(PROBE_CHOICES), default="logistic",
                    help="estimator; non-default variants write to their own namespace")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    vkey = checkpoint.variant_key(args.model, args.probe)
    out_path = rd / f"{vkey}_templates.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; pass --force to recompute")
        return 0
    main_json = json.loads((rd / f"{vkey}_main.json").read_text())
    blocks_all = json.loads((rd / f"{vkey}_main_blocks.json").read_text())
    d = load_design(Path(FREEZE) / args.model / "meta.jsonl")
    feats = main_json["header"]["template_features"]

    tpls = d.tpl_names
    markup = [t for t in tpls if feats[t]["serialization"] != "plain"]
    plain = [t for t in tpls if feats[t]["serialization"] == "plain"]
    bench = [t for t in tpls if feats[t]["format"] == "benchmark"]
    casual = [t for t in tpls if feats[t]["format"] == "casual"]
    legacy_tpls = sorted({m["template_id"] for m in d.meta if m.get("legacy")})

    out = {"header": main_json["header"] | {
        "analysis": "per-template dose-response on the purpose axis",
        "groups": {"markup_serialized": markup, "plain_serialized": plain,
                   "benchmark_family": bench, "casual_family": casual},
    }, "layers": []}

    for row in main_json["layers"]:
        L = row["layer"]
        blocks = {k: np.array(v, dtype=float) for k, v in blocks_all[str(L)].items()}
        mD = np.nanmean(blocks["deploy"])
        d_stated = np.nanmean(blocks["eval"]) - mD
        pi = {t: float((np.nanmean(blocks[f"none_tpl::{t}"]) - mD) / d_stated) for t in tpls}
        pi_vec = [pi[t] for t in tpls]
        is_markup = [feats[t]["serialization"] != "plain" for t in tpls]

        # Robustness variants. The two v1 anchors are legacy wrappers, which the
        # format lint exempts from the purpose-vocabulary rule it applies to v2
        # templates, and the catalog pre-declares bench_preamble_v2 (the
        # deliberate MMLU-fingerprint template) as excluded from pooled format
        # estimates. Both are plain-serialized, so dropping them can only work
        # against the markup contrast.
        variants = {}
        for vname, drop in (("exclude_legacy", set(legacy_tpls)),
                            ("exclude_legacy_and_preamble",
                             set(legacy_tpls) | {"bench_preamble_v2"})):
            p2 = [t for t in plain if t not in drop]
            m2 = [t for t in markup if t not in drop]
            if p2 and m2:
                variants[vname] = group_contrast(blocks, m2, p2, args.n_boot, args.seed)

        out["layers"].append({
            "layer": L,
            "purpose_auc_oof": row["purpose_auc_oof"],
            "pi_by_template": pi,
            "markup_vs_plain": group_contrast(blocks, markup, plain, args.n_boot, args.seed),
            "markup_vs_plain_robustness": variants,
            "benchmark_vs_casual_templates": group_contrast(blocks, bench, casual,
                                                            args.n_boot, args.seed),
            "variance_decomposition": variance_decomposition(pi_vec, d.tpl_format),
            "markup_rank_test": rank_exact_p(pi_vec, is_markup),
        })

    path = out_path
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({len(out['layers'])} layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
