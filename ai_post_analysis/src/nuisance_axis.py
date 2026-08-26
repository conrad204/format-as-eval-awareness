"""Nuisance-axis control: is the no-cue format shift specific to *purpose*?

The worry behind every projection result is that markup-serialized and plain
prompts simply differ in text, so any decodable direction would separate them
and the purpose axis is not special.

`task_family` is the right comparison axis. It is a property of the payload
block, and payload blocks are crossed with every template by construction, so
task family is exactly orthogonal to template identity in the design. It is
also semantically real and well decodable, which makes it a fair stand-in for
"an arbitrary axis of comparable strength".

If the normalized markup-minus-plain shift along the task-family axis is as
large as along the purpose axis, the headline effect is generic. If it is much
smaller, the shift is specific to the purpose direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402
from axis import PROBE_KW, _block_means  # noqa: E402
from axis import pinned  # noqa: E402
from design import load_design  # noqa: E402

FREEZE = "activations/full320-v3"
CONTRAST = ("rule_application", "numeric_reasoning")
N_CV = 5


def task_family_axis(Xc, Xn, d, meta_cued, meta_none, seed=0):
    """Block-grouped CV probe for task family, then project the no-cue rows."""
    fam = np.array([m["task_family"] for m in meta_cued])
    keep = np.isin(fam, CONTRAST)
    y = (fam == CONTRAST[0]).astype(np.int8)

    # Task family is constant within a payload block, so folds must be grouped
    # by block; a deterministic permutation assigns blocks to the 5 folds.
    rng = np.random.default_rng(seed)
    block_fold = rng.permutation(d.n_blocks) % N_CV
    row_fold = block_fold[d.cued_block]

    oof = np.full(len(Xc), np.nan)
    none_by_fold = np.zeros((N_CV, len(Xn)))
    aucs = []
    for f in range(N_CV):
        tr = keep & (row_fold != f)
        te = keep & (row_fold == f)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            raise RuntimeError("degenerate task-family fold")
        sc = StandardScaler().fit(Xc[tr])
        clf = LogisticRegression(**PROBE_KW).fit(sc.transform(Xc[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(Xc[te]))
        none_by_fold[f] = clf.decision_function(sc.transform(Xn))
        aucs.append(float(roc_auc_score(y[te], oof[te])))
    return oof, none_by_fold.mean(axis=0), y, keep, aucs


@pinned
def run_layer(model, layer, n_boot, seed, out_dir=None, stem=None):
    mdir = Path(FREEZE) / model
    X = np.load(mdir / "X_all_layers.npy", mmap_mode="r")
    d = load_design(mdir / "meta.jsonl")
    Xc = np.ascontiguousarray(X[d.cued_idx, layer, :])
    Xn = np.ascontiguousarray(X[d.none_idx, layer, :])
    meta_cued = [d.meta[i] for i in d.cued_idx]
    meta_none = [d.meta[i] for i in d.none_idx]

    oof, none, y, keep, aucs = task_family_axis(Xc, Xn, d, meta_cued, meta_none, seed)

    nb = d.n_blocks
    pos = _block_means(oof, d.cued_block, keep & (y == 1), nb)
    neg = _block_means(oof, d.cued_block, keep & (y == 0), nb)
    # Task family is a block-level property, so the class means come from
    # disjoint block sets and cannot be differenced within a block.
    scale = np.nanmean(pos) - np.nanmean(neg)

    feats = d.tpl_feature_table
    markup_t = [i for i, t in enumerate(d.tpl_names) if feats[t]["serialization"] != "plain"]
    is_markup = np.isin(d.none_tpl, markup_t)

    mk = _block_means(none, d.none_block, is_markup, nb)
    pl = _block_means(none, d.none_block, ~is_markup, nb)
    bn = _block_means(none, d.none_block, d.none_fmt == 1, nb)
    cs = _block_means(none, d.none_block, d.none_fmt == 0, nb)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))

    def ratio_ci(diff):
        with np.errstate(invalid="ignore"):
            num = np.nanmean(diff[idx], axis=1)
            den = np.nanmean(pos[idx], axis=1) - np.nanmean(neg[idx], axis=1)
            r = (num / den)
        r = r[np.isfinite(r)]
        return {"ratio_task_units": float(np.nanmean(diff) / scale),
                "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}

    row = {
        "layer": int(layer),
        "task_family_auc_oof": float(np.mean(aucs)),
        "task_family_auc_by_fold": aucs,
        "scale_task_axis": float(scale),
        "markup_vs_plain": ratio_ci(mk - pl),
        "benchmark_vs_casual": ratio_ci(bn - cs),
    }
    if out_dir is not None:
        checkpoint.write_json(out_dir, stem, layer, row)
    return row

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="*", type=int)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true",
                    help="discard existing per-layer checkpoints and recompute")
    args = ap.parse_args()

    X = np.load(Path(FREEZE) / args.model / "X_all_layers.npy", mmap_mode="r")
    layers = args.layers or list(range(X.shape[1]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_nuisance"

    checkpoint.guard_params(out_dir, stem,
                            {"n_boot": args.n_boot, "seed": args.seed}, args.force)
    todo = checkpoint.pending(layers, out_dir, stem, args.force)
    checkpoint.report(stem, layers, todo)
    if todo:
        Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(run_layer)(args.model, L, args.n_boot, args.seed, str(out_dir), stem)
            for L in todo)
    have = sorted(checkpoint.done_layers(out_dir, stem))
    res = checkpoint.read_json_shards(out_dir, stem, have)

    out = {"header": {"model_key": args.model, "contrast": list(CONTRAST),
                      "n_cv_block_folds": N_CV, "n_boot": args.n_boot, "seed": args.seed,
                      "note": "task family is orthogonal to template identity by design"},
           "layers": res}
    path = Path(args.out_dir) / f"{args.model}_nuisance.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
