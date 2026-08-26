"""Control-task null for the no-cue format contrasts.

A single permuted-label run is not a null distribution, and the ratio `rho`
is unusable under permutation because its denominator - the stated-cue effect -
collapses to zero. This builds the null properly: for S label permutations the
whole held-fold pipeline is refit and the *standardized* block-paired
contrasts are recomputed, giving an empirical distribution against which the
observed shift is scored.

Labels are permuted within each (training fold x format cell), so every
nuisance marginal the real probe could exploit survives and only the purpose
assignment is destroyed. Following Hewitt and Liang (2019), the question is
not whether the real probe beats chance but whether it beats a probe of
identical capacity fitted to a structureless target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402
from axis import _block_means, held_fold_scores  # noqa: E402
from axis import pinned  # noqa: E402
from design import load_design  # noqa: E402

FREEZE = "activations/full320-v3"


def _block_demean(scores, blocks, nb):
    means = _block_means(scores, blocks, np.ones(len(scores), dtype=bool), nb)
    return scores - means[blocks]


def contrasts(oof, none, d):
    """Standardized block-paired shifts, plus the no-cue-specific differential.

    The differential subtracts the same template contrast measured on the cued
    rows, so a per-template offset shared by cued and uncued rows cancels. It
    is the statistic the headline claim rests on, so it needs its own null.
    """
    nb = d.n_blocks
    feats = d.tpl_feature_table
    markup_t = [i for i, t in enumerate(d.tpl_names) if feats[t]["serialization"] != "plain"]
    is_markup_n = np.isin(d.none_tpl, markup_t)
    is_markup_c = np.isin(d.cued_tpl, markup_t)

    dc = _block_demean(oof, d.cued_block, nb)
    dn = _block_demean(none, d.none_block, nb)

    def smd(vals, blocks, ma, mb):
        diff = (_block_means(vals, blocks, ma, nb) - _block_means(vals, blocks, mb, nb))
        diff = diff[~np.isnan(diff)]
        return diff, float(diff.mean() / diff.std(ddof=1))

    out = {}
    cuts = (("benchmark_minus_casual", d.none_fmt == 1, d.none_fmt == 0,
             d.cued_fmt == 1, d.cued_fmt == 0),
            ("markup_minus_plain", is_markup_n, ~is_markup_n, is_markup_c, ~is_markup_c))
    for name, na, nb_, ca, cb in cuts:
        _, out[name] = smd(none, d.none_block, na, nb_)
        dn_diff, _ = smd(dn, d.none_block, na, nb_)
        dc_diff, _ = smd(dc, d.cued_block, ca, cb)
        delta = dn_diff - dc_diff
        out[name + "__differential"] = float(delta.mean() / delta.std(ddof=1))
    return out


def task_id(layer: int, draw: int) -> int:
    """Shard id for one (layer, draw) cell; draw = -1 is the observed fit."""
    return layer * 1000 + (draw + 1)


@pinned
def run_task(model, layer, draw, out_dir, stem):
    """One cell of the (layer x draw) grid — the unit that gets checkpointed.

    Checkpointing at draw level rather than layer level matters here: a layer
    is ~20 permutation fits, so a run killed mid-layer would otherwise throw
    away most of an hour.
    """
    mdir = Path(FREEZE) / model
    X = np.load(mdir / "X_all_layers.npy", mmap_mode="r")
    d = load_design(mdir / "meta.jsonl")
    Xc = np.ascontiguousarray(X[d.cued_idx, layer, :])
    Xn = np.ascontiguousarray(X[d.none_idx, layer, :])

    # Seed is layer- and draw-specific so no two cells share a permutation.
    seed = None if draw < 0 else 1_000_000 + layer * 1000 + draw
    oof, none, _ = held_fold_scores(Xc, Xn, d, permute_seed=seed)

    from sklearn.metrics import roc_auc_score
    payload = {"layer": int(layer), "draw": int(draw),
               "purpose_auc": float(roc_auc_score(d.cued_y, oof)),
               "contrasts": contrasts(oof, none, d)}
    checkpoint.write_json(out_dir, stem, task_id(layer, draw), payload)
    return payload


def assemble(out_dir, stem, layers, n_perm):
    rows = []
    for layer in layers:
        cells = checkpoint.read_json_shards(
            out_dir, stem, [task_id(layer, s) for s in range(-1, n_perm)])
        obs = next(c for c in cells if c["draw"] == -1)
        nulls = [c for c in cells if c["draw"] >= 0]
        res = {"layer": int(layer), "n_perm": len(nulls),
               "observed_purpose_auc": obs["purpose_auc"],
               "null_purpose_auc_mean": float(np.mean([c["purpose_auc"] for c in nulls])),
               "null_purpose_auc_max": float(np.max([c["purpose_auc"] for c in nulls]))}
        for k, o in obs["contrasts"].items():
            arr = np.array([c["contrasts"][k] for c in nulls])
            # Two-sided: how often does a structureless probe move the no-cue
            # rows at least as far, in either direction, as the real one?
            p = (1 + int((np.abs(arr) >= abs(o)).sum())) / (len(arr) + 1)
            res[k] = {
                "observed_smd": float(o),
                "null_mean": float(arr.mean()), "null_sd": float(arr.std(ddof=1)),
                "null_abs_max": float(np.abs(arr).max()),
                "ratio_obs_to_null_abs_max": float(abs(o) / np.abs(arr).max()),
                "p_two_sided": p,
                "null_draws": [float(v) for v in arr],
            }
        rows.append(res)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--out-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true",
                    help="discard existing per-cell checkpoints and recompute")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_permutation_null"

    cells = [(L, s) for L in args.layers for s in range(-1, args.n_perm)]
    ids = [task_id(L, s) for L, s in cells]
    # Deliberately NOT guarded on --n-perm. Cells are one-per-draw and each
    # draw is independently seeded, so raising the draw count only adds new
    # cells and leaves existing ones valid. Guarding on it would discard
    # completed draws every time the count is raised. Only the seeding
    # scheme changes what an existing shard means.
    checkpoint.guard_params(out_dir, stem, {"seed_scheme": "1e6 + layer*1000 + draw"},
                            args.force)
    todo_ids = set(checkpoint.pending(ids, out_dir, stem, args.force))
    todo = [(L, s) for (L, s), i in zip(cells, ids) if i in todo_ids]
    checkpoint.report(stem, ids, todo)

    if todo:
        Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(run_task)(args.model, L, s, str(out_dir), stem) for L, s in todo)

    path = out_dir / f"{args.model}_permutation_null.json"
    path.write_text(json.dumps({
        "header": {"model_key": args.model, "n_perm": args.n_perm,
                   "permutation": "purpose labels shuffled within (training fold x format cell)",
                   "statistic": "standardized block-paired no-cue shift (smd)",
                   "reference": "Hewitt and Liang (2019), control tasks"},
        "layers": assemble(out_dir, stem, args.layers, args.n_perm)}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
