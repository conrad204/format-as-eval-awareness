"""Direction 1 runner: how much implicit purpose does prompt format carry?

Stage `main` fits the decorrelated held-fold purpose axis at every layer,
projects the 3,840 no-cue items onto it, and reports the block-bootstrapped
implicit-purpose estimands.

Stage `controls` re-runs the same estimands on a preregistered normalized-depth
grid under three control conditions: format-erased activations (INLP), permuted
purpose labels, and cross-format-trained axes; it also records the
within/cross cue-vocabulary abstraction gap used to interpret the depth
profile.

The depth grid is declared here, before any control outcome is inspected.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402
from axis import (PROBE_CHOICES, abstraction_gap, block_bootstrap,  # noqa: E402
                  block_table, estimands_from_blocks, held_fold_scores,
                  paired_stats, pinned)
from design import load_design  # noqa: E402

DEPTH_GRID = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
FREEZE = "activations/full320-v3"


def repo_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_model(model: str):
    mdir = Path(FREEZE) / model
    X = np.load(mdir / "X_all_layers.npy", mmap_mode="r")
    ids = np.load(mdir / "activations_all_layers.npz")["item_ids"] if False else None
    d = load_design(mdir / "meta.jsonl", ids)
    cfg = json.loads((mdir / "config.json").read_text())
    return X, d, cfg, mdir


def layer_slices(X, d, layer):
    Xc = np.ascontiguousarray(X[d.cued_idx, layer, :])
    Xn = np.ascontiguousarray(X[d.none_idx, layer, :])
    return Xc, Xn


def summarise(oof, none, d, n_boot, seed):
    t = block_table(oof, none, d)
    tpl_keys = [k for k in t if k.startswith("none_tpl::")]
    point = estimands_from_blocks(t, tpl_keys)
    point.update(paired_stats(t))
    ci = block_bootstrap(t, n_boot=n_boot, seed=seed, keys=tpl_keys)
    return point, ci, t


@pinned
def run_layer_main(model, layer, n_boot, seed, out_dir=None, stem=None, probe="logistic"):
    X, d, _, _ = load_model(model)
    Xc, Xn = layer_slices(X, d, layer)
    t0 = time.time()
    oof, none, diag = held_fold_scores(Xc, Xn, d, probe=probe)
    point, ci, t = summarise(oof, none, d, n_boot, seed)
    from sklearn.metrics import roc_auc_score
    row = {
        "layer": int(layer),
        "purpose_auc_oof": float(roc_auc_score(d.cued_y, oof)),
        "purpose_auc_by_fold": diag["purpose_auc_by_fold"],
        "format_auc_by_fold": diag["format_auc_by_fold"],
        "point": point,
        "ci95": ci,
        "seconds": round(time.time() - t0, 1),
    }
    blocks = {"layer": int(layer), "blocks": {k: v.tolist() for k, v in t.items()}}
    if out_dir is not None:
        checkpoint.write_json(out_dir, stem, layer, {"row": row, "blocks": blocks})
    return row, blocks


@pinned
def run_layer_controls(model, layer, n_boot, seed, out_dir=None, stem=None):
    X, d, _, _ = load_model(model)
    Xc, Xn = layer_slices(X, d, layer)
    from sklearn.metrics import roc_auc_score
    out = {"layer": int(layer)}
    t0 = time.time()

    # (a) format-erased axis
    oof, none, diag = held_fold_scores(Xc, Xn, d, erase_format=True)
    point, ci, _ = summarise(oof, none, d, n_boot, seed)
    out["erased"] = {"purpose_auc_oof": float(roc_auc_score(d.cued_y, oof)),
                     "point": point, "ci95": ci,
                     "erasure": diag["erasure"]}

    # (b) permuted-purpose control task
    oof, none, diag = held_fold_scores(Xc, Xn, d, permute_seed=20260823 + layer)
    point, ci, _ = summarise(oof, none, d, n_boot, seed)
    out["permuted"] = {"purpose_auc_oof": float(roc_auc_score(d.cued_y, oof)),
                       "point": point, "ci95": ci}

    # (c) cross-format-trained axes
    for name, tf in (("train_benchmark", 1), ("train_casual", 0)):
        oof, none, diag = held_fold_scores(Xc, Xn, d, train_fmt=tf)
        te_other = d.cued_fmt != tf
        point, ci, _ = summarise(oof, none, d, n_boot, seed)
        out[name] = {
            "purpose_auc_oof": float(roc_auc_score(d.cued_y, oof)),
            "transfer_auc_other_format": float(
                roc_auc_score(d.cued_y[te_other], oof[te_other])),
            "point": point, "ci95": ci,
        }

    # (d) lexical vs abstract readout at matched training set
    out["abstraction"] = abstraction_gap(Xc, d)
    out["seconds"] = round(time.time() - t0, 1)
    if out_dir is not None:
        checkpoint.write_json(out_dir, stem, layer, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=["main", "controls"], default="main")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--layers", nargs="*", type=int)
    ap.add_argument("--out-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true",
                    help="discard existing per-layer checkpoints and recompute")
    ap.add_argument("--probe", choices=list(PROBE_CHOICES), default="logistic",
                    help="estimator; non-default variants write to their own namespace")
    args = ap.parse_args()

    X, d, cfg, mdir = load_model(args.model)
    n_layers = X.shape[1]

    if args.layers:
        layers = args.layers
    elif args.stage == "main":
        layers = list(range(n_layers))
    else:
        layers = sorted({max(0, min(n_layers - 1, int(round(q * n_layers)) - 1))
                         for q in DEPTH_GRID})

    header = {
        "model": cfg["model"],
        "model_key": args.model,
        "model_revision": cfg["model_revision"],
        "n_transformer_layers": n_layers,
        "hidden_size": int(X.shape[2]),
        "n_items": int(X.shape[0]),
        "items_sha256": cfg["items_sha256"],
        "activations_all_layers_sha256": cfg["activations_all_layers_sha256"],
        "layer_semantics": cfg["layer_semantics"],
        "position": cfg["position"],
        "stage": args.stage,
        "layers": layers,
        "depth_grid": list(DEPTH_GRID) if args.stage == "controls" else None,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_blocks": d.n_blocks,
        "n_cued": int(len(d.cued_idx)),
        "n_none": int(len(d.none_idx)),
        "probe": args.probe,
        "probe_spec": ("StandardScaler + LogisticRegression(C=0.1, max_iter=3000, random_state=0)"
                       if args.probe == "logistic" else "diagonal LDA, ridge 0.1, internal scaling"),
        "split": "leave-one-cue-vocabulary-family-out; no-cue scored by mean of 5 held-fold models",
        "bootstrap_unit": "payload_block_id",
        "repo_commit": repo_commit(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "template_features": d.tpl_feature_table,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vkey = checkpoint.variant_key(args.model, args.probe)
    stem = f"{vkey}_{args.stage}"

    checkpoint.guard_params(out_dir, stem,
                            {"n_boot": args.n_boot, "seed": args.seed}, args.force)
    todo = checkpoint.pending(layers, out_dir, stem, args.force)
    checkpoint.report(stem, layers, todo)

    fn = run_layer_main if args.stage == "main" else run_layer_controls
    t0 = time.time()
    if todo:
        Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(fn)(args.model, L, args.n_boot, args.seed, str(out_dir), stem, args.probe)
            for L in todo
        )

    # Assemble from every shard on disk, not just the layers requested this
    # invocation: a targeted `--layers 17` rerun must refresh that layer inside
    # the complete artifact rather than replace the artifact with one layer.
    have = sorted(checkpoint.done_layers(out_dir, stem))
    shards = checkpoint.read_json_shards(out_dir, stem, have)
    if args.stage == "main":
        layer_rows = [s["row"] for s in shards]
        blocks = {str(s["blocks"]["layer"]): s["blocks"]["blocks"] for s in shards}
        (out_dir / f"{stem}_blocks.json").write_text(json.dumps(blocks))
    else:
        layer_rows = shards

    payload = {"header": header | {"wall_seconds": round(time.time() - t0, 1),
                                   "layers": have},
               "layers": layer_rows}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_dir / f'{stem}.json'} ({len(layer_rows)} layers, "
          f"{round(time.time() - t0, 1)}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
