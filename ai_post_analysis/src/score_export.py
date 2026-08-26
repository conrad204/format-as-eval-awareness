"""Export item-level purpose-axis scores once, so analyses need no refitting.

Writes, per model, an .npz holding

  oof_cued [n_layers, n_cued]   out-of-fold decision values for cued items
  none     [n_layers, n_none]   no-cue decision values (mean of the 5 held-fold
                                models, per the frozen plan)

Every downstream estimand - block tables, template contrasts, the fixed-effects
regressions - is a function of these arrays plus the design frame, so the fits
happen exactly once per model.
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
from axis import held_fold_scores  # noqa: E402
from axis import PROBE_CHOICES  # noqa: E402
from axis import pinned  # noqa: E402
from design import load_design  # noqa: E402

FREEZE = "activations/full320-v3"


@pinned
def one_layer(model, layer, out_dir, stem, probe="logistic"):
    mdir = Path(FREEZE) / model
    X = np.load(mdir / "X_all_layers.npy", mmap_mode="r")
    d = load_design(mdir / "meta.jsonl")
    Xc = np.ascontiguousarray(X[d.cued_idx, layer, :])
    Xn = np.ascontiguousarray(X[d.none_idx, layer, :])
    oof, none, diag = held_fold_scores(Xc, Xn, d, probe=probe)
    checkpoint.write_npz(out_dir, stem, layer,
                         oof=oof.astype(np.float32), none=none.astype(np.float32),
                         diag=np.array(json.dumps(diag)))
    return layer

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--out-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true",
                    help="discard existing per-layer checkpoints and recompute")
    ap.add_argument("--probe", choices=list(PROBE_CHOICES), default="logistic",
                    help="estimator; non-default variants write to their own namespace")
    args = ap.parse_args()

    mdir = Path(FREEZE) / args.model
    X = np.load(mdir / "X_all_layers.npy", mmap_mode="r")
    d = load_design(mdir / "meta.jsonl")
    cfg = json.loads((mdir / "config.json").read_text())
    n_layers = X.shape[1]
    layers = list(range(n_layers))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vkey = checkpoint.variant_key(args.model, args.probe)
    stem = f"{vkey}_scores"

    checkpoint.guard_params(out_dir, stem, {"probe": "frozen-recipe"}, args.force)
    todo = checkpoint.pending(layers, out_dir, stem, args.force, suffix=".npz")
    checkpoint.report(stem, layers, todo)
    if todo:
        Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(one_layer)(args.model, L, str(out_dir), stem, args.probe) for L in todo)

    oof = np.stack(checkpoint.read_npz_shards(out_dir, stem, layers, "oof"))
    none = np.stack(checkpoint.read_npz_shards(out_dir, stem, layers, "none"))
    diags = {str(L): json.loads(str(np.load(
        checkpoint.shard_path(out_dir, stem, L, ".npz"))["diag"])) for L in layers}

    tok = np.load(mdir / "activations_all_layers.npz")["token_lengths"]

    out = Path(args.out_dir) / f"{vkey}_scores.npz"
    np.savez_compressed(
        out, oof_cued=oof, none=none,
        cued_idx=d.cued_idx, none_idx=d.none_idx,
        token_lengths=tok.astype(np.int32),
        layers=np.arange(n_layers, dtype=np.int16),
    )
    (Path(args.out_dir) / f"{vkey}_scores_meta.json").write_text(json.dumps({
        "model": cfg["model"], "model_key": args.model, "probe": args.probe,
        "model_revision": cfg["model_revision"],
        "activations_all_layers_sha256": cfg["activations_all_layers_sha256"],
        "n_layers": n_layers, "n_cued": int(len(d.cued_idx)), "n_none": int(len(d.none_idx)),
        "probe": "StandardScaler + LogisticRegression(C=0.1, max_iter=3000, random_state=0)",
        "split": "leave-one-cue-vocabulary-family-out; no-cue = mean of 5 held-fold models",
        "fold_diagnostics": diags,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
