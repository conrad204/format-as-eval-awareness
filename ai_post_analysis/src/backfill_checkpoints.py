"""Reconstruct per-layer checkpoints from already-finished artifacts.

Checkpointing was added after the first runs completed, so the existing
`*_main.json`, `*_controls.json` and `*_scores.npz` have no shards behind them
and would be recomputed from scratch on the next run. This walks whatever
final artifacts are present and writes the shards they imply, which makes
those layers count as done.

Idempotent and safe to re-run; it only writes shards that are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402

MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")


def backfill_main(rd: Path, model: str) -> int:
    stem = f"{model}_main"
    main_p = rd / f"{stem}.json"
    blocks_p = rd / f"{stem}_blocks.json"
    if not (main_p.exists() and blocks_p.exists()):
        return 0
    main = json.loads(main_p.read_text())
    blocks = json.loads(blocks_p.read_text())
    done = checkpoint.done_layers(rd, stem)
    n = 0
    for row in main["layers"]:
        L = row["layer"]
        if L in done or str(L) not in blocks:
            continue
        checkpoint.write_json(rd, stem, L, {
            "row": row, "blocks": {"layer": L, "blocks": blocks[str(L)]}})
        n += 1
    return n


def backfill_flat(rd: Path, model: str, suffix: str) -> int:
    """Stages whose final JSON is just {'header':..., 'layers':[rows]}."""
    stem = f"{model}_{suffix}"
    p = rd / f"{stem}.json"
    if not p.exists():
        return 0
    rows = json.loads(p.read_text())["layers"]
    done = checkpoint.done_layers(rd, stem)
    n = 0
    for row in rows:
        if row["layer"] in done:
            continue
        checkpoint.write_json(rd, stem, row["layer"], row)
        n += 1
    return n


def backfill_scores(rd: Path, model: str) -> int:
    stem = f"{model}_scores"
    p = rd / f"{stem}.npz"
    meta_p = rd / f"{stem}_meta.json"
    if not p.exists():
        return 0
    z = np.load(p)
    diags = json.loads(meta_p.read_text()).get("fold_diagnostics", {}) if meta_p.exists() else {}
    done = checkpoint.done_layers(rd, stem, ".npz")
    n = 0
    for L in range(z["oof_cued"].shape[0]):
        if L in done:
            continue
        checkpoint.write_npz(rd, stem, L,
                             oof=z["oof_cued"][L], none=z["none"][L],
                             diag=np.array(json.dumps(diags.get(str(L), {}))))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    rd = Path(args.results_dir)

    total = 0
    for model in args.models:
        counts = {
            "main": backfill_main(rd, model),
            "controls": backfill_flat(rd, model, "controls"),
            "nuisance": backfill_flat(rd, model, "nuisance"),
            "regression": backfill_flat(rd, model, "regression"),
            "scores": backfill_scores(rd, model),
        }
        counts = {k: v for k, v in counts.items() if v}
        if counts:
            print(f"{model}: backfilled shards {counts}")
            total += sum(counts.values())
    print(f"total shards written: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
