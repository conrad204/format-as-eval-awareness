"""Aggregate progress across the Direction 1 pipeline.

Each layer-wise stage drops a shard per finished layer, so counting shards
gives an exact completion figure without touching the running processes.
Cheap post-processing stages are done-or-not, judged by their output file.

    python ai_post_analysis/src/progress.py           # once
    python ai_post_analysis/src/progress.py --watch   # refresh every 30s
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402

MODELS = {"llama31_8b": 32, "llama31_70b": 80, "llama33_70b": 80}
# stem suffix -> (expected units, shard suffix); None means file-existence only
SHARDED = {"main": ".json", "scores": ".npz", "regression": ".json",
           "nuisance": ".json", "contracts": ".json"}
FILE_ONLY = {"templates": "_templates.json", "offsets": "_offsets.json",
             "cue_response": "_cue_response.json"}
CONTROL_LAYERS = 8
PERM_CELLS = 5 * 21  # 5 layers x (1 observed + 20 draws)


def bar(done: int, total: int, width: int = 22) -> str:
    if total <= 0:
        return " " * width
    n = int(round(width * done / total))
    return "#" * n + "." * (width - n)


def report(rd: Path) -> bool:
    lines = []
    total_done = total_all = 0
    for model, n_layers in MODELS.items():
        for suffix, ext in SHARDED.items():
            stem = f"{model}_{suffix}"
            done = len(checkpoint.done_layers(rd, stem, ext))
            if done == 0 and not (rd / f"{stem}.json").exists():
                if not checkpoint.shard_dir(rd, stem).exists():
                    lines.append((f"{model} {suffix}", 0, n_layers))
                    total_all += n_layers
                    continue
            lines.append((f"{model} {suffix}", min(done, n_layers), n_layers))
            total_done += min(done, n_layers)
            total_all += n_layers
        for name, fname in FILE_ONLY.items():
            ok = (rd / f"{model}{fname}").exists()
            lines.append((f"{model} {name}", int(ok), 1))
            total_done += int(ok)
            total_all += 1

    ctl = len(checkpoint.done_layers(rd, "llama31_8b_controls"))
    lines.append(("llama31_8b controls", min(ctl, CONTROL_LAYERS), CONTROL_LAYERS))
    total_done += min(ctl, CONTROL_LAYERS)
    total_all += CONTROL_LAYERS

    perm = len(checkpoint.done_layers(rd, "llama31_8b_permutation_null"))
    lines.append(("llama31_8b perm-null", min(perm, PERM_CELLS), PERM_CELLS))
    total_done += min(perm, PERM_CELLS)
    total_all += PERM_CELLS

    width = max(len(n) for n, _, _ in lines)
    for name, done, tot in lines:
        mark = "done" if done >= tot else f"{done}/{tot}"
        print(f"  {name:<{width}}  {bar(done, tot)}  {mark}")
    pct = 100 * total_done / total_all if total_all else 0
    print(f"\n  {'OVERALL':<{width}}  {bar(total_done, total_all)}  "
          f"{total_done}/{total_all} units ({pct:.1f}%)")
    return total_done >= total_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()
    rd = Path(args.results_dir)
    while True:
        print(f"\n=== Direction 1 pipeline - {time.strftime('%H:%M:%S')} ===")
        finished = report(rd)
        if not args.watch or finished:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
