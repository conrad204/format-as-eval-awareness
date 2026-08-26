"""Per-layer checkpointing so an interrupted run resumes instead of restarting.

Every expensive stage in this folder is a map over layers. Each finished layer
is written immediately as its own shard under

    results/_partial/<stem>/<layer>.json      (or .npz for array payloads)

and the final combined artifact is assembled from the shards once all layers
are present. Re-running a stage therefore recomputes only the layers whose
shard is missing; `--force` clears the shards first and starts over.

Shards are written by the worker that produced them, so a run killed part-way
through keeps every layer that had already completed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

PARTIAL_DIRNAME = "_partial"


def shard_dir(out_dir: str | Path, stem: str) -> Path:
    return Path(out_dir) / PARTIAL_DIRNAME / stem


def shard_path(out_dir: str | Path, stem: str, layer: int, suffix: str = ".json") -> Path:
    return shard_dir(out_dir, stem) / f"{layer:04d}{suffix}"


def clear(out_dir: str | Path, stem: str) -> None:
    d = shard_dir(out_dir, stem)
    if d.exists():
        shutil.rmtree(d)


def write_json(out_dir: str | Path, stem: str, layer: int, payload) -> None:
    p = shard_path(out_dir, stem, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)  # atomic, so a kill mid-write cannot leave a truncated shard


def write_npz(out_dir: str | Path, stem: str, layer: int, **arrays) -> None:
    p = shard_path(out_dir, stem, layer, ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" unless the name already ends with it, so the
    # temp name has to keep that extension or the rename target won't exist.
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(p)


def done_layers(out_dir: str | Path, stem: str, suffix: str = ".json") -> set[int]:
    d = shard_dir(out_dir, stem)
    if not d.exists():
        return set()
    out = set()
    for p in d.glob(f"*{suffix}"):
        try:
            out.add(int(p.name.split(".")[0]))
        except ValueError:
            continue
    return out


def pending(layers, out_dir: str | Path, stem: str, force: bool = False,
            suffix: str = ".json") -> list[int]:
    if force:
        clear(out_dir, stem)
        return list(layers)
    done = done_layers(out_dir, stem, suffix)
    return [L for L in layers if L not in done]


def read_json_shards(out_dir: str | Path, stem: str, layers) -> list:
    return [json.loads(shard_path(out_dir, stem, L).read_text()) for L in sorted(layers)]


def read_npz_shards(out_dir: str | Path, stem: str, layers, key: str):
    return [np.load(shard_path(out_dir, stem, L, ".npz"))[key] for L in sorted(layers)]


def report(stem: str, layers, todo) -> None:
    skipped = len(layers) - len(todo)
    if skipped:
        print(f"[{stem}] resuming: {skipped}/{len(layers)} layers already done, "
              f"{len(todo)} to compute", flush=True)
    else:
        print(f"[{stem}] computing all {len(layers)} layers", flush=True)


def guard_params(out_dir: str | Path, stem: str, params: dict, force: bool = False) -> bool:
    """Invalidate shards that were produced under different run parameters.

    Without this, re-running a stage with a different `--n-boot` (or seed)
    silently mixes shards computed under both settings into one artifact. The
    parameters are recorded next to the shards and compared on every run;
    a mismatch clears the directory so the stage recomputes cleanly.

    Returns True if existing shards were discarded.
    """
    d = shard_dir(out_dir, stem)
    p = d / "params.json"
    if force:
        clear(out_dir, stem)
        d.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(params, sort_keys=True))
        return False
    if p.exists():
        if json.loads(p.read_text()) == params:
            return False
        print(f"[{stem}] run parameters changed; discarding existing shards", flush=True)
        clear(out_dir, stem)
    d.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params, sort_keys=True))
    return True


def variant_key(model: str, probe: str) -> str:
    """Output namespace for a probe variant.

    The frozen logistic recipe keeps the bare model name so existing artifacts
    and filenames are unchanged; any other estimator gets its own namespace so
    the two never overwrite one another.
    """
    return model if probe == "logistic" else f"{model}_{probe}"
