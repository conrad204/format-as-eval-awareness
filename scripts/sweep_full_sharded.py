"""pilot80_sweep_full.py for per-layer shards (405B via NDIF) — same analysis,
different X source, so the 405B numbers are directly comparable to the
checked-in full320-v3 8B/70B sweeps.

Identical to pilot80_sweep_full.py in every analysis detail (stated stratum
only, fg1-fg5 leave-one-vocabulary-fold-out, StandardScaler +
LogisticRegression(C=0.1, max_iter=3000, random_state=0), same regimes and
paired stats). The only difference: X for each layer is loaded on demand from
layers/layer_XXX.npz shards (bf16 bits as uint16), either from a local
directory or streamed from the HF dataset repo with `hf://<repo_id>`, so the
full [n_items, 126, 16384] matrix never has to exist in memory.

Usage:
  sweep_full_sharded.py items.jsonl hf://Conradf/v3-llama-3.1-405B-instruct out.json [n_workers]
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)

_G: dict = {}


def _list_layer_files(source: str):
    if source.startswith("hf://"):
        from huggingface_hub import HfApi

        repo = source[len("hf://"):]
        names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if re.fullmatch(r"layers/layer_\d{3}\.npz", f))
        return repo, names
    return None, sorted(Path(source).glob("layer_*.npz"))


def _load_layer(layer: int) -> np.ndarray:
    repo, files = _G["repo"], _G["layer_files"]
    if repo:
        from huggingface_hub import HfFileSystem

        # HF CDN throttles/drops streaming reads under burst load; each layer
        # is independent, so retry with backoff rather than losing the run.
        for attempt in range(6):
            try:
                with HfFileSystem().open(f"datasets/{repo}/{files[layer]}", "rb") as f:
                    d = np.load(io.BytesIO(f.read()))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"L{layer} fetch attempt {attempt + 1} failed ({exc}); retrying", flush=True)
                time.sleep(60 * (attempt + 1))
    else:
        d = np.load(files[layer])
    assert int(d["layer"]) == layer
    assert [str(x) for x in d["item_ids"]] == _G["ids_expected"], (
        f"layer {layer} item_ids do not match items jsonl order"
    )
    u16 = d["X_bf16_u16"]
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _fit_scores(X, train, test, y):
    sc = StandardScaler().fit(X[train])
    model = LogisticRegression(**PROBE).fit(sc.transform(X[train]), y[train])
    return model.decision_function(sc.transform(X[test]))


def _cells(scores, test_idx, purpose, fmt):
    out = {}
    for p in ("evaluation", "deployment"):
        for f in ("benchmark", "casual"):
            m = (purpose[test_idx] == p) & (fmt[test_idx] == f)
            if m.any():
                out[f"{p}::{f}"] = float((scores[m] > 0).mean() * 100)
    return out


def _paired(scores_full, mask, pair_key, label):
    """Fraction of matched (block, pair, format) pairs with eval > deploy."""
    got = {}
    for i in np.flatnonzero(mask & ~np.isnan(scores_full)):
        got.setdefault(pair_key[i], {})[label[i]] = scores_full[i]
    diffs = [d[1] - d[0] for d in got.values() if 0 in d and 1 in d]
    if not diffs:
        return float("nan"), float("nan")
    return float(np.mean(np.asarray(diffs) > 0)), float(np.mean(diffs))


def sweep_layer(layer: int) -> dict:
    X = _load_layer(layer)
    stated, label, fam = _G["stated"], _G["label"], _G["fam"]
    purpose, fmt, pair_key = _G["purpose"], _G["fmt"], _G["pair_key"]
    n = len(label)
    is_bench = fmt == "benchmark"
    fmt_label = is_bench.astype(int)

    train_of = {
        "confounded": stated & ((label == 1) & is_bench | (label == 0) & ~is_bench),
        "decorrelated": stated,
        "b2c": stated & is_bench,
        "c2b": stated & ~is_bench,
    }
    test_of = {
        "confounded": stated,
        "decorrelated": stated,
        "b2c": stated & ~is_bench,
        "c2b": stated & is_bench,
    }

    row: dict = {"layer": layer}
    for reg in ("confounded", "decorrelated", "b2c", "c2b"):
        scores = np.full(n, np.nan)
        for fold in FOLDS:
            train = train_of[reg] & (fam != fold)
            test = test_of[reg] & (fam == fold)
            scores[test] = _fit_scores(X, train, test, label)
        m = ~np.isnan(scores)
        pp, pm = _paired(scores, m, pair_key, label)
        row[reg] = {
            "auc": float(roc_auc_score(label[m], scores[m])),
            "bal_acc": float(balanced_accuracy_score(label[m], scores[m] > 0)),
            "paired_pos": pp,
            "paired_mean": pm,
            "cells": _cells(scores[m], m, purpose, fmt),
        }

    fscores = np.full(n, np.nan)
    for fold in FOLDS:
        train = stated & (fam != fold)
        test = stated & (fam == fold)
        fscores[test] = _fit_scores(X, train, test, fmt_label)
    m = ~np.isnan(fscores)
    row["format"] = {
        "auc": float(roc_auc_score(fmt_label[m], fscores[m])),
        "bal_acc": float(balanced_accuracy_score(fmt_label[m], fscores[m] > 0)),
    }
    print(f"L{layer:03d} conf {row['confounded']['auc']:.3f} "
          f"decorr {row['decorrelated']['auc']:.3f} "
          f"b2c {row['b2c']['auc']:.3f} c2b {row['c2b']['auc']:.3f} "
          f"fmt {row['format']['auc']:.3f}", flush=True)
    return row


def main() -> None:
    items_path, source, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else min(12, os.cpu_count() or 1)
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    _G["repo"], _G["layer_files"] = _list_layer_files(source)
    n_layers = len(_G["layer_files"])
    assert n_layers > 0, f"no layer shards found at {source}"
    _G["ids_expected"] = [r["item_id"] for r in rows]
    _G["stated"] = np.array([r["render_slot"] == "stated" for r in rows])
    _G["label"] = np.array([-1 if r["label"] is None else int(r["label"]) for r in rows])
    _G["fam"] = np.array([r["purpose_family_id"] for r in rows])
    _G["purpose"] = np.array([r["intended_purpose"] for r in rows])
    _G["fmt"] = np.array([r["format"] for r in rows])
    _G["pair_key"] = np.array(
        [f'{r["payload_block_id"]}::{r["purpose_pair_id"]}::{r["format"]}' for r in rows]
    )

    with Pool(n_workers) as pool:
        table = pool.map(sweep_layer, range(n_layers))
    out_path.write_text(json.dumps(table, indent=1) + "\n")
    best = max(table, key=lambda r: r["decorrelated"]["auc"])
    print(f"wrote {out_path} ({n_layers} layers); "
          f"decorrelated argmax L{best['layer']} auc {best['decorrelated']['auc']:.4f}")
    print("SWEEP DONE")


if __name__ == "__main__":
    main()
