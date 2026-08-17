"""Full-regime layer sweep for the v2 (pilot80) crossed design — the v2
analogue of sweep_full.py, so the v1 plots have honestly computed counterparts.

Differences from v1, by design (see docs/pilot_design_v2.md):
  - CV folds are the five disjoint cue-vocabulary folds (fg1-fg5), not the
    v1 purpose families: every test row's cue lexicon is unseen in training.
  - Only the stated third-person stratum is swept (it is the analogue of the
    whole v1 dataset); indirect/second-person/no-cue strata are endpoint-level
    analyses in confirmation_analysis.py, not layer sweeps.
  - Paired stats pair rows by (payload_block_id, purpose_pair_id, format),
    the v2 minimal-pair unit.

Regimes per layer (all leave-one-vocabulary-fold-out, probe recipe frozen:
StandardScaler + LogisticRegression(C=0.1, max_iter=3000, random_state=0)):
  confounded    train eval::benchmark + deploy::casual only, test all 4 cells
  decorrelated  train all 4 cells, test all 4 cells
  b2c           train benchmark rows, test casual rows
  c2b           train casual rows, test benchmark rows
  format        same splits as decorrelated but predicting format

Usage: pilot80_sweep_full.py items.jsonl activations_all_layers.npz out.json
"""

from __future__ import annotations

import json
import os
import sys
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


def _init_worker(global_state: dict) -> None:
    _G.update(global_state)

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
    X = _G["X_all"][:, layer, :].astype(np.float32)
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
    return row


def main() -> None:
    items_path, npz_path, out_path = map(Path, sys.argv[1:4])
    rows = [json.loads(l) for l in items_path.read_text().splitlines() if l.strip()]
    npz = np.load(npz_path)
    assert [str(x) for x in npz["item_ids"]] == [r["item_id"] for r in rows], (
        "npz item_ids do not match items jsonl order"
    )
    global_state = {
        "X_all": npz["X"],
        "stated": np.array([r["render_slot"] == "stated" for r in rows]),
        "label": np.array([-1 if r["label"] is None else int(r["label"]) for r in rows]),
        "fam": np.array([r["purpose_family_id"] for r in rows]),
        "purpose": np.array([r["intended_purpose"] for r in rows]),
        "fmt": np.array([r["format"] for r in rows]),
        "pair_key": np.array(
            [f'{r["payload_block_id"]}::{r["purpose_pair_id"]}::{r["format"]}' for r in rows]
        ),
    }
    _G.update(global_state)
    n_layers = _G["X_all"].shape[1]
    if os.name == "nt":
        # Spawn would pickle the multi-gigabyte activation tensor per worker.
        table = [sweep_layer(layer) for layer in range(n_layers)]
    else:
        with Pool(min(12, os.cpu_count() or 1),
                  initializer=_init_worker, initargs=(global_state,)) as pool:
            table = pool.map(sweep_layer, range(n_layers))
    out_path.write_text(json.dumps(table, indent=1) + "\n")
    best = max(table, key=lambda r: r["decorrelated"]["auc"])
    print(f"wrote {out_path} ({n_layers} layers); "
          f"decorrelated argmax L{best['layer']} auc {best['decorrelated']['auc']:.4f}")


if __name__ == "__main__":
    main()
