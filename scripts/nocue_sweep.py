"""No-cue projection sweep for the v3 full320 design.

The stated-stratum half of every layer row is computed by code copied verbatim
from the scripts that produced the checked-in v3 sweeps — pilot80_sweep_full.py
(monolithic 8B/70B npz) and sweep_full_sharded.py (405B per-layer shards) — so
this script must reproduce results/layer_sweep_full320-v3_*.json exactly. That
equality is the regression check; verify it with scripts/verify_nocue_sweep.py.

What is added: the 3,840 `render_slot == "no_cue"` items carry no purpose label
and no vocabulary fold, so the sweeps never train on them and never score them.
Here each layer's five held-fold models are additionally applied to every no-cue
item and averaged over folds (the confirmation_analysis.py E6 convention, which
is the only defensible one for rows that belong to no fold). The question:
absent any stated purpose, does benchmark formatting alone move an item along
the purpose direction the probe learned from stated items?

Per layer, per regime, over no-cue items only:
  mean/sd of the projection, and the same split by format
  shift_sd         (mean_benchmark - mean_casual) / pooled_sd
  shift_sd_paired  the same contrast taken within payload block, then averaged
                   (blocks are the independent unit; content is held fixed)
  *_ci95           cluster bootstrap over payload blocks, shared resample draws
  format_auc       how well the purpose direction alone ranks benchmark > casual
  pct_eval         share of no-cue items landing on the "evaluation" side

Also writes <out>.scores.npz: the per-item projections behind every number
above, for all layers and regimes, so downstream analysis never has to touch
the activations again.

Usage:
  nocue_sweep.py items.jsonl <all_layers.npz | shard_dir | hf://repo> out.json [n_workers]
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
REGIMES = ("confounded", "decorrelated", "b2c", "c2b")
N_BOOT = 2000
BOOT_SEED = 0

_G: dict = {}


# --------------------------------------------------------------------------
# activation sources: monolithic npz (fp32 "X"), local shards, or hf:// shards
# --------------------------------------------------------------------------

def _list_layer_files(source: str):
    if source.startswith("hf://"):
        from huggingface_hub import HfApi

        repo = source[len("hf://"):]
        names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if re.fullmatch(r"layers/layer_\d{3}\.npz", f))
        return repo, names
    return None, sorted(Path(source).glob("layer_*.npz"))


def _load_layer(layer: int) -> np.ndarray:
    if _G["mono"]:
        return _G["X_all"][:, layer, :].astype(np.float32)

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


# --------------------------------------------------------------------------
# probe + stated-stratum statistics (semantics frozen: do not edit)
# --------------------------------------------------------------------------

def _fit_scores(X, train, test, y, extra=None):
    """Reference _fit_scores, plus an optional second block to score."""
    sc = StandardScaler().fit(X[train])
    model = LogisticRegression(**PROBE).fit(sc.transform(X[train]), y[train])
    scores = model.decision_function(sc.transform(X[test]))
    if extra is None:
        return scores
    return scores, model.decision_function(sc.transform(X[extra]))


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


# --------------------------------------------------------------------------
# no-cue projection statistics
# --------------------------------------------------------------------------

def _block_sums(s: np.ndarray):
    """Per-block (sum, count) of `s`, split by format. Blocks index rows of the
    returned arrays; the same block ordering is reused by the bootstrap."""
    nb = _G["n_blocks"]
    bi, is_b = _G["nc_block"], _G["nc_is_bench"]
    sb = np.bincount(bi[is_b], weights=s[is_b], minlength=nb)
    cb = np.bincount(bi[is_b], minlength=nb)
    sc = np.bincount(bi[~is_b], weights=s[~is_b], minlength=nb)
    cc = np.bincount(bi[~is_b], minlength=nb)
    return sb, cb, sc, cc


def _shift_stats(s: np.ndarray) -> dict:
    """Benchmark-minus-casual shift on the no-cue projection, pooled and
    block-paired, each with a cluster bootstrap CI over payload blocks.

    pooled_sd is held fixed at its observed value across bootstrap draws (the
    E6 convention): the resampling is about the mean contrast, not the scale
    used to express it."""
    is_b = _G["nc_is_bench"]
    sd = float(s.std(ddof=1))
    sb, cb, sc, cc = _block_sums(s)
    both = (cb > 0) & (cc > 0)

    pooled = (s[is_b].mean() - s[~is_b].mean()) / sd
    paired = float(np.mean(sb[both] / cb[both] - sc[both] / cc[both]) / sd)

    draws = _G["boot"]                      # (N_BOOT, n_blocks) block indices
    dsb, dcb = sb[draws].sum(1), cb[draws].sum(1)
    dsc, dcc = sc[draws].sum(1), cc[draws].sum(1)
    boot_pooled = (dsb / dcb - dsc / dcc) / sd

    per_block = np.where(both, sb / np.maximum(cb, 1) - sc / np.maximum(cc, 1), np.nan)
    drawn = per_block[draws]
    boot_paired = np.nanmean(drawn, axis=1) / sd

    return {
        "mean": float(s.mean()),
        "sd": sd,
        "mean_benchmark": float(s[is_b].mean()),
        "mean_casual": float(s[~is_b].mean()),
        "shift_sd": float(pooled),
        "shift_sd_ci95": [float(np.percentile(boot_pooled, 2.5)),
                          float(np.percentile(boot_pooled, 97.5))],
        "shift_sd_paired": paired,
        "shift_sd_paired_ci95": [float(np.percentile(boot_paired, 2.5)),
                                 float(np.percentile(boot_paired, 97.5))],
        "format_auc": float(roc_auc_score(is_b.astype(int), s)),
        "pct_eval": float((s > 0).mean() * 100),
        "pct_eval_benchmark": float((s[is_b] > 0).mean() * 100),
        "pct_eval_casual": float((s[~is_b] > 0).mean() * 100),
    }


def sweep_layer(layer: int):
    X = _load_layer(layer)
    stated, label, fam = _G["stated"], _G["label"], _G["fam"]
    purpose, fmt, pair_key = _G["purpose"], _G["fmt"], _G["pair_key"]
    nocue = _G["nocue"]
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

    row: dict = {"layer": layer, "nocue": {}}
    oof_out, nc_out = {}, {}
    for reg in REGIMES:
        scores = np.full(n, np.nan)
        nc_folds = []
        for fold in FOLDS:
            train = train_of[reg] & (fam != fold)
            test = test_of[reg] & (fam == fold)
            s_test, s_nc = _fit_scores(X, train, test, label, extra=nocue)
            scores[test] = s_test
            nc_folds.append(s_nc)
        m = ~np.isnan(scores)
        pp, pm = _paired(scores, m, pair_key, label)
        row[reg] = {
            "auc": float(roc_auc_score(label[m], scores[m])),
            "bal_acc": float(balanced_accuracy_score(label[m], scores[m] > 0)),
            "paired_pos": pp,
            "paired_mean": pm,
            "cells": _cells(scores[m], m, purpose, fmt),
        }
        # No-cue rows belong to no vocabulary fold, so no single held-fold model
        # owns them: average the five, as confirmation_analysis.py E6 does.
        nc = np.mean(nc_folds, axis=0)
        row["nocue"][reg] = _shift_stats(nc)
        oof_out[reg], nc_out[reg] = scores.astype(np.float32), nc.astype(np.float32)

    fscores = np.full(n, np.nan)
    nc_folds = []
    for fold in FOLDS:
        train = stated & (fam != fold)
        test = stated & (fam == fold)
        s_test, s_nc = _fit_scores(X, train, test, fmt_label, extra=nocue)
        fscores[test] = s_test
        nc_folds.append(s_nc)
    m = ~np.isnan(fscores)
    row["format"] = {
        "auc": float(roc_auc_score(fmt_label[m], fscores[m])),
        "bal_acc": float(balanced_accuracy_score(fmt_label[m], fscores[m] > 0)),
    }
    # Sanity anchor: the format direction must separate no-cue benchmark from
    # no-cue casual. If it does not, the projection machinery is broken.
    nc = np.mean(nc_folds, axis=0)
    row["nocue"]["format"] = _shift_stats(nc)
    oof_out["format"], nc_out["format"] = fscores.astype(np.float32), nc.astype(np.float32)

    d = row["nocue"]["decorrelated"]
    print(f"L{layer:03d} decorr {row['decorrelated']['auc']:.3f} "
          f"fmt {row['format']['auc']:.3f} | nocue shift {d['shift_sd']:+.3f} sd "
          f"[{d['shift_sd_ci95'][0]:+.3f},{d['shift_sd_ci95'][1]:+.3f}] "
          f"fmt_auc {d['format_auc']:.3f} eval% b{d['pct_eval_benchmark']:.1f}/"
          f"c{d['pct_eval_casual']:.1f}", flush=True)
    return row, oof_out, nc_out


def main() -> None:
    items_path, source, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else min(12, os.cpu_count() or 1)
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    _G["ids_expected"] = [r["item_id"] for r in rows]

    mono = not source.startswith("hf://") and Path(source).is_file()
    _G["mono"] = mono
    if mono:
        npz = np.load(source)
        assert [str(x) for x in npz["item_ids"]] == _G["ids_expected"], (
            "npz item_ids do not match items jsonl order"
        )
        _G["X_all"] = npz["X"]
        n_layers = _G["X_all"].shape[1]
    else:
        _G["repo"], _G["layer_files"] = _list_layer_files(source)
        n_layers = len(_G["layer_files"])
        assert n_layers > 0, f"no layer shards found at {source}"

    _G["stated"] = np.array([r["render_slot"] == "stated" for r in rows])
    _G["nocue"] = np.array([r["render_slot"] == "no_cue" for r in rows])
    _G["label"] = np.array([-1 if r["label"] is None else int(r["label"]) for r in rows])
    _G["fam"] = np.array([r["purpose_family_id"] for r in rows])
    _G["purpose"] = np.array([r["intended_purpose"] for r in rows])
    _G["fmt"] = np.array([r["format"] for r in rows])
    _G["pair_key"] = np.array(
        [f'{r["payload_block_id"]}::{r["purpose_pair_id"]}::{r["format"]}' for r in rows]
    )
    assert _G["stated"].sum() + _G["nocue"].sum() == len(rows), "unexpected render_slot"

    nc_blocks = np.array([r["payload_block_id"] for r in rows])[_G["nocue"]]
    uniq, _G["nc_block"] = np.unique(nc_blocks, return_inverse=True)
    _G["n_blocks"] = len(uniq)
    _G["nc_is_bench"] = _G["fmt"][_G["nocue"]] == "benchmark"
    # One shared set of block resamples for every layer and regime, so CIs are
    # driven by the scores rather than by bootstrap noise varying between them.
    _G["boot"] = np.random.default_rng(BOOT_SEED).integers(
        0, _G["n_blocks"], size=(N_BOOT, _G["n_blocks"]), dtype=np.int32
    )
    print(f"{len(rows)} items: {_G['stated'].sum()} stated, {_G['nocue'].sum()} no_cue "
          f"({_G['nc_is_bench'].sum()} benchmark / {(~_G['nc_is_bench']).sum()} casual) "
          f"over {_G['n_blocks']} payload blocks; {n_layers} layers, {n_workers} workers",
          flush=True)

    with Pool(n_workers) as pool:
        out = pool.map(sweep_layer, range(n_layers))
    table = [r for r, _, _ in out]
    out_path.write_text(json.dumps(table, indent=1) + "\n")

    arrays = {"item_ids": np.array(_G["ids_expected"]),
              "nocue_item_ids": np.array(_G["ids_expected"])[_G["nocue"]],
              "layers": np.arange(n_layers, dtype=np.int16)}
    for reg in REGIMES + ("format",):
        arrays[f"stated_oof::{reg}"] = np.stack([o[reg] for _, o, _ in out])
        arrays[f"nocue::{reg}"] = np.stack([c[reg] for _, _, c in out])
    scores_path = out_path.with_suffix(".scores.npz")
    np.savez_compressed(scores_path, **arrays)

    best = max(table, key=lambda r: r["decorrelated"]["auc"])
    peak = max(table, key=lambda r: abs(r["nocue"]["decorrelated"]["shift_sd"]))
    print(f"wrote {out_path} ({n_layers} layers) + {scores_path}")
    print(f"decorrelated argmax L{best['layer']} auc {best['decorrelated']['auc']:.4f}; "
          f"no-cue shift at that layer "
          f"{best['nocue']['decorrelated']['shift_sd']:+.3f} sd")
    print(f"largest |no-cue shift| L{peak['layer']} "
          f"{peak['nocue']['decorrelated']['shift_sd']:+.3f} sd")
    print("SWEEP DONE")


if __name__ == "__main__":
    main()
