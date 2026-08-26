"""Cross-dataset decomposition of Devbunova's eval-awareness signal.

Frozen design: docs/devbunova_decomposition_plan.md — every analytic choice is
pre-registered there; deviations must be recorded in the output JSON. Per
model this script

1. rebuilds the full320-v3 stated-stratum probes at the two frozen layers with
   code copied verbatim from nocue_sweep.py / pilot80_sweep_full.py, and gates
   on exact agreement with the checked-in layer-sweep record (G1) — the same
   guarantee verify_nocue_sweep.py provides for the no-cue sweep;
2. streams Devbunova's activations (Conradf/devbunova-2x2-activations,
   <model>/<regime>/chunks/, bf16 bits in uint16) into per-layer fp32 slices
   cached under activations/devbunova-2x2/<model>/<regime>/;
3. scores every Devbunova item with the fold-averaged purpose_decorr /
   purpose_conf / format probes (five held-fold models averaged — the E6
   convention for rows that belong to no fold) in both regimes;
4. trains her-protocol probe per regime/layer (StandardScaler +
   LogisticRegression(max_iter=1000, random_state=42), bench_eval = class 0 vs
   casual_deploy_matched = class 1, per external/eval-awareness-format/
   scripts/evaluate_probe_cross.py; sign flipped so positive = eval) and gates
   on the format trap reproducing on 8b raw (G2, 8b only);
5. writes endpoints P1-P4 with bootstrap CIs, full provenance, and the
   persisted fold models (coef/intercept/scaler) to results/devbunova_decomp/.

Sign convention everywhere in the outputs: score > 0 means "called eval"
(resp. "called benchmark" for the format probe).

Usage: devbunova_decomposition.py <8b|70b|405b> [our_activations_source] [n_workers]
  our_activations_source: monolithic npz path, shard dir, or hf://repo
  (defaults per model below).
"""

from __future__ import annotations

import io
import json
import os
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_format_mvp.io import sha256_file  # noqa: E402

OUR_ITEMS = ROOT / "data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl"
HER_ITEMS = ROOT / "data/devbunova-2x2/items_devbunova_2x2.jsonl"
HER_REPO = "Conradf/devbunova-2x2-activations"
HER_REGIMES = ("template", "raw")
OUT_DIR = ROOT / "results/devbunova_decomp"
CACHE_DIR = ROOT / "activations/devbunova-2x2"

# model_key -> (n_layers, hidden, (primary layer, robustness layer),
#               default our-activations source, reference sweep json)
MODELS = {
    "8b": (32, 4096, (8, 16),
           "activations/full320-v3/llama31_8b/activations_all_layers.npz",
           "results/layer_sweep_full320-v3_llama31_8b.json"),
    "70b": (80, 8192, (18, 40),
            "activations/full320-v3/llama31_70b/activations_all_layers.npz",
            "results/layer_sweep_full320-v3_llama31_70b.json"),
    "405b": (126, 16384, (37, 63),
             "hf://Conradf/v3-llama-3.1-405B-instruct",
             "results/layer_sweep_full320-v3_llama31_405b_ndif.json"),
}

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
REGIMES = ("confounded", "decorrelated", "b2c", "c2b")
HER_SCORED = ("confounded", "decorrelated", "format")  # probes projected onto her data
N_BOOT = 2000
BOOT_SEED = 0
# G1 tolerance. Plan default: bit-identity (1e-9 on every leaf), which holds on
# the reference environment class (RunPod EPYC/Zen2, numpy 2.5.2, sklearn 1.9.0
# — where verify_nocue_sweep.py measured 0.0). DEVB_G1_MODE=local-fp-jitter is
# the recorded amendment for other microarchitectures (see plan doc): lbfgs
# coefficient jitter of ~1e-6 leaves continuous leaves within 1e-5 but lets a
# handful of near-zero decision scores flip sign, moving the discrete leaves
# (cells %, bal_acc, paired_pos) in single-item steps. The tolerances below
# bound that jitter while still catching any logic error (which shifts AUCs by
# >= 1e-3). The mode + per-leaf-class deviation profile land in the output JSON.
G1_MODE = os.environ.get("DEVB_G1_MODE", "reference")
if G1_MODE == "reference":
    # bit-identity: every leaf, every regime
    G1_GATED_REGIMES = REGIMES_ALL = ("confounded", "decorrelated", "b2c",
                                      "c2b", "format")
    G1_TOL = dict.fromkeys(
        ("auc", "bal_acc", "paired_pos", "paired_mean", "cells"), 1e-9)
else:
    # local-fp-jitter: gate only the probes the endpoints project with
    # (confounded / decorrelated / format); the b2c/c2b transfer probes are
    # never applied to her data and their near-separable fits amplify FP
    # jitter into ||w|| (paired_mean moves in raw score units) — they are
    # reported informationally. paired_mean is unscaled, so it is reported
    # but not gated; discrimination is pinned by auc/bal_acc/paired_pos/cells.
    # Tolerances set once from the complete measured 8B profile (plan doc
    # amendment table): observed jitter <= 3.1e-4 auc / 1.2e-3 bal_acc /
    # 3.4e-1 cells across all regimes; a logic error moves auc by >= 1e-2.
    G1_GATED_REGIMES = ("confounded", "decorrelated", "format")
    G1_TOL = {"auc": 1e-3, "bal_acc": 5e-3, "paired_pos": 5e-3,
              "paired_mean": float("inf"), "cells": 1.0}

_G: dict = {}


# --------------------------------------------------------------------------
# our activations: monolithic npz (fp32 "X"), local shards, or hf:// shards
# (copied from nocue_sweep.py)
# --------------------------------------------------------------------------

def _hf_read(path: str) -> bytes:
    from huggingface_hub import HfFileSystem

    for attempt in range(6):
        try:
            with HfFileSystem().open(path, "rb") as f:
                return f.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"{path} fetch attempt {attempt + 1} failed ({exc}); retrying",
                  flush=True)
            time.sleep(60 * (attempt + 1))
    raise RuntimeError("unreachable")


def _load_our_layer(source: str, layer: int, ids_expected: list[str]) -> np.ndarray:
    if not source.startswith("hf://") and Path(source).is_file():
        if "X_all" not in _G:
            npz = np.load(source)
            assert [str(x) for x in npz["item_ids"]] == ids_expected, (
                "npz item_ids do not match items jsonl order"
            )
            _G["X_all"] = npz["X"]
        return _G["X_all"][:, layer, :].astype(np.float32)

    name = f"layers/layer_{layer:03d}.npz"
    if source.startswith("hf://"):
        d = np.load(io.BytesIO(_hf_read(f"datasets/{source[5:]}/{name}")))
    else:
        d = np.load(Path(source) / f"layer_{layer:03d}.npz")
    assert int(d["layer"]) == layer
    assert [str(x) for x in d["item_ids"]] == ids_expected, (
        f"layer {layer} item_ids do not match items jsonl order"
    )
    if "X_fp32" in getattr(d, "files", ()):  # npz_stream_slice.py output
        return d["X_fp32"]
    u16 = d["X_bf16_u16"]
    return (u16.astype(np.uint32) << 16).view(np.float32)


# --------------------------------------------------------------------------
# probe + stated-stratum statistics (semantics frozen: do not edit)
# (copied from nocue_sweep.py / pilot80_sweep_full.py)
# --------------------------------------------------------------------------

def _cells(scores, test_idx, purpose, fmt):
    out = {}
    for p in ("evaluation", "deployment"):
        for f in ("benchmark", "casual"):
            m = (purpose[test_idx] == p) & (fmt[test_idx] == f)
            if m.any():
                out[f"{p}::{f}"] = float((scores[m] > 0).mean() * 100)
    return out


def _paired(scores_full, mask, pair_key, label):
    got = {}
    for i in np.flatnonzero(mask & ~np.isnan(scores_full)):
        got.setdefault(pair_key[i], {})[label[i]] = scores_full[i]
    diffs = [d[1] - d[0] for d in got.values() if 0 in d and 1 in d]
    if not diffs:
        return float("nan"), float("nan")
    return float(np.mean(np.asarray(diffs) > 0)), float(np.mean(diffs))


def _masks(reg: str, fold: str):
    stated, label, fam = _G["stated"], _G["label"], _G["fam"]
    is_bench = _G["fmt"] == "benchmark"
    if reg == "format":
        train, test = stated & (fam != fold), stated & (fam == fold)
        y = is_bench.astype(int)
        return train, test, y
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
    return train_of[reg] & (fam != fold), test_of[reg] & (fam == fold), label


def _fit_job(args):
    """One (layer, regime, fold) probe: OOF scores on our stated stratum, plus
    decision scores on every Devbunova item in both regimes (E6 inputs)."""
    layer, reg, fold = args
    X = _G["X"][layer]
    train, test, y = _masks(reg, fold)
    sc = StandardScaler().fit(X[train])
    model = LogisticRegression(**PROBE).fit(sc.transform(X[train]), y[train])
    scores = model.decision_function(sc.transform(X[test]))
    her = {}
    if reg in HER_SCORED:
        for hreg in _G["her_regimes"]:
            her[hreg] = model.decision_function(
                sc.transform(_G["her"][(hreg, layer)])).astype(np.float32)
    return dict(layer=layer, reg=reg, fold=fold, test=np.flatnonzero(test),
                oof=scores.astype(np.float64), her=her,
                coef=model.coef_[0].astype(np.float32),
                intercept=float(model.intercept_[0]),
                mean=sc.mean_.astype(np.float32),
                scale=sc.scale_.astype(np.float32))


def _layer_record(layer: int, fits: list[dict]) -> dict:
    """Assemble the sweep-JSON layer record from the fold fits (G1 shape)."""
    n = len(_G["label"])
    label, fmt = _G["label"], _G["fmt"]
    fmt_label = (fmt == "benchmark").astype(int)
    row: dict = {"layer": layer}
    for reg in REGIMES + ("format",):
        scores = np.full(n, np.nan)
        for f in fits:
            if f["layer"] == layer and f["reg"] == reg:
                scores[f["test"]] = f["oof"]
        m = ~np.isnan(scores)
        yy = fmt_label if reg == "format" else label
        row[reg] = {
            "auc": float(roc_auc_score(yy[m], scores[m])),
            "bal_acc": float(balanced_accuracy_score(yy[m], scores[m] > 0)),
        }
        if reg != "format":
            pp, pm = _paired(scores, m, _G["pair_key"], label)
            row[reg].update(paired_pos=pp, paired_mean=pm,
                            cells=_cells(scores[m], m, _G["purpose"], fmt))
        row.setdefault("_oof", {})[reg] = scores
    return row


def _compare(new, ref, path="", worst=None, leaf_class=None):
    """Collect the worst deviation per leaf class vs the reference record
    (gated against G1_TOL by the caller, after the full profile is known)."""
    if worst is None:
        worst = {}
    if isinstance(ref, dict):
        for k in ref:
            assert k in new, f"G1: missing {path}/{k}"
            _compare(new[k], ref[k], f"{path}/{k}", worst,
                     k if leaf_class is None else leaf_class)
    elif isinstance(ref, float):
        d = abs(new - ref)
        if d > worst.get(leaf_class, (0.0, ""))[0]:
            worst[leaf_class] = (d, path)
    else:
        assert new == ref, f"G1: {path}: {new} != {ref}"
    return worst


# --------------------------------------------------------------------------
# Devbunova activations: chunk streaming with a per-layer fp32 cache
# --------------------------------------------------------------------------

def her_layer_slices(model_key: str, regime: str, layers: tuple[int, ...],
                     her_ids: list[str], hidden: int):
    d = CACHE_DIR / model_key / regime
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "cache_meta.json"
    missing = [L for L in layers if not (d / f"layer_{L:03d}.npy").exists()]
    if not missing and meta_p.exists():
        meta = json.loads(meta_p.read_text())
    else:
        from huggingface_hub import HfApi

        sha = HfApi().dataset_info(HER_REPO).sha
        want = missing or list(layers)
        out = {L: np.empty((len(her_ids), hidden), np.float32) for L in want}
        pos = 0
        for c in range(133):
            raw = _hf_read(f"datasets/{HER_REPO}/{model_key}/{regime}/chunks/chunk_{c:04d}.npz")
            data = np.load(io.BytesIO(raw))
            ids = [str(x) for x in data["item_ids"]]
            n = len(ids)
            assert ids == her_ids[pos:pos + n], (
                f"{model_key}/{regime} chunk {c}: item_ids do not match jsonl order"
            )
            u16 = data["X_bf16_u16"]
            assert u16.shape[1:] == (_G["n_layers"], hidden)
            for L in want:
                out[L][pos:pos + n] = (
                    u16[:, L, :].astype(np.uint32) << 16).view(np.float32)
            pos += n
            if c % 20 == 0:
                print(f"  {model_key}/{regime}: chunk {c}/132", flush=True)
        assert pos == len(her_ids)
        for L in want:
            np.save(d / f"layer_{L:03d}.npy", out[L])
        meta = {"repo": HER_REPO, "revision": sha, "n_items": pos,
                "layers": sorted(set(json.loads(meta_p.read_text())["layers"]
                                     if meta_p.exists() else []) | set(want)),
                "alignment": "item_ids verified against adapter jsonl row order"}
        meta_p.write_text(json.dumps(meta, indent=1) + "\n")
    arrs = {L: np.load(d / f"layer_{L:03d}.npy") for L in layers}
    for L in layers:
        assert arrs[L].shape == (len(her_ids), hidden)
    return arrs, meta


# --------------------------------------------------------------------------
# local cache of the deterministic fold fits (same machine + env => same fit;
# lets gate/endpoint iterations skip the expensive LR refits)
# --------------------------------------------------------------------------

def _save_fits(path: Path, fits: list[dict]) -> None:
    arrays = {"her_regimes": np.array(list(HER_REGIMES))}
    for f in fits:
        tag = f'{f["layer"]}::{f["reg"]}::{f["fold"]}'
        arrays[f"{tag}::test"] = f["test"]
        arrays[f"{tag}::oof"] = f["oof"]
        arrays[f"{tag}::coef"] = f["coef"]
        arrays[f"{tag}::intercept"] = np.float64(f["intercept"])
        arrays[f"{tag}::mean"] = f["mean"]
        arrays[f"{tag}::scale"] = f["scale"]
        for hreg, s in f["her"].items():
            arrays[f"{tag}::her::{hreg}"] = s
    np.savez_compressed(path, **arrays)


def _load_fits(path: Path, jobs: list[tuple]) -> list[dict]:
    z = np.load(path)
    assert list(z["her_regimes"]) == list(HER_REGIMES)
    fits = []
    for layer, reg, fold in jobs:
        tag = f"{layer}::{reg}::{fold}"
        her = {hreg: z[f"{tag}::her::{hreg}"] for hreg in HER_REGIMES
               if f"{tag}::her::{hreg}" in z}
        fits.append(dict(layer=layer, reg=reg, fold=fold,
                         test=z[f"{tag}::test"], oof=z[f"{tag}::oof"],
                         her=her, coef=z[f"{tag}::coef"],
                         intercept=float(z[f"{tag}::intercept"]),
                         mean=z[f"{tag}::mean"], scale=z[f"{tag}::scale"]))
    return fits


# --------------------------------------------------------------------------
# bootstrap helpers (2,000 draws, seed 0, percentile 95% CIs)
# --------------------------------------------------------------------------

def _ci_pct(s: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    vals = (s[draws] > 0).mean(axis=1) * 100
    return {"pct_eval": float((s > 0).mean() * 100), "n": int(len(s)),
            "ci95": [float(np.percentile(vals, 2.5)),
                     float(np.percentile(vals, 97.5))]}


def _ci_auc(s_pos: np.ndarray, s_neg: np.ndarray) -> dict:
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    s = np.r_[s_pos, s_neg]
    rng = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        p = s_pos[rng.integers(0, len(s_pos), len(s_pos))]
        q = s_neg[rng.integers(0, len(s_neg), len(s_neg))]
        vals.append(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))],
                                  np.r_[p, q]))
    return {"auc": float(roc_auc_score(y, s)),
            "n_pos": int(len(s_pos)), "n_neg": int(len(s_neg)),
            "ci95": [float(np.percentile(vals, 2.5)),
                     float(np.percentile(vals, 97.5))]}


def _ci_mean(d: np.ndarray, sd: float) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    vals = d[draws].mean(axis=1) / sd
    return {"mean_sd": float(d.mean() / sd), "n_pairs": int(len(d)),
            "ci95": [float(np.percentile(vals, 2.5)),
                     float(np.percentile(vals, 97.5))]}


def _ols_decomp(y: np.ndarray, cols: dict[str, np.ndarray]) -> dict:
    """OLS of y on the z-scored columns; R^2, betas, partial R^2 per column."""
    yz = (y - y.mean()) / y.std(ddof=1)
    Z = {k: (v - v.mean()) / v.std(ddof=1) for k, v in cols.items()}

    def r2(names):
        A = np.c_[np.ones(len(yz)), *(Z[k] for k in names)]
        beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
        resid = yz - A @ beta
        return 1.0 - float(resid @ resid) / float(yz @ yz), beta

    names = list(cols)
    full, beta = r2(names)
    out = {"r2_full": full,
           "beta_std": {k: float(b) for k, b in zip(names, beta[1:])},
           "partial_r2": {}}
    for k in names:
        rest, _ = r2([n for n in names if n != k])
        out["partial_r2"][k] = full - rest
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    model_key = sys.argv[1]
    n_layers, hidden, layers, default_src, ref_path = MODELS[model_key]
    source = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / default_src)
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else min(10, os.cpu_count() or 1)
    _G["n_layers"] = n_layers

    # ---- our items + masks (field semantics as in nocue_sweep.py) ----
    rows = [json.loads(l) for l in open(OUR_ITEMS) if l.strip()]
    ids_expected = [r["item_id"] for r in rows]
    _G["stated"] = np.array([r["render_slot"] == "stated" for r in rows])
    _G["label"] = np.array([-1 if r["label"] is None else int(r["label"]) for r in rows])
    _G["fam"] = np.array([r["purpose_family_id"] for r in rows])
    _G["purpose"] = np.array([r["intended_purpose"] for r in rows])
    _G["fmt"] = np.array([r["format"] for r in rows])
    _G["pair_key"] = np.array(
        [f'{r["payload_block_id"]}::{r["purpose_pair_id"]}::{r["format"]}' for r in rows])

    # ---- her items ----
    hrows = [json.loads(l) for l in open(HER_ITEMS) if l.strip()]
    her_ids = [r["item_id"] for r in hrows]
    cell = np.array([r["cell"] for r in hrows])
    variant = np.array([r["variant"] for r in hrows])
    h_ctx = np.array([r["context"] for r in hrows])
    h_fmt = np.array([r["format"] for r in hrows])
    asis = variant == "asis"
    bench_deploy = np.isin(cell, ["bench_deploy_rewritten", "bench_deploy_formal"])

    # ---- her activations (streams once, then cached per-layer fp32) ----
    _G["her_regimes"] = HER_REGIMES
    her_meta = {}
    _G["her"] = {}
    for hreg in HER_REGIMES:
        arrs, meta = her_layer_slices(model_key, hreg, layers, her_ids, hidden)
        her_meta[hreg] = meta
        for L in layers:
            _G["her"][(hreg, L)] = arrs[L]

    # ---- our activations at the frozen layers ----
    print(f"loading our layers {layers} from {source}", flush=True)
    _G["X"] = {L: _load_our_layer(source, L, ids_expected) for L in layers}
    our_src_prov = {"source": source}
    if "X_all" in _G:
        del _G["X_all"]
        our_src_prov["sha256"] = sha256_file(Path(source))
    slice_meta = Path(source) / "slice_meta.json" if not source.startswith("hf://") else None
    if slice_meta and slice_meta.exists():
        our_src_prov["slice_meta"] = json.loads(slice_meta.read_text())

    # ---- fit all fold probes (Pool; fork shares _G copy-on-write) ----
    jobs = [(L, reg, fold) for L in layers
            for reg in REGIMES + ("format",) for fold in FOLDS]
    fit_cache = CACHE_DIR / model_key / "_fold_fit_cache.npz"
    if fit_cache.exists():
        print(f"loading {len(jobs)} fold fits from {fit_cache}", flush=True)
        fits = _load_fits(fit_cache, jobs)
    else:
        print(f"fitting {len(jobs)} fold probes on {n_workers} workers", flush=True)
        with Pool(n_workers) as pool:
            fits = pool.map(_fit_job, jobs)
        _save_fits(fit_cache, fits)

    # ---- G1: the rebuilt stated-stratum records must match the sweep ----
    ref = {r["layer"]: r for r in json.load(open(ROOT / ref_path))}
    g1 = {}
    oof = {}
    for L in layers:
        rec = _layer_record(L, fits)
        oof[L] = rec.pop("_oof")
        profile = {}
        fails = []
        for reg in REGIMES + ("format",):
            w = _compare(rec[reg], ref[L][reg], f"L{L}/{reg}")
            gated = reg in G1_GATED_REGIMES
            profile[reg] = {"gated": gated,
                            "worst": {c: {"max_abs_delta": d, "at": p}
                                      for c, (d, p) in w.items()}}
            if gated:
                fails += [f"{reg}/{c}" for c, (d, _) in w.items()
                          if d > G1_TOL[c]]
        g1[f"L{L}"] = {"pass": not fails, "mode": G1_MODE, "tol": G1_TOL,
                       "gated_regimes": list(G1_GATED_REGIMES),
                       "profile": profile, "over_tolerance": fails}
        print(f"G1 L{L}: {'PASS' if not fails else 'FAIL'} vs checked-in "
              f"record (mode {G1_MODE})", flush=True)
        for reg, p in profile.items():
            tagline = ", ".join(f"{c} {v['max_abs_delta']:.1e}"
                                for c, v in sorted(p["worst"].items()))
            print(f"  {reg:13s} {'[gated]' if p['gated'] else '[info ]'} {tagline}",
                  flush=True)
        if fails:
            raise SystemExit(
                f"G1 failed in mode {G1_MODE}: {fails} over tolerance. "
                f"Run on the reference environment, or use "
                f"DEVB_G1_MODE=local-fp-jitter only per the recorded amendment "
                f"in the plan doc.")

    # sd units: stated-stratum OOF score sd per (layer, probe)
    sd_units = {f"L{L}": {p: float(np.nanstd(oof[L][r], ddof=1))
                          for p, r in (("purpose_decorr", "decorrelated"),
                                       ("purpose_conf", "confounded"),
                                       ("format", "format"))}
                for L in layers}

    # ---- E6 fold-averaged projections of her items ----
    proj = {}  # (hreg, L, probe) -> (8479,) float32
    for hreg in HER_REGIMES:
        for L in layers:
            for probe, reg in (("purpose_decorr", "decorrelated"),
                               ("purpose_conf", "confounded"),
                               ("format", "format")):
                sel = [f["her"][hreg] for f in fits
                       if f["layer"] == L and f["reg"] == reg]
                assert len(sel) == len(FOLDS)
                proj[(hreg, L, probe)] = np.mean(sel, axis=0)

    # ---- her-protocol probe per (regime, layer); sign: positive = eval ----
    train = (cell == "bench_eval") & asis | (cell == "casual_deploy_matched")
    y_her = (cell[train] == "casual_deploy_matched").astype(int)  # eval=0, deploy=1
    her_protocol_info = {}
    for hreg in HER_REGIMES:
        for L in layers:
            Xh = _G["her"][(hreg, L)]
            sc = StandardScaler()
            Xt = sc.fit_transform(Xh[train])
            clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xt, y_her)
            proj[(hreg, L, "her_protocol")] = (
                -clf.decision_function(sc.transform(Xh))).astype(np.float32)
            her_protocol_info[f"{hreg}::L{L}"] = {
                "train_acc": float(clf.score(Xt, y_her)),
                "train_cells": ["bench_eval (asis, class 0=eval)",
                                "casual_deploy_matched (class 1=deploy)"],
            }

    # ---- G2 (8b only): the format trap must reproduce on raw ----
    g2 = None
    if model_key == "8b":
        s = proj[("raw", layers[0], "her_protocol")]
        fp = float((s[bench_deploy & asis] > 0).mean() * 100)
        g2 = {"fp_pct_bench_deploy_asis_raw": fp, "threshold": 85.0,
              "pass": fp >= 85.0, "reference": "her RESULTS.md: 94.3% (L8-31)"}
        print(f"G2: her-protocol FP on bench_deploy (8b raw L{layers[0]}): "
              f"{fp:.1f}% -> {'PASS' if g2['pass'] else 'FAIL'}", flush=True)
        assert g2["pass"], "G2 failed: format trap did not reproduce; stopping."

    # ---- minimal pairs (asis vs stripped, stripped_changed only) ----
    idx_of = {i: k for k, i in enumerate(her_ids)}
    pairs = []  # (asis_row, stripped_row)
    for k, r in enumerate(hrows):
        if r["variant"] == "stripped" and r.get("stripped_changed"):
            a = r["item_id"].replace("::stripped::", "::asis::")
            pairs.append((idx_of[a], k))
    pairs = np.array(pairs)
    changed_bd_stripped = np.array([k for _, k in pairs if bench_deploy[k]])
    changed_bd_asis = np.array([a for a, k in pairs if bench_deploy[k]])

    # ---- endpoints ----
    cells_asis = {c: (cell == c) & asis for c in np.unique(cell)}
    eval_asis = (h_ctx == "eval") & asis
    deploy_asis = (h_ctx == "deploy") & asis
    bench_asis = (h_fmt == "bench") & asis
    casual_asis = (h_fmt == "casual") & asis

    def endpoints(hreg: str, L: int) -> dict:
        P = {p: proj[(hreg, L, p)] for p in
             ("purpose_decorr", "purpose_conf", "format", "her_protocol")}
        sd = dict(sd_units[f"L{L}"],
                  her_protocol=float(P["her_protocol"][asis].std(ddof=1)))
        e: dict = {"P1_trap": {}, "P2_auc": {}, "P3_format_attribution": {},
                   "P4_scaffold": {}}
        for probe in ("purpose_conf", "purpose_decorr", "her_protocol"):
            e["P1_trap"][probe] = {
                "bench_deploy_asis": _ci_pct(P[probe][bench_deploy & asis]),
                "per_cell": {c: {"pct_eval": float((P[probe][m] > 0).mean() * 100),
                                 "n": int(m.sum())}
                             for c, m in cells_asis.items()},
            }
            e["P2_auc"][probe] = {
                "within_bench_eval_vs_deploy": _ci_auc(
                    P[probe][cells_asis["bench_eval"]],
                    P[probe][bench_deploy & asis]),
                "within_casual_eval_vs_deploy": _ci_auc(
                    P[probe][cells_asis["casual_eval_mutual"]],
                    P[probe][casual_asis & (h_ctx == "deploy")]),
                "overall_eval_vs_deploy": _ci_auc(
                    P[probe][eval_asis], P[probe][deploy_asis]),
            }
        e["P3_format_attribution"] = {
            "format_probe_auc_bench_vs_casual": _ci_auc(
                P["format"][bench_asis], P["format"][casual_asis]),
            "ols_her_score_on_projections": _ols_decomp(
                P["her_protocol"][asis].astype(np.float64),
                {"purpose_decorr": P["purpose_decorr"][asis].astype(np.float64),
                 "format": P["format"][asis].astype(np.float64)}),
        }
        a, k = pairs[:, 0], pairs[:, 1]
        e["P4_scaffold"] = {
            "delta_asis_minus_stripped_sd": {
                probe: _ci_mean(P[probe][a] - P[probe][k], sd[probe])
                for probe in ("her_protocol", "format",
                              "purpose_conf", "purpose_decorr")},
            "her_protocol_fp_bench_deploy": {
                "asis_changed_pairs": _ci_pct(P["her_protocol"][changed_bd_asis]),
                "stripped_changed_pairs": _ci_pct(P["her_protocol"][changed_bd_stripped]),
            },
            "sd_units_used": sd,
        }
        return e

    results = {hreg: {f"L{L}": endpoints(hreg, L) for L in layers}
               for hreg in HER_REGIMES}

    # ---- outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": model_key,
        "generated_by": "scripts/devbunova_decomposition.py",
        "plan": "docs/devbunova_decomposition_plan.md",
        "layers": {"primary": layers[0], "robustness": layers[1]},
        "regimes": {"primary": "template", "secondary": "raw"},
        "gates": {"G1": g1, "G2": g2,
                  "alignment": "our npz/shard and her chunk item_ids verified "
                               "against the respective jsonl row orders"},
        "sd_units_stated_oof": sd_units,
        "her_protocol": her_protocol_info,
        "endpoints": results,
        "n_minimal_pairs_changed": int(len(pairs)),
        "provenance": {
            "our_items": {"path": str(OUR_ITEMS.relative_to(ROOT)),
                          "sha256": sha256_file(OUR_ITEMS)},
            "her_items": {"path": str(HER_ITEMS.relative_to(ROOT)),
                          "sha256": sha256_file(HER_ITEMS)},
            "our_activations": our_src_prov,
            "her_activations": her_meta,
            "reference_sweep": ref_path,
            "probe": dict(PROBE, folds=list(FOLDS),
                          scaler="StandardScaler", e6_fold_average=True),
            "her_protocol_probe": {"max_iter": 1000, "random_state": 42,
                                   "source": "external/eval-awareness-format/"
                                             "scripts/evaluate_probe_cross.py"},
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
        },
        "notes": [
            "score > 0 = called eval (format probe: called benchmark); "
            "her-protocol decision sign flipped so positive = eval",
            "sd unit for her_protocol deltas = sd of its scores over asis rows "
            "(our stated-OOF sd does not apply to a probe trained on her data)",
            "her-protocol probe trained on the full 1,076+1,076 cells "
            "(deterministic; her published run sampled n=1076 per cell "
            "irreproducibly)",
            "405b raw chunks 0-76 vs 77-132 came from different NDIF "
            "deployment instances (bf16 caveat; template unaffected)"
            if model_key == "405b" else
            "cross-cell contrasts inherit her dataset's non-format confounds "
            "(MuTual length skew, DeepSeek rewriter fingerprint); the clean "
            "contrasts are the within-format AUCs and the minimal pairs",
        ],
    }
    out_json = OUT_DIR / f"decomp_{model_key}.json"
    out_json.write_text(json.dumps(out, indent=1) + "\n")

    arrays = {"item_ids": np.array(her_ids)}
    for (hreg, L, probe), s in proj.items():
        arrays[f"{hreg}::L{L}::{probe}"] = s
    for f in fits:
        if f["reg"] not in HER_SCORED:
            continue
        tag = f'L{f["layer"]:03d}::{f["reg"]}::fold_{f["fold"]}'
        arrays[f"{tag}::coef"] = f["coef"]
        arrays[f"{tag}::intercept"] = np.float32(f["intercept"])
        arrays[f"{tag}::scaler_mean"] = f["mean"]
        arrays[f"{tag}::scaler_scale"] = f["scale"]
    np.savez_compressed(OUT_DIR / f"decomp_{model_key}.scores.npz", **arrays)
    print(f"wrote {out_json} + scores npz", flush=True)

    # quick console summary (primary regime/layer)
    L = layers[0]
    for probe in ("her_protocol", "purpose_conf", "purpose_decorr"):
        t = results["template"][f"L{L}"]["P1_trap"][probe]["bench_deploy_asis"]
        a = results["template"][f"L{L}"]["P2_auc"][probe]
        print(f"[template L{L}] {probe:15s} trap FP {t['pct_eval']:5.1f}% "
              f"[{t['ci95'][0]:.1f},{t['ci95'][1]:.1f}] | "
              f"within-bench AUC {a['within_bench_eval_vs_deploy']['auc']:.3f} "
              f"within-casual AUC {a['within_casual_eval_vs_deploy']['auc']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
