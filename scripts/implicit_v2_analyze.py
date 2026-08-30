"""Analyze frozen implicit-v2 activations with scenario family as the unit.

Analysis A projects fixed explicit-purpose probes; no implicit examples enter
those models. Analysis B fits implicit probes with leave-one-entire-scenario-
family-out CV. All uncertainty resamples the six family-level estimates, never
individual prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
REGIMES = ("joint", "benchmark", "casual")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
SEED = 20260830
EXPECTED_ITEMS_SHA256 = "ba29c2cb55c96664a957aa135801343d4c77409b65d1251ccee5d0626a0b4743"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float:
    return float(roc_auc_score(y[mask], score[mask]))


def metrics(y: np.ndarray, fmt: np.ndarray, joint: np.ndarray,
            benchmark: np.ndarray, casual: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    b = mask & (fmt == "benchmark")
    c = mask & (fmt == "casual")
    out = {
        "joint_format_auc": auc(y, joint, mask),
        "benchmark_self_auc": auc(y, benchmark, b),
        "benchmark_to_casual_auc": auc(y, benchmark, c),
        "casual_self_auc": auc(y, casual, c),
        "casual_to_benchmark_auc": auc(y, casual, b),
    }
    out["cross_format_mean_auc"] = (out["benchmark_to_casual_auc"] + out["casual_to_benchmark_auc"]) / 2
    out["transfer_gap"] = out["joint_format_auc"] - out["cross_format_mean_auc"]
    return out


def summarize_families(per_family: dict[str, dict[str, float]]) -> dict:
    names = sorted(per_family)
    keys = list(next(iter(per_family.values())))
    rng = np.random.default_rng(SEED)
    summary = {}
    for key in keys:
        values = np.array([per_family[name][key] for name in names])
        boot = np.array([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(10_000)])
        rec = {
            "mean": float(np.mean(values)),
            "family_bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "n_independent_families": len(values),
        }
        if key == "transfer_gap":
            wins = int(np.sum(values > 0))
            rec["families_gap_positive"] = wins
            rec["exact_family_sign_test_p_one_sided"] = float(
                binomtest(wins, len(values), 0.5, alternative="greater").pvalue)
        summary[key] = rec
    return summary


def fixed_affine(bank: np.lib.npyio.NpzFile, layer: int, regime: str) -> tuple[np.ndarray, float]:
    weights, offsets = [], []
    for fold in FOLDS:
        p = f"L{layer:03d}::{regime}::fold_{fold}"
        coef = bank[f"{p}::coef"].astype(np.float64)
        mean = bank[f"{p}::scaler_mean"].astype(np.float64)
        scale = bank[f"{p}::scaler_scale"].astype(np.float64)
        intercept = float(bank[f"{p}::intercept"])
        weight = coef / scale
        weights.append(weight)
        offsets.append(intercept - float(mean @ weight))
    return np.mean(weights, axis=0), float(np.mean(offsets))


def fit_score(X: np.ndarray, y: np.ndarray, fmt: np.ndarray, train: np.ndarray,
              test: np.ndarray, regime: str) -> tuple[str, np.ndarray]:
    selected = train if regime == "joint" else train & (fmt == regime)
    scaler = StandardScaler().fit(X[selected])
    model = LogisticRegression(**PROBE).fit(scaler.transform(X[selected]), y[selected])
    return regime, model.decision_function(scaler.transform(X[test])).astype(np.float32)


def analyze_layer(layer: int, Xl: np.ndarray, y: np.ndarray, fmt: np.ndarray,
                  family: np.ndarray, bank_path: Path, jobs: int) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    families = sorted(set(family))
    with np.load(bank_path, allow_pickle=False) as bank:
        fixed_scores = {}
        for regime in REGIMES:
            weight, offset = fixed_affine(bank, layer, regime)
            fixed_scores[regime] = (Xl @ weight + offset).astype(np.float32)
    fixed_per_family = {name: metrics(y, fmt, fixed_scores["joint"], fixed_scores["benchmark"],
                                      fixed_scores["casual"], family == name)
                        for name in families}

    implicit_scores = {regime: np.full(len(y), np.nan, dtype=np.float32) for regime in REGIMES}
    tasks = []
    for held in families:
        test = family == held
        for regime in REGIMES:
            tasks.append((held, test, regime))
    fitted = Parallel(n_jobs=min(jobs, len(tasks)), backend="threading")(
        delayed(fit_score)(Xl, y, fmt, family != held, test, regime)
        for held, test, regime in tasks)
    for (held, test, regime), (got_regime, score) in zip(tasks, fitted):
        assert regime == got_regime
        implicit_scores[regime][test] = score
    if any(np.isnan(values).any() for values in implicit_scores.values()):
        raise RuntimeError(f"incomplete family-held-out predictions at layer {layer}")
    implicit_per_family = {name: metrics(y, fmt, implicit_scores["joint"], implicit_scores["benchmark"],
                                         implicit_scores["casual"], family == name)
                           for name in families}
    fixed_stack = np.stack([fixed_scores[r] for r in REGIMES])
    implicit_stack = np.stack([implicit_scores[r] for r in REGIMES])
    return fixed_per_family, implicit_per_family, fixed_stack, implicit_stack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--explicit-bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=18)
    args = ap.parse_args()
    if sha256(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("implicit-v2 frozen bank hash mismatch")
    manifest_path = args.items.with_name("freeze_manifest.json")
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_ACTIVATIONS" or freeze["items_sha256"] != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("invalid implicit-v2 freeze manifest")
    rows_raw = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(args.activations, allow_pickle=False) as z:
        ids = z["item_ids"].astype(str)
        X = z["X"].astype(np.float32)
    by_id = {r["item_id"]: r for r in rows_raw}
    if len(ids) != len(rows_raw) or set(ids) != set(by_id):
        raise RuntimeError("activation IDs do not exactly match frozen implicit-v2 bank")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    family = np.array([r["implicit_family"] for r in rows])
    if len(set(family)) != 6 or X.shape[:2] != (3840, 32):
        raise RuntimeError(f"unexpected implicit-v2 tensor/strata: {X.shape}, {set(family)}")

    fixed_layers, implicit_layers = [], []
    fixed_scores_all, implicit_scores_all = [], []
    for layer in range(X.shape[1]):
        fixed_family, implicit_family, fixed_scores, implicit_scores = analyze_layer(
            layer, X[:, layer, :], y, fmt, family, args.explicit_bank, args.jobs)
        fixed_layers.append({"layer": layer, "per_family": fixed_family,
                             "family_unit_summary": summarize_families(fixed_family)})
        implicit_layers.append({"layer": layer, "per_family": implicit_family,
                                "family_unit_summary": summarize_families(implicit_family)})
        fixed_scores_all.append(fixed_scores)
        implicit_scores_all.append(implicit_scores)
        print(f"layer {layer}/31 complete", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    score_path = args.out.with_suffix(".scores.npz")
    np.savez_compressed(score_path, item_ids=ids, regimes=np.asarray(REGIMES),
                        fixed_explicit=np.stack(fixed_scores_all, axis=1),
                        implicit_family_heldout=np.stack(implicit_scores_all, axis=1))
    result = {
        "schema_version": 1,
        "mode": "development",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "n_items": len(rows),
        "n_scenario_pairs": 24,
        "n_independent_scenario_families": 6,
        "inference_unit": "scenario_family",
        "items_sha256": sha256(args.items),
        "freeze_manifest_sha256": sha256(manifest_path),
        "activations_sha256": sha256(args.activations),
        "explicit_probe_bank_sha256": sha256(args.explicit_bank),
        "fixed_explicit_to_implicit": {
            "probe_retrained_on_implicit": False,
            "regimes": list(REGIMES),
            "layers": fixed_layers,
        },
        "implicit_to_implicit": {
            "cv": "leave-one-entire-scenario-family-out",
            "regimes": list(REGIMES),
            "layers": implicit_layers,
        },
        "scores": str(score_path),
        "scores_sha256": sha256(score_path),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "scores": str(score_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
