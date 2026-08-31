"""Analyze binding_v2: semantic ablation and indirection-depth transfer.

The label is the same factorial interaction in every arm/hop. Analyses:
  1. frozen stated-purpose probes -> binding_v2, without retraining;
  2. binding_v2 -> binding_v2 with leave-one-entire-family-out CV;
  3. train at hop 0, test at hops 1/2 (same arm);
  4. consequence -> neutral and neutral -> consequence at fixed hop.

Every refit excludes the held relational family. A fitted model is reused across
its within-condition, hop-transfer, and cross-arm target sets, avoiding redundant
fits without changing any train/test split.
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
from sklearn.preprocessing import StandardScaler

ARMS = ("consequence", "neutral")
HOPS = (0, 1, 2)
REGIMES = ("joint", "benchmark", "casual")
FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
SEED = 20260904
N_BOOT = 10_000
EXPECTED_ITEMS_SHA256 = "d7ad085991345750384f8877a380e6ca5c23e93dccffa1cf0712137063824cc2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc_or_nan(y, score, mask) -> float:
    from sklearn.metrics import roc_auc_score
    if mask.sum() == 0 or len(set(y[mask].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], score[mask]))


def probe_metrics(y, fmt, scores, mask) -> dict:
    joint, bench, casual = (scores[r] for r in REGIMES)
    b, c = mask & (fmt == "benchmark"), mask & (fmt == "casual")
    out = {
        "joint_format_auc": auc_or_nan(y, joint, mask),
        "joint_benchmark_items_auc": auc_or_nan(y, joint, b),
        "joint_casual_items_auc": auc_or_nan(y, joint, c),
        "benchmark_self_auc": auc_or_nan(y, bench, b),
        "benchmark_to_casual_auc": auc_or_nan(y, bench, c),
        "casual_self_auc": auc_or_nan(y, casual, c),
        "casual_to_benchmark_auc": auc_or_nan(y, casual, b),
    }
    out["cross_format_mean_auc"] = (
        out["benchmark_to_casual_auc"] + out["casual_to_benchmark_auc"]
    ) / 2
    out["transfer_gap"] = out["joint_format_auc"] - out["cross_format_mean_auc"]
    return out


def summarize_units(per_unit: dict[str, dict[str, float]]) -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    for key in next(iter(per_unit.values())):
        vals = np.array([v[key] for v in per_unit.values()], dtype=float)
        vals = vals[np.isfinite(vals)]
        boot = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(1)
        reference = 0.5 if key.endswith("_auc") else 0.0
        wins = int(np.sum(vals > reference))
        out[key] = {
            "family_mean": float(np.mean(vals)),
            "family_min": float(np.min(vals)),
            "family_max": float(np.max(vals)),
            "family_values": {u: float(v[key]) for u, v in per_unit.items()},
            "family_bootstrap_ci95": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "n_families": int(len(vals)),
            "families_above_reference": wins,
            "reference": reference,
            "exact_family_sign_test_p_one_sided": float(
                binomtest(
                    wins, len(vals), 0.5, alternative="greater"
                ).pvalue
            ),
        }
    return out


def fixed_affine(bank, layer: int, regime: str) -> tuple[np.ndarray, float]:
    weights, offsets = [], []
    for fold in FOLDS:
        p = f"L{layer:03d}::{regime}::fold_{fold}"
        coef = bank[f"{p}::coef"].astype(np.float64)
        mean = bank[f"{p}::scaler_mean"].astype(np.float64)
        scale = bank[f"{p}::scaler_scale"].astype(np.float64)
        w = coef / scale
        weights.append(w)
        offsets.append(float(bank[f"{p}::intercept"]) - float(mean @ w))
    return np.mean(weights, axis=0), float(np.mean(offsets))


def task_fit(
    X, y, fmt, family, arm, hop, held, regime, train_arm, train_hop
):
    train = (family != held) & (arm == train_arm) & (hop == train_hop)
    fit = train if regime == "joint" else train & (fmt == regime)
    scaler = StandardScaler().fit(X[fit])
    model = LogisticRegression(**PROBE).fit(scaler.transform(X[fit]), y[fit])
    targets = {
        f"within:{train_arm}:h{train_hop}": (
            (family == held) & (arm == train_arm) & (hop == train_hop)
        ),
        f"arm:{train_arm}_to_{ARMS[1 - ARMS.index(train_arm)]}:h{train_hop}": (
            (family == held)
            & (arm == ARMS[1 - ARMS.index(train_arm)])
            & (hop == train_hop)
        ),
    }
    if train_hop == 0:
        for target_hop in (1, 2):
            targets[f"hop:{train_arm}:0to{target_hop}"] = (
                (family == held) & (arm == train_arm) & (hop == target_hop)
            )
    out = {}
    for key, mask in targets.items():
        idx = np.flatnonzero(mask)
        out[key] = (
            idx,
            model.decision_function(scaler.transform(X[idx])).astype(np.float32),
        )
    return held, regime, out


def evaluate_prediction_set(key, scores, y, fmt, family, target_mask) -> dict:
    families = sorted(set(family[target_mask].tolist()))
    per = {
        f: probe_metrics(y, fmt, scores, target_mask & (family == f))
        for f in families
    }
    return {
        "split": key,
        "cv": "train excludes the entire held relational family",
        "pooled_descriptive": probe_metrics(y, fmt, scores, target_mask),
        "per_family": per,
        "family_unit_summary": summarize_units(per),
    }


def analyze_layer(layer, X, y, fmt, family, arm, hop, explicit_bank, jobs):
    families = sorted(set(family.tolist()))
    tasks = [
        (held, regime, train_arm, train_hop)
        for held in families
        for train_arm in ARMS
        for train_hop in HOPS
        for regime in REGIMES
    ]
    fitted = Parallel(n_jobs=min(jobs, len(tasks)), backend="threading")(
        delayed(task_fit)(
            X, y, fmt, family, arm, hop, held, regime, train_arm, train_hop
        )
        for held, regime, train_arm, train_hop in tasks
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for held, regime, outputs in fitted:
        for key, (idx, value) in outputs.items():
            predictions.setdefault(
                key, {r: np.full(len(y), np.nan, dtype=np.float32) for r in REGIMES}
            )[regime][idx] = value

    within, hop_transfer, cross_arm = {}, {}, {}
    for key, score in sorted(predictions.items()):
        if key.startswith("within:"):
            _, a, hs = key.split(":")
            target = (arm == a) & (hop == int(hs[1:]))
            dest = within
        elif key.startswith("hop:"):
            _, a, path = key.split(":")
            target = (arm == a) & (hop == int(path.split("to")[1]))
            dest = hop_transfer
        else:
            _, path, hs = key.split(":")
            target_arm = path.split("_to_")[1]
            target = (arm == target_arm) & (hop == int(hs[1:]))
            dest = cross_arm
        if any(np.isnan(score[r][target]).any() for r in REGIMES):
            raise RuntimeError(f"incomplete predictions for {key}")
        dest[key] = evaluate_prediction_set(key, score, y, fmt, family, target)

    fixed = None
    if explicit_bank is not None:
        with np.load(explicit_bank, allow_pickle=False) as bank:
            prefix = f"L{layer:03d}::joint::fold_fg1::coef"
            if prefix in bank.files:
                fs = {}
                for regime in REGIMES:
                    w, offset = fixed_affine(bank, layer, regime)
                    fs[regime] = (X.astype(np.float64) @ w + offset).astype(np.float32)
                by_condition = {}
                for a in ARMS:
                    for h in HOPS:
                        m = (arm == a) & (hop == h)
                        per = {
                            f: probe_metrics(y, fmt, fs, m & (family == f))
                            for f in families
                        }
                        by_condition[f"{a}:h{h}"] = {
                            "pooled_descriptive": probe_metrics(y, fmt, fs, m),
                            "per_family": per,
                            "family_unit_summary": summarize_units(per),
                        }
                fixed = {
                    "probe_retrained_on_binding_v2": False,
                    "positive_score_convention": "evaluation in stated-purpose bank",
                    "by_arm_hop": by_condition,
                }
    return {
        "layer": layer,
        "fixed_explicit_to_binding_v2": fixed,
        "binding_v2_within_condition": within,
        "hop_transfer": hop_transfer,
        "cross_arm_transfer": cross_arm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--explicit-bank", type=Path)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if sha256_file(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("frozen binding_v2 items hash mismatch")
    manifest = json.loads(args.items.with_name("freeze_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_ACTIVATIONS":
        raise RuntimeError("freeze status invalid")
    raw = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {r["item_id"]: r for r in raw}
    with np.load(args.activations, allow_pickle=False) as z:
        ids, X = z["item_ids"].astype(str), z["X"].astype(np.float32)
    if X.ndim != 3:
        raise RuntimeError(f"expected all-layer tensor, got {X.shape}")
    if len(ids) != len(raw) or set(ids) != set(by_id):
        raise RuntimeError("activation IDs do not match frozen binding_v2 bank")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    family = np.array([r["binding_family"] for r in rows])
    arm = np.array([r["binding_arm"] for r in rows])
    hop = np.array([r["binding_hop"] for r in rows], dtype=np.int8)
    layers = list(range(X.shape[1])) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    result_layers = []
    for layer in layers:
        result_layers.append(
            analyze_layer(layer, X[:, layer, :], y, fmt, family, arm, hop,
                          args.explicit_bank, args.jobs)
        )
        print(f"layer {layer} complete", flush=True)
    result = {
        "schema_version": 1,
        "experiment": "binding_v2",
        "model_key": args.model_key,
        "n_items": len(rows),
        "n_families": len(set(family.tolist())),
        "inference_unit": "relational family",
        "label_definition": manifest["label_definition"],
        "items_sha256": sha256_file(args.items),
        "activations_sha256": sha256_file(args.activations),
        "explicit_probe_bank_sha256": sha256_file(args.explicit_bank) if args.explicit_bank else None,
        "layers": result_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "n_layers": len(result_layers)}))


if __name__ == "__main__":
    main()
