"""Primary binding_v1 analyses.

A. Fixed explicit -> binding: persisted explicit-purpose probes (joint,
   benchmark-trained, casual-trained) are applied without any retraining.
B. Binding -> binding: new probes with leave-one-entire-relational-family-out CV.
C. Factorial geometry: matched contrasts for mapping, selected route, format,
   the bound-purpose interaction P, and the P x format interaction C.

The inferential unit is the relational-template family. Item-level pooled AUCs
are reported as descriptive only.
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
SEED = 20260831
EXPECTED_ITEMS_SHA256 = "bdfd3d4360d1704511f7c3451432c5127b1dbecad75e73654ab4d301f1018efd"
N_BOOT = 10_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_families(per_family: dict[str, dict[str, float]]) -> dict:
    names = sorted(per_family)
    rng = np.random.default_rng(SEED)
    out = {}
    for key in next(iter(per_family.values())):
        values = np.array([per_family[n][key] for n in names], dtype=float)
        finite = values[np.isfinite(values)]
        boot = np.array([np.mean(rng.choice(finite, len(finite), replace=True))
                         for _ in range(N_BOOT)]) if len(finite) else np.array([np.nan])
        reference = 0.5 if key.endswith("_auc") else 0.0
        wins = int(np.sum(finite > reference))
        out[key] = {
            "family_mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "family_values": {n: float(per_family[n][key]) for n in names},
            "family_bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "n_families": int(len(finite)),
            "families_above_reference": wins,
            "reference": reference,
            "exact_family_sign_test_p_one_sided": float(
                binomtest(wins, len(finite), 0.5, alternative="greater").pvalue) if len(finite) else float("nan"),
        }
    return out


def auc_or_nan(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0 or len(set(y[mask].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], score[mask]))


def fixed_affine(bank: np.lib.npyio.NpzFile, layer: int, regime: str) -> tuple[np.ndarray, float]:
    weights, offsets = [], []
    for fold in FOLDS:
        p = f"L{layer:03d}::{regime}::fold_{fold}"
        coef = bank[f"{p}::coef"].astype(np.float64)
        mean = bank[f"{p}::scaler_mean"].astype(np.float64)
        scale = bank[f"{p}::scaler_scale"].astype(np.float64)
        weight = coef / scale
        weights.append(weight)
        offsets.append(float(bank[f"{p}::intercept"]) - float(mean @ weight))
    return np.mean(weights, axis=0), float(np.mean(offsets))


def interaction_stats(score: np.ndarray, block: np.ndarray, mapping: np.ndarray,
                      selected: np.ndarray, fmt: np.ndarray) -> dict[str, np.ndarray]:
    """Per (block, format) factorial interaction I = mean over cells of (m*s)*score."""
    keys: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    for i in range(len(score)):
        keys.setdefault((block[i], fmt[i]), {})[(int(mapping[i]), int(selected[i]))] = float(score[i])
    out_block, out_fmt, out_I = [], [], []
    for (blk, f), cells in keys.items():
        if len(cells) != 4:
            raise RuntimeError(f"incomplete factorial cell set for {blk}/{f}")
        I = sum(m * s * v for (m, s), v in cells.items()) / 4.0
        out_block.append(blk)
        out_fmt.append(f)
        out_I.append(I)
    return {"block": np.asarray(out_block), "format": np.asarray(out_fmt),
            "I": np.asarray(out_I, dtype=float)}


def fit_score(X: np.ndarray, y: np.ndarray, fmt: np.ndarray, train: np.ndarray,
              test: np.ndarray, regime: str) -> tuple[str, np.ndarray]:
    sel = train if regime == "joint" else train & (fmt == regime)
    scaler = StandardScaler().fit(X[sel])
    model = LogisticRegression(**PROBE).fit(scaler.transform(X[sel]), y[sel])
    return regime, model.decision_function(scaler.transform(X[test])).astype(np.float32)


def probe_metrics(y: np.ndarray, fmt: np.ndarray, joint: np.ndarray, bench: np.ndarray,
                  casual: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    b = mask & (fmt == "benchmark")
    c = mask & (fmt == "casual")
    out = {
        "joint_format_auc": auc_or_nan(y, joint, mask),
        "joint_benchmark_items_auc": auc_or_nan(y, joint, b),
        "joint_casual_items_auc": auc_or_nan(y, joint, c),
        "benchmark_self_auc": auc_or_nan(y, bench, b),
        "benchmark_to_casual_auc": auc_or_nan(y, bench, c),
        "casual_self_auc": auc_or_nan(y, casual, c),
        "casual_to_benchmark_auc": auc_or_nan(y, casual, b),
    }
    out["cross_format_mean_auc"] = (out["benchmark_to_casual_auc"] + out["casual_to_benchmark_auc"]) / 2
    out["transfer_gap"] = out["joint_format_auc"] - out["cross_format_mean_auc"]
    return out


def factorial_vectors(Xl: np.ndarray, block: np.ndarray, mapping: np.ndarray,
                      selected: np.ndarray, fmt: np.ndarray,
                      rows_mask: np.ndarray, expect: int = 8) -> dict[str, np.ndarray]:
    f = np.where(fmt == "benchmark", 1, -1)
    m, s = mapping, selected
    idx = np.flatnonzero(rows_mask)
    cells: dict[str, list[np.ndarray]] = {k: [] for k in ("M", "S", "F", "P", "C")}
    by_block: dict[str, list[int]] = {}
    for i in idx:
        by_block.setdefault(block[i], []).append(i)
    for blk, members in by_block.items():
        if len(members) != expect:
            continue
        ii = np.asarray(members)
        H = Xl[ii].astype(np.float64)
        cells["M"].append((m[ii, None] * H).mean(axis=0))
        cells["S"].append((s[ii, None] * H).mean(axis=0))
        cells["P"].append(((m[ii] * s[ii])[:, None] * H).mean(axis=0))
        if expect == 8:
            cells["F"].append((f[ii, None] * H).mean(axis=0))
            cells["C"].append(((m[ii] * s[ii] * f[ii])[:, None] * H).mean(axis=0))
    return {k: np.mean(np.stack(v), axis=0) for k, v in cells.items() if v}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def analyze_layer(layer: int, Xl: np.ndarray, y: np.ndarray, fmt: np.ndarray,
                  family: np.ndarray, block: np.ndarray, mapping: np.ndarray,
                  selected: np.ndarray, bank_path: Path, jobs: int) -> dict:
    families = sorted(set(family.tolist()))
    with np.load(bank_path, allow_pickle=False) as bank:
        fixed_w, fixed_scores = {}, {}
        for regime in REGIMES:
            weight, offset = fixed_affine(bank, layer, regime)
            fixed_w[regime] = weight
            fixed_scores[regime] = (Xl @ weight + offset).astype(np.float64)

    fixed_family = {n: probe_metrics(y, fmt, fixed_scores["joint"], fixed_scores["benchmark"],
                                     fixed_scores["casual"], family == n) for n in families}
    inter = interaction_stats(fixed_scores["joint"], block, mapping, selected, fmt)
    inter_family = {}
    for n in families:
        sel = np.array([b.startswith(f"{n}::") for b in inter["block"]])
        vals = inter["I"][sel]
        bench = inter["I"][sel & (inter["format"] == "benchmark")]
        cas = inter["I"][sel & (inter["format"] == "casual")]
        inter_family[n] = {
            "mean_interaction_I": float(np.mean(vals)),
            "mean_interaction_I_benchmark": float(np.mean(bench)),
            "mean_interaction_I_casual": float(np.mean(cas)),
            "fraction_blocks_I_positive": float(np.mean(vals > 0)),
        }

    implicit = {r: np.full(len(y), np.nan, dtype=np.float32) for r in REGIMES}
    tasks = [(n, family == n, r) for n in families for r in REGIMES]
    fitted = Parallel(n_jobs=min(jobs, len(tasks)), backend="threading")(
        delayed(fit_score)(Xl, y, fmt, family != n, test, r) for n, test, r in tasks)
    for (n, test, r), (got, values) in zip(tasks, fitted):
        assert r == got
        implicit[r][test] = values
    if any(np.isnan(v).any() for v in implicit.values()):
        raise RuntimeError(f"incomplete family-held-out predictions at layer {layer}")
    binding_family = {n: probe_metrics(y, fmt, implicit["joint"], implicit["benchmark"],
                                       implicit["casual"], family == n) for n in families}

    vecs = factorial_vectors(Xl, block, mapping, selected, fmt, np.ones(len(y), bool))
    per_family_P = {}
    for n in families:
        v = factorial_vectors(Xl, block, mapping, selected, fmt, family == n)
        per_family_P[n] = v["P"]
    fam_names = sorted(per_family_P)
    cross = [cosine(per_family_P[a], per_family_P[b])
             for i, a in enumerate(fam_names) for b in fam_names[i + 1:]]
    bench_vecs = factorial_vectors(Xl, block, mapping, selected, fmt, fmt == "benchmark", expect=4)
    casual_vecs = factorial_vectors(Xl, block, mapping, selected, fmt, fmt == "casual", expect=4)
    geometry = {
        "norms": {k: float(np.linalg.norm(v)) for k, v in vecs.items()},
        "cos_P_explicit_joint_probe": cosine(vecs["P"], fixed_w["joint"]),
        "cos_P_explicit_benchmark_probe": cosine(vecs["P"], fixed_w["benchmark"]),
        "cos_P_explicit_casual_probe": cosine(vecs["P"], fixed_w["casual"]),
        "cos_P_M": cosine(vecs["P"], vecs["M"]),
        "cos_P_S": cosine(vecs["P"], vecs["S"]),
        "cos_P_F": cosine(vecs["P"], vecs["F"]),
        "cos_P_benchmark_vs_casual": cosine(bench_vecs["P"], casual_vecs["P"]),
        "cos_P_across_families_mean": float(np.mean(cross)),
        "cos_P_across_families_min": float(np.min(cross)),
        "cos_P_family_vs_explicit_joint": {n: cosine(per_family_P[n], fixed_w["joint"])
                                           for n in fam_names},
    }
    return {
        "layer": layer,
        "fixed_explicit_to_binding": {
            "probe_retrained_on_binding": False,
            "pooled_descriptive": probe_metrics(y, fmt, fixed_scores["joint"],
                                                fixed_scores["benchmark"], fixed_scores["casual"],
                                                np.ones(len(y), bool)),
            "per_family": fixed_family,
            "family_unit_summary": summarize_families(fixed_family),
            "interaction_per_family": inter_family,
            "interaction_family_unit_summary": summarize_families(inter_family),
        },
        "binding_to_binding": {
            "cv": "leave-one-entire-relational-family-out",
            "pooled_descriptive": probe_metrics(y, fmt, implicit["joint"], implicit["benchmark"],
                                                implicit["casual"], np.ones(len(y), bool)),
            "per_family": binding_family,
            "family_unit_summary": summarize_families(binding_family),
        },
        "factorial_geometry": geometry,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--explicit-bank", type=Path, required=True)
    ap.add_argument("--fixed-layer", type=int, help="layer number for 2-D activation input")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    if sha256_file(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("frozen binding_v1 items hash mismatch")
    manifest = json.loads(args.items.with_name("freeze_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_ACTIVATIONS":
        raise RuntimeError("binding_v1 freeze status invalid")
    rows_raw = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    with np.load(args.activations, allow_pickle=False) as z:
        ids = z["item_ids"].astype(str)
        X = z["X"].astype(np.float32)
    by_id = {r["item_id"]: r for r in rows_raw}
    if len(ids) != len(rows_raw) or set(ids) != set(by_id):
        raise RuntimeError("activation IDs do not match the frozen binding_v1 bank")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    family = np.array([r["binding_family"] for r in rows])
    block = np.array([r["binding_block_id"] for r in rows])
    mapping = np.array([r["binding_mapping"] for r in rows], dtype=np.int8)
    selected = np.array([r["binding_selected"] for r in rows], dtype=np.int8)
    if len(set(family.tolist())) != manifest["n_families"]:
        raise RuntimeError("unexpected family count")

    if X.ndim == 3:
        layers = list(range(X.shape[1]))
        get = lambda L: X[:, L, :]
    elif X.ndim == 2:
        if args.fixed_layer is None:
            raise RuntimeError("--fixed-layer required for 2-D activations")
        layers = [args.fixed_layer]
        get = lambda L: X
    else:
        raise RuntimeError(f"unexpected activation shape {X.shape}")

    out_layers = []
    for L in layers:
        out_layers.append(analyze_layer(L, get(L), y, fmt, family, block, mapping,
                                        selected, args.explicit_bank, args.jobs))
        print(f"layer {L} complete", flush=True)

    result = {
        "schema_version": 1,
        "mode": "development",
        "experiment": "binding_v1",
        "model_key": args.model_key,
        "n_items": len(rows),
        "n_families": manifest["n_families"],
        "n_payloads": manifest["n_payloads"],
        "inference_unit": "binding_family",
        "label_definition": manifest["label_definition"],
        "items_sha256": sha256_file(args.items),
        "activations_sha256": sha256_file(args.activations),
        "explicit_probe_bank_sha256": sha256_file(args.explicit_bank),
        "layers": out_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "n_layers": len(out_layers)}, indent=2))


if __name__ == "__main__":
    main()
