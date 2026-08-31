"""Score a frozen provenance bank with pre-existing stated-purpose probes.

No probe is trained or selected on the provenance corpus. The persisted
stated-purpose bank is applied as an affine map. Positive scores mean evaluation
under the original explicit-purpose training convention.

Primary quantities:
  * provenance AUC pooled and separately within benchmark/casual format;
  * all four eval-source x deployment-source AUCs (natural source robustness);
  * paired within-text benchmark-minus-casual score shift;
  * the provenance x format interaction in that paired shift.

Bootstrap resamples complete underlying-text blocks, stratified by source
corpus. One block sample is reused for every layer and regime in a replicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
REGIMES = ("joint", "benchmark", "casual")
SEED = 20260903
EXPECTED_ITEMS_SHA256 = "1040383665d716d71f087ac78ee50da563dada1a696fa790fd48c7bd259b814f"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed_affine(bank, layer: int, regime: str) -> tuple[np.ndarray, float]:
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


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def point_metrics(
    score: np.ndarray,
    y: np.ndarray,
    fmt: np.ndarray,
    source: np.ndarray,
    pair: np.ndarray,
) -> dict:
    b, c = fmt == "benchmark", fmt == "casual"
    eval_sources = sorted(set(source[y == 1].tolist()))
    deploy_sources = sorted(set(source[y == 0].tolist()))
    source_pair_auc = {}
    for es in eval_sources:
        for ds in deploy_sources:
            m = (source == es) | (source == ds)
            source_pair_auc[f"{es}|{ds}"] = auc(y[m], score[m])

    # A bootstrap sample can contain the same pair multiple times. Pair each
    # format's k-th occurrence rather than collapsing duplicate pair IDs.
    occurrences: dict[tuple[str, str], int] = {}
    by_occurrence: dict[tuple[str, int], dict[str, int]] = {}
    for i, (p_raw, f_raw) in enumerate(zip(pair, fmt)):
        p, f = str(p_raw), str(f_raw)
        count_key = (p, f)
        occurrence = occurrences.get(count_key, 0)
        occurrences[count_key] = occurrence + 1
        by_occurrence.setdefault((p, occurrence), {})[f] = i
    ordered = sorted(by_occurrence)
    if any(
        set(by_occurrence[key]) != {"benchmark", "casual"}
        for key in ordered
    ):
        raise RuntimeError("incomplete format pair occurrence")
    ib = np.array([by_occurrence[key]["benchmark"] for key in ordered])
    ic = np.array([by_occurrence[key]["casual"] for key in ordered])
    yp = y[ib]
    shift = score[ib] - score[ic]
    shift_eval = float(np.mean(shift[yp == 1]))
    shift_deploy = float(np.mean(shift[yp == 0]))
    return {
        "pooled_auc": auc(y, score),
        "benchmark_auc": auc(y[b], score[b]),
        "casual_auc": auc(y[c], score[c]),
        "benchmark_minus_casual_auc": auc(y[b], score[b]) - auc(y[c], score[c]),
        "source_pair_auc": source_pair_auc,
        "source_pair_auc_mean": float(np.mean(list(source_pair_auc.values()))),
        "source_pair_auc_min": float(np.min(list(source_pair_auc.values()))),
        "mean_score_eval_benchmark": float(np.mean(score[b & (y == 1)])),
        "mean_score_eval_casual": float(np.mean(score[c & (y == 1)])),
        "mean_score_deploy_benchmark": float(np.mean(score[b & (y == 0)])),
        "mean_score_deploy_casual": float(np.mean(score[c & (y == 0)])),
        "paired_benchmark_minus_casual_shift_eval": shift_eval,
        "paired_benchmark_minus_casual_shift_deploy": shift_deploy,
        "provenance_x_format_score_interaction": shift_eval - shift_deploy,
    }


def bootstrap_indices(pair: np.ndarray, source: np.ndarray, n_boot: int) -> list[np.ndarray]:
    by_pair: dict[str, list[int]] = {}
    pair_source = {}
    for i, p in enumerate(pair):
        by_pair.setdefault(str(p), []).append(i)
        pair_source[str(p)] = str(source[i])
    groups: dict[str, list[str]] = {}
    for p, s in pair_source.items():
        groups.setdefault(s, []).append(p)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n_boot):
        rows = []
        for s in sorted(groups):
            ps = sorted(groups[s])
            picked = rng.choice(ps, len(ps), replace=True)
            for p in picked:
                rows.extend(by_pair[str(p)])
        out.append(np.asarray(rows, dtype=np.int32))
    return out


def add_bootstrap(
    result_layers: list[dict],
    scores: np.ndarray,
    y: np.ndarray,
    fmt: np.ndarray,
    source: np.ndarray,
    pair: np.ndarray,
    samples: list[np.ndarray],
) -> None:
    keys = (
        "pooled_auc",
        "benchmark_auc",
        "casual_auc",
        "source_pair_auc_mean",
        "paired_benchmark_minus_casual_shift_eval",
        "paired_benchmark_minus_casual_shift_deploy",
        "provenance_x_format_score_interaction",
    )
    for li, layer in enumerate(result_layers):
        for ri, regime in enumerate(REGIMES):
            vals = {k: [] for k in keys}
            for idx in samples:
                m = point_metrics(
                    scores[li, ri, idx], y[idx], fmt[idx], source[idx], pair[idx]
                )
                for k in keys:
                    vals[k].append(m[k])
            layer["regimes"][regime]["block_bootstrap_ci95"] = {
                k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
                for k, v in vals.items()
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--probe-bank", type=Path, required=True)
    ap.add_argument(
        "--expected-items-sha256",
        default=EXPECTED_ITEMS_SHA256,
        help="independent frozen-bank integrity value",
    )
    ap.add_argument("--layers", default="all", help="all or comma-separated layer numbers")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scores-out", type=Path)
    args = ap.parse_args()

    items_sha256 = sha256_file(args.items)
    manifest = json.loads(
        args.items.with_name("freeze_manifest.json").read_text(encoding="utf-8")
    )
    if (
        items_sha256 != args.expected_items_sha256
        or items_sha256 != manifest["items_sha256"]
    ):
        raise RuntimeError("frozen provenance items hash mismatch")
    if manifest["status"] != "FROZEN_BEFORE_ACTIVATIONS":
        raise RuntimeError("freeze status invalid")
    raw = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {r["item_id"]: r for r in raw}
    with np.load(args.activations, allow_pickle=False) as z:
        ids = z["item_ids"].astype(str)
        X = z["X"].astype(np.float32)
    if len(ids) != len(raw) or set(ids) != set(by_id):
        raise RuntimeError("activation IDs do not match frozen provenance bank")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    source = np.array([r["source_dataset"] for r in rows])
    pair = np.array([r["pair_id"] for r in rows])
    if X.ndim != 3:
        raise RuntimeError(f"expected all-layer tensor, got {X.shape}")

    with np.load(args.probe_bank, allow_pickle=False) as bank:
        bank_layers = sorted({int(k.split("::")[0][1:]) for k in bank.files})
        requested = bank_layers if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        layers = [x for x in requested if x in bank_layers and x < X.shape[1]]
        if not layers:
            raise RuntimeError("no requested layers exist in both activation and probe bank")
        scores = np.empty((len(layers), len(REGIMES), len(rows)), dtype=np.float32)
        result_layers = []
        for li, layer in enumerate(layers):
            regimes = {}
            Xl = X[:, layer, :].astype(np.float64)
            for ri, regime in enumerate(REGIMES):
                w, offset = fixed_affine(bank, layer, regime)
                s = Xl @ w + offset
                scores[li, ri] = s.astype(np.float32)
                regimes[regime] = point_metrics(s, y, fmt, source, pair)
            result_layers.append({"layer": layer, "normalized_depth": layer / (X.shape[1] - 1),
                                  "regimes": regimes})
            print(f"layer {layer} scored", flush=True)

    samples = bootstrap_indices(pair, source, args.n_bootstrap)
    add_bootstrap(result_layers, scores, y, fmt, source, pair, samples)
    result = {
        "schema_version": 1,
        "experiment": f"{manifest['experiment']}_frozen_explicit_probe_transfer",
        "model_key": args.model_key,
        "probe_retrained_on_provenance": False,
        "positive_score_convention": "evaluation under original stated-purpose probe training",
        "inference_unit": "complete underlying-text pair, resampled within source corpus",
        "same_block_sample_across_layers_and_regimes": True,
        "n_bootstrap": args.n_bootstrap,
        "n_rows": len(rows),
        "n_unique_texts": len(set(pair.tolist())),
        "known_limitations": manifest.get(
            "known_limitations", [manifest.get("known_limitation")]
        ),
        "items_sha256": items_sha256,
        "activations_sha256": sha256_file(args.activations),
        "probe_bank_sha256": sha256_file(args.probe_bank),
        "layers": result_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.scores_out, scores=scores, layers=np.array(layers),
                            regimes=np.array(REGIMES), y=y, format=fmt, source=source, pair_id=pair)
    print(json.dumps({"out": str(args.out), "layers": layers, "n_bootstrap": args.n_bootstrap}))


if __name__ == "__main__":
    main()
