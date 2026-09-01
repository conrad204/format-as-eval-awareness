"""Test selective external transfer of a controlled evaluation-purpose direction.

This script never fits on provenance_v3. Directions are estimated at one layer
from the frozen full320-v3 factorial bank, then applied unchanged to the frozen
provenance bank. See results/ea_specificity_protocol_20260831.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_bound_regrowth_bootstrap import stream_extract_layers  # noqa: E402

TRAIN_SHA256 = "5528819d74e696d1a595062047a829836dfde1d2a45287b06a5ac1ba5d1b6c67"
PROV_SHA256 = "2a574bf1bf0cd5cc8ec53c97e2bf69e9246316b1a3d5364e370acc0f8fcddc7e"
SEED = 20260831
NAMED = ("purpose", "purpose_orthogonal_to_format", "format", "nocue_format")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_layer(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    X, _ = stream_extract_layers(path, [layer])
    with zipfile.ZipFile(path) as z:
        ids = np.load(z.open("item_ids.npy")).astype(str)
    return np.ascontiguousarray(X[:, 0, :], dtype=np.float64), ids


def ordered_rows(items: Path, ids: np.ndarray) -> list[dict]:
    raw = load_rows(items)
    by_id = {str(r["item_id"]): r for r in raw}
    if len(ids) != len(raw) or set(ids.tolist()) != set(by_id):
        raise RuntimeError(f"activation/item mismatch for {items}")
    return [by_id[x] for x in ids]


def grouped_contrasts(
    X: np.ndarray,
    rows: list[dict],
    mask: np.ndarray,
    positive,
    negative,
) -> tuple[np.ndarray, list[str]]:
    blocks = sorted({str(r["payload_block_id"]) for r, keep in zip(rows, mask) if keep})
    out = np.empty((len(blocks), X.shape[1]), dtype=np.float64)
    block_arr = np.array([str(r["payload_block_id"]) for r in rows])
    for i, block in enumerate(blocks):
        base = mask & (block_arr == block)
        ip = base & positive
        im = base & negative
        if not np.any(ip) or not np.any(im):
            raise RuntimeError(f"incomplete contrast block {block}")
        out[i] = X[ip].mean(axis=0) - X[im].mean(axis=0)
    return out, blocks


def fast_auc_matrix(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact row-wise AUC for a score matrix with a shared binary label vector."""
    y = np.asarray(y, dtype=np.int8)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[0], np.nan)
    order = np.argsort(scores, axis=1, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1, dtype=np.float64)
    # Ties are vanishingly rare for affine scores; detect rather than silently bias.
    sorted_s = np.take_along_axis(scores, order, axis=1)
    if np.any(np.diff(sorted_s, axis=1) == 0):
        from scipy.stats import rankdata
        ranks = np.vstack([rankdata(s, method="average") for s in scores])
    return (ranks[:, y == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def source_pair_masks(y: np.ndarray, source: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for es in sorted(set(source[y == 1])):
        for ds in sorted(set(source[y == 0])):
            out[f"{es}|{ds}"] = (source == es) | (source == ds)
    return out


def metrics(scores: np.ndarray, y: np.ndarray, fmt: np.ndarray, source: np.ndarray) -> dict:
    pair_masks = source_pair_masks(y, source)
    pair = {k: fast_auc_matrix(scores[:, m], y[m]) for k, m in pair_masks.items()}
    pair_stack = np.stack(list(pair.values()), axis=1)
    return {
        "pooled_auc": fast_auc_matrix(scores, y),
        "benchmark_auc": fast_auc_matrix(scores[:, fmt == "benchmark"], y[fmt == "benchmark"]),
        "casual_auc": fast_auc_matrix(scores[:, fmt == "casual"], y[fmt == "casual"]),
        "source_pair_auc": pair,
        "source_pair_auc_min": pair_stack.min(axis=1),
        "source_pair_auc_mean": pair_stack.mean(axis=1),
    }


def bootstrap_indices(pair: np.ndarray, source: np.ndarray, n_boot: int) -> list[np.ndarray]:
    by_pair: dict[str, list[int]] = {}
    pair_source = {}
    for i, (p, s) in enumerate(zip(pair, source)):
        by_pair.setdefault(str(p), []).append(i)
        pair_source[str(p)] = str(s)
    groups: dict[str, list[str]] = {}
    for p, s in pair_source.items():
        groups.setdefault(s, []).append(p)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(n_boot):
        idx = []
        for s in sorted(groups):
            ps = sorted(groups[s])
            for p in rng.choice(ps, len(ps), replace=True):
                idx.extend(by_pair[str(p)])
        samples.append(np.asarray(idx, dtype=np.int32))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--train-items", type=Path, required=True)
    ap.add_argument("--train-activations", type=Path, required=True)
    ap.add_argument("--provenance-items", type=Path, required=True)
    ap.add_argument("--provenance-activations", type=Path, required=True)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train_hash = sha256(args.train_items)
    prov_hash = sha256(args.provenance_items)
    if train_hash != TRAIN_SHA256:
        raise RuntimeError(f"training items hash mismatch: {train_hash}")
    if prov_hash != PROV_SHA256:
        raise RuntimeError(f"provenance items hash mismatch: {prov_hash}")

    Xt, train_ids = load_layer(args.train_activations, args.layer)
    train_rows = ordered_rows(args.train_items, train_ids)
    cue = np.array([r.get("cue_style") for r in train_rows])
    y_train = np.array([
        -1 if r.get("label") is None else int(r["label"]) for r in train_rows
    ], dtype=np.int8)
    fmt_train = np.array([r["format"] for r in train_rows])
    stated = cue == "stated"
    nocue = cue == "none"
    if stated.sum() != 7680 or nocue.sum() != 3840:
        raise RuntimeError(f"unexpected train strata: stated={stated.sum()} nocue={nocue.sum()}")

    variance = Xt[stated].var(axis=0)
    variance = np.maximum(variance, np.finfo(np.float64).eps)
    purpose_c, purpose_blocks = grouped_contrasts(
        Xt, train_rows, stated, y_train == 1, y_train == 0
    )
    format_c, format_blocks = grouped_contrasts(
        Xt, train_rows, stated, fmt_train == "benchmark", fmt_train == "casual"
    )
    nocue_c, nocue_blocks = grouped_contrasts(
        Xt, train_rows, nocue, fmt_train == "benchmark", fmt_train == "casual"
    )
    if purpose_blocks != format_blocks or purpose_blocks != nocue_blocks:
        raise RuntimeError("training block sets differ across contrasts")

    w_purpose = purpose_c.mean(axis=0) / variance
    w_format = format_c.mean(axis=0) / variance
    w_nocue = nocue_c.mean(axis=0) / variance
    w_orth = w_purpose - w_format * (w_purpose @ w_format) / (w_format @ w_format)

    rng = np.random.default_rng(SEED)
    base_signs = np.r_[np.full(len(purpose_blocks) // 2, -1.0),
                       np.full(len(purpose_blocks) - len(purpose_blocks) // 2, 1.0)]
    signs = np.vstack([rng.permutation(base_signs) for _ in range(args.n_null)])
    # Exact balance prevents global-label imbalance from defining the null.
    null_delta = signs @ purpose_c / len(purpose_blocks)
    null_w = null_delta / variance[None, :]
    names = list(NAMED) + [f"purpose_block_signflip_{i:03d}" for i in range(args.n_null)]
    W = np.vstack([w_purpose, w_orth, w_format, w_nocue, null_w])
    norms = np.linalg.norm(W, axis=1)
    if np.any(norms == 0) or not np.all(np.isfinite(W)):
        raise RuntimeError("invalid direction")
    W /= norms[:, None]

    del Xt, purpose_c, format_c, nocue_c, null_delta, null_w
    Xp, prov_ids = load_layer(args.provenance_activations, args.layer)
    prov_rows = ordered_rows(args.provenance_items, prov_ids)
    y = np.array([r["label"] for r in prov_rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in prov_rows])
    source = np.array([r["source_dataset"] for r in prov_rows])
    pair = np.array([r["pair_id"] for r in prov_rows])
    scores = W @ Xp.T
    point = metrics(scores, y, fmt, source)

    samples = bootstrap_indices(pair, source, args.n_bootstrap)
    boot_named = {k: [] for k in ("pooled_auc", "benchmark_auc", "casual_auc", "source_pair_auc_min", "source_pair_auc_mean")}
    delta_boot = {"purpose_minus_format_pooled": [], "purpose_minus_format_min_pair": [],
                  "orth_minus_format_pooled": [], "orth_minus_format_min_pair": []}
    for rep, idx in enumerate(samples):
        m = metrics(scores[:4, idx], y[idx], fmt[idx], source[idx])
        for key in boot_named:
            boot_named[key].append(m[key])
        delta_boot["purpose_minus_format_pooled"].append(m["pooled_auc"][0] - m["pooled_auc"][2])
        delta_boot["purpose_minus_format_min_pair"].append(m["source_pair_auc_min"][0] - m["source_pair_auc_min"][2])
        delta_boot["orth_minus_format_pooled"].append(m["pooled_auc"][1] - m["pooled_auc"][2])
        delta_boot["orth_minus_format_min_pair"].append(m["source_pair_auc_min"][1] - m["source_pair_auc_min"][2])
        if (rep + 1) % 100 == 0:
            print(f"bootstrap {rep + 1}/{args.n_bootstrap}", flush=True)

    records = []
    for i, name in enumerate(names):
        rec = {
            "name": name,
            "pooled_auc": float(point["pooled_auc"][i]),
            "benchmark_auc": float(point["benchmark_auc"][i]),
            "casual_auc": float(point["casual_auc"][i]),
            "source_pair_auc_min": float(point["source_pair_auc_min"][i]),
            "source_pair_auc_mean": float(point["source_pair_auc_mean"][i]),
            "source_pair_auc": {k: float(v[i]) for k, v in point["source_pair_auc"].items()},
        }
        if i < 4:
            rec["block_bootstrap_ci95"] = {
                k: [float(x) for x in np.quantile(np.asarray(v)[:, i], [0.025, 0.975])]
                for k, v in boot_named.items()
            }
        records.append(rec)

    null_min = point["source_pair_auc_min"][4:]
    null_pool = point["pooled_auc"][4:]
    p_min = (1 + int(np.sum(null_min >= point["source_pair_auc_min"][0]))) / (args.n_null + 1)
    p_pool = (1 + int(np.sum(null_pool >= point["pooled_auc"][0]))) / (args.n_null + 1)
    result = {
        "schema_version": 1,
        "experiment": "controlled_ea_direction_external_provenance_specificity",
        "status": "EXPLORATORY_FROZEN_BEFORE_CONTROL_OUTCOMES",
        "model_key": args.model_key,
        "layer": args.layer,
        "direction_fit_on_provenance": False,
        "layer_selected_on_provenance": False,
        "positive_score_convention": "evaluation minus deployment in controlled full320 training",
        "n_training_blocks": len(purpose_blocks),
        "n_null": args.n_null,
        "n_bootstrap": args.n_bootstrap,
        "same_block_sample_across_named_directions": True,
        "train_items_sha256": train_hash,
        "train_activations_sha256": sha256(args.train_activations),
        "provenance_items_sha256": prov_hash,
        "provenance_activations_sha256": sha256(args.provenance_activations),
        "directions": records,
        "null_summary": {
            "purpose_min_pair_empirical_p_greater_equal": p_min,
            "purpose_pooled_empirical_p_greater_equal": p_pool,
            "min_pair_auc_quantiles": {str(q): float(np.quantile(null_min, q)) for q in (0.025, 0.5, 0.975)},
            "pooled_auc_quantiles": {str(q): float(np.quantile(null_pool, q)) for q in (0.025, 0.5, 0.975)},
        },
        "named_deltas": {
            "purpose_minus_format_pooled": {
                "point": float(point["pooled_auc"][0] - point["pooled_auc"][2]),
                "ci95": [float(x) for x in np.quantile(
                    delta_boot["purpose_minus_format_pooled"], [0.025, 0.975]
                )],
            },
            "purpose_minus_format_min_pair": {
                "point": float(
                    point["source_pair_auc_min"][0] - point["source_pair_auc_min"][2]
                ),
                "ci95": [float(x) for x in np.quantile(
                    delta_boot["purpose_minus_format_min_pair"], [0.025, 0.975]
                )],
            },
            "orth_minus_format_pooled": {
                "point": float(point["pooled_auc"][1] - point["pooled_auc"][2]),
                "ci95": [float(x) for x in np.quantile(
                    delta_boot["orth_minus_format_pooled"], [0.025, 0.975]
                )],
            },
            "orth_minus_format_min_pair": {
                "point": float(
                    point["source_pair_auc_min"][1] - point["source_pair_auc_min"][2]
                ),
                "ci95": [float(x) for x in np.quantile(
                    delta_boot["orth_minus_format_min_pair"], [0.025, 0.975]
                )],
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "purpose": records[0], "null": result["null_summary"]}, indent=2))


if __name__ == "__main__":
    main()
