"""Template-held-out matched-format subspace test for binding_v2.

For each held scaffold template, remove every complete factorial block that uses
that template. Estimate an uncentered-SVD basis from matched benchmark-minus-
casual block contrasts in the remaining blocks only. Retrain raw and projected
binding probes on those same training blocks and score the complete held blocks.

A row belongs to two scaffold folds (its block has one benchmark and one casual
template), so fold metrics are kept separate and summarized at the template
unit; they are not concatenated as if they were independent OOF predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REGIMES = ("joint", "benchmark", "casual")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)
EXPECTED_ITEMS_SHA256 = "d7ad085991345750384f8877a380e6ca5c23e93dccffa1cf0712137063824cc2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remove(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return X if Q.shape[1] == 0 else X - (X @ Q) @ Q.T


def orth_basis(D: np.ndarray, rank: int) -> np.ndarray:
    """Exact uncentered SVD, matching the existing binding_v1 implementation."""
    _, s, vh = np.linalg.svd(np.asarray(D, dtype=np.float64), full_matrices=False)
    keep = min(rank, int(np.sum(s > max(s[0], 1.0) * 1e-10)), vh.shape[0])
    return vh[:keep].T


def format_contrasts(X, block, fmt, train) -> np.ndarray:
    rows = []
    for blk in sorted(set(block[train].tolist())):
        ii = np.flatnonzero(train & (block == blk))
        b, c = ii[fmt[ii] == "benchmark"], ii[fmt[ii] == "casual"]
        if len(b) != 4 or len(c) != 4:
            raise RuntimeError(f"incomplete factorial block {blk}: {len(b)} / {len(c)}")
        rows.append(X[b].astype(np.float64).mean(0) - X[c].astype(np.float64).mean(0))
    return np.stack(rows)


def fit_scores(Xtr, ytr, ftr, Xte, regime) -> np.ndarray:
    sel = np.ones(len(ytr), bool) if regime == "joint" else ftr == regime
    scaler = StandardScaler().fit(Xtr[sel])
    model = LogisticRegression(**PROBE).fit(scaler.transform(Xtr[sel]), ytr[sel])
    return model.decision_function(scaler.transform(Xte))


def metrics(y, fmt, scores, mask=None) -> dict:
    if mask is None:
        mask = np.ones(len(y), bool)
    b, c = mask & (fmt == "benchmark"), mask & (fmt == "casual")
    def a(score, m):
        return float(roc_auc_score(y[m], score[m]))
    out = {
        "joint_format_auc": a(scores["joint"], mask),
        "benchmark_self_auc": a(scores["benchmark"], b),
        "benchmark_to_casual_auc": a(scores["benchmark"], c),
        "casual_self_auc": a(scores["casual"], c),
        "casual_to_benchmark_auc": a(scores["casual"], b),
    }
    out["cross_format_mean_auc"] = (
        out["benchmark_to_casual_auc"] + out["casual_to_benchmark_auc"]
    ) / 2
    out["transfer_gap"] = out["joint_format_auc"] - out["cross_format_mean_auc"]
    return out


def format_auc(Xtr, ftr, Xte, fte) -> float:
    ytr, yte = (ftr == "benchmark").astype(int), (fte == "benchmark").astype(int)
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(**PROBE).fit(scaler.transform(Xtr), ytr)
    return float(roc_auc_score(yte, model.decision_function(scaler.transform(Xte))))


def summarize(per_template: dict[str, dict]) -> dict:
    scalar_keys = sorted(per_template[next(iter(per_template))])
    out = {}
    for key in scalar_keys:
        vals = np.array([per_template[t][key] for t in per_template], dtype=float)
        out[key] = {
            "template_mean": float(np.mean(vals)),
            "template_min": float(np.min(vals)),
            "template_max": float(np.max(vals)),
            "template_values": {t: float(per_template[t][key]) for t in sorted(per_template)},
        }
    return out


def flatten_delta(projected: dict, raw: dict) -> dict:
    return {f"delta_{k}": projected[k] - raw[k] for k in raw}


def analyze_layer(layer, X, y, fmt, arm, hop, block, template, ranks):
    block_templates: dict[str, set[str]] = {}
    template_format = {}
    for b, t, f in zip(block, template, fmt):
        block_templates.setdefault(str(b), set()).add(str(t))
        template_format[str(t)] = str(f)
    templates = sorted(template_format)
    by_rank = {str(rank): {} for rank in ranks}
    max_rank = max(ranks)
    for held in templates:
        held_blocks = {b for b, ts in block_templates.items() if held in ts}
        test = np.array([b in held_blocks for b in block])
        train = ~test
        D = format_contrasts(X, block, fmt, train)
        Qmax = orth_basis(D, max_rank)
        if Qmax.shape[1] < max_rank:
            raise RuntimeError(f"rank-deficient basis for {held}: {Qmax.shape}")
        raw_train, raw_test = X[train], X[test]
        raw_scores = {
            r: fit_scores(raw_train, y[train], fmt[train], raw_test, r)
            for r in REGIMES
        }
        raw_all = metrics(y[test], fmt[test], raw_scores)
        raw_conditions = {
            f"{a}:h{h}": metrics(
                y[test], fmt[test], raw_scores,
                (arm[test] == a) & (hop[test] == h),
            )
            for a in ("consequence", "neutral") for h in (0, 1, 2)
        }
        raw_fmt_auc = format_auc(raw_train, fmt[train], raw_test, fmt[test])
        for rank in ranks:
            Q = Qmax[:, :rank]
            proj_train, proj_test = remove(raw_train, Q), remove(raw_test, Q)
            proj_scores = {
                r: fit_scores(proj_train, y[train], fmt[train], proj_test, r)
                for r in REGIMES
            }
            projected = metrics(y[test], fmt[test], proj_scores)
            projected_conditions = {
                f"{a}:h{h}": metrics(
                    y[test], fmt[test], proj_scores,
                    (arm[test] == a) & (hop[test] == h),
                )
                for a in ("consequence", "neutral") for h in (0, 1, 2)
            }
            rec = {
                "held_template_format": template_format[held],
                "n_held_blocks": len(held_blocks),
                "n_train_blocks": len(block_templates) - len(held_blocks),
                "basis_rank": rank,
                "basis_n_training_contrasts": len(D),
                "raw": raw_all,
                "projected": projected,
                "projected_minus_raw": flatten_delta(projected, raw_all),
                "raw_format_auc": raw_fmt_auc,
                "projected_format_auc": format_auc(proj_train, fmt[train], proj_test, fmt[test]),
                "delta_format_auc": format_auc(proj_train, fmt[train], proj_test, fmt[test]) - raw_fmt_auc,
                "raw_by_arm_hop": raw_conditions,
                "projected_by_arm_hop": projected_conditions,
            }
            by_rank[str(rank)][held] = rec
        print(f"layer {layer} held template {held} complete", flush=True)

    result = {"layer": layer, "by_rank": {}}
    for rank in ranks:
        records = by_rank[str(rank)]
        pooled_raw = {t: r["raw"] for t, r in records.items()}
        pooled_proj = {t: r["projected"] for t, r in records.items()}
        pooled_delta = {t: r["projected_minus_raw"] for t, r in records.items()}
        format_values = {
            t: {"raw_format_auc": r["raw_format_auc"],
                "projected_format_auc": r["projected_format_auc"],
                "delta_format_auc": r["delta_format_auc"]}
            for t, r in records.items()
        }
        result["by_rank"][str(rank)] = {
            "per_template": records,
            "raw_template_summary": summarize(pooled_raw),
            "projected_template_summary": summarize(pooled_proj),
            "projected_minus_raw_template_summary": summarize(pooled_delta),
            "format_auc_template_summary": summarize(format_values),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--layers", required=True, help="comma-separated layer numbers")
    ap.add_argument("--ranks", default="64")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if sha256_file(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("frozen binding_v2 items hash mismatch")
    raw = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {r["item_id"]: r for r in raw}
    with np.load(args.activations, allow_pickle=False) as z:
        ids, X = z["item_ids"].astype(str), z["X"].astype(np.float32)
    if X.ndim != 3 or len(ids) != len(raw) or set(ids) != set(by_id):
        raise RuntimeError("activation tensor does not match frozen bank")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    arm = np.array([r["binding_arm"] for r in rows])
    hop = np.array([r["binding_hop"] for r in rows], dtype=np.int8)
    block = np.array([r["binding_block_id"] for r in rows])
    template = np.array([r["template_id"] for r in rows])
    layers = [int(x) for x in args.layers.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    out_layers = []
    for layer in layers:
        out_layers.append(analyze_layer(layer, X[:, layer, :], y, fmt, arm, hop,
                                        block, template, ranks))
    result = {
        "schema_version": 1,
        "experiment": "binding_v2_template_heldout_format_subspace",
        "model_key": args.model_key,
        "basis": "exact uncentered SVD of matched benchmark-minus-casual complete-block contrasts",
        "cross_fitting": "held template's complete blocks excluded from basis and probe training",
        "inference_unit": "scaffold template",
        "items_sha256": sha256_file(args.items),
        "activations_sha256": sha256_file(args.activations),
        "ranks": ranks,
        "layers": out_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "layers": layers, "ranks": ranks}))


if __name__ == "__main__":
    main()
