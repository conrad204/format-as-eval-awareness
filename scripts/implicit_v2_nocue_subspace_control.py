"""Project implicit-v2 through a payload-disjoint no-cue format subspace.

Q_nocue uses only original no-purpose benchmark-minus-casual contrasts and
excludes all 40 payloads in implicit-v2. Implicit probes remain
leave-one-entire-scenario-family-out. Family is the inference unit.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import orth_basis, remove  # noqa: E402
from implicit_v2_analyze import REGIMES, fit_score, metrics, sha256, summarize_families  # noqa: E402


def projected_layer(X: np.ndarray, Q: np.ndarray, y: np.ndarray, fmt: np.ndarray,
                    family: np.ndarray, jobs: int) -> tuple[dict, np.ndarray]:
    Xt = remove(X, Q).astype(np.float32)
    families = sorted(set(family))
    scores = {regime: np.full(len(y), np.nan, np.float32) for regime in REGIMES}
    tasks = [(held, family == held, regime) for held in families for regime in REGIMES]
    fitted = Parallel(n_jobs=min(jobs, len(tasks)), backend="threading")(
        delayed(fit_score)(Xt, y, fmt, family != held, test, regime)
        for held, test, regime in tasks)
    for (_, test, regime), (got, values) in zip(tasks, fitted):
        assert regime == got
        scores[regime][test] = values
    per_family = {held: metrics(y, fmt, scores["joint"], scores["benchmark"], scores["casual"],
                                family == held) for held in families}
    return per_family, np.stack([scores[r] for r in REGIMES])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True, choices=(1, 4, 16, 64))
    ap.add_argument("--original-items", type=Path, required=True)
    ap.add_argument("--original-activations", type=Path, required=True)
    ap.add_argument("--implicit-items", type=Path, required=True)
    ap.add_argument("--implicit-activations", type=Path, required=True)
    ap.add_argument("--raw-analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=18)
    args = ap.parse_args()

    implicit_rows_raw = [json.loads(x) for x in args.implicit_items.read_text(encoding="utf-8").splitlines() if x.strip()]
    freeze = json.loads(args.implicit_items.with_name("freeze_manifest.json").read_text(encoding="utf-8"))
    excluded_payloads = set(freeze["selected_payload_block_ids"])
    with np.load(args.implicit_activations, allow_pickle=False) as z:
        implicit_ids, Xi = z["item_ids"].astype(str), z["X"].astype(np.float32)
    iby = {r["item_id"]: r for r in implicit_rows_raw}
    rows = [iby[x] for x in implicit_ids]
    y = np.array([r["label"] for r in rows], np.int8)
    fmt = np.array([r["format"] for r in rows])
    family = np.array([r["implicit_family"] for r in rows])

    original_rows_raw = [json.loads(x) for x in args.original_items.read_text(encoding="utf-8").splitlines() if x.strip()]
    oby = {r["item_id"]: r for r in original_rows_raw}
    with zipfile.ZipFile(args.original_activations) as z:
        original_ids = np.load(z.open("item_ids.npy")).astype(str)
    original_rows = [oby[x] for x in original_ids]
    with np.load(args.original_activations, allow_pickle=False) as z:
        Xo = z["X"].astype(np.float32)
    by_payload_format: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(original_rows):
        if row["render_slot"] == "no_cue" and row["payload_block_id"] not in excluded_payloads:
            by_payload_format.setdefault((row["payload_block_id"], row["format"]), []).append(i)
    basis_payloads = sorted({p for p, _ in by_payload_format}
                            - excluded_payloads)
    D_by_layer = []
    for layer in range(Xo.shape[1]):
        D = np.asarray([Xo[by_payload_format[(p, "benchmark")], layer].mean(axis=0)
                        - Xo[by_payload_format[(p, "casual")], layer].mean(axis=0)
                        for p in basis_payloads])
        D_by_layer.append(orth_basis(D, args.rank))
    del Xo

    raw = json.loads(args.raw_analysis.read_text(encoding="utf-8"))
    raw_layers = raw["implicit_to_implicit"]["layers"]
    layers, scores = [], []
    for layer in range(Xi.shape[1]):
        per_family, layer_scores = projected_layer(Xi[:, layer], D_by_layer[layer], y, fmt,
                                                   family, args.jobs)
        projected_summary = summarize_families(per_family)
        raw_family = raw_layers[layer]["per_family"]
        delta_family = {}
        for name in sorted(per_family):
            delta_family[name] = {
                "delta_joint_format_auc": per_family[name]["joint_format_auc"] - raw_family[name]["joint_format_auc"],
                "delta_cross_format_mean_auc": per_family[name]["cross_format_mean_auc"] - raw_family[name]["cross_format_mean_auc"],
                "delta_transfer_gap": per_family[name]["transfer_gap"] - raw_family[name]["transfer_gap"],
            }
        layers.append({"layer": layer, "projected_per_family": per_family,
                       "projected_family_unit_summary": projected_summary,
                       "projected_minus_raw_per_family": delta_family,
                       "projected_minus_raw_family_unit_summary": summarize_families(delta_family)})
        scores.append(layer_scores)
        print(f"rank={args.rank} layer={layer}/31", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    score_path = args.out.with_suffix(".scores.npz")
    np.savez_compressed(score_path, item_ids=implicit_ids, regimes=np.asarray(REGIMES),
                        projected_scores=np.stack(scores, axis=1))
    result = {"schema_version": 1, "mode": "development", "model": "llama31_8b",
              "rank": args.rank, "basis": "payload-disjoint original no-cue benchmark-minus-casual contrasts",
              "n_basis_payloads": len(basis_payloads), "n_excluded_implicit_payloads": len(excluded_payloads),
              "implicit_cv": "leave-one-entire-scenario-family-out", "inference_unit": "scenario_family",
              "n_independent_families": 6, "original_activations_sha256": sha256(args.original_activations),
              "implicit_activations_sha256": sha256(args.implicit_activations), "layers": layers,
              "scores": str(score_path), "scores_sha256": sha256(score_path)}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
