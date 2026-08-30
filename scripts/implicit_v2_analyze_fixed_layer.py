"""Run both implicit-v2 analyses at one pre-specified validation layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from implicit_v2_analyze import (EXPECTED_ITEMS_SHA256, REGIMES, analyze_layer,
                                 sha256, summarize_families)  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--explicit-bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=18)
    args = ap.parse_args()
    if sha256(args.items) != EXPECTED_ITEMS_SHA256:
        raise RuntimeError("implicit-v2 frozen bank hash mismatch")
    rows_raw = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(args.activations, allow_pickle=False) as z:
        ids, X = z["item_ids"].astype(str), z["X"].astype(np.float32)
    by_id = {r["item_id"]: r for r in rows_raw}
    if len(ids) != 3840 or set(ids) != set(by_id) or X.ndim != 2:
        raise RuntimeError(f"activation/frozen-bank mismatch: ids={len(ids)} X={X.shape}")
    rows = [by_id[x] for x in ids]
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    family = np.array([r["implicit_family"] for r in rows])
    fixed_family, implicit_family, fixed_scores, implicit_scores = analyze_layer(
        args.layer, X, y, fmt, family, args.explicit_bank, args.jobs)
    score_path = args.out.with_suffix(".scores.npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_path, item_ids=ids, regimes=np.asarray(REGIMES),
                        fixed_explicit=fixed_scores, implicit_family_heldout=implicit_scores)
    result = {
        "schema_version": 1, "mode": "development", "model_key": args.model_key,
        "pre_specified_layer": args.layer, "n_items": 3840,
        "n_scenario_pairs": 24, "n_independent_scenario_families": 6,
        "inference_unit": "scenario_family", "items_sha256": sha256(args.items),
        "activations_sha256": sha256(args.activations),
        "explicit_probe_bank_sha256": sha256(args.explicit_bank),
        "fixed_explicit_to_implicit": {
            "probe_retrained_on_implicit": False, "per_family": fixed_family,
            "family_unit_summary": summarize_families(fixed_family)},
        "implicit_to_implicit": {
            "cv": "leave-one-entire-scenario-family-out", "per_family": implicit_family,
            "family_unit_summary": summarize_families(implicit_family)},
        "scores": str(score_path), "scores_sha256": sha256(score_path),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "scores": str(score_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
