#!/usr/bin/env python3
"""Residualize every frozen C direction against span(A, B) without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EXPERIMENT_TAG = "prospective_model_extension_C_residualized_against_A_B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("cosine received a zero vector")
    return float(np.dot(left, right) / denominator)


def transform(source_json: Path, source_npz: Path, out_dir: Path) -> dict[str, Any]:
    plan = json.loads(source_json.read_text(encoding="utf-8"))
    if plan.get("plan_npz_sha256") != sha256_file(source_npz):
        raise ValueError("source plan NPZ hash mismatch")
    out_dir.mkdir(parents=True, exist_ok=False)
    out_npz = out_dir / "causal_plan_orthc.npz"
    out_json = out_dir / "causal_plan_orthc.json"
    diagnostics: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    with np.load(source_npz, allow_pickle=False) as archive:
        for name in archive.files:
            arrays[name] = np.asarray(archive[name])
    for fold in plan["folds"]:
        for layer, source_meta in fold["source_layers"].items():
            prefix = f"fold_{fold['fold_id']}_layer_{layer}_"
            a = arrays[prefix + "A"].astype(np.float64)
            b = arrays[prefix + "B"].astype(np.float64)
            c = arrays[prefix + "C"].astype(np.float64)
            basis = np.column_stack([a, b])
            coefficients, _, rank, singular_values = np.linalg.lstsq(basis, c, rcond=None)
            if rank < 2:
                raise ValueError(f"rank-deficient A/B span: {fold['fold_id']} layer {layer}")
            residual = c - basis @ coefficients
            residual_norm = float(np.linalg.norm(residual))
            if not np.isfinite(residual).all() or residual_norm <= 0:
                raise ValueError(f"invalid residual C: {fold['fold_id']} layer {layer}")
            residual32 = residual.astype(np.float32)
            cos_a_after = cosine(residual32.astype(np.float64), a)
            cos_b_after = cosine(residual32.astype(np.float64), b)
            if max(abs(cos_a_after), abs(cos_b_after)) > 1e-6:
                raise ValueError(f"float32 residual orthogonality failure: {fold['fold_id']} layer {layer}")
            arrays[prefix + "C"] = residual32
            diagnostic = {
                "fold_id": fold["fold_id"],
                "layer": int(layer),
                "rank_A_B": int(rank),
                "singular_values_A_B": [float(value) for value in singular_values],
                "coefficients_A_B": [float(value) for value in coefficients],
                "norm_C_before": float(np.linalg.norm(c)),
                "norm_C_after": residual_norm,
                "norm_residual_over_original": residual_norm / float(np.linalg.norm(c)),
                "cos_C_A_before": cosine(c, a),
                "cos_C_B_before": cosine(c, b),
                "cos_C_A_after": cos_a_after,
                "cos_C_B_after": cos_b_after,
            }
            diagnostics.append(diagnostic)
            source_meta["orthogonalized_C"] = diagnostic
            source_meta["norm_C_before_orthogonalization"] = source_meta["norm_C"]
            source_meta["norm_C"] = residual_norm
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, out_npz)
    finally:
        temporary.unlink(missing_ok=True)
    plan["experiment_tag"] = EXPERIMENT_TAG
    plan["orthogonalization"] = {
        "method": "C_residual_equals_C_minus_least_squares_projection_onto_span_A_B",
        "source_plan_json": source_json.as_posix(),
        "source_plan_json_sha256": sha256_file(source_json),
        "source_plan_npz": source_npz.as_posix(),
        "source_plan_npz_sha256": sha256_file(source_npz),
        "script_sha256": sha256_file(Path(__file__)),
        "diagnostics": diagnostics,
    }
    plan["plan_npz"] = out_npz.name
    plan["plan_npz_sha256"] = sha256_file(out_npz)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan-json", type=Path, required=True)
    parser.add_argument("--source-plan-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = transform(args.source_plan_json.resolve(), args.source_plan_npz.resolve(), args.out_dir.resolve())
    diagnostics = plan["orthogonalization"]["diagnostics"]
    print(json.dumps({
        "fold_layers": len(diagnostics),
        "maximum_abs_cosine_after": max(max(abs(row["cos_C_A_after"]), abs(row["cos_C_B_after"])) for row in diagnostics),
        "plan_npz_sha256": plan["plan_npz_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
