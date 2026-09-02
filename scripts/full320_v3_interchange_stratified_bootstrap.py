#!/usr/bin/env python3
"""Descriptive stratification of the running full320-v3 interchange records.

This is deliberately not a causal specificity test: ``real_swap`` projects the
natural B±C donor difference through Q_B, while ``random_control`` is not
magnitude matched.  The analysis asks only whether downstream consequences of
that real Q_B-mediated swap differ when stated purpose is evaluation versus
deployment and when the swap is benchmark->casual versus casual->benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 2000
RNG_SEED = 20260829
PURPOSES = ("all", "evaluation", "deployment")
DIRECTIONS = ("all", "b2c", "c2b")
CONDITIONS = ("real_swap", "random_control")
MIN_ROWS = 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: Path, allow_partial_tail: bool) -> tuple[list[dict], bool]:
    records = []
    skipped_partial_tail = False
    seen = set()
    with path.open("rb") as handle:
        lines = handle.readlines()
    for pos, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            if allow_partial_tail and pos == len(lines) - 1 and not raw.endswith(b"\n"):
                skipped_partial_tail = True
                continue
            raise
        key = (row["layer"], row["fold"], row["src_row"], row["condition"])
        if key in seen:
            raise ValueError(f"duplicate interchange record {key}")
        seen.add(key)
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"unexpected condition {row['condition']}")
        if row["intended_purpose"] not in PURPOSES[1:]:
            raise ValueError(f"unexpected purpose {row['intended_purpose']}")
        if row["direction"] not in DIRECTIONS[1:]:
            raise ValueError(f"unexpected direction {row['direction']}")
        records.append(row)
    return records, skipped_partial_tail


def row_metrics(row: dict) -> dict[str, float]:
    purpose_shift = abs(float(row["purpose_score_patched"]) - float(row["purpose_score_baseline"]))
    target_gap = float(row["format_score_baseline_tgt"]) - float(row["format_score_baseline_src"])
    actual_shift = float(row["format_score_patched"]) - float(row["format_score_baseline_src"])
    format_iia = np.nan if target_gap == 0 else float(np.sign(actual_shift) == np.sign(target_gap))
    return {"purpose_abs_shift": purpose_shift, "format_iia": format_iia}


def subgroup_keys(row: dict):
    purpose, direction = row["intended_purpose"], row["direction"]
    return (
        (int(row["layer"]), "all", "all"),
        (int(row["layer"]), purpose, "all"),
        (int(row["layer"]), "all", direction),
        (int(row["layer"]), purpose, direction),
    )


def summarize(records: list[dict], n_boot: int) -> dict:
    layers = sorted({int(r["layer"]) for r in records})
    blocks = np.asarray(sorted({str(r["payload_block_id"]) for r in records}))
    block_i = {block: i for i, block in enumerate(blocks)}
    group_keys = sorted({key for row in records for key in subgroup_keys(row)})
    group_i = {key: i for i, key in enumerate(group_keys)}
    condition_i = {name: i for i, name in enumerate(CONDITIONS)}
    metric_i = {name: i for i, name in enumerate(("purpose_abs_shift", "format_iia"))}
    shape = (len(group_keys), len(CONDITIONS), len(metric_i), len(blocks))
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)

    for row in records:
        ci = condition_i[row["condition"]]
        bi = block_i[str(row["payload_block_id"])]
        metrics = row_metrics(row)
        for key in subgroup_keys(row):
            gi = group_i[key]
            for metric, value in metrics.items():
                if np.isfinite(value):
                    mi = metric_i[metric]
                    sums[gi, ci, mi, bi] += value
                    counts[gi, ci, mi, bi] += 1

    total_sum = sums.sum(axis=3)
    total_count = counts.sum(axis=3)
    if np.any((total_count[:, 0] == 0) | (total_count[:, 1] == 0)):
        raise ValueError("a reported subgroup is missing a condition or metric")
    means = total_sum / total_count

    boot_means = np.empty((len(group_keys), len(CONDITIONS), len(metric_i), n_boot), dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    for rep in range(n_boot):
        sampled = rng.choice(len(blocks), size=len(blocks), replace=True)
        weights = np.bincount(sampled, minlength=len(blocks)).astype(np.float64)
        numerator = np.tensordot(sums, weights, axes=(3, 0))
        denominator = np.tensordot(counts, weights, axes=(3, 0))
        boot_means[:, :, :, rep] = numerator / denominator

    def ci(values):
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    by_layer: dict[str, dict] = defaultdict(dict)
    for key, gi in group_i.items():
        layer, purpose, direction = key
        label = f"purpose={purpose}|direction={direction}"
        entry = {
            "purpose": purpose,
            "direction": direction,
            "adequate": bool(np.all(total_count[gi] >= MIN_ROWS)),
            "minimum_required_rows_per_condition_metric": MIN_ROWS,
            "metrics": {},
        }
        for metric, mi in metric_i.items():
            real = means[gi, condition_i["real_swap"], mi]
            control = means[gi, condition_i["random_control"], mi]
            boot_real = boot_means[gi, condition_i["real_swap"], mi]
            boot_control = boot_means[gi, condition_i["random_control"], mi]
            if metric == "purpose_abs_shift":
                gap = control - real
                boot_gap = boot_control - boot_real
                gap_definition = "random_control - real_swap; positive means less purpose disruption under real swap"
            else:
                gap = real - control
                boot_gap = boot_real - boot_control
                gap_definition = "real_swap - random_control; positive means stronger format interchange alignment under real swap"
            entry["metrics"][metric] = {
                "real_swap": float(real),
                "real_swap_ci95": ci(boot_real),
                "real_swap_n": int(total_count[gi, condition_i["real_swap"], mi]),
                "random_control": float(control),
                "random_control_ci95": ci(boot_control),
                "random_control_n": int(total_count[gi, condition_i["random_control"], mi]),
                "gap": float(gap),
                "gap_ci95": ci(boot_gap),
                "gap_definition": gap_definition,
            }
        by_layer[str(layer)][label] = entry

    def contrast_entry(key_a, key_b, metric: str) -> dict:
        ai, bi = group_i[key_a], group_i[key_b]
        mi = metric_i[metric]
        real_i, control_i = condition_i["real_swap"], condition_i["random_control"]
        real_difference = means[ai, real_i, mi] - means[bi, real_i, mi]
        control_difference = means[ai, control_i, mi] - means[bi, control_i, mi]
        boot_real_difference = boot_means[ai, real_i, mi] - boot_means[bi, real_i, mi]
        boot_control_difference = boot_means[ai, control_i, mi] - boot_means[bi, control_i, mi]
        if metric == "purpose_abs_shift":
            gap_difference = control_difference - real_difference
            boot_gap_difference = boot_control_difference - boot_real_difference
        else:
            gap_difference = real_difference - control_difference
            boot_gap_difference = boot_real_difference - boot_control_difference
        return {
            "real_swap_difference": float(real_difference),
            "real_swap_difference_ci95": ci(boot_real_difference),
            "random_control_difference": float(control_difference),
            "random_control_difference_ci95": ci(boot_control_difference),
            "gap_difference": float(gap_difference),
            "gap_difference_ci95": ci(boot_gap_difference),
        }

    contrasts_by_layer: dict[str, dict] = defaultdict(dict)
    for layer in layers:
        for direction in DIRECTIONS:
            label = f"evaluation_minus_deployment|direction={direction}"
            contrasts_by_layer[str(layer)][label] = {
                metric: contrast_entry(
                    (layer, "evaluation", direction),
                    (layer, "deployment", direction),
                    metric,
                )
                for metric in metric_i
            }
        for purpose in PURPOSES:
            label = f"b2c_minus_c2b|purpose={purpose}"
            contrasts_by_layer[str(layer)][label] = {
                metric: contrast_entry(
                    (layer, purpose, "b2c"),
                    (layer, purpose, "c2b"),
                    metric,
                )
                for metric in metric_i
            }

    return {
        "layers": layers,
        "n_layers": len(layers),
        "n_blocks": len(blocks),
        "n_records": len(records),
        "by_layer": dict(by_layer),
        "contrasts_by_layer": dict(contrasts_by_layer),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--allow-partial-tail", action="store_true")
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise ValueError("n_boot must be >=1000")
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    records, skipped = load_records(args.input, args.allow_partial_tail)
    analysis = summarize(records, args.n_boot)
    result = {
        "schema_version": 1,
        "mode": "development_descriptive",
        "input_path": str(args.input),
        "input_sha256": sha256_file(args.input),
        "partial_tail_skipped": skipped,
        "n_boot": args.n_boot,
        "rng_seed": RNG_SEED,
        "bootstrap": "one payload-block resample per replicate reused across every completed layer, purpose stratum, direction stratum, condition, and metric",
        "interpretation_boundary": "Descriptive/hypothesis-generating only. real_swap contains natural B±C donor content projected through Q_B; random_control is rank-matched but not magnitude-matched. No pure-C or behavioral-purpose causal claim.",
        **analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
