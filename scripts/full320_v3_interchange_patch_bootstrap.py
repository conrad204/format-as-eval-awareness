"""Correlation-aware analysis of the full-layer interchange-patching raw records
produced by scripts/full320_v3_interchange_patch_full_layer.py.

Accepts one or more input JSONL files (raw records may be split across files
by condition, e.g. real_swap/random_control in one file and matched_control
in a companion file from a `--conditions matched_control` run) and reports,
for every control condition present against real_swap:

  H1 (purpose protected): mean |purpose_shift| SMALLER under real_swap than
     under the control.
  H2 (format carried): format interchange-intervention accuracy (IIA) --
     whether sign(format_score_patched - format_score_baseline_src) matches
     sign(format_score_baseline_tgt - format_score_baseline_src) -- HIGHER
     under real_swap than under the control.

Layers are NOT independent replicates (residual-stream continuity means
adjacent layers share almost all their variance): every bootstrap replicate
resamples PAYLOAD BLOCKS ONCE and reuses that SAME resampled block set to
recompute every statistic at EVERY layer AND for EVERY control comparison, so
all reported intervals share the true cross-layer correlation structure
instead of pretending each layer (or each comparison) is an independent draw
(the same fix already applied to
scripts/full320_v3_erasure_dose_response_bootstrap.py's paired gap CI). This
script also reports, per control comparison, a joint family-wise-valid
worst-layer omnibus statistic derived from the SAME shared draws -- it does
NOT report or threshold a per-layer significance count, which would be
exactly the layers-as-independent-replicates pseudoreplication RULES.md
prohibits.

Record-count sanity: prints, per condition, the observed record count against
the expected n_layers * n_pairs (pairs counted from the real_swap condition,
since every condition is applied to the identical pair set) and aborts if any
present condition's count doesn't match -- catches a truncated/incomplete
sweep before it silently produces a CI from partial data.

Usage:
  python scripts/full320_v3_interchange_patch_bootstrap.py \\
      results/tae_ixpatch_full_llama31_8b.jsonl \\
      results/tae_ixpatch_matched_llama31_8b.jsonl \\
      -- results/tae_ixpatch_full_llama31_8b_bootstrap.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 0
CONTROL_CONDITIONS = ("random_control", "matched_control")


def load_records(paths: list[Path]) -> list[dict]:
    recs = []
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def per_row_stats(records: list[dict]) -> dict:
    """One row per (layer, condition, src_row): purpose_shift, format_iia_hit,
    payload_block_id -- collapsed over the 4 (direction, intended_purpose)
    combinations that share a source row (there is exactly one per source row
    per layer per condition in this design, so this is an identity reshape,
    not an aggregation)."""
    by_layer_cond: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in records:
        purpose_shift = r["purpose_score_patched"] - r["purpose_score_baseline"]
        target_gap = r["format_score_baseline_tgt"] - r["format_score_baseline_src"]
        actual_shift = r["format_score_patched"] - r["format_score_baseline_src"]
        iia_hit = float(np.sign(actual_shift) == np.sign(target_gap)) if target_gap != 0 else np.nan
        by_layer_cond[(r["layer"], r["condition"])].append({
            "block": r["payload_block_id"], "purpose_shift": abs(purpose_shift), "iia_hit": iia_hit,
        })
    return by_layer_cond


def sanity_check_counts(by_layer_cond: dict, layers: list[int], conditions: list[str]) -> None:
    expected = None
    for cond in conditions:
        for layer in layers:
            n = len(by_layer_cond.get((layer, cond), []))
            if expected is None:
                expected = n
            elif n != expected:
                raise ValueError(
                    f"record-count mismatch: condition={cond} layer={layer} has {n} rows, "
                    f"expected {expected} (from the first (layer, condition) cell) -- "
                    f"sweep is truncated or incomplete, refusing to compute a CI from partial data"
                )
    print(f"record-count check: {expected} rows per (layer, condition) cell, "
          f"{len(layers)} layers x {len(conditions)} conditions -- consistent")


def main() -> None:
    args = sys.argv[1:]
    if "--" in args:
        sep = args.index("--")
        in_paths, out_path = [Path(p) for p in args[:sep]], Path(args[sep + 1])
    else:
        in_paths, out_path = [Path(args[0])], Path(args[1])
    records = load_records(in_paths)
    if not records:
        raise ValueError(f"no records in {in_paths}")
    model = records[0]["model"]
    rank = records[0]["rank"]
    layers = sorted({r["layer"] for r in records})
    by_layer_cond = per_row_stats(records)
    present_controls = [c for c in CONTROL_CONDITIONS if any((L, c) in by_layer_cond for L in layers)]
    sanity_check_counts(by_layer_cond, layers, ["real_swap"] + present_controls)

    blocks = sorted({r["payload_block_id"] for r in records})
    block_to_pos: dict[tuple[int, str], dict[str, list[int]]] = {}
    for (layer, cond), rows in by_layer_cond.items():
        bp: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            bp.setdefault(row["block"], []).append(i)
        block_to_pos[(layer, cond)] = bp

    def stat(rows: list[dict], idx: np.ndarray) -> tuple[float, float]:
        purpose = np.array([rows[i]["purpose_shift"] for i in idx])
        iia = np.array([rows[i]["iia_hit"] for i in idx])
        iia = iia[~np.isnan(iia)]
        return float(np.mean(purpose)), float(np.mean(iia)) if len(iia) else float("nan")

    point = {c: {} for c in present_controls}
    for control in present_controls:
        for layer in layers:
            real_rows = by_layer_cond[(layer, "real_swap")]
            ctrl_rows = by_layer_cond[(layer, control)]
            p_real, iia_real = stat(real_rows, np.arange(len(real_rows)))
            p_ctrl, iia_ctrl = stat(ctrl_rows, np.arange(len(ctrl_rows)))
            point[control][layer] = {
                "purpose_disruption_real": p_real, "purpose_disruption_control": p_ctrl,
                "purpose_disruption_gap": p_ctrl - p_real,  # H1: want > 0
                "format_iia_real": iia_real, "format_iia_control": iia_ctrl,
                "format_iia_gap": iia_real - iia_ctrl,  # H2: want > 0
            }

    rng = np.random.default_rng(SEED)
    boot = {control: {"purpose_gap": {L: np.empty(N_BOOT) for L in layers},
                       "iia_gap": {L: np.empty(N_BOOT) for L in layers}}
            for control in present_controls}
    for b in range(N_BOOT):
        sample_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        for control in present_controls:
            for layer in layers:
                real_rows = by_layer_cond[(layer, "real_swap")]
                ctrl_rows = by_layer_cond[(layer, control)]
                real_bp = block_to_pos[(layer, "real_swap")]
                ctrl_bp = block_to_pos[(layer, control)]
                real_idx = np.concatenate([real_bp[blk] for blk in sample_blocks if blk in real_bp]) if any(blk in real_bp for blk in sample_blocks) else np.array([], int)
                ctrl_idx = np.concatenate([ctrl_bp[blk] for blk in sample_blocks if blk in ctrl_bp]) if any(blk in ctrl_bp for blk in sample_blocks) else np.array([], int)
                p_real, iia_real = stat(real_rows, real_idx)
                p_ctrl, iia_ctrl = stat(ctrl_rows, ctrl_idx)
                boot[control]["purpose_gap"][layer][b] = p_ctrl - p_real
                boot[control]["iia_gap"][layer][b] = iia_real - iia_ctrl

    by_control_out = {}
    for control in present_controls:
        per_layer_out = {}
        for layer in layers:
            def ci(a):
                return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
            per_layer_out[str(layer)] = {
                **point[control][layer],
                "purpose_disruption_gap_ci95": ci(boot[control]["purpose_gap"][layer]),
                "format_iia_gap_ci95": ci(boot[control]["iia_gap"][layer]),
            }
        worst_purpose_gap = np.min(np.stack([boot[control]["purpose_gap"][L] for L in layers]), axis=0)
        worst_iia_gap = np.min(np.stack([boot[control]["iia_gap"][L] for L in layers]), axis=0)
        by_control_out[control] = {
            "joint_worst_layer_purpose_disruption_gap_ci95": [float(np.percentile(worst_purpose_gap, 2.5)), float(np.percentile(worst_purpose_gap, 97.5))],
            "joint_worst_layer_format_iia_gap_ci95": [float(np.percentile(worst_iia_gap, 2.5)), float(np.percentile(worst_iia_gap, 97.5))],
            "by_layer": per_layer_out,
        }

    result = {
        "schema_version": 2, "mode": "development",
        "purpose": "Correlation-aware (shared block-resample across all layers AND all "
                   "control comparisons per bootstrap replicate) analysis of the "
                   "interchange-patching intervention, against every control condition "
                   "present (random_control and/or matched_control). Per-layer CIs are "
                   "NOT independent tests; per-layer significance counts are intentionally "
                   "not reported. The joint statistics use the same shared draws to make "
                   "one family-wise-valid claim per control across the whole depth profile.",
        "model": model, "rank": rank, "n_layers_tested": len(layers), "n_boot": N_BOOT,
        "n_blocks": len(blocks), "controls_present": present_controls,
        "by_control": by_control_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for control in present_controls:
        c = by_control_out[control]
        print(f"[{control}] joint worst-layer purpose_disruption_gap 95% CI (want > 0): {c['joint_worst_layer_purpose_disruption_gap_ci95']}")
        print(f"[{control}] joint worst-layer format_iia_gap 95% CI (want > 0): {c['joint_worst_layer_format_iia_gap_ci95']}")


if __name__ == "__main__":
    main()
