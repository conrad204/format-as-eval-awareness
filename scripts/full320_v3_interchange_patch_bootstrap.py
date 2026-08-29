"""Correlation-aware analysis of the full-layer interchange-patching raw records
produced by scripts/full320_v3_interchange_patch_full_layer.py.

Two predictions, both required for a mechanism claim:
  H1 (purpose protected): mean |purpose_shift| is SMALLER under real_swap
     (patch with Q_format(r)) than under random_control (patch with a random
     rank-matched subspace, same real counterpart content).
  H2 (format carried): format interchange-intervention accuracy (IIA) --
     whether sign(format_score_patched - format_score_baseline_src) matches
     sign(format_score_baseline_tgt - format_score_baseline_src) -- is HIGHER
     under real_swap than under random_control.

Layers are NOT independent replicates (residual-stream continuity means
adjacent layers share almost all their variance): every bootstrap replicate
resamples PAYLOAD BLOCKS ONCE and reuses that SAME resampled block set to
recompute the statistic at EVERY layer, so the resulting per-layer confidence
intervals share the true cross-layer correlation structure instead of
pretending each layer is an independent draw (the same fix already applied to
scripts/full320_v3_erasure_dose_response_bootstrap.py's paired gap CI). This
script also reports two joint, family-wise-valid omnibus statistics derived
from the SAME shared draws (worst-layer purpose disruption gap, worst-layer
format-IIA gap) for a single depth-independent claim -- it does NOT report or
threshold a per-layer significance count, which would be exactly the
layers-as-independent-replicates pseudoreplication RULES.md prohibits.

Usage:
  python scripts/full320_v3_interchange_patch_bootstrap.py \\
      results/full320_v3_interchange_patch_llama31_8b.jsonl \\
      results/full320_v3_interchange_patch_llama31_8b_bootstrap.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 0


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def main() -> None:
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    records = load_records(in_path)
    if not records:
        raise ValueError(f"no records in {in_path}")
    model = records[0]["model"]
    rank = records[0]["rank"]
    layers = sorted({r["layer"] for r in records})
    by_layer_cond = per_row_stats(records)
    blocks = sorted({r["payload_block_id"] for r in records})
    block_to_pos: dict[int, dict[str, list[int]]] = {}
    for (layer, cond), rows in by_layer_cond.items():
        block_to_pos.setdefault((layer, cond), {})
        for i, row in enumerate(rows):
            block_to_pos[(layer, cond)].setdefault(row["block"], []).append(i)

    def stat(rows: list[dict], idx: np.ndarray) -> tuple[float, float]:
        purpose = np.array([rows[i]["purpose_shift"] for i in idx])
        iia = np.array([rows[i]["iia_hit"] for i in idx])
        iia = iia[~np.isnan(iia)]
        return float(np.mean(purpose)), float(np.mean(iia)) if len(iia) else float("nan")

    point = {}
    for layer in layers:
        real_rows = by_layer_cond[(layer, "real_swap")]
        ctrl_rows = by_layer_cond[(layer, "random_control")]
        real_idx = np.arange(len(real_rows))
        ctrl_idx = np.arange(len(ctrl_rows))
        p_real, iia_real = stat(real_rows, real_idx)
        p_ctrl, iia_ctrl = stat(ctrl_rows, ctrl_idx)
        point[layer] = {
            "purpose_disruption_real": p_real, "purpose_disruption_control": p_ctrl,
            "purpose_disruption_gap": p_ctrl - p_real,  # H1: want > 0
            "format_iia_real": iia_real, "format_iia_control": iia_ctrl,
            "format_iia_gap": iia_real - iia_ctrl,  # H2: want > 0
        }

    rng = np.random.default_rng(SEED)
    boot_purpose_gap = {layer: np.empty(N_BOOT) for layer in layers}
    boot_iia_gap = {layer: np.empty(N_BOOT) for layer in layers}
    for b in range(N_BOOT):
        sample_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        for layer in layers:
            real_rows = by_layer_cond[(layer, "real_swap")]
            ctrl_rows = by_layer_cond[(layer, "random_control")]
            real_bp = block_to_pos[(layer, "real_swap")]
            ctrl_bp = block_to_pos[(layer, "random_control")]
            real_idx = np.concatenate([real_bp[blk] for blk in sample_blocks if blk in real_bp]) if any(blk in real_bp for blk in sample_blocks) else np.array([], int)
            ctrl_idx = np.concatenate([ctrl_bp[blk] for blk in sample_blocks if blk in ctrl_bp]) if any(blk in ctrl_bp for blk in sample_blocks) else np.array([], int)
            p_real, iia_real = stat(real_rows, real_idx)
            p_ctrl, iia_ctrl = stat(ctrl_rows, ctrl_idx)
            boot_purpose_gap[layer][b] = p_ctrl - p_real
            boot_iia_gap[layer][b] = iia_real - iia_ctrl

    per_layer_out = {}
    for layer in layers:
        def ci(a):
            return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
        per_layer_out[str(layer)] = {
            **point[layer],
            "purpose_disruption_gap_ci95": ci(boot_purpose_gap[layer]),
            "format_iia_gap_ci95": ci(boot_iia_gap[layer]),
        }

    # Joint (correlation-honest) omnibus statistics from the SAME shared draws:
    # the worst layer's gap, per replicate -- a single family-wise-valid claim
    # ("the effect holds at every layer with 95% confidence") rather than a
    # per-layer significance count.
    worst_purpose_gap = np.min(np.stack([boot_purpose_gap[layer] for layer in layers]), axis=0)
    worst_iia_gap = np.min(np.stack([boot_iia_gap[layer] for layer in layers]), axis=0)

    result = {
        "schema_version": 1, "mode": "development",
        "purpose": "Correlation-aware (shared block-resample across all layers per "
                   "bootstrap replicate) analysis of the interchange-patching intervention. "
                   "Per-layer CIs are NOT independent tests; per-layer significance counts "
                   "are intentionally not reported. The two joint statistics use the same "
                   "shared draws to make one family-wise-valid claim across the whole depth "
                   "profile instead.",
        "model": model, "rank": rank, "n_layers_tested": len(layers), "n_boot": N_BOOT,
        "n_blocks": len(blocks),
        "joint_worst_layer_purpose_disruption_gap_ci95": [float(np.percentile(worst_purpose_gap, 2.5)), float(np.percentile(worst_purpose_gap, 97.5))],
        "joint_worst_layer_format_iia_gap_ci95": [float(np.percentile(worst_iia_gap, 2.5)), float(np.percentile(worst_iia_gap, 97.5))],
        "by_layer": per_layer_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"joint worst-layer purpose_disruption_gap 95% CI (want > 0): {result['joint_worst_layer_purpose_disruption_gap_ci95']}")
    print(f"joint worst-layer format_iia_gap 95% CI (want > 0): {result['joint_worst_layer_format_iia_gap_ci95']}")


if __name__ == "__main__":
    main()
