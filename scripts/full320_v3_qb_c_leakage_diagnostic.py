"""Offline-only diagnostic (no forward passes): how much of the purpose x format
interaction C is captured by the B-derived matched-format subspace Q_B(r), at
every layer and rank, held-out-family cross-fitted -- plus a purpose-stratified
readout of the already-running interchange-patching records.

Three parts, matching the requested diagnostic exactly:
  1. capture_C(r, layer) = held-out mean ||Q_B(r)^T C_g||^2 / ||C_g||^2,
     r in {1,4,16,64}, vs. isotropic chance r/d.
  2. Purpose-stratified (evaluation-fixed vs deployment-fixed) format-IIA and
     mean |purpose shift| from the CURRENT (partial) interchange-patch raw
     records, per completed layer, block-bootstrapped per layer (NOT a joint
     cross-layer claim -- purely descriptive, per instructions).
  3. Purpose-stratified donor-delta capture at rank 16: how much of the RAW
     natural (h_EB-h_EC) [eval] and (h_DB-h_DC) [deploy] group deltas fall
     inside Q_B(16), same held-out cross-fitting as (1).

Reads only frozen activations + the already-written interchange-patch JSONL;
fits no new probes beyond the existing factorial_vectors/orth_basis helpers;
launches no model, no GPU.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from format_erasure_development import factorial_vectors, groups, orth_basis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"
NPZ_PATH = ROOT / "v3-activations" / "activations" / "llama31_8b" / "activations_all_layers.npz"
DEFAULT_RECORDS_PATH = Path("/scratch/jppatton/langea/results/tae_ixpatch_full_llama31_8b.jsonl")
RANKS = (1, 4, 16, 64)
D = 4096
N_BOOT = 1000
SEED = 0


def load_data():
    rows = [json.loads(l) for l in open(ITEMS_PATH) if l.strip()]
    with np.load(NPZ_PATH, allow_pickle=False) as arc:
        item_ids = arc["item_ids"].astype(str).tolist()
        assert item_ids == [r["item_id"] for r in rows]
        X_all = arc["X"]
    fam = np.array([r["purpose_family_id"] for r in rows])
    folds = sorted(set(f for f in fam if f != "none"))
    return rows, X_all, fam, folds


def part1_and_part3(rows, X_all, fam, folds, n_layers=32):
    """Held-out capture_C(r, layer) for r in RANKS, plus purpose-stratified
    donor-delta capture at rank 16."""
    part1 = {r: np.full(n_layers, np.nan) for r in RANKS}
    part3 = {"eval": np.full(n_layers, np.nan), "deploy": np.full(n_layers, np.nan)}
    for layer in range(n_layers):
        Xl = X_all[:, layer, :].astype(np.float64)
        cap_by_rank = {r: [] for r in RANKS}
        cap_eval, cap_deploy = [], []
        for fold in folds:
            train_mask = fam != fold
            test_mask = fam == fold
            _, B_train, _, _ = factorial_vectors(Xl, rows, train_mask)
            gs_test = groups(rows, np.array([r["render_slot"] == "stated" for r in rows]), test_mask)
            if not gs_test:
                continue
            _, _, C_test, _ = factorial_vectors(Xl, rows, test_mask)
            for r in RANKS:
                Q = orth_basis(B_train, r)
                for c in C_test:
                    denom = float(np.dot(c, c))
                    if denom > 0:
                        cap_by_rank[r].append(float(np.sum((c @ Q) ** 2)) / denom)
            Q16 = orth_basis(B_train, 16)
            for _, eb, ec, db, dc in gs_test:
                d_eval = Xl[eb] - Xl[ec]
                d_deploy = Xl[db] - Xl[dc]
                cap_eval.append(float(np.sum((d_eval @ Q16) ** 2)) / float(np.dot(d_eval, d_eval)))
                cap_deploy.append(float(np.sum((d_deploy @ Q16) ** 2)) / float(np.dot(d_deploy, d_deploy)))
        for r in RANKS:
            if cap_by_rank[r]:
                part1[r][layer] = float(np.mean(cap_by_rank[r]))
        if cap_eval:
            part3["eval"][layer] = float(np.mean(cap_eval))
            part3["deploy"][layer] = float(np.mean(cap_deploy))
    return part1, part3


def part2(records_path):
    recs = []
    for line in open(records_path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a trailing line still being written by the live job
    by_layer_purpose_cond = defaultdict(list)
    for r in recs:
        target_gap = r["format_score_baseline_tgt"] - r["format_score_baseline_src"]
        actual_shift = r["format_score_patched"] - r["format_score_baseline_src"]
        iia = float(np.sign(actual_shift) == np.sign(target_gap)) if target_gap != 0 else np.nan
        purpose_shift = abs(r["purpose_score_patched"] - r["purpose_score_baseline"])
        by_layer_purpose_cond[(r["layer"], r["intended_purpose"], r["condition"])].append(
            {"block": r["payload_block_id"], "iia": iia, "purpose_shift": purpose_shift}
        )
    layers = sorted({k[0] for k in by_layer_purpose_cond})
    rng = np.random.default_rng(SEED)
    out = {}
    for layer in layers:
        cell = {}
        for purpose in ("evaluation", "deployment"):
            for cond in ("real_swap", "random_control"):
                rows_ = by_layer_purpose_cond.get((layer, purpose, cond), [])
                if not rows_:
                    continue
                iia_vals = np.array([r["iia"] for r in rows_])
                iia_vals = iia_vals[~np.isnan(iia_vals)]
                pshift_vals = np.array([r["purpose_shift"] for r in rows_])
                blocks = sorted({r["block"] for r in rows_})
                block_to_pos = defaultdict(list)
                for i, r in enumerate(rows_):
                    block_to_pos[r["block"]].append(i)
                boot_iia = np.empty(N_BOOT)
                boot_pshift = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    sample_blocks = rng.choice(blocks, size=len(blocks), replace=True)
                    idx = np.concatenate([block_to_pos[blk] for blk in sample_blocks if blk in block_to_pos])
                    iv = np.array([rows_[i]["iia"] for i in idx])
                    iv = iv[~np.isnan(iv)]
                    boot_iia[b] = np.mean(iv) if len(iv) else np.nan
                    boot_pshift[b] = np.mean([rows_[i]["purpose_shift"] for i in idx])
                cell[f"{purpose[:4]}_{cond}"] = {
                    "n": len(rows_),
                    "format_iia": float(np.mean(iia_vals)) if len(iia_vals) else float("nan"),
                    "format_iia_ci95": [float(np.nanpercentile(boot_iia, 2.5)), float(np.nanpercentile(boot_iia, 97.5))],
                    "mean_abs_purpose_shift": float(np.mean(pshift_vals)),
                    "mean_abs_purpose_shift_ci95": [float(np.percentile(boot_pshift, 2.5)), float(np.percentile(boot_pshift, 97.5))],
                }
        out[str(layer)] = cell
    return out, layers


def main():
    records_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RECORDS_PATH
    rows, X_all, fam, folds = load_data()
    print("=== Part 1: held-out capture_C(r, layer) ===")
    part1, part3 = part1_and_part3(rows, X_all, fam, folds)
    header = "layer " + " ".join(f"r={r:<2}" for r in RANKS) + "  [isotropic chance]"
    print(header)
    for layer in range(32):
        vals = " ".join(f"{part1[r][layer]*100:5.1f}%" for r in RANKS)
        print(f"{layer:5d} {vals}")
    print()
    print("isotropic chance (r/d): " + " ".join(f"r={r}:{r/D*100:.2f}%" for r in RANKS))

    print()
    print("=== Part 3: purpose-stratified rank-16 donor-delta capture ===")
    print("layer  eval_delta_capture  deploy_delta_capture")
    for layer in range(32):
        print(f"{layer:5d}  {part3['eval'][layer]*100:7.2f}%            {part3['deploy'][layer]*100:7.2f}%")

    print()
    print("=== Part 2: purpose-stratified interchange readout (partial records) ===")
    part2_out, layers2 = part2(records_path)
    for layer in layers2:
        cell = part2_out[str(layer)]
        print(f"layer={layer}")
        for key in ("eval_real_swap", "eval_random_control", "depl_real_swap", "depl_random_control"):
            if key in cell:
                c = cell[key]
                print(f"  {key:22s} n={c['n']:4d} IIA={c['format_iia']:.2f} CI={[round(x,2) for x in c['format_iia_ci95']]}"
                      f"  |purpose_shift|={c['mean_abs_purpose_shift']:.3f} CI={[round(x,2) for x in c['mean_abs_purpose_shift_ci95']]}")

    out = {
        "schema_version": 1, "mode": "development",
        "part1_capture_C_by_rank_layer": {str(r): part1[r].tolist() for r in RANKS},
        "isotropic_chance_by_rank": {str(r): r / D for r in RANKS},
        "part2_purpose_stratified_interchange": part2_out,
        "part3_donor_delta_capture_rank16": {"eval": part3["eval"].tolist(), "deploy": part3["deploy"].tolist()},
    }
    out_path = ROOT / "results" / "full320_v3_qb_c_leakage_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
