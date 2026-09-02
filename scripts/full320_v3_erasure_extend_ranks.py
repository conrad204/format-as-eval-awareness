"""Extend the 70B matched_pca rank grid in results/full320_v3_format_erasure_retrain.json
with ranks 4 and 64 (matching the 8B grid), without recomputing the already-computed
ranks 1 and 16. Reuses the exact same per-fold cross-fitted erasure+retrain function
as scripts/full320_v3_format_erasure_retrain.py.

Usage: python scripts/full320_v3_erasure_extend_ranks.py
"""
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full320_v3_format_erasure_retrain import (  # noqa: E402
    ITEMS_PATH, MODELS, JOBLIB_TEMP_FOLDER, aggregate, erase_and_score_fold, stream_extract_layers,
)

TARGET_MODELS = ("llama31_70b", "llama33_70b")
NEW_RANKS = (4, 64)
OUT_PATH = "results/full320_v3_format_erasure_retrain.json"


def main():
    results = json.loads(Path(OUT_PATH).read_text())
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_by_id = {r["item_id"]: r for r in rows_all if r.get("label") is not None}

    for model in TARGET_MODELS:
        cfg = MODELS[model]
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        depth_list0 = [cfg["min_layer"], cfg["final_layer"], cfg["spike_layer"]]
        X, layer_list = stream_extract_layers(cfg["npz"], sorted(set(depth_list0)))
        layer_pos = {L: j for j, L in enumerate(layer_list)}
        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        keep_mask = np.array([iid in rows_by_id for iid in raw_ids])
        assert keep_mask.sum() == len(rows_by_id)
        X = X[keep_mask].astype(np.float64)
        kept_rows = [rows_by_id[i] for i in raw_ids[keep_mask]]
        y = np.array([int(r["label"]) for r in kept_rows])
        fam = np.array([r["purpose_family_id"] for r in kept_rows])
        fmt = np.array([r["format"] for r in kept_rows])
        fmt_bin = (fmt == "benchmark").astype(int)
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"
        folds = sorted(set(fam))

        depth_list = depth_list0
        Xl_by_layer = {L: np.ascontiguousarray(X[:, layer_pos[L], :]) for L in depth_list}
        del X

        jobs = [(L, rank, fold) for L in depth_list for rank in NEW_RANKS for fold in folds]
        outs = Parallel(n_jobs=2, temp_folder=JOBLIB_TEMP_FOLDER)(
            delayed(erase_and_score_fold)(
                Xl_by_layer[L], kept_rows, y, fam, fmt_bin, is_bench, is_casual, fold, rank, "matched_pca")
            for L, rank, fold in jobs
        )
        by_layer_rank = {}
        for (L, rank, fold), (_, _, fold_result) in zip(jobs, outs):
            by_layer_rank.setdefault(L, {}).setdefault(rank, []).append(fold_result)

        for L in depth_list:
            for rank in NEW_RANKS:
                fold_results = by_layer_rank[L][rank]
                purpose_auc = aggregate(fold_results, "purpose")
                format_auc = aggregate(fold_results, "format")
                b2c_auc = aggregate(fold_results, "b2c")
                c2b_auc = aggregate(fold_results, "c2b")
                gap = purpose_auc - (b2c_auc + c2b_auc) / 2
                results[model]["by_layer"][str(L)]["matched_pca"][str(rank)] = {
                    "purpose_auc_after_erasure": purpose_auc,
                    "format_auc_after_erasure": format_auc,
                    "b2c_auc_after_erasure": b2c_auc,
                    "c2b_auc_after_erasure": c2b_auc,
                    "gap_after_erasure": gap,
                }
                print(f"  layer {L} matched_pca rank {rank}: purpose={purpose_auc:.4f} "
                      f"format(sanity)={format_auc:.4f} b2c={b2c_auc:.4f} c2b={c2b_auc:.4f} "
                      f"gap={gap:.4f}", file=sys.stderr, flush=True)
        results[model]["method_grid"]["matched_pca"] = sorted(
            set(results[model]["method_grid"]["matched_pca"]) | set(NEW_RANKS))
        del Xl_by_layer

    Path(OUT_PATH).write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
