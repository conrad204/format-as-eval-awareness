"""Targeted, pre-registered intervention testing two specific hypotheses from
the signal/noise failure-mode decomposition (format_probe_signal_noise_decomposition.py):

  A. Casual->benchmark transfer fails via SIGNAL LOSS (probe direction stops
     tracking true separation). Hypothesis: this is a coordinate-ROTATION
     problem. Test: fit an orthogonal Procrustes map between formats using
     ONLY matched no-purpose (no-cue) payloads -- no eval/deploy label or cue
     information enters the fit. Transport benchmark test representations
     into casual coordinates, then score with the UNCHANGED casual-trained
     probe. An orthogonal map cannot shrink within-class variance (it
     preserves all pairwise distances/norms), so if this selectively repairs
     the SIGNAL term (not the noise term) for casual->benchmark, that is
     evidence for coordinate misalignment as the mechanism.

  B. Benchmark->casual transfer fails via NOISE GROWTH (within-class scatter
     inflates; signal usually preserved/improves). Hypothesis: this is a
     COVARIANCE mismatch problem. Test: fit a regularized CORAL
     (whitening-coloring) affine map between formats, again from matched
     no-cue payloads only. If this selectively repairs the NOISE term (not
     signal) for benchmark->casual, that is evidence for covariance mismatch.

Selectivity is the actual test, not "does anything help": each transform is
ALSO applied to the OTHER direction (rotation applied to benchmark->casual;
covariance-alignment applied to casual->benchmark) as a built-in specificity
control. Additional controls: identity (no-op sanity check), shuffled-pair
Procrustes (breaks true block correspondence -- tests whether ANY orthogonal
map fit on mismatched pairs looks like it helps, which would indicate
overfitting/artifact rather than real coordinate alignment).

Payload blocks are split 75/25 TRAIN/HELD-OUT (seeded, fixed before any
result is inspected). Transport maps are fit ONLY on no-cue rows belonging to
TRAIN blocks. All evaluation (signal/noise decomposition, real AUC) is
computed ONLY on purpose rows belonging to HELD-OUT blocks -- disjoint from
fitting, per the pre-registration. Fixed depth anchors (27%, 50%, final),
not retrospectively chosen.

Regularization: CORAL covariance estimates use sklearn's LedoitWolf shrinkage
(automatic, data-driven shrinkage intensity, reported per model/layer/format)
before eigendecomposition-based matrix square roots, with a floor on
eigenvalues to avoid unstable inverses. Conditioning is reported.
"""
import json
import sys
import time
import zipfile

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
from format_bound_regrowth_bootstrap import ITEMS_PATH, stream_extract_layers
from format_probe_signal_noise_decomposition import depth_to_layer

MODELS = {
    "llama31_8b": {"npz": "v3-activations/activations/llama31_8b/activations_all_layers.npz", "n_layers": 32},
    "llama31_70b": {"npz": "v3-activations/activations/llama31_70b/activations_all_layers.npz", "n_layers": 80},
    "llama33_70b": {"npz": "v3-activations/activations/llama33_70b/activations_all_layers.npz", "n_layers": 80},
}
FIXED_ANCHOR_DEPTH = 0.27
MID_DEPTH = 0.5
BLOCK_SPLIT_SEED = 20260828
TRAIN_FRACTION = 0.75


def split_blocks(all_blocks: list[str]) -> tuple[set, set]:
    rng = np.random.default_rng(BLOCK_SPLIT_SEED)
    blocks = sorted(all_blocks)
    perm = rng.permutation(len(blocks))
    n_train = int(round(TRAIN_FRACTION * len(blocks)))
    train = {blocks[i] for i in perm[:n_train]}
    held = {blocks[i] for i in perm[n_train:]}
    return train, held


def fit_shared_pca_basis(X_pool: np.ndarray, K: int) -> dict:
    """Shared ambient subspace (fit on POOLED, both-format, no-cue TRAIN data)
    that both format's Procrustes rotation operates within. Using a SHARED
    basis for source and target -- rather than each format's own separate
    subspace -- is what makes the resulting transform EXACTLY globally
    orthogonal (see apply_transport_linear): the correction is a rotation
    strictly within this K-dim subspace, identity on its orthogonal
    complement, which is provably norm-preserving for any K, unlike a
    full-rank (d x d) SVD-based Procrustes fit from n << d paired points
    (which is only exactly orthogonal within-subspace, not globally --
    verified and rejected during development)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=K, random_state=0)
    pca.fit(X_pool)
    return {"mean": pca.mean_, "P": pca.components_.T}  # P: (d, K), orthonormal columns


def fit_procrustes(Xs: np.ndarray, Xt: np.ndarray, pca_basis: dict) -> dict:
    P, mean = pca_basis["P"], pca_basis["mean"]
    cs = (Xs - mean) @ P
    ct = (Xt - mean) @ P
    cs_mean, ct_mean = cs.mean(0), ct.mean(0)
    csc, ctc = cs - cs_mean, ct - ct_mean
    M = csc.T @ ctc  # K x K, cheap regardless of d
    U, S, Vt = np.linalg.svd(M)
    R_small = U @ Vt  # K x K, exactly orthogonal
    return {"P": P, "mean": mean, "cs_mean": cs_mean, "ct_mean": ct_mean, "R_small": R_small}


def apply_transport_linear(X: np.ndarray, transport: dict) -> np.ndarray:
    """x_new = x + P @ (R_small - I) @ P.T @ (x - mean) + P @ (ct_mean - cs_mean @ R_small)
    Implemented as: rotate the K-dim projection, reconstruct the delta, add
    to x -- the (d-K)-dim orthogonal complement is left untouched (identity),
    which is exactly what keeps the overall map orthogonal."""
    P, mean = transport["P"], transport["mean"]
    coeffs = (X - mean) @ P
    coeffs_centered = coeffs - transport["cs_mean"]
    coeffs_rotated = coeffs_centered @ transport["R_small"] + transport["ct_mean"]
    delta = (coeffs_rotated - coeffs) @ P.T
    return X + delta


def sqrtm_eigh(Sigma: np.ndarray, floor_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray, float]:
    w, V = np.linalg.eigh(Sigma)
    floor = floor_frac * w.max()
    w_clipped = np.clip(w, floor, None)
    cond = float(w_clipped.max() / w_clipped.min())
    sqrt_S = V @ np.diag(np.sqrt(w_clipped)) @ V.T
    invsqrt_S = V @ np.diag(1.0 / np.sqrt(w_clipped)) @ V.T
    return sqrt_S, invsqrt_S, cond


def fit_coral(Xs: np.ndarray, Xt: np.ndarray, pca_basis: dict) -> dict:
    """CORAL (whitening-coloring) restricted to the same shared PCA subspace
    used for Procrustes. Covariance estimation from n_rows << d samples is
    only measurable within an n_rows-dimensional subspace anyway, so a full
    d x d eigendecomposition (a) wastes compute (O(d^3), infeasible at
    d=8192) and (b) estimates most of its degrees of freedom from pure
    sampling noise. Operating within the K-dim shared subspace and leaving
    the orthogonal complement untouched is faster and more statistically
    honest about what the data can actually determine."""
    P, mean = pca_basis["P"], pca_basis["mean"]
    cs = (Xs - mean) @ P
    ct = (Xt - mean) @ P
    mean_s, mean_t = cs.mean(0), ct.mean(0)
    lw_s = LedoitWolf().fit(cs - mean_s)
    lw_t = LedoitWolf().fit(ct - mean_t)
    sqrt_t, _, cond_t = sqrtm_eigh(lw_t.covariance_)
    _, invsqrt_s, cond_s = sqrtm_eigh(lw_s.covariance_)
    W_small = invsqrt_s @ sqrt_t
    return {
        "P": P, "mean": mean, "mean_s": mean_s, "mean_t": mean_t, "W_small": W_small,
        "shrinkage_s": float(lw_s.shrinkage_), "shrinkage_t": float(lw_t.shrinkage_),
        "cond_s": cond_s, "cond_t": cond_t,
    }


def apply_transport_affine(X: np.ndarray, transport: dict) -> np.ndarray:
    P, mean = transport["P"], transport["mean"]
    coeffs = (X - mean) @ P
    coeffs_centered = coeffs - transport["mean_s"]
    coeffs_transformed = coeffs_centered @ transport["W_small"] + transport["mean_t"]
    delta = (coeffs_transformed - coeffs) @ P.T
    return X + delta


def fit_fold_models(X: np.ndarray, y: np.ndarray, fam: np.ndarray, tr_mask: np.ndarray) -> dict:
    """Fit one (scaler, logistic model) pair per held-out family, trained only
    on rows satisfying tr_mask, matching the project's OOF fold convention."""
    out = {}
    for f in sorted(set(fam)):
        trm = (fam != f) & tr_mask
        if trm.sum() < 10:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[trm])
        m = LogisticRegression(C=0.1, max_iter=3000, random_state=0).fit(Xtr, y[trm])
        out[f] = (sc, m)
    return out


def score_with_fold_models(fold_models: dict, X: np.ndarray, fam: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Score rows at idx using the fold model whose held-out family matches
    that row's family (OOF scoring against a fixed set of fold models).
    Vectorized by family group rather than one sklearn call per row."""
    s = np.full(len(idx), np.nan)
    idx_fam = fam[idx]
    for f in set(idx_fam):
        if f not in fold_models:
            continue
        sc, m = fold_models[f]
        sel = idx_fam == f
        rows = idx[sel]
        s[sel] = m.decision_function(sc.transform(X[rows]))
    return s


def signal_noise_dprime(scores: np.ndarray, purpose_subset: np.ndarray) -> dict:
    valid = ~np.isnan(scores)
    e = scores[valid & (purpose_subset == "evaluation")]
    d = scores[valid & (purpose_subset == "deployment")]
    if len(e) < 2 or len(d) < 2:
        return {"signal": np.nan, "sd": np.nan, "d_prime": np.nan, "n_e": len(e), "n_d": len(d)}
    signal = float(np.mean(e) - np.mean(d))
    sd = float(np.sqrt((np.var(e, ddof=1) + np.var(d, ddof=1)) / 2))
    return {"signal": signal, "sd": sd, "d_prime": signal / sd if sd > 0 else np.nan, "n_e": len(e), "n_d": len(d)}


def real_auc(scores: np.ndarray, y_subset: np.ndarray) -> float:
    valid = ~np.isnan(scores)
    if valid.sum() < 2 or len(set(y_subset[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y_subset[valid], scores[valid]))


def main():
    rows_all = [json.loads(line) for line in open(ITEMS_PATH) if line.strip()]
    rows_purpose = {r["item_id"]: r for r in rows_all if r.get("label") is not None}
    rows_nocue = {r["item_id"]: r for r in rows_all if r.get("label") is None}
    all_blocks = sorted({r["payload_block_id"] for r in rows_all})
    train_blocks, held_blocks = split_blocks(all_blocks)
    print(f"payload blocks: {len(train_blocks)} train / {len(held_blocks)} held-out", file=sys.stderr)

    results = {"train_blocks": sorted(train_blocks), "held_blocks": sorted(held_blocks)}
    for model, cfg in MODELS.items():
        print(f"=== {model} ===", file=sys.stderr, flush=True)
        n_layers = cfg["n_layers"]
        anchor_L = depth_to_layer(FIXED_ANCHOR_DEPTH, n_layers)
        mid_L = depth_to_layer(MID_DEPTH, n_layers)
        final_L = n_layers - 1
        layer_list = sorted({anchor_L, mid_L, final_L})

        X, layer_list = stream_extract_layers(cfg["npz"], layer_list)
        layer_pos = {L: j for j, L in enumerate(layer_list)}

        z = zipfile.ZipFile(cfg["npz"])
        raw_ids = np.load(z.open("item_ids.npy"))
        raw_ids = [str(i) for i in raw_ids]

        purpose_idx = np.array([i for i, iid in enumerate(raw_ids) if iid in rows_purpose])
        nocue_idx = np.array([i for i, iid in enumerate(raw_ids) if iid in rows_nocue])

        p_rows = [rows_purpose[raw_ids[i]] for i in purpose_idx]
        y = np.array([int(r["label"]) for r in p_rows])
        fam = np.array([r["purpose_family_id"] for r in p_rows])
        fmt = np.array([r["format"] for r in p_rows])
        blk = np.array([r["payload_block_id"] for r in p_rows])
        purpose = np.array([r["intended_purpose"] for r in p_rows])
        is_bench, is_casual = fmt == "benchmark", fmt == "casual"
        blk_is_train = np.array([b in train_blocks for b in blk])
        blk_is_held = ~blk_is_train

        n_rows = [rows_nocue[raw_ids[i]] for i in nocue_idx]
        n_fmt = np.array([r["format"] for r in n_rows])
        n_blk = np.array([r["payload_block_id"] for r in n_rows])
        n_is_train = np.array([b in train_blocks for b in n_blk])

        model_result = {"anchor_layer": int(anchor_L), "mid_layer": int(mid_L), "final_layer": int(final_L), "layers": {}}

        for L in [anchor_L, mid_L, final_L]:
            print(f"  --- layer {L} ---", file=sys.stderr, flush=True)
            Xl_full = X[:, layer_pos[L], :]
            Xp = Xl_full[purpose_idx]  # purpose rows
            Xn = Xl_full[nocue_idx]    # no-cue rows

            # --- fit transports on TRAIN-block no-cue rows only ---
            nb_train_mask = n_is_train & (n_fmt == "benchmark")
            nc_train_mask = n_is_train & (n_fmt == "casual")
            nb_train = Xn[nb_train_mask]
            nc_train = Xn[nc_train_mask]
            print(f"    no-cue train rows: benchmark={len(nb_train)} casual={len(nc_train)}", file=sys.stderr)

            # Procrustes needs TRUE point-to-point correspondence: average the
            # ~6 no-cue renderings per (block, format) into one centroid per
            # block, matched by payload_block_id -- not an arbitrary
            # equal-length truncation, which would pair unrelated content.
            train_block_list = sorted(train_blocks)
            nb_centroids, nc_centroids = [], []
            for b in train_block_list:
                bm = nb_train_mask & (n_blk == b)
                cm = nc_train_mask & (n_blk == b)
                if bm.sum() and cm.sum():
                    nb_centroids.append(Xn[bm].mean(0))
                    nc_centroids.append(Xn[cm].mean(0))
            nb_centroids = np.asarray(nb_centroids)
            nc_centroids = np.asarray(nc_centroids)
            n_pairs = len(nb_centroids)
            print(f"    matched block centroids for procrustes: {n_pairs}", file=sys.stderr)

            t0 = time.time()
            K = min(n_pairs - 1, 250)
            pca_basis = fit_shared_pca_basis(np.vstack([nb_train, nc_train]), K)
            procrustes_b2c = fit_procrustes(nb_centroids, nc_centroids, pca_basis)
            procrustes_c2b = fit_procrustes(nc_centroids, nb_centroids, pca_basis)
            rng = np.random.default_rng(BLOCK_SPLIT_SEED + hash(L) % 1000)
            shuffle_idx = rng.permutation(n_pairs)
            procrustes_b2c_shuffled = fit_procrustes(nb_centroids, nc_centroids[shuffle_idx], pca_basis)
            procrustes_c2b_shuffled = fit_procrustes(nc_centroids, nb_centroids[shuffle_idx], pca_basis)
            print(f"    procrustes fit (K={K} shared PCA subspace) in {time.time() - t0:.1f}s", file=sys.stderr)

            t0 = time.time()
            coral_b2c = fit_coral(nb_train, nc_train, pca_basis)
            coral_c2b = fit_coral(nc_train, nb_train, pca_basis)
            print(f"    coral fit in {time.time() - t0:.1f}s | shrinkage_b={coral_b2c['shrinkage_s']:.3f} "
                  f"shrinkage_c={coral_b2c['shrinkage_t']:.3f} cond_b={coral_b2c['cond_s']:.1f} cond_c={coral_b2c['cond_t']:.1f}",
                  file=sys.stderr)

            # --- fit purpose probes (self regimes) on ALL rows (standard OOF family CV) ---
            bb_models = fit_fold_models(Xp, y, fam, is_bench)
            cc_models = fit_fold_models(Xp, y, fam, is_casual)

            # --- evaluate on HELD-OUT block purpose rows only ---
            bench_held_idx = np.where(is_bench & blk_is_held)[0]
            casual_held_idx = np.where(is_casual & blk_is_held)[0]

            def eval_direction(test_idx, X_test_transformed, fold_models, y_all, purpose_all):
                scores = score_with_fold_models(fold_models, X_test_transformed, fam, test_idx)
                sn = signal_noise_dprime(scores, purpose_all[test_idx])
                sn["real_auc"] = real_auc(scores, y_all[test_idx])
                sn["_scores_full"] = scores  # aligned to test_idx order; consumed by bootstrap below, stripped before saving
                return sn

            conditions_c2b = {
                "baseline": Xp,
                "identity": Xp,
                "procrustes": apply_transport_linear(Xp, procrustes_c2b),
                "procrustes_shuffled": apply_transport_linear(Xp, procrustes_c2b_shuffled),
                "coral": apply_transport_affine(Xp, coral_c2b),
            }
            conditions_b2c = {
                "baseline": Xp,
                "identity": Xp,
                "procrustes": apply_transport_linear(Xp, procrustes_b2c),
                "procrustes_shuffled": apply_transport_linear(Xp, procrustes_b2c_shuffled),
                "coral": apply_transport_affine(Xp, coral_b2c),
            }

            layer_result = {
                "self_bench": eval_direction(bench_held_idx, Xp, bb_models, y, purpose),
                "self_casual": eval_direction(casual_held_idx, Xp, cc_models, y, purpose),
                "c2b": {}, "b2c": {},
                "coral_conditioning": {
                    "b2c": {"shrinkage_s": coral_b2c["shrinkage_s"], "shrinkage_t": coral_b2c["shrinkage_t"],
                            "cond_s": coral_b2c["cond_s"], "cond_t": coral_b2c["cond_t"]},
                    "c2b": {"shrinkage_s": coral_c2b["shrinkage_s"], "shrinkage_t": coral_c2b["shrinkage_t"],
                            "cond_s": coral_c2b["cond_s"], "cond_t": coral_c2b["cond_t"]},
                },
                "n_no_cue_train_pairs": int(n_pairs),
            }
            for cond_name, Xc in conditions_c2b.items():
                layer_result["c2b"][cond_name] = eval_direction(bench_held_idx, Xc, cc_models, y, purpose)
            for cond_name, Xc in conditions_b2c.items():
                layer_result["b2c"][cond_name] = eval_direction(casual_held_idx, Xc, bb_models, y, purpose)

            # --- payload-block bootstrap CIs on delta-AUC and the preregistered
            # selectivity (directional-interaction) prediction, using the raw
            # scores already computed above (no refitting per replicate) ---
            N_BOOT = 1000
            boot_rng = np.random.default_rng(BLOCK_SPLIT_SEED + 7 * (hash(L) % 1000))
            held_block_arr = np.array(sorted(held_blocks))
            # scores arrays are POSITION-indexed (aligned to bench_held_idx /
            # casual_held_idx order), not global-row-indexed -- map blocks to
            # POSITIONS, not global row ids, to avoid an index-space mismatch.
            bench_blk_of_pos = blk[bench_held_idx]
            casual_blk_of_pos = blk[casual_held_idx]
            block_to_bench_pos = {b: np.where(bench_blk_of_pos == b)[0] for b in held_block_arr}
            block_to_casual_pos = {b: np.where(casual_blk_of_pos == b)[0] for b in held_block_arr}

            def boot_auc(scores, positions, held_idx, y_all):
                if positions is None or len(positions) < 2:
                    return np.nan
                s = scores[positions]
                yy = y_all[held_idx[positions]]
                valid = ~np.isnan(s)
                if valid.sum() < 2 or len(set(yy[valid])) < 2:
                    return np.nan
                return roc_auc_score(yy[valid], s[valid])

            delta_auc = {("c2b", "procrustes"): [], ("c2b", "coral"): [], ("b2c", "procrustes"): [], ("b2c", "coral"): []}
            interaction_procrustes, interaction_coral = [], []
            for _ in range(N_BOOT):
                sample_blocks = boot_rng.choice(held_block_arr, size=len(held_block_arr), replace=True)
                bench_pos = np.concatenate([block_to_bench_pos[b] for b in sample_blocks])
                casual_pos = np.concatenate([block_to_casual_pos[b] for b in sample_blocks])

                base_c2b = boot_auc(layer_result["c2b"]["baseline"]["_scores_full"], bench_pos, bench_held_idx, y)
                base_b2c = boot_auc(layer_result["b2c"]["baseline"]["_scores_full"], casual_pos, casual_held_idx, y)
                proc_c2b = boot_auc(layer_result["c2b"]["procrustes"]["_scores_full"], bench_pos, bench_held_idx, y)
                proc_b2c = boot_auc(layer_result["b2c"]["procrustes"]["_scores_full"], casual_pos, casual_held_idx, y)
                cor_c2b = boot_auc(layer_result["c2b"]["coral"]["_scores_full"], bench_pos, bench_held_idx, y)
                cor_b2c = boot_auc(layer_result["b2c"]["coral"]["_scores_full"], casual_pos, casual_held_idx, y)

                d_proc_c2b, d_proc_b2c = proc_c2b - base_c2b, proc_b2c - base_b2c
                d_cor_c2b, d_cor_b2c = cor_c2b - base_c2b, cor_b2c - base_b2c
                delta_auc[("c2b", "procrustes")].append(d_proc_c2b)
                delta_auc[("b2c", "procrustes")].append(d_proc_b2c)
                delta_auc[("c2b", "coral")].append(d_cor_c2b)
                delta_auc[("b2c", "coral")].append(d_cor_b2c)
                # preregistered selectivity: Procrustes should help c2b MORE than b2c;
                # CORAL should help b2c MORE than c2b
                interaction_procrustes.append(d_proc_c2b - d_proc_b2c)
                interaction_coral.append(d_cor_b2c - d_cor_c2b)

            def ci95(a):
                a = np.asarray(a, dtype=float)
                a = a[~np.isnan(a)]
                return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]

            layer_result["bootstrap"] = {
                "n_boot": N_BOOT,
                "delta_auc_ci95": {f"{d}_{t}": ci95(v) for (d, t), v in delta_auc.items()},
                "delta_auc_point": {f"{d}_{t}": float(layer_result[d][t]["real_auc"] - layer_result[d]["baseline"]["real_auc"]) for (d, t) in delta_auc},
                "interaction_procrustes_ci95": ci95(interaction_procrustes),
                "interaction_procrustes_point": float(
                    (layer_result["c2b"]["procrustes"]["real_auc"] - layer_result["c2b"]["baseline"]["real_auc"])
                    - (layer_result["b2c"]["procrustes"]["real_auc"] - layer_result["b2c"]["baseline"]["real_auc"])
                ),
                "interaction_coral_ci95": ci95(interaction_coral),
                "interaction_coral_point": float(
                    (layer_result["b2c"]["coral"]["real_auc"] - layer_result["b2c"]["baseline"]["real_auc"])
                    - (layer_result["c2b"]["coral"]["real_auc"] - layer_result["c2b"]["baseline"]["real_auc"])
                ),
            }
            print(f"    bootstrap: interaction_procrustes(c2b-b2c delta)={layer_result['bootstrap']['interaction_procrustes_point']:+.4f} "
                  f"CI={layer_result['bootstrap']['interaction_procrustes_ci95']} | "
                  f"interaction_coral(b2c-c2b delta)={layer_result['bootstrap']['interaction_coral_point']:+.4f} "
                  f"CI={layer_result['bootstrap']['interaction_coral_ci95']}", file=sys.stderr)

            # strip raw score arrays before saving (not JSON-serializable / not needed downstream)
            for direction in ("c2b", "b2c", "self_bench", "self_casual"):
                d = layer_result[direction]
                if direction in ("self_bench", "self_casual"):
                    d.pop("_scores_full", None)
                else:
                    for cond in d.values():
                        cond.pop("_scores_full", None)

            model_result["layers"][str(L)] = layer_result

        results[model] = model_result
        del X

    out_path = "artifacts/format_transport_intervention.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
