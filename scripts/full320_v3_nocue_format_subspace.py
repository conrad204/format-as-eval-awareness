"""Cross-fitted no-cue-derived format-associated subspace analysis.

Uses only frozen full320-v3 activations. For each held purpose-family fold and
layer, no-cue benchmark/casual activations are averaged within payload block.
The resulting matched format contrast is repeated once for every stated
(payload_block_id, purpose_pair_id) group in that fold's training set, then an
UNCENTERED SVD supplies Q_nocue. Stated-purpose activations are projected with
I-QQ^T and the repository's exact probes are retrained.

The repetition is intentional: it is the no-cue equivalent of the existing
matched-format B matrix, whose rows are training (payload,purpose-pair) groups.
No-cue rows have no purpose family, so this maps each eligible training group
to its payload's cue-free contrast without assigning labels to no-cue rows.

Writes per-item OOF scores (NPZ), a JSON/CSV summary, a figure, and a concise
Markdown report. No model forward pass is performed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import factorial_vectors, fit_probe, orth_basis, remove  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
MODELS = {
    "llama31_8b": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
    "llama31_70b": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
    "llama33_70b": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
}
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nocue_payload_contrasts(X: np.ndarray, rows: list[dict]) -> dict[str, np.ndarray]:
    """Benchmark-minus-casual no-cue centroid for each complete payload."""
    by: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        if row["render_slot"] == "no_cue":
            by.setdefault((row["payload_block_id"], row["format"]), []).append(i)
    blocks = sorted({b for b, _ in by})
    out = {}
    for block in blocks:
        b, c = by.get((block, "benchmark"), []), by.get((block, "casual"), [])
        if b and c:
            out[block] = X[b].mean(axis=0) - X[c].mean(axis=0)
    return out


def training_group_keys(rows: list[dict], train_mask: np.ndarray) -> list[tuple[str, str]]:
    return sorted({(r["payload_block_id"], r["purpose_pair_id"])
                   for r, keep in zip(rows, train_mask)
                   if keep and r["render_slot"] == "stated"})


def score_layer(layer: int, Xfull: np.ndarray, rows: list[dict], stated_idx: np.ndarray,
                y: np.ndarray, fam: np.ndarray, bench: np.ndarray, casual: np.ndarray,
                rank: int):
    t0 = time.time()
    # Slice the layer inside the job: materializing every layer up front peaks at
    # 2x the full tensor and exhausts host memory on a 16 GB workstation.
    X = np.ascontiguousarray(Xfull[:, layer, :])
    Xs = X[stated_idx]
    stated_rows = [rows[i] for i in stated_idx]
    nc = nocue_payload_contrasts(X, rows)
    del X
    n = len(y)
    sp = np.full(n, np.nan, np.float32)
    sb = np.full(n, np.nan, np.float32)
    sc = np.full(n, np.nan, np.float32)
    sf = np.full(n, np.nan, np.float32)
    overlaps, var_qb, var_qn, basis_rows = [], [], [], []

    for fold in sorted(set(fam)):
        train, test = fam != fold, fam == fold
        keys = training_group_keys(stated_rows, train)
        D = np.asarray([nc[block] for block, _ in keys])
        if len(D) != len(keys):
            raise AssertionError("missing no-cue payload contrast for a training group")
        Qn = orth_basis(D, rank)
        _, B, _, groups = factorial_vectors(Xs, stated_rows, train)
        Qb = orth_basis(B, rank)
        if Qn.shape[1] != rank or Qb.shape[1] != rank:
            raise AssertionError((fold, Qn.shape, Qb.shape))
        Xt = remove(Xs, Qn)
        sp[test] = fit_probe(Xt[train], y[train], Xt[test])
        sf[test] = fit_probe(Xt[train], bench[train].astype(int), Xt[test])
        bt, be = train & bench, test & casual
        ct, ce = train & casual, test & bench
        sb[be] = fit_probe(Xt[bt], y[bt], Xt[be])
        sc[ce] = fit_probe(Xt[ct], y[ct], Xt[ce])

        overlaps.append(float(np.linalg.norm(Qb.T @ Qn, "fro") ** 2 / rank))
        Xh = Xs[test].astype(np.float64, copy=False)
        Xh = Xh - Xh.mean(axis=0, keepdims=True)
        denom = float(np.sum(Xh * Xh))
        var_qb.append(float(np.sum((Xh @ Qb) ** 2) / denom))
        var_qn.append(float(np.sum((Xh @ Qn) ** 2) / denom))
        basis_rows.append({"fold": fold, "n_training_groups": len(keys),
                           "n_unique_payloads": len({k[0] for k in keys}),
                           "qb_rows": len(groups)})

    return layer, sp, sb, sc, sf, {
        "trace_overlap_over_rank_mean": float(np.mean(overlaps)),
        "trace_overlap_over_rank_by_fold": overlaps,
        "heldout_stated_centered_variance_qb_mean": float(np.mean(var_qb)),
        "heldout_stated_centered_variance_qnocue_mean": float(np.mean(var_qn)),
        "basis_rows_by_fold": basis_rows,
    }, time.time() - t0


def fast_auc(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(S, axis=1, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(S.shape[1], dtype=np.float64), axis=1)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return (ranks[:, pos].sum(axis=1) - npos * (npos - 1) / 2) / (npos * nneg)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(scores_path: Path, raw_scores_path: Path, n_boot: int, seed: int) -> dict:
    d, raw = np.load(scores_path, allow_pickle=True), np.load(raw_scores_path, allow_pickle=True)
    y, blk = d["y"].astype(int), d["payload_block_id"]
    P, B, C, F = (d[k].astype(float) for k in ("purpose_scores", "b2c_scores", "c2b_scores", "format_scores"))
    ranks_raw = [int(x) for x in raw["ranks"]]
    ri0 = ranks_raw.index(0)
    RP, RB, RC = (raw[k][ri0].astype(float) for k in ("purpose_scores", "b2c_scores", "c2b_scores"))
    n_layers = P.shape[0]
    q = np.arange(n_layers) / (n_layers - 1)
    masks = [~np.isnan(x[0]) for x in (P, B, C, F)]
    rmasks = [~np.isnan(x[0]) for x in (RP, RB, RC)]

    def layer_auc(S, mask, labels=y):
        return np.array([roc_auc_score(labels[mask], S[L, mask]) for L in range(n_layers)])
    p, b, c = layer_auc(P, masks[0]), layer_auc(B, masks[1]), layer_auc(C, masks[2])
    fmt_y = (d["format"] == "benchmark").astype(int)
    f = layer_auc(F, masks[3], fmt_y)
    rp, rb, rc = layer_auc(RP, rmasks[0]), layer_auc(RB, rmasks[1]), layer_auc(RC, rmasks[2])
    gap, raw_gap = p - (b + c) / 2, rp - (rb + rc) / 2
    G, raw_G = float(TRAPZ(gap, q)), float(TRAPZ(raw_gap, q))

    by_block = {}
    for i, block in enumerate(blk):
        by_block.setdefault(block, []).append(i)
    blocks = np.array(sorted(by_block))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 8))
    for rep in range(n_boot):
        draw = rng.choice(blocks, len(blocks), replace=True)
        idx = np.concatenate([by_block[x] for x in draw])
        def auc_matrix(S, mask, labels):
            ii = idx[mask[idx]]
            return fast_auc(S[:, ii], labels[ii])
        ap, ab, ac = auc_matrix(P, masks[0], y), auc_matrix(B, masks[1], y), auc_matrix(C, masks[2], y)
        arp, arb, arc = auc_matrix(RP, rmasks[0], y), auc_matrix(RB, rmasks[1], y), auc_matrix(RC, rmasks[2], y)
        af = auc_matrix(F, masks[3], fmt_y)
        gg, rgg = TRAPZ(ap - (ab + ac) / 2, q), TRAPZ(arp - (arb + arc) / 2, q)
        boot[rep] = [rgg, gg, 1 - gg / rgg, TRAPZ(ap - arp, q),
                     TRAPZ(ab - arb, q), TRAPZ(ac - arc, q), TRAPZ(af, q), gg - rgg]

    geometry = json.loads(str(d["geometry_json"]))
    layers = []
    for L in range(n_layers):
        layers.append({"layer": L, "normalized_depth": float(q[L]),
                       "raw_gap": float(raw_gap[L]), "projected_gap": float(gap[L]),
                       "raw_joint_auc": float(rp[L]), "projected_joint_auc": float(p[L]),
                       "raw_b2c_auc": float(rb[L]), "projected_b2c_auc": float(b[L]),
                       "raw_c2b_auc": float(rc[L]), "projected_c2b_auc": float(c[L]),
                       "projected_format_auc": float(f[L]), **geometry[str(L)]})
    return {
        "model": str(d["model"]), "rank": int(d["rank"]), "n_layers": n_layers,
        "n_items_stated": len(y), "n_payload_blocks": len(blocks), "n_bootstrap": n_boot,
        "bootstrap_seed": seed, "raw_score_artifact": str(raw_scores_path),
        "raw_integrated_gap": raw_G, "raw_integrated_gap_ci95": ci(boot[:, 0]),
        "projected_integrated_gap": G, "projected_integrated_gap_ci95": ci(boot[:, 1]),
        "fraction_integrated_gap_removed": 1 - G / raw_G,
        "fraction_integrated_gap_removed_ci95": ci(boot[:, 2]),
        "integrated_delta_joint_auc": float(TRAPZ(p-rp, q)), "integrated_delta_joint_auc_ci95": ci(boot[:, 3]),
        "integrated_delta_b2c_auc": float(TRAPZ(b-rb, q)), "integrated_delta_b2c_auc_ci95": ci(boot[:, 4]),
        "integrated_delta_c2b_auc": float(TRAPZ(c-rc, q)), "integrated_delta_c2b_auc_ci95": ci(boot[:, 5]),
        "integrated_projected_format_auc": float(TRAPZ(f, q)), "integrated_projected_format_auc_ci95": ci(boot[:, 6]),
        "layers": layers,
    }


def emit_outputs(summary: dict, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = out_prefix.with_suffix(".json"), out_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = list(summary["layers"][0])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary["layers"])
    try:
        import matplotlib.pyplot as plt
        q = [x["normalized_depth"] for x in summary["layers"]]
        plt.figure(figsize=(6.4, 3.8))
        plt.plot(q, [x["raw_gap"] for x in summary["layers"]], label="Raw", lw=2)
        plt.plot(q, [x["projected_gap"] for x in summary["layers"]], label="No-cue projected", lw=2)
        plt.axhline(0, color="black", lw=.7); plt.xlabel("Normalized depth"); plt.ylabel("Cross-format transfer gap (AUC)")
        plt.legend(frameon=False); plt.tight_layout(); plt.savefig(out_prefix.with_suffix(".png"), dpi=180); plt.close()
    except ImportError:
        pass
    s = summary
    verdict = ("strengthens" if s["fraction_integrated_gap_removed_ci95"][0] > 0.2
               else "weakens" if s["fraction_integrated_gap_removed_ci95"][1] < 0.2 else "leaves uncertain")
    md = f"""# No-cue-derived format-associated subspace result

Model: `{s['model']}`; rank {s['rank']}; {s['n_payload_blocks']} complete payload blocks; {s['n_bootstrap']} paired block-bootstrap replicates.

- Raw integrated gap: **{s['raw_integrated_gap']:.4f}** (95% CI {s['raw_integrated_gap_ci95'][0]:.4f}, {s['raw_integrated_gap_ci95'][1]:.4f}).
- Projected integrated gap: **{s['projected_integrated_gap']:.4f}** (95% CI {s['projected_integrated_gap_ci95'][0]:.4f}, {s['projected_integrated_gap_ci95'][1]:.4f}).
- Fraction removed: **{s['fraction_integrated_gap_removed']:.1%}** (95% CI {s['fraction_integrated_gap_removed_ci95'][0]:.1%}, {s['fraction_integrated_gap_removed_ci95'][1]:.1%}).
- Integrated change in joint / B->C / C->B AUC: **{s['integrated_delta_joint_auc']:+.4f} / {s['integrated_delta_b2c_auc']:+.4f} / {s['integrated_delta_c2b_auc']:+.4f}**.
- Integrated projected format AUC: **{s['integrated_projected_format_auc']:.4f}**.

By the prespecified, conservative rule encoded in this script, this result **{verdict}** the current format-subspace claim. This classification is about whether a format-associated subspace estimated without purpose cues explains explicitly stated-purpose probe transfer failure; it is not evidence of spontaneous evaluation awareness.

Construction and limits: no-cue rows have no evaluation/deployment labels. The basis is an uncentered SVD of payload-matched benchmark-minus-casual no-cue centroid contrasts, cross-fitted by the existing held-purpose-family folds and projected only before exact probe retraining. Frozen-activation projection is not a behavioral causal intervention. The no-cue-derived subspace is not claimed to be “pure format,” and overlap with the stated-purpose-derived basis is not geometric identity.
"""
    out_prefix.with_suffix(".md").write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_8b", choices=sorted(MODELS))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", "4")))
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--out-prefix", type=Path)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    scores = args.scores or ROOT / "results" / f"nocue_format_subspace_scores_{args.model}_r{args.rank}.npz"
    prefix = args.out_prefix or ROOT / "results" / f"nocue_format_subspace_{args.model}_r{args.rank}"
    raw_scores = ROOT / "results" / f"full_layer_gap_erasure_scores_{args.model}.npz"

    if not args.summarize_only:
        rows = [json.loads(x) for x in ITEMS.open(encoding="utf-8") if x.strip()]
        by_id = {r["item_id"]: r for r in rows}
        with zipfile.ZipFile(MODELS[args.model]) as z:
            raw_ids = np.load(z.open("item_ids.npy"))
        if set(raw_ids) != set(by_id):
            raise ValueError("activation/item ID mismatch")
        ordered = [by_id[x] for x in raw_ids]
        stated_idx = np.flatnonzero([r["label"] is not None for r in ordered])
        stated_rows = [ordered[i] for i in stated_idx]
        y = np.array([int(r["label"]) for r in stated_rows])
        fam = np.array([r["purpose_family_id"] for r in stated_rows])
        fmt = np.array([r["format"] for r in stated_rows])
        bench, casual = fmt == "benchmark", fmt == "casual"
        print("loading frozen activation tensor", file=sys.stderr, flush=True)
        Xall = np.load(MODELS[args.model])["X"]
        n_layers = Xall.shape[1]
        gc.collect()
        outs = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(score_layer)(L, Xall, ordered, stated_idx,
                                 y, fam, bench, casual, args.rank)
            for L in range(n_layers)
        )
        n = len(y)
        arrays = [np.full((n_layers, n), np.nan, np.float32) for _ in range(4)]
        geometry = {}
        for L, p, b, c, f, g, dt in outs:
            arrays[0][L], arrays[1][L], arrays[2][L], arrays[3][L] = p, b, c, f
            geometry[str(L)] = g
            print(f"layer {L} done in {dt:.1f}s", file=sys.stderr, flush=True)
        scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores, model=args.model, rank=args.rank, y=y,
            payload_block_id=np.array([r["payload_block_id"] for r in stated_rows]),
            purpose_family_id=fam, format=fmt, purpose_scores=arrays[0], b2c_scores=arrays[1],
            c2b_scores=arrays[2], format_scores=arrays[3], geometry_json=json.dumps(geometry),
            items_sha256=sha256(ITEMS), activations_sha256=sha256(MODELS[args.model]))
    summary = summarize(scores, raw_scores, args.n_bootstrap, args.seed)
    emit_outputs(summary, prefix)
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
