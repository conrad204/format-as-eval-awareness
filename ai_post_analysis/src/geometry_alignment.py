"""Counterfactual geometry audit for stated purpose and prompt format.

The score-based analysis asks whether uncued formats move along a fitted
purpose probe.  This module asks the stricter geometric question: does the
activation displacement specific to an uncued format contrast align with the
matched evaluation-minus-deployment displacement itself?

All estimands use the eleven non-legacy templates, which form a complete
320-block x template x {evaluation, deployment, no-cue} panel.  For block b:

    P_b = mean_template(h_eval - h_deploy)
    V_bt = h_no_cue - 0.5 * (h_eval + h_deploy)
    D_b(group) = mean_group(V_bt) - mean_complement(V_bt)

The subtraction in V removes a format offset shared by cued and uncued rows.
We report raw Euclidean and per-feature-standardized geometry.  Purpose
directions used for scalar projection are cross-fitted by cue-vocabulary
family, and uncertainty is clustered by payload block.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")
MODEL_LABELS = {
    "llama31_8b": "Llama 3.1 8B",
    "llama31_70b": "Llama 3.1 70B",
    "llama33_70b": "Llama 3.3 70B",
}
COLORS = {"benchmark_minus_casual": "#1b8a6b", "markup_minus_plain": "#d95f32"}
LABELS = {"benchmark_minus_casual": "benchmark - casual",
          "markup_minus_plain": "markup - plain"}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else math.nan


def _stable_half(name: str) -> int:
    return hashlib.sha256(("geometry-half-v1\0" + name).encode()).digest()[0] & 1


def _percentile_ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _jackknife_cosine(p: np.ndarray, d: np.ndarray) -> dict:
    """Joint delete-one-block jackknife for cosine(mean(P), mean(D))."""
    n = len(p)
    ps, ds = p.sum(axis=0), d.sum(axis=0)
    point = _cosine(ps, ds)
    leave = np.empty(n, dtype=np.float64)
    for i in range(n):
        leave[i] = _cosine(ps - p[i], ds - d[i])
    center = float(leave.mean())
    se = float(np.sqrt((n - 1) / n * np.square(leave - center).sum()))
    return {
        "point": point,
        "jackknife_se": se,
        "ci95_normal": [max(-1.0, point - 1.96 * se),
                         min(1.0, point + 1.96 * se)],
        "leave_one_min": float(leave.min()),
        "leave_one_max": float(leave.max()),
    }


def _crossfit_projection(
    d: np.ndarray,
    excluded_purpose: np.ndarray,
    held_purpose: np.ndarray,
    fold_names: list[str],
    boot_idx: np.ndarray,
) -> dict:
    """Project D blocks on purpose directions fitted without each cue family."""
    per_fold_ratios = []
    fold_rows = []
    for fi, name in enumerate(fold_names):
        p_train = excluded_purpose[fi]
        p_norm = float(np.linalg.norm(p_train))
        if p_norm == 0:
            raise RuntimeError(f"zero purpose displacement outside fold {name}")
        ratios = (d @ (p_train / p_norm)) / p_norm
        per_fold_ratios.append(ratios)
        fold_rows.append({
            "fold": name,
            "mean_projection_ratio": float(ratios.mean()),
            "cos_p_excluding_fold_d": _cosine(p_train, d.mean(axis=0)),
            "cos_p_held_fold_d": _cosine(held_purpose[fi], d.mean(axis=0)),
            "cos_excluded_vs_held_purpose": _cosine(p_train, held_purpose[fi]),
        })
    ratios = np.mean(np.stack(per_fold_ratios), axis=0)
    draws = ratios[boot_idx].mean(axis=1)
    return {
        "point": float(ratios.mean()),
        "ci95_block_bootstrap_conditional": _percentile_ci(draws),
        "block_sd": float(ratios.std(ddof=1)),
        "positive_block_fraction": float(np.mean(ratios > 0)),
        "folds": fold_rows,
        "mean_cos_p_excluding_fold_d": float(np.mean(
            [row["cos_p_excluding_fold_d"] for row in fold_rows])),
        "mean_cos_p_held_fold_d": float(np.mean(
            [row["cos_p_held_fold_d"] for row in fold_rows])),
        "mean_cos_excluded_vs_held_purpose": float(np.mean(
            [row["cos_excluded_vs_held_purpose"] for row in fold_rows])),
    }


def _split_half(p: np.ndarray, d: np.ndarray, half: np.ndarray) -> dict:
    return {
        "n_half_0": int(np.sum(half == 0)),
        "n_half_1": int(np.sum(half == 1)),
        "purpose_direction_cosine": _cosine(p[half == 0].mean(0), p[half == 1].mean(0)),
        "format_differential_cosine": _cosine(d[half == 0].mean(0), d[half == 1].mean(0)),
    }


def _catalog_null(
    p_mean: np.ndarray,
    template_vectors: np.ndarray,
    observed_mask: np.ndarray,
    expected_sign: int,
) -> dict:
    """Exact null over all template subsets with the observed group size."""
    n, k = len(observed_mask), int(observed_mask.sum())
    weights = []
    combos = list(itertools.combinations(range(n), k))
    for combo in combos:
        mask = np.zeros(n, dtype=bool)
        mask[list(combo)] = True
        weights.append(np.where(mask, 1.0 / k, -1.0 / (n - k)))
    contrast = np.asarray(weights, dtype=np.float64) @ template_vectors
    nums = contrast @ p_mean
    dens = np.linalg.norm(contrast, axis=1) * np.linalg.norm(p_mean)
    vals = nums / dens
    observed = _cosine(
        p_mean,
        template_vectors[observed_mask].mean(0) - template_vectors[~observed_mask].mean(0),
    )
    tol = 1e-12
    one_sided = (np.mean(vals >= observed - tol) if expected_sign > 0
                 else np.mean(vals <= observed + tol))
    return {
        "n_exact_assignments": len(combos),
        "observed_cosine": observed,
        "expected_sign": expected_sign,
        "one_sided_p": float(one_sided),
        "two_sided_p": float(np.mean(np.abs(vals) >= abs(observed) - tol)),
        "null_min": float(vals.min()),
        "null_median": float(np.median(vals)),
        "null_max": float(vals.max()),
    }


class GeometryDesign:
    def __init__(self, meta_path: Path):
        self.meta = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        self.blocks = sorted({row["payload_block_id"] for row in self.meta})
        self.templates = sorted({row["template_id"] for row in self.meta if not row["legacy"]})
        self.block_code = {name: i for i, name in enumerate(self.blocks)}
        self.template_code = {name: i for i, name in enumerate(self.templates)}
        nb, nt = len(self.blocks), len(self.templates)
        if (nb, nt) != (320, 11):
            raise RuntimeError(f"expected complete 320 x 11 v2 panel, got {nb} x {nt}")

        self.eval_idx = np.full((nb, nt), -1, dtype=np.int32)
        self.deploy_idx = np.full((nb, nt), -1, dtype=np.int32)
        self.none_idx = np.full((nb, nt), -1, dtype=np.int32)
        self.features: dict[str, dict] = {}
        self.family_idx = np.full((nb, nt), -1, dtype=np.int8)
        family_names = sorted({row["purpose_family_id"] for row in self.meta
                               if not row["legacy"] and row["intended_purpose"]
                               in ("evaluation", "deployment")})
        family_code = {name: i for i, name in enumerate(family_names)}
        contract_cells: dict[tuple[str, str], set] = {}

        for i, row in enumerate(self.meta):
            if row["legacy"]:
                continue
            b = self.block_code[row["payload_block_id"]]
            t = self.template_code[row["template_id"]]
            purpose = row["intended_purpose"]
            target = {"evaluation": self.eval_idx, "deployment": self.deploy_idx,
                      "none": self.none_idx}.get(purpose)
            if target is None:
                raise RuntimeError(f"unexpected purpose {purpose!r}")
            if target[b, t] != -1:
                raise RuntimeError(f"duplicate panel cell block={b} template={t} purpose={purpose}")
            target[b, t] = i
            self.features.setdefault(row["template_id"], {
                "format": row["format"], "serialization": row["serialization"]})
            contract_cells.setdefault((row["payload_block_id"], row["template_id"]), set()).add(
                row.get("contract_id"))
            if purpose in ("evaluation", "deployment"):
                fi = family_code[row["purpose_family_id"]]
                prior = self.family_idx[b, t]
                if prior not in (-1, fi):
                    raise RuntimeError("evaluation/deployment pair crosses cue families")
                self.family_idx[b, t] = fi

        for name, arr in (("evaluation", self.eval_idx), ("deployment", self.deploy_idx),
                          ("no-cue", self.none_idx)):
            if np.any(arr < 0):
                raise RuntimeError(f"incomplete {name} panel: {int(np.sum(arr < 0))} missing cells")
        bad_contracts = [cell for cell, values in contract_cells.items() if len(values) != 1]
        if bad_contracts:
            raise RuntimeError(f"contract differs within purpose triplet for {bad_contracts[:3]}")

        if np.any(self.family_idx < 0):
            raise RuntimeError("missing cue-family assignment in v2 panel")
        self.fold_names = family_names
        self.half = np.array([_stable_half(b) for b in self.blocks], dtype=np.int8)
        self.group_masks = {
            "benchmark_minus_casual": np.array([
                self.features[t]["format"] == "benchmark" for t in self.templates]),
            "markup_minus_plain": np.array([
                self.features[t]["serialization"] != "plain" for t in self.templates]),
        }
        if [int(self.group_masks[k].sum()) for k in self.group_masks] != [6, 3]:
            raise RuntimeError("unexpected v2 template group sizes")


def _layer_vectors(h: np.ndarray, design: GeometryDesign) -> tuple:
    """Return P blocks, D blocks, template means, and cued feature scale."""
    nb, nt = design.eval_idx.shape
    width = h.shape[1]
    p = np.zeros((nb, width), dtype=np.float64)
    differentials = {name: np.zeros((nb, width), dtype=np.float64)
                     for name in design.group_masks}
    template_means = np.zeros((nt, width), dtype=np.float64)
    feature_sum = np.zeros(width, dtype=np.float64)
    feature_sumsq = np.zeros(width, dtype=np.float64)
    n_cued = 0
    family_sum = np.zeros((len(design.fold_names), width), dtype=np.float64)
    family_count = np.zeros(len(design.fold_names), dtype=np.int64)

    weights = {
        name: np.where(mask, 1.0 / mask.sum(), -1.0 / (~mask).sum())
        for name, mask in design.group_masks.items()
    }
    for t in range(nt):
        e = np.asarray(h[design.eval_idx[:, t]], dtype=np.float64)
        dep = np.asarray(h[design.deploy_idx[:, t]], dtype=np.float64)
        none = np.asarray(h[design.none_idx[:, t]], dtype=np.float64)
        pair_delta = e - dep
        p += pair_delta
        for fi in range(len(design.fold_names)):
            mask = design.family_idx[:, t] == fi
            family_sum[fi] += pair_delta[mask].sum(axis=0)
            family_count[fi] += int(mask.sum())
        v = none - 0.5 * (e + dep)
        template_means[t] = v.mean(axis=0)
        for name in differentials:
            differentials[name] += weights[name][t] * v
        feature_sum += e.sum(axis=0) + dep.sum(axis=0)
        feature_sumsq += np.square(e).sum(axis=0) + np.square(dep).sum(axis=0)
        n_cued += 2 * nb
    p /= nt
    total_sum = family_sum.sum(axis=0)
    total_count = int(family_count.sum())
    held_purpose = family_sum / family_count[:, None]
    excluded_purpose = ((total_sum[None, :] - family_sum)
                        / (total_count - family_count)[:, None])
    mean = feature_sum / n_cued
    variance = np.maximum(feature_sumsq / n_cued - np.square(mean), 1e-12)
    return (p, differentials, template_means, np.sqrt(variance),
            excluded_purpose, held_purpose)


def _analyze_metric(
    p: np.ndarray,
    differentials: dict[str, np.ndarray],
    template_means: np.ndarray,
    excluded_purpose: np.ndarray,
    held_purpose: np.ndarray,
    design: GeometryDesign,
    boot_idx: np.ndarray,
    do_catalog_null: bool,
) -> dict:
    out = {}
    for name, d in differentials.items():
        row = {
            "cosine": _jackknife_cosine(p, d),
            "norm_ratio_format_to_purpose": float(
                np.linalg.norm(d.mean(axis=0)) / np.linalg.norm(p.mean(axis=0))),
            "split_half": _split_half(p, d, design.half),
            "crossfit_projection_ratio": _crossfit_projection(
                d, excluded_purpose, held_purpose, design.fold_names, boot_idx),
        }
        if do_catalog_null:
            row["exact_template_catalog_null"] = _catalog_null(
                p.mean(axis=0), template_means, design.group_masks[name],
                expected_sign=1 if name == "benchmark_minus_casual" else -1)
        out[name] = row
    return out


def analyze_model(
    model: str,
    freeze_dir: Path,
    results_dir: Path,
    layers: list[int] | None,
    n_boot: int,
    seed: int,
    null_depths: list[float],
    force: bool,
) -> dict:
    out_path = results_dir / f"{model}_geometry.json"
    if out_path.exists() and not force and layers is None:
        print(f"[skip] {out_path} exists; pass --force to recompute", flush=True)
        return json.loads(out_path.read_text())

    model_dir = freeze_dir / model
    config = json.loads((model_dir / "config.json").read_text())
    x = np.load(model_dir / "X_all_layers.npy", mmap_mode="r")
    design = GeometryDesign(model_dir / "meta.jsonl")
    if x.shape[0] != len(design.meta) or x.shape[1] != config["n_transformer_layers"]:
        raise RuntimeError(f"activation/config mismatch for {model}: {x.shape}")
    use_layers = layers if layers is not None else list(range(x.shape[1]))
    null_layers = {int(round(q * (x.shape[1] - 1))) for q in null_depths}
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(design.blocks), size=(n_boot, len(design.blocks)),
                            dtype=np.int32)

    rows = []
    for layer in use_layers:
        print(f"[{model}] layer {layer}/{x.shape[1] - 1}", flush=True)
        (p, diffs, template_means, scale,
         excluded_purpose, held_purpose) = _layer_vectors(x[:, layer, :], design)
        raw = _analyze_metric(p, diffs, template_means, excluded_purpose,
                              held_purpose, design, boot_idx, layer in null_layers)
        standardized = _analyze_metric(
            p / scale, {k: v / scale for k, v in diffs.items()},
            template_means / scale, excluded_purpose / scale,
            held_purpose / scale, design, boot_idx, layer in null_layers)
        rows.append({"layer": int(layer), "normalized_depth": float(layer / (x.shape[1] - 1)),
                     "raw": raw, "standardized": standardized})

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "header": {
            "analysis": "counterfactual purpose-format geometry",
            "status": "post-outcome exploratory analysis of spent full320-v3 freeze",
            "model": model,
            "model_id": config["model"],
            "model_revision": config["model_revision"],
            "activation_sha256": config["activations_all_layers_sha256"],
            "script_sha256": script_hash,
            "n_layers": int(x.shape[1]),
            "hidden_size": int(x.shape[2]),
            "n_blocks": len(design.blocks),
            "n_templates": len(design.templates),
            "templates": design.templates,
            "template_group_sizes": {k: int(v.sum()) for k, v in design.group_masks.items()},
            "fold_names": design.fold_names,
            "n_boot": n_boot,
            "seed": seed,
            "null_depths": null_depths,
            "null_layers": sorted(null_layers),
            "metrics": ["raw_euclidean", "cued_feature_standardized_euclidean"],
            "estimand": "alignment of matched purpose displacement with no-cue-minus-cued format displacement",
            "claim_boundary": "representational geometry only; not behavior, causality, or evaluation awareness",
        },
        "layers": rows,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}", flush=True)
    return result


def make_figure(results: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(5.2 * len(results), 7.2),
                             sharex="col", squeeze=False)
    for col, (model, result) in enumerate(results.items()):
        q = np.array([row["normalized_depth"] for row in result["layers"]])
        family_alignment = np.array([
            row["standardized"]["benchmark_minus_casual"]
            ["crossfit_projection_ratio"]["mean_cos_excluded_vs_held_purpose"]
            for row in result["layers"]])
        for name in COLORS:
            raw = [row["raw"][name]["cosine"] for row in result["layers"]]
            point = np.array([r["point"] for r in raw])
            lo = np.array([r["ci95_normal"][0] for r in raw])
            hi = np.array([r["ci95_normal"][1] for r in raw])
            axes[0, col].fill_between(q, lo, hi, color=COLORS[name], alpha=0.16, lw=0)
            axes[0, col].plot(q, point, color=COLORS[name], lw=2, label=LABELS[name])

            proj = [row["standardized"][name]["crossfit_projection_ratio"]
                    for row in result["layers"]]
            pp = np.array([r["point"] for r in proj])
            plo = np.array([r["ci95_block_bootstrap_conditional"][0] for r in proj])
            phi = np.array([r["ci95_block_bootstrap_conditional"][1] for r in proj])
            axes[1, col].fill_between(q, plo, phi, color=COLORS[name], alpha=0.16, lw=0)
            axes[1, col].plot(q, pp, color=COLORS[name], lw=2)

        for row in range(2):
            axes[row, col].axhline(0, color="0.35", lw=0.9)
            axes[row, col].grid(alpha=0.16, lw=0.6)
            axes[row, col].set_xlim(0, 1)
            unstable = family_alignment < 0.5
            if unstable.any():
                edges = np.flatnonzero(np.diff(unstable.astype(np.int8)) != 0) + 1
                for span in np.split(np.arange(len(q)), edges):
                    if len(span) and unstable[span[0]]:
                        axes[row, col].axvspan(q[span[0]], q[span[-1]],
                                              color="#9aa0a6", alpha=0.12,
                                              lw=0, zorder=0)
        axes[0, col].set_title(MODEL_LABELS.get(model, model), fontsize=11)
        axes[1, col].set_xlabel("normalized depth")
    axes[0, 0].set_ylabel("raw displacement cosine")
    axes[1, 0].set_ylabel("held-family projection /\nexplicit-purpose magnitude")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("Counterfactual geometry of uncued prompt format", fontsize=13)
    fig.text(0.5, 0.01,
             "Top: joint block-jackknife 95% intervals. Bottom: purpose direction fit without "
             "each cue family; conditional block-bootstrap 95% intervals. Grey: held-family "
             "purpose alignment < 0.5.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}", flush=True)


def write_summary(results: dict[str, dict], path: Path, figure_path: Path) -> dict:
    summary = {
        "analysis": "counterfactual purpose-format geometry",
        "status": "post-outcome exploratory",
        "figure": str(figure_path),
        "fixed_depths": {},
    }
    requested = (0.25, 0.375, 0.5, 0.75, 1.0)
    for model, result in results.items():
        rows = result["layers"]
        by_layer = {row["layer"]: row for row in rows}
        last_layer = result["header"]["n_layers"] - 1
        picked = []
        for q in requested:
            row = by_layer[int(round(q * last_layer))]
            picked.append({
                "requested_depth": q,
                "layer": row["layer"],
                "normalized_depth": row["normalized_depth"],
                "raw_cosine": {name: row["raw"][name]["cosine"] for name in COLORS},
                "standardized_crossfit_projection_ratio": {
                    name: row["standardized"][name]["crossfit_projection_ratio"]
                    for name in COLORS},
                "standardized_exact_template_catalog_null": {
                    name: row["standardized"][name].get("exact_template_catalog_null")
                    for name in COLORS},
            })
        summary["fixed_depths"][model] = picked
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[wrote] {path}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="activations/full320-v3")
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--figures-dir", default="ai_post_analysis/figures")
    ap.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    ap.add_argument("--layers", nargs="*", type=int,
                    help="development/debug subset; omit for all layers")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-depths", nargs="*", type=float,
                    default=[0.25, 0.375, 0.5, 0.75, 1.0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    results = {}
    for model in args.models:
        results[model] = analyze_model(
            model, Path(args.freeze_dir), results_dir, args.layers,
            args.n_boot, args.seed, args.null_depths, args.force)

    if args.layers is None:
        figure = figures_dir / "08_counterfactual_geometry.png"
        make_figure(results, figure)
        write_summary(results, results_dir / "geometry_summary.json", figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
