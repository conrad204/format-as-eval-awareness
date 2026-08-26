"""Figures for the implicit-purpose (Direction 1) analysis.

The ladder and control panels are drawn at a fixed normalized depth q = 0.375
rather than at each model's peak-AUC layer. That depth is not chosen here: it
is the one the repository's own estimator-reversal analysis
(`artifacts/exploratory/full320_exact_logistic_estimator_reversal_*`) already
pinned, so using it avoids selecting a layer on the outcome being plotted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_PURPOSE = "#2a78d6"
C_MARKUP = "#eb6834"
C_FAMILY = "#1baf7a"
C_MUTED = "#9aa0a6"
Q_FIXED = 0.375
AUC_FLOOR = 0.95
MODEL_LABEL = {"llama31_8b": "Llama 3.1 8B", "llama31_70b": "Llama 3.1 70B",
               "llama33_70b": "Llama 3.3 70B"}


def load(rd: Path, model: str, kind: str):
    p = rd / f"{model}_{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fixed_layer(n_layers: int) -> int:
    return max(0, min(n_layers - 1, int(round(Q_FIXED * n_layers)) - 1))


def shade_unreliable(ax, q, auc):
    """Grey out depths where the axis itself is too weak to interpret a ratio."""
    bad = auc < AUC_FLOOR
    if not bad.any():
        return
    edges = np.flatnonzero(np.diff(bad.astype(int)) != 0) + 1
    spans = np.split(np.arange(len(bad)), edges)
    for s in spans:
        if len(s) and bad[s[0]]:
            ax.axvspan(q[s[0]], q[s[-1]], color=C_MUTED, alpha=0.13, lw=0, zorder=0)


def fig_depth(rd: Path, models, out: Path):
    have = [(m, load(rd, m, "main"), load(rd, m, "templates")) for m in models]
    have = [(m, a, b) for m, a, b in have if a and b]
    fig, axes = plt.subplots(1, len(have), figsize=(5.4 * len(have), 4.2), squeeze=False)

    for ax, (model, main, tpl) in zip(axes[0], have):
        n = main["header"]["n_transformer_layers"]
        q = np.array([r["layer"] for r in main["layers"]]) / (n - 1)
        auc = np.array([r["purpose_auc_oof"] for r in main["layers"]])
        rho = np.array([r["point"]["rho"] for r in main["layers"]])
        rlo = np.array([r["ci95"]["rho"]["ci95"][0] for r in main["layers"]])
        rhi = np.array([r["ci95"]["rho"]["ci95"][1] for r in main["layers"]])
        mk = np.array([r["markup_vs_plain"]["ratio_stated_units"] for r in tpl["layers"]])
        mlo = np.array([r["markup_vs_plain"]["ratio_ci95"][0] for r in tpl["layers"]])
        mhi = np.array([r["markup_vs_plain"]["ratio_ci95"][1] for r in tpl["layers"]])

        ax.set_ylim(-0.32, 0.22)
        shade_unreliable(ax, q, auc)
        ax.axhline(0, color="0.35", lw=1.0, zorder=1)
        ax.fill_between(q, rlo, rhi, color=C_FAMILY, alpha=0.2, lw=0)
        ax.plot(q, rho, color=C_FAMILY, lw=2.0, label="benchmark − casual", zorder=3)
        ax.fill_between(q, mlo, mhi, color=C_MARKUP, alpha=0.2, lw=0)
        ax.plot(q, mk, color=C_MARKUP, lw=2.0, label="markup − plain", zorder=3)

        peak = int(np.argmax(auc))
        ax.axvline(q[peak], color=C_PURPOSE, lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.text(q[peak] + 0.012, -0.30,
                f"peak AUC {auc[peak]:.3f}\n(L{main['layers'][peak]['layer']})",
                fontsize=7.5, color=C_PURPOSE, va="bottom")

        ax2 = ax.twinx()
        ax2.plot(q, auc, color=C_MUTED, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax2.set_ylim(0.5, 1.02)
        ax2.tick_params(labelsize=7.5, colors=C_MUTED)
        if ax is axes[0][-1]:
            ax2.set_ylabel("purpose AUC (dotted)", fontsize=8, color=C_MUTED)

        ax.set_title(MODEL_LABEL.get(model, model), fontsize=11)
        ax.set_xlabel("normalized depth")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.15, lw=0.6)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    axes[0][0].set_ylabel("no-cue shift (stated-cue units)")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Uncued prompts drift along the purpose axis, and serialization "
                 "drives it further than template family", fontsize=12.5)
    fig.text(0.5, 0.005, f"shaded: purpose AUC < {AUC_FLOOR}, where the ratio's "
                         "denominator is unreliable", ha="center", fontsize=7.5,
             color=C_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=190)
    plt.close(fig)


def fig_ladder(rd: Path, models, out: Path):
    have = [(m, load(rd, m, "templates")) for m in models]
    have = [(m, t) for m, t in have if t]
    fig, axes = plt.subplots(1, len(have), figsize=(5.0 * len(have), 4.4), squeeze=False)

    for ax, (model, tpl) in zip(axes[0], have):
        rows = {r["layer"]: r for r in tpl["layers"]}
        L = fixed_layer(tpl["header"]["n_transformer_layers"])
        r = rows[L]
        feats = tpl["header"]["template_features"]
        items = sorted(r["pi_by_template"].items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]

        y = np.arange(len(names))
        ax.axvline(0, color="0.35", lw=1.0)
        for yi, n_, v in zip(y, names, vals):
            c = C_MARKUP if feats[n_]["serialization"] != "plain" else C_PURPOSE
            mk = "s" if feats[n_]["format"] == "benchmark" else "o"
            ax.plot([0, v], [yi, yi], color=c, lw=1.5, alpha=0.5, zorder=2)
            ax.scatter([v], [yi], color=c, marker=mk, s=46, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("_v2", "").replace("_v1", "") for n in names],
                           fontsize=8)
        ax.set_xlabel("no-cue placement π (stated-cue units)")
        ax.set_title(f"{MODEL_LABEL.get(model, model)} — L{L} (q≈{Q_FIXED})", fontsize=11)
        ax.grid(axis="x", alpha=0.15, lw=0.6)

    handles = [plt.Line2D([], [], color=C_MARKUP, marker="o", ls="", label="markup serialization"),
               plt.Line2D([], [], color=C_PURPOSE, marker="o", ls="", label="plain serialization"),
               plt.Line2D([], [], color=C_MUTED, marker="s", ls="", label="benchmark-family template"),
               plt.Line2D([], [], color=C_MUTED, marker="o", ls="", label="casual-family template")]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Serialization orders the template ladder; template family does not",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def fig_differential_ladder(rd: Path, models, out: Path):
    """Per-template ladder of the no-cue-specific part, not raw placement.

    pi(t) mixes the no-cue-specific shift with whatever offset the template
    imposes on every row. This ladder plots the difference of the two strata's
    offsets, which is the quantity the headline claim actually rests on.
    """
    have = [(m, load(rd, m, "offsets"), load(rd, m, "templates")) for m in models]
    have = [(m, o, t) for m, o, t in have if o and t]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.0 * len(have), 4.4), squeeze=False)

    for ax, (model, off, tpl) in zip(axes[0], have):
        L = fixed_layer(tpl["header"]["n_transformer_layers"])
        row = {r["layer"]: r for r in off["layers"]}[L]
        feats = tpl["header"]["template_features"]
        diff = {t: row["offset_none_by_template"][t] - row["offset_cued_by_template"][t]
                for t in row["offset_none_by_template"]}
        items = sorted(diff.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]

        y = np.arange(len(names))
        ax.axvline(0, color="0.35", lw=1.0)
        for yi, n_, v in zip(y, names, vals):
            c = C_MARKUP if feats[n_]["serialization"] != "plain" else C_PURPOSE
            mk = "s" if feats[n_]["format"] == "benchmark" else "o"
            ax.plot([0, v], [yi, yi], color=c, lw=1.5, alpha=0.5, zorder=2)
            ax.scatter([v], [yi], color=c, marker=mk, s=46, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("_v2", "").replace("_v1", "") for n in names],
                           fontsize=8)
        ax.set_xlabel("no-cue-specific shift (stated-cue units)")
        ax.set_title(f"{MODEL_LABEL.get(model, model)} — L{L} (q≈{Q_FIXED})", fontsize=11)
        ax.grid(axis="x", alpha=0.15, lw=0.6)

    handles = [plt.Line2D([], [], color=C_MARKUP, marker="o", ls="", label="markup serialization"),
               plt.Line2D([], [], color=C_PURPOSE, marker="o", ls="", label="plain serialization"),
               plt.Line2D([], [], color=C_MUTED, marker="s", ls="", label="benchmark-family template"),
               plt.Line2D([], [], color=C_MUTED, marker="o", ls="", label="casual-family template")]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Template ladder after removing each wrapper's effect on cued rows",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def fig_differential(rd: Path, models, out: Path):
    """The headline control: the shift exists only where no purpose is stated."""
    have = [(m, load(rd, m, "offsets"), load(rd, m, "main")) for m in models]
    have = [(m, o, mn) for m, o, mn in have if o and mn]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.2), squeeze=False)

    for ax, (model, off, main) in zip(axes[0], have):
        n = main["header"]["n_transformer_layers"]
        q = np.array([r["layer"] for r in off["layers"]]) / (n - 1)
        auc = np.array([r["purpose_auc_oof"] for r in main["layers"]])
        g = "markup_minus_plain"
        cued = np.array([r[g]["cued"]["point"] for r in off["layers"]])
        none = np.array([r[g]["no_cue"]["point"] for r in off["layers"]])
        dlo = np.array([r[g]["differential"]["ci95"][0] for r in off["layers"]])
        dhi = np.array([r[g]["differential"]["ci95"][1] for r in off["layers"]])
        diff = np.array([r[g]["differential"]["point"] for r in off["layers"]])

        ax.set_ylim(-0.42, 0.22)
        shade_unreliable(ax, q, auc[: len(q)])
        ax.axhline(0, color="0.35", lw=1.0, zorder=1)
        ax.plot(q, cued, color=C_PURPOSE, lw=2.0, label="on cued rows", zorder=3)
        ax.plot(q, none, color=C_MARKUP, lw=2.0, label="on no-cue rows", zorder=3)
        ax.fill_between(q, dlo, dhi, color="0.35", alpha=0.18, lw=0)
        ax.plot(q, diff, color="0.25", lw=1.5, ls=(0, (3, 2)),
                label="differential (no-cue − cued)", zorder=3)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=11)
        ax.set_xlabel("normalized depth")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.15, lw=0.6)

    axes[0][0].set_ylabel("markup − plain (stated-cue units)")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("The markup shift is specific to prompts that state no purpose",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def fig_controls(rd: Path, models, out: Path):
    """Format erasure and the permutation null, on the scale-free statistic."""
    have = [(m, load(rd, m, "controls"), load(rd, m, "permutation_null"),
             load(rd, m, "main")) for m in models]
    have = [h for h in have if h[1] or h[2]]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.4 * len(have), 4.2), squeeze=False)

    for ax, (model, ctl, perm, main) in zip(axes[0], have):
        raw = {r["layer"]: r["point"]["smd_paired"] for r in main["layers"]}
        ax.axhline(0, color="0.35", lw=1.0)
        if ctl:
            L = [r["layer"] for r in ctl["layers"]]
            ax.plot(L, [raw[l] for l in L], "o-", color=C_FAMILY, lw=1.6, ms=5,
                    label="observed axis")
            ax.plot(L, [r["erased"]["point"]["smd_paired"] for r in ctl["layers"]],
                    "^--", color=C_PURPOSE, lw=1.4, ms=5,
                    label="after INLP format erasure")
        if perm:
            for r in perm["layers"]:
                draws = r["benchmark_minus_casual"]["null_draws"]
                ax.scatter([r["layer"]] * len(draws), draws, s=9, color=C_MUTED,
                           alpha=0.55, zorder=2,
                           label="permuted-label null draws"
                           if r is perm["layers"][0] else None)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=11)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.15, lw=0.6)

    axes[0][0].set_ylabel("no-cue benchmark − casual (standardized)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Controls: format erasure and the control-task null", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def fig_contract(rd: Path, models, out: Path):
    """The within-template replication: structured-output contract C4."""
    have = [(m, load(rd, m, "contracts"), load(rd, m, "main")) for m in models]
    have = [(m, c, mn) for m, c, mn in have if c and mn]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.2), squeeze=False)

    for ax, (model, ct, main) in zip(axes[0], have):
        n = main["header"]["n_transformer_layers"]
        rows = ct["layers"]
        q = np.array([r["layer"] for r in rows]) / (n - 1)
        auc = np.array([r["purpose_auc_oof"] for r in main["layers"]])
        s = [r["structured_vs_rest"] for r in rows]
        none = np.array([x["no_cue"]["point"] for x in s])
        cued = np.array([x["cued"]["point"] for x in s])
        dif = np.array([x["differential"]["point"] for x in s])
        dlo = np.array([x["differential"]["ci95"][0] for x in s])
        dhi = np.array([x["differential"]["ci95"][1] for x in s])

        ax.set_ylim(-0.30, 0.20)
        shade_unreliable(ax, q, auc[: len(q)])
        ax.axhline(0, color="0.35", lw=1.0, zorder=1)
        ax.plot(q, cued, color=C_PURPOSE, lw=2.0, label="on cued rows", zorder=3)
        ax.plot(q, none, color=C_MARKUP, lw=2.0, label="on no-cue rows", zorder=3)
        ax.fill_between(q, dlo, dhi, color="0.35", alpha=0.18, lw=0)
        ax.plot(q, dif, color="0.25", lw=1.5, ls=(0, (3, 2)),
                label="differential (no-cue − cued)", zorder=3)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=11)
        ax.set_xlabel("normalized depth")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.15, lw=0.6)

    axes[0][0].set_ylabel('C4 ("return an object…") − C0–C3\n(stated-cue units)')
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Same signature from the response contract, measured inside "
                 "each template", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def fig_nuisance(rd: Path, models, out: Path):
    """Would any equally decodable axis separate these prompts? No."""
    have = [(m, load(rd, m, "nuisance"), load(rd, m, "templates"),
             load(rd, m, "main")) for m in models]
    have = [h for h in have if h[1] and h[2] and h[3]]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.2), squeeze=False)

    for ax, (model, nui, tpl, main) in zip(axes[0], have):
        n = main["header"]["n_transformer_layers"]
        trow = {r["layer"]: r for r in tpl["layers"]}
        rows = [r for r in nui["layers"] if r["layer"] in trow]
        q = np.array([r["layer"] for r in rows]) / (n - 1)
        auc = np.array([r["purpose_auc_oof"] for r in main["layers"]])
        task = np.array([r["markup_vs_plain"]["ratio_task_units"] for r in rows])
        tlo = np.array([r["markup_vs_plain"]["ci95"][0] for r in rows])
        thi = np.array([r["markup_vs_plain"]["ci95"][1] for r in rows])
        purp = np.array([trow[r["layer"]]["markup_vs_plain"]["ratio_stated_units"]
                         for r in rows])

        ax.set_ylim(-0.32, 0.12)
        shade_unreliable(ax, q, auc[: len(q)])
        ax.axhline(0, color="0.35", lw=1.0, zorder=1)
        ax.plot(q, purp, color=C_MARKUP, lw=2.0, label="on the purpose axis", zorder=3)
        ax.fill_between(q, tlo, thi, color=C_PURPOSE, alpha=0.25, lw=0)
        ax.plot(q, task, color=C_PURPOSE, lw=2.0,
                label="on a task-family axis", zorder=3)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=11)
        ax.set_xlabel("normalized depth")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.15, lw=0.6)

    axes[0][0].set_ylabel("markup − plain, in each axis's own units")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("The shift is specific to the purpose direction, not to any "
                 "decodable axis", fontsize=12.5)
    fig.text(0.5, 0.005, "the task-family axis is orthogonal to template by design "
                         "and is decoded at AUC ≈ 0.999", ha="center", fontsize=7.5,
             color=C_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=190)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--fig-dir", default="ai_post_analysis/figures")
    ap.add_argument("--models", nargs="*",
                    default=["llama31_8b", "llama31_70b", "llama33_70b"])
    args = ap.parse_args()

    rd, fd = Path(args.results_dir), Path(args.fig_dir)
    fd.mkdir(parents=True, exist_ok=True)
    fig_depth(rd, args.models, fd / "01_depth_profile.png")
    fig_ladder(rd, args.models, fd / "02_template_ladder.png")
    fig_differential_ladder(rd, args.models, fd / "03_differential_ladder.png")
    fig_differential(rd, args.models, fd / "04_no_cue_specific.png")
    fig_contract(rd, args.models, fd / "05_contract_replication.png")
    fig_nuisance(rd, args.models, fd / "06_nuisance_axis.png")
    fig_controls(rd, args.models, fd / "07_controls.png")
    print(f"figures written to {fd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
