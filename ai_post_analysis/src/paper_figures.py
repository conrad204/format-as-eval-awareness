"""Paper variants of the three figures used in the workshop submission.

Redraws from the same `results/*.json` artifacts as `make_figures.py`, with
three differences that matter for print:

* no baked-in suptitle -- the claim belongs in the LaTeX caption, and one of
  the README suptitles states a serialization claim the paper walks back;
* smaller type and tighter panels, sized for a single text column;
* PDF output, so the figures stay vector in the compiled paper.

This is a separate script rather than a flag on `make_figures.py` or
`geometry_alignment.py` on purpose: those files' sha256 is recorded inside the
result artifacts they produced, and editing them would invalidate that record.
Nothing here recomputes an estimand; every value is read from disk.
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
AUC_FLOOR = 0.95
MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")
MODEL_LABEL = {
    "llama31_8b": "Llama 3.1 8B",
    "llama31_70b": "Llama 3.1 70B",
    "llama33_70b": "Llama 3.3 70B",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
})


def load(rd: Path, model: str, kind: str):
    p = rd / f"{model}_{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _top_legend(fig, ax, ncol: int) -> None:
    """One horizontal legend placed entirely above the panel grid.

    Anchored to the top edge of the figure with loc="lower center", so it sits
    clear of both the data and the panel titles; bbox_inches="tight" at save
    time grows the canvas to include it. Call after tight_layout.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, frameon=False, handlelength=1.5, columnspacing=1.6,
               borderpad=0.0, handletextpad=0.5)


def shade_unreliable(ax, q, auc):
    bad = auc < AUC_FLOOR
    if not bad.any():
        return
    edges = np.flatnonzero(np.diff(bad.astype(int)) != 0) + 1
    for s in np.split(np.arange(len(bad)), edges):
        if len(s) and bad[s[0]]:
            ax.axvspan(q[s[0]], q[s[-1]], color=C_MUTED, alpha=0.13, lw=0, zorder=0)


def fig_depth(rd: Path, out: Path) -> None:
    have = [(m, load(rd, m, "main"), load(rd, m, "templates")) for m in MODELS]
    have = [(m, a, b) for m, a, b in have if a and b]
    fig, axes = plt.subplots(1, len(have), figsize=(2.35 * len(have), 2.05),
                             squeeze=False, sharey=True)

    for ax, (model, main, tpl) in zip(axes[0], have):
        n = main["header"]["n_transformer_layers"]
        rows = main["layers"]
        q = np.array([r["layer"] for r in rows]) / (n - 1)
        auc = np.array([r["purpose_auc_oof"] for r in rows])
        rho = np.array([r["point"]["rho"] for r in rows])
        rlo = np.array([r["ci95"]["rho"]["ci95"][0] for r in rows])
        rhi = np.array([r["ci95"]["rho"]["ci95"][1] for r in rows])
        trows = tpl["layers"]
        mk = np.array([r["markup_vs_plain"]["ratio_stated_units"] for r in trows])
        mlo = np.array([r["markup_vs_plain"]["ratio_ci95"][0] for r in trows])
        mhi = np.array([r["markup_vs_plain"]["ratio_ci95"][1] for r in trows])

        ax.set_ylim(-0.30, 0.20)
        shade_unreliable(ax, q, auc)
        ax.axhline(0, color="0.35", lw=0.8, zorder=1)
        ax.fill_between(q, rlo, rhi, color=C_FAMILY, alpha=0.20, lw=0)
        ax.plot(q, rho, color=C_FAMILY, lw=1.5, label="benchmark $-$ casual", zorder=3)
        ax.fill_between(q[: len(mk)], mlo, mhi, color=C_MARKUP, alpha=0.20, lw=0)
        ax.plot(q[: len(mk)], mk, color=C_MARKUP, lw=1.5, label="markup $-$ plain", zorder=3)

        peak = int(np.argmax(auc))
        ax.axvline(q[peak], color=C_PURPOSE, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(q[peak] + 0.02, -0.285, f"L{rows[peak]['layer']}", fontsize=6.5,
                color=C_PURPOSE, va="bottom")

        ax.set_title(MODEL_LABEL.get(model, model), fontsize=8.5, pad=4)
        ax.set_xlabel("normalized depth", labelpad=2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.15, lw=0.5)

    axes[0][0].set_ylabel("no-cue shift\n(stated-cue units)", fontsize=7.5)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    _top_legend(fig, axes[0][0], ncol=2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def fig_differential(rd: Path, out: Path) -> None:
    have = [(m, load(rd, m, "offsets"), load(rd, m, "main")) for m in MODELS]
    have = [(m, o, mn) for m, o, mn in have if o and mn]
    fig, axes = plt.subplots(1, len(have), figsize=(2.35 * len(have), 2.05),
                             squeeze=False, sharey=True)

    for ax, (model, off, main) in zip(axes[0], have):
        n = main["header"]["n_transformer_layers"]
        rows = off["layers"]
        q = np.array([r["layer"] for r in rows]) / (n - 1)
        auc = np.array([r["purpose_auc_oof"] for r in main["layers"]])
        g = "markup_minus_plain"
        cued = np.array([r[g]["cued"]["point"] for r in rows])
        none = np.array([r[g]["no_cue"]["point"] for r in rows])
        diff = np.array([r[g]["differential"]["point"] for r in rows])
        dlo = np.array([r[g]["differential"]["ci95"][0] for r in rows])
        dhi = np.array([r[g]["differential"]["ci95"][1] for r in rows])

        ax.set_ylim(-0.40, 0.20)
        shade_unreliable(ax, q, auc[: len(q)])
        ax.axhline(0, color="0.35", lw=0.8, zorder=1)
        ax.plot(q, cued, color=C_PURPOSE, lw=1.5, label="on cued rows", zorder=3)
        ax.plot(q, none, color=C_MARKUP, lw=1.5, label="on no-cue rows", zorder=3)
        ax.fill_between(q, dlo, dhi, color="0.35", alpha=0.18, lw=0)
        ax.plot(q, diff, color="0.25", lw=1.1, ls=(0, (3, 2)),
                label="differential", zorder=3)

        ax.set_title(MODEL_LABEL.get(model, model), fontsize=8.5, pad=4)
        ax.set_xlabel("normalized depth", labelpad=2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.15, lw=0.5)

    axes[0][0].set_ylabel("markup $-$ plain\n(stated-cue units)", fontsize=7.5)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    _top_legend(fig, axes[0][0], ncol=3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def fig_geometry(rd: Path, out: Path) -> None:
    results = {}
    for m in MODELS:
        g = load(rd, m, "geometry")
        if g:
            results[m] = g
    colors = {"benchmark_minus_casual": "#1b8a6b", "markup_minus_plain": "#d95f32"}
    labels = {"benchmark_minus_casual": "benchmark $-$ casual",
              "markup_minus_plain": "markup $-$ plain"}

    fig, axes = plt.subplots(2, len(results), figsize=(2.35 * len(results), 3.3),
                             sharex="col", squeeze=False)
    for col, (model, res) in enumerate(results.items()):
        rows = res["layers"]
        q = np.array([r["normalized_depth"] for r in rows])
        stability = np.array([
            r["standardized"]["benchmark_minus_casual"]["crossfit_projection_ratio"]
            ["mean_cos_excluded_vs_held_purpose"] for r in rows])

        for name, color in colors.items():
            raw = [r["raw"][name]["cosine"] for r in rows]
            pt = np.array([x["point"] for x in raw])
            lo = np.array([x["ci95_normal"][0] for x in raw])
            hi = np.array([x["ci95_normal"][1] for x in raw])
            axes[0, col].fill_between(q, lo, hi, color=color, alpha=0.16, lw=0)
            axes[0, col].plot(q, pt, color=color, lw=1.5, label=labels[name])

            proj = [r["standardized"][name]["crossfit_projection_ratio"] for r in rows]
            pp = np.array([x["point"] for x in proj])
            plo = np.array([x["ci95_block_bootstrap_conditional"][0] for x in proj])
            phi = np.array([x["ci95_block_bootstrap_conditional"][1] for x in proj])
            axes[1, col].fill_between(q, plo, phi, color=color, alpha=0.16, lw=0)
            axes[1, col].plot(q, pp, color=color, lw=1.5)

        for r in range(2):
            ax = axes[r, col]
            ax.axhline(0, color="0.35", lw=0.8)
            ax.grid(alpha=0.15, lw=0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            unstable = stability < 0.5
            if unstable.any():
                edges = np.flatnonzero(np.diff(unstable.astype(np.int8)) != 0) + 1
                for span in np.split(np.arange(len(q)), edges):
                    if len(span) and unstable[span[0]]:
                        ax.axvspan(q[span[0]], q[span[-1]], color=C_MUTED,
                                   alpha=0.13, lw=0, zorder=0)
        axes[0, col].set_title(MODEL_LABEL.get(model, model), fontsize=8.5, pad=4)
        axes[1, col].set_xlabel("normalized depth", labelpad=2)

    axes[0, 0].set_ylabel("raw displacement\ncosine", fontsize=7.5)
    axes[1, 0].set_ylabel("held-family projection /\nexplicit-purpose magnitude",
                          fontsize=7.5)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.5)
    _top_legend(fig, axes[0, 0], ncol=2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--out-dir", default="ai_post_analysis/paper/figures")
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png"])
    args = ap.parse_args()

    rd, od = Path(args.results_dir), Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    e = args.ext
    fig_depth(rd, od / f"fig1_depth_profile.{e}")
    fig_differential(rd, od / f"fig2_no_cue_specific.{e}")
    fig_geometry(rd, od / f"fig3_counterfactual_geometry.{e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
