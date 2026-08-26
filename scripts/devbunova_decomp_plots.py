"""Plots for the cross-dataset decomposition of Devbunova's eval-awareness
signal (results/devbunova_decomp/decomp_{8b,70b,405b}.json; design frozen in
docs/devbunova_decomposition_plan.md).

All numbers are the template-regime endpoints at each model's frozen primary
layer (8B L8, 70B L18, 405B L37), pilot (local-fp-jitter) run. Style matches
scripts/full320_plots*.py. Probe colors are fixed across figures: her probe
violet, our confounded probe orange, our decorrelated probe green (validated
categorical trio); model colors are the house 3.1-family set with marker +
direct-label secondary encoding wherever they identify a series.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "devbunova decomp 8-24-2026"
OUT.mkdir(parents=True, exist_ok=True)
OUT = str(OUT)

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
S8B = "#2a78d6"; S71 = "#8a5cd6"; S405 = "#d6336c"
P_HER = "#8a5cd6"; P_CONF = "#eb6834"; P_DEC = "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": BASE, "font.size": 10})

MODELS = [("8b", "Llama-3.1-8B", S8B, "o"), ("70b", "Llama-3.1-70B", S71, "s"),
          ("405b", "Llama-3.1-405B", S405, "^")]
SHORT = ["8B", "3.1-70B", "405B"]
PROBES = [("her_protocol", "Devbunova protocol", P_HER),
          ("purpose_conf", "ours — confounded", P_CONF),
          ("purpose_decorr", "ours — decorrelated", P_DEC)]

D, LAY = [], []
for key, _, _, _ in MODELS:
    d = json.load(open(ROOT / f"results/devbunova_decomp/decomp_{key}.json"))
    D.append(d["endpoints"]["template"][f"L{d['layers']['primary']}"])
    LAY.append(d["layers"]["primary"])
SUB_TAIL = (f"template regime, frozen layers L{LAY[0]}/L{LAY[1]}/L{LAY[2]}; "
            "95% item-bootstrap CIs; pilot run")


def header(fig, title, sub, handles, ncol):
    fig.text(0.008, 0.975, title, fontsize=13, fontweight="bold", color=INK,
             va="top", ha="left")
    fig.text(0.008, 0.917, sub, fontsize=9.5, color=INK2, va="top", ha="left")
    if handles:
        fig.legend(handles=handles, loc="upper right",
                   bbox_to_anchor=(0.995, 0.995), ncol=ncol, frameon=False,
                   fontsize=9.5, labelcolor=INK2, handlelength=1.8,
                   columnspacing=1.4)


def axchrome(ax, xlab, ylab):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- 01: the format trap under each probe -------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(3)
w = 0.24
for j, (pk, plab, pc) in enumerate(PROBES):
    v = [D[i]["P1_trap"][pk]["bench_deploy_asis"] for i in range(3)]
    pos = x + (j - 1) * (w + 0.03)
    ax.bar(pos, [e["pct_eval"] for e in v], w, color=pc, zorder=3)
    ax.errorbar(pos, [e["pct_eval"] for e in v],
                yerr=[[e["pct_eval"] - e["ci95"][0] for e in v],
                      [e["ci95"][1] - e["pct_eval"] for e in v]],
                fmt="none", ecolor=INK2, lw=1.1, capsize=2.5, zorder=4)
    for p, e in zip(pos, v):
        ax.text(p, e["pct_eval"] + 3.5, f'{e["pct_eval"]:.0f}', ha="center",
                fontsize=9, color=INK2)
ax.axhline(94.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
ax.set_xlim(-0.85, 2.6)
ax.text(-0.82, 96.2, "94.5% — her published\nfalse-positive rate", fontsize=8.2,
        color=MUTED, ha="left", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(SHORT)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
axchrome(ax, "", "bench-deploy items called “eval” (%)")
header(fig, "The format trap follows the training design",
       "% of Devbunova's benchmark-formatted deployment items (n=1,670) scored\n"
       "as “evaluation” — " + SUB_TAIL,
       [Patch(color=c, label=l) for _, l, c in PROBES], 1)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.savefig(f"{OUT}/01_format_trap.png", dpi=200); plt.close(fig)

# ---- 02: what her probe's score is made of ------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.4))
fmt_r2 = [D[i]["P3_format_attribution"]["ols_her_score_on_projections"]
          ["partial_r2"]["format"] for i in range(3)]
dec_r2 = [D[i]["P3_format_attribution"]["ols_her_score_on_projections"]
          ["partial_r2"]["purpose_decorr"] for i in range(3)]
full_r2 = [D[i]["P3_format_attribution"]["ols_her_score_on_projections"]
           ["r2_full"] for i in range(3)]
w = 0.3
ax.bar(x - w / 2 - 0.02, fmt_r2, w, color=P_CONF, zorder=3)
ax.bar(x + w / 2 + 0.02, dec_r2, w, color=P_DEC, zorder=3)
for i in range(3):
    ax.text(x[i] - w / 2 - 0.02, fmt_r2[i] + 0.02, f"{fmt_r2[i]:.2f}",
            ha="center", fontsize=9, color=INK2)
    ax.text(x[i] + w / 2 + 0.02, dec_r2[i] + 0.02, f"{dec_r2[i]:.2f}",
            ha="center", fontsize=9, color=INK2)
    ax.plot([x[i] - 0.42, x[i] + 0.42], [full_r2[i]] * 2, color=INK2, lw=1.1,
            ls=(0, (4, 3)), zorder=4)
ax.set_xticks(x); ax.set_xticklabels(SHORT)
ax.set_ylim(0, 1.0)
axchrome(ax, "", "partial R² of her probe's decision score")
header(fig, "Her probe's score is mostly the format direction",
       "OLS of her probe's per-item score on our two directions; dashes mark total R²\n"
       f"({full_r2[0]:.2f} / {full_r2[1]:.2f} / {full_r2[2]:.2f}) — " + SUB_TAIL,
       [Patch(color=P_CONF, label="format direction"),
        Patch(color=P_DEC, label="purpose direction (decorrelated)")], 1)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.savefig(f"{OUT}/02_signal_decomposition.png", dpi=200); plt.close(fig)

# ---- 03: the scaffold stops mattering at 70B ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for i, (key, name, c, m) in enumerate(MODELS):
    fp = D[i]["P4_scaffold"]["her_protocol_fp_bench_deploy"]
    a, s = fp["asis_changed_pairs"]["pct_eval"], fp["stripped_changed_pairs"]["pct_eval"]
    ax.plot([0, 1], [a, s], color=c, lw=2, marker=m, ms=7,
            markerfacecolor=c, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
    dy = {"8b": 0, "70b": 2.4, "405b": -3.4}[key]
    ax.text(1.04, s + dy, f"{name}  {s:.0f}%", fontsize=9.5, color=INK2,
            va="center")
ax.set_xlim(-0.35, 1.85); ax.set_ylim(-6, 110)
ax.set_xticks([0, 1])
ax.set_xticklabels(["with “Answer:” scaffold", "scaffold stripped"])
ax.set_yticks([0, 25, 50, 75, 100])
axchrome(ax, "", "her-protocol probe: bench-deploy called “eval” (%)")
header(fig, "At 70B+, the trap no longer lives in the scaffold token",
       "Her probe on minimal pairs differing only in the trailing “Answer:/Solution:”\n"
       "scaffold (1,580 bench-deploy pairs) — template regime,\n"
       f"frozen layers L{LAY[0]}/L{LAY[1]}/L{LAY[2]}; pilot run", None, 1)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(f"{OUT}/03_scaffold_slope.png", dpi=200); plt.close(fig)

# ---- 04: within-format sensitivity stays near chance --------------------
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=True)
for ax, ck, title in [
        (axes[0], "within_bench_eval_vs_deploy",
         "within benchmark format\n(bench-eval vs bench-deploy)"),
        (axes[1], "within_casual_eval_vs_deploy",
         "within casual format\n(casual-eval vs casual-deploy)")]:
    for j, (pk, _, pc) in enumerate(PROBES):
        v = [D[i]["P2_auc"][pk][ck] for i in range(3)]
        pos = x + (j - 1) * 0.22
        ax.errorbar(pos, [e["auc"] for e in v],
                    yerr=[[e["auc"] - e["ci95"][0] for e in v],
                          [e["ci95"][1] - e["auc"] for e in v]],
                    fmt="none", ecolor=pc, lw=1.4, capsize=2.5, zorder=3)
        ax.plot(pos, [e["auc"] for e in v], ls="none", marker="o", ms=7,
                color=pc, markeredgecolor=SURF, markeredgewidth=1.1, zorder=4)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(SHORT)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9.5, color=INK2)
    axchrome(ax, "", "AUC (eval vs deploy)" if ax is axes[0] else "")
axes[1].text(2.42, 0.52, "chance", fontsize=8.2, color=MUTED, ha="right")
header(fig, "Once format is held fixed, no probe finds much purpose signal",
       "Eval-vs-deploy AUC inside each of her format strata —\n" + SUB_TAIL,
       [Line2D([], [], ls="none", marker="o", color=c, label=l)
        for _, l, c in PROBES], 1)
fig.tight_layout(rect=[0, 0, 1, 0.82])
fig.savefig(f"{OUT}/04_within_format_auc.png", dpi=200); plt.close(fig)

print(f"wrote 4 plots to {OUT}")
