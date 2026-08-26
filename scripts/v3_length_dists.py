"""Prompt-length distributions for the v3 full320 freeze (with cue).

Reads the frozen rendered items directly (data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl,
11,520 rows) -- not the sweep JSONs -- and plots the length of each item's
`text` field exactly as it goes into extraction, i.e. cue sentence included.

Two figures, each rendered in both units:
  01  by intended_purpose   evaluation / deployment / none (3,840 each)
  02  by template_id        13 wrapper templates, hue-coded benchmark vs casual

Lengths are measured two ways: Llama-3.1 tokens (meta-llama/Llama-3.1-8B-Instruct
tokenizer, add_special_tokens=False, no chat template -- the template adds the
same fixed prefix to every item, so it shifts all curves identically) and raw
characters.  Style block matches scripts/full320_plots.py.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "full320-v3" / "rendered_items_v3_full320_8-17-2026.jsonl"
OUT = ROOT / "plots" / "v3-8-19-2026-data-analysis-w-cue"
OUT.mkdir(parents=True, exist_ok=True)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S71="#8a5cd6"; S70="#eb6834"; S405="#d6336c"; S3="#1baf7a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

def header(fig,title,sub,handles,ncol,y=0.975,ys=0.917):
    fig.text(0.008,y,title,fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
    fig.text(0.008,ys,sub,fontsize=9.5,color=INK2,va="top",ha="left")
    if handles: fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(0.995,0.995),
                           ncol=ncol,frameon=False,fontsize=9.5,labelcolor=INK2,
                           handlelength=1.8,columnspacing=1.4)
def axchrome(ax,xlab,ylab):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)

# ---------------------------------------------------------------- load + measure
rows = [json.loads(l) for l in open(ITEMS)]
texts = [r["text"] for r in rows]
chars = np.array([len(t) for t in texts])
from transformers import AutoTokenizer          # cached locally; no network needed
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
toks = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])

purpose = np.array([r["intended_purpose"] for r in rows])
template = np.array([r["template_id"] for r in rows])
fmt = np.array([r["format"] for r in rows])

UNITS = [("tokens", toks, "prompt length (Llama-3.1 tokens)"),
         ("chars",  chars, "prompt length (characters)")]

def stats(v): return dict(n=len(v), med=float(np.median(v)), mean=float(v.mean()),
                          p5=float(np.percentile(v,5)), p95=float(np.percentile(v,95)))

# ---------------------------------------------------------------- 01 by purpose
PURPOSES = [("evaluation", S8B, "evaluation cue"),
            ("deployment", S70, "deployment cue"),
            ("none",       MUTED, "no cue")]

for uname, vals, xlab in UNITS:
    bins = np.linspace(vals.min(), vals.max(), 61)
    fig, ax = plt.subplots(figsize=(9.4,5.8))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.10)
    handles = []
    for p, c, lab in PURPOSES:
        v = vals[purpose == p]; s = stats(v)
        ax.hist(v, bins=bins, color=c, alpha=0.38, zorder=3)
        ax.hist(v, bins=bins, histtype="step", color=c, lw=1.6, alpha=0.95, zorder=4)
        ax.axvline(s["med"], color=c, lw=1.4, ls=(0,(4,3)), alpha=0.9, zorder=5)
        handles.append(Patch(facecolor=c, alpha=0.55, edgecolor=c, lw=1.4,
                             label=f"{lab}  ·  n={s['n']:,}  ·  median {s['med']:.0f}"))
    axchrome(ax, xlab, "items")
    ax.set_xlim(vals.min(), np.percentile(vals, 99.8))
    ymax = ax.get_ylim()[1]
    # the three medians sit within a few units of each other -- stagger vertically
    for (p, c, _), fr in zip(PURPOSES, (0.985, 0.930, 0.875)):
        m = float(np.median(vals[purpose == p]))
        ax.annotate(f"{m:.0f}", (m, ymax*fr), color=c, fontsize=9, fontweight="bold",
                    ha="center", va="top", zorder=6,
                    bbox=dict(fc=SURF, ec="none", pad=1.2))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5,
              labelcolor=INK2, handlelength=1.8, borderaxespad=0.6)
    med_cue = np.median(vals[purpose != "none"]); med_no = np.median(vals[purpose == "none"])
    header(fig, "Prompt-length distribution by intended purpose",
           f"full320-v3 frozen renders, cue included · 11,520 items, 3,840 per arm · "
           f"the cue sentence costs a median {med_cue-med_no:+.0f} {uname} vs the no-cue arm",
           None, 1)
    fig.savefig(OUT / f"01_length_by_purpose_{uname}.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- 02 by template
BENCH = sorted({t for t, f in zip(template, fmt) if f == "benchmark"})
CASUAL = sorted({t for t, f in zip(template, fmt) if f == "casual"})
def shades(cmap, k): return [plt.get_cmap(cmap)(x) for x in np.linspace(0.42, 0.92, k)]
COLOR = dict(zip(BENCH, shades("Blues", len(BENCH))))
COLOR.update(dict(zip(CASUAL, shades("Oranges", len(CASUAL)))))
FAMCOL = {"benchmark": S8B, "casual": S70}
FAM = {t: "benchmark" for t in BENCH}; FAM.update({t: "casual" for t in CASUAL})

for uname, vals, xlab in UNITS:
    bins = np.linspace(vals.min(), vals.max(), 61)
    fig = plt.figure(figsize=(10.6,7.8))
    # separate margins: only the lower panel needs room for the template names
    ax  = fig.add_axes([0.075, 0.500, 0.910, 0.335])
    ax2 = fig.add_axes([0.175, 0.075, 0.810, 0.285])

    order = BENCH + CASUAL
    for t in order:
        v = vals[template == t]; c = COLOR[t]
        ax.hist(v, bins=bins, color=c, alpha=0.16, zorder=3)
        ax.hist(v, bins=bins, histtype="step", color=c, lw=1.3, alpha=0.95, zorder=4)
    axchrome(ax, xlab, "items")
    ax.set_xlim(vals.min(), np.percentile(vals, 99.8))
    ax.set_title("all 13 templates overlaid", fontsize=10, color=INK2, loc="left", pad=6)

    # readable companion: one row per template, sorted by median, family-colored
    srt = sorted(order, key=lambda t: np.median(vals[template == t]))
    for i, t in enumerate(srt):
        v = vals[template == t]; c = COLOR[t]
        q05, q25, med, q75, q95 = np.percentile(v, [5,25,50,75,95])
        ax2.plot([q05,q95],[i,i], color=c, lw=1.6, alpha=0.75, solid_capstyle="round", zorder=3)
        ax2.plot([q25,q75],[i,i], color=c, lw=7.5, alpha=0.55, solid_capstyle="butt", zorder=4)
        ax2.plot([med],[i], "|", ms=13, mew=2.2, color=c, zorder=5)
        ax2.annotate(f"{med:.0f}", (q95, i), textcoords="offset points", xytext=(6,0),
                     color=c, fontsize=8.5, va="center", ha="left", zorder=6)
    ax2.set_yticks(range(len(srt)))
    ax2.set_yticklabels(srt, fontsize=8.6)
    for lbl, t in zip(ax2.get_yticklabels(), srt):
        lbl.set_color(FAMCOL[FAM[t]])
    ax2.set_ylim(-0.7, len(srt)-0.3)
    axchrome(ax2, xlab, "")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_title("per template: median tick, interquartile bar, 5–95% whisker  "
                  "(label colour = format family)", fontsize=10, color=INK2, loc="left", pad=6)

    handles = [Patch(facecolor=S8B, alpha=0.55, edgecolor=S8B, lw=1.4,
                     label=f"benchmark family — {len(BENCH)} templates"),
               Patch(facecolor=S70, alpha=0.55, edgecolor=S70, lw=1.4,
                     label=f"casual family — {len(CASUAL)} templates")]
    header(fig, "Prompt-length distribution across all 13 wrapper templates",
           "full320-v3 frozen renders, cue included · 960 items per v2 template, 480 per legacy v1 template · "
           "shade within a family darkens with template order",
           handles, 1)
    fig.savefig(OUT / f"02_length_by_template_{uname}.png", dpi=200)
    plt.close(fig)

# ------------------------------------------------- 03 format x purpose contrasts
# The freeze is a clean 2x3 (format x intended_purpose); the three requested
# contrasts are all 2-cell slices of it.  Cells are unbalanced by construction --
# benchmark spans 7 templates (2,080 items) and casual 6 (1,760) -- but all 320
# payload blocks appear in every cell, so a block-paired difference is also
# available and is reported alongside the unpaired one.
from scipy.stats import mannwhitneyu, ks_2samp

CELL = {("benchmark","evaluation"): ("benchmark · evaluation", "#2a78d6"),
        ("benchmark","deployment"): ("benchmark · deployment", "#8fbdec"),
        ("casual",   "evaluation"): ("casual · evaluation",    "#eb6834"),
        ("casual",   "deployment"): ("casual · deployment",    "#f6b08c")}
CONTRASTS = [
    (("benchmark","evaluation"), ("casual","deployment"),
     "A · the confounded diagonal — format and purpose move together"),
    (("benchmark","evaluation"), ("casual","evaluation"),
     "B · format effect with purpose held at evaluation"),
    (("benchmark","deployment"), ("casual","deployment"),
     "C · format effect with purpose held at deployment"),
]

block = np.array([r["payload_block_id"] for r in rows])
def cell_mask(k): return (fmt == k[0]) & (purpose == k[1])
def paired_delta(vals, ka, kb):
    """median over payload blocks of (mean length in cell A) - (mean length in cell B)."""
    a, b = {}, {}
    for m, d in ((cell_mask(ka), a), (cell_mask(kb), b)):
        for bl, v in zip(block[m], vals[m]): d.setdefault(bl, []).append(v)
    common = sorted(set(a) & set(b))
    return float(np.median([np.mean(a[k]) - np.mean(b[k]) for k in common])), len(common)

contrast_stats = {}
for uname, vals, xlab in UNITS:
    bins = np.linspace(vals.min(), vals.max(), 61)
    fig = plt.figure(figsize=(9.8, 9.4))
    axes = [fig.add_axes([0.075, 0.635 - i*0.283, 0.910, 0.212]) for i in range(3)]
    rowstats = []
    for ax, (ka, kb, cap) in zip(axes, CONTRASTS):
        handles = []
        for k in (ka, kb):
            v = vals[cell_mask(k)]; lab, c = CELL[k]
            ax.hist(v, bins=bins, color=c, alpha=0.42, zorder=3)
            ax.hist(v, bins=bins, histtype="step", color=c, lw=1.6, alpha=0.95, zorder=4)
            ax.axvline(np.median(v), color=c, lw=1.4, ls=(0,(4,3)), alpha=0.9, zorder=5)
            handles.append(Patch(facecolor=c, alpha=0.6, edgecolor=c, lw=1.4,
                                 label=f"{lab}  ·  n={len(v):,}  ·  median {np.median(v):.0f}"))
        va, vb = vals[cell_mask(ka)], vals[cell_mask(kb)]
        u = mannwhitneyu(va, vb, alternative="two-sided")
        auc = u.statistic / (len(va) * len(vb))          # P(random A longer than random B)
        ks = ks_2samp(va, vb)
        pd_, nblk = paired_delta(vals, ka, kb)
        rowstats.append((cap, ka, kb, len(va), len(vb), float(np.median(va)), float(np.median(vb)),
                         float(np.median(va)-np.median(vb)), pd_, nblk, auc, float(ks.statistic), float(u.pvalue)))
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
                  labelcolor=INK2, handlelength=1.8, borderaxespad=0.4)
        ax.set_title(cap, fontsize=10, color=INK2, loc="left", pad=6)
        axchrome(ax, xlab if ax is axes[-1] else "", "items")
        ax.set_xlim(vals.min(), np.percentile(vals, 99.8))
        if ax is not axes[-1]: ax.set_xticklabels([])
        pstr = "p < 1e-4" if u.pvalue < 1e-4 else f"p = {u.pvalue:.3g}"
        ax.text(0.995, 0.055,
                f"median gap {np.median(va)-np.median(vb):+.0f} {uname}   ·   "
                f"block-paired gap {pd_:+.1f} (n={nblk})   ·   "
                f"P(A longer) {auc:.3f}   ·   KS {ks.statistic:.3f}   ·   {pstr}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, color=INK2,
                bbox=dict(fc=SURF, ec=BASE, lw=0.8, pad=3.2), zorder=7)
    contrast_stats[uname] = rowstats
    header(fig, "Prompt length across the format x purpose cells",
           "full320-v3 frozen renders, cue included · benchmark = 7 templates (n=2,080), casual = 6 (n=1,760) "
           "· every cell holds all 320 payload blocks",
           None, 1, y=0.982, ys=0.940)
    fig.savefig(OUT / f"03_format_x_purpose_{uname}.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- console summary
print(f"wrote 6 PNGs to {OUT}")
for uname, vals, _ in UNITS:
    print(f"\n[{uname}]")
    for p, _, lab in PURPOSES:
        s = stats(vals[purpose == p])
        print(f"  {lab:<16} n={s['n']:>6,}  median {s['med']:>7.1f}  mean {s['mean']:>7.1f}  "
              f"p5-p95 {s['p5']:.0f}-{s['p95']:.0f}")
    for t in BENCH + CASUAL:
        s = stats(vals[template == t])
        print(f"  {FAM[t][:5]:<5} {t:<24} n={s['n']:>5,}  median {s['med']:>7.1f}  "
              f"p5-p95 {s['p5']:.0f}-{s['p95']:.0f}")
    for (cap, ka, kb, na, nb, ma, mb, gap, pd_, nblk, auc, ksv, pv) in contrast_stats[uname]:
        print(f"  {cap}")
        print(f"      A {ka[0]}/{ka[1]:<10} n={na:>5,} median {ma:>7.1f}   "
              f"B {kb[0]}/{kb[1]:<10} n={nb:>5,} median {mb:>7.1f}")
        print(f"      median gap {gap:+.1f}   block-paired gap {pd_:+.2f} (n={nblk})   "
              f"P(A longer) {auc:.3f}   KS {ksv:.3f}   p {pv:.3g}")
