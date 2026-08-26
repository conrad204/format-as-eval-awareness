"""Scale figure: format pull on the purpose direction vs model size.

Reads only checked-in results:
  results/nocue_sweep_full320-v3_*.json      (no-cue stratum, 3,840 items, 4 models)
  results/devbunova_decomp/decomp_*.json     (Devbunova minimal pairs, 2,656, 3 models)

Statistic (ours): decorrelated-probe score shift, no-cue benchmark minus
no-cue casual, in sd units of the stated-stratum OOF scores, with block
cluster-bootstrap 95% CI (nocue_sweep.py). Purpose is held fixed (no cue),
so any shift is format alone moving the item along the purpose direction.
Statistic (hers): P4 scaffold share -- mean Delta(asis - stripped) of the
purpose_decorr projection on her bench-cell minimal pairs, sd units.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "plots" / "scale law 8-25-2026"
OUT.mkdir(parents=True, exist_ok=True)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S71="#8a5cd6"; S70="#eb6834"; S405="#d6336c"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

MODELS = [  # (nocue json, decomp key or None, short, params (B), color)
    ("nocue_sweep_full320-v3_llama31_8b.json",        "8b",   "8B",      8,   S8B),
    ("nocue_sweep_full320-v3_llama31_70b.json",       "70b",  "3.1-70B", 70,  S71),
    ("nocue_sweep_full320-v3_llama33_70b.json",       None,   "3.3-70B", 70,  S70),
    ("nocue_sweep_full320-v3_llama31_405b_ndif.json", "405b", "405B",    405, S405),
]
NC = [json.load(open(R / f)) for f, *_ in MODELS]
PEAK = [max(d, key=lambda r: r["decorrelated"]["auc"])["layer"] for d in NC]

def header(fig, title, sub, handles, ncol):
    fig.text(0.008,0.975,title,fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
    fig.text(0.008,0.935,sub,fontsize=9.5,color=INK2,va="top",ha="left",linespacing=1.4)
    if handles: fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(0.004,0.845),
                           ncol=ncol,frameon=False,fontsize=9.5,labelcolor=INK2,
                           handlelength=1.8,columnspacing=1.4)
def axchrome(ax, xlab, ylab):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)

fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 6.6), gridspec_kw={"width_ratios":[1.45,1]})

# --- A: no-cue format shift on the decorrelated purpose direction, by depth ---
for (f, dk, short, p, c), d, pk in zip(MODELS, NC, PEAK):
    n = len(d); x = np.array([r["layer"]/(n-1)*100 for r in d])
    y  = np.array([r["nocue"]["decorrelated"]["shift_sd"] for r in d])
    lo = np.array([r["nocue"]["decorrelated"]["shift_sd_ci95"][0] for r in d])
    hi = np.array([r["nocue"]["decorrelated"]["shift_sd_ci95"][1] for r in d])
    a.fill_between(x, lo, hi, color=c, alpha=.15, lw=0, zorder=2)
    a.plot(x, y, color=c, lw=2, zorder=3)
    a.plot([pk/(n-1)*100], [y[pk]], "o", color=c, ms=7, mec=SURF, mew=1.2, zorder=4)
a.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)
axchrome(a, "depth (% of layers)", "no-cue shift along purpose direction (sd, benchmark − casual)")
a.set_title("A  Our no-cue stratum (3,840 items, purpose unstated)", loc="left", fontsize=10.5, color=INK2)

# --- B: peak-layer value vs parameter count, both datasets ---
ours = [(p, d[pk]["nocue"]["decorrelated"]["shift_sd"], d[pk]["nocue"]["decorrelated"]["shift_sd_ci95"], c, short)
        for (f, dk, short, p, c), d, pk in zip(MODELS, NC, PEAK)]
hers = []
for f, dk, short, p, c in MODELS:
    if dk is None: continue
    e = json.load(open(R / "devbunova_decomp" / f"decomp_{dk}.json"))
    L = list(e["endpoints"]["template"].keys())[0]
    v = e["endpoints"]["template"][L]["P4_scaffold"]["delta_asis_minus_stripped_sd"]["purpose_decorr"]
    hers.append((p, v["mean_sd"], v["ci95"], c, short))
# jitter the two 70Bs apart on the log axis
def xj(p, short): return p * (0.93 if short=="3.1-70B" else 1.07 if short=="3.3-70B" else 1)
for p, y, ci, c, s in ours:
    b.errorbar(xj(p,s), y, yerr=[[y-ci[0]],[ci[1]-y]], fmt="o", color=c, ms=8, mec=SURF, mew=1.2, capsize=3, lw=1.4, zorder=4)
for p, y, ci, c, s in hers:
    b.errorbar(xj(p,s), y, yerr=[[y-ci[0]],[ci[1]-y]], fmt="s", color=c, ms=8, mfc=SURF, mew=1.8, capsize=3, lw=1.4, zorder=4)
o31 = [(p,y) for p,y,_,_,s in ours if s!="3.3-70B"]
b.plot([p for p,_ in o31],[y for _,y in o31], color=INK2, lw=1, alpha=.5, zorder=2)
b.plot([p for p,*_ in hers],[y for _,y,*_ in hers], color=INK2, lw=1, ls=(0,(3,2)), alpha=.5, zorder=2)
b.axhline(0, color=BASE, lw=1.2, ls=":", zorder=1)
b.set_xscale("log"); b.set_xticks([8,70,405]); b.set_xticklabels(["8B","70B","405B"]); b.minorticks_off()
axchrome(b, "parameters (Llama-3.x-Instruct)", "shift along purpose direction (sd)")
b.set_title("B  At each model's peak purpose layer", loc="left", fontsize=10.5, color=INK2)
b.set_xlim(5, 650)

handles = [Line2D([],[],color=c,lw=2,label=s) for *_, s, p, c in [(m[2],m[3],m[4]) for m in MODELS]]
handles = [Line2D([],[],color=m[4],lw=2,label=m[2]) for m in MODELS] + [
    Line2D([],[],marker="o",color=INK2,lw=0,ms=7,label="ours: no-cue benchmark − casual"),
    Line2D([],[],marker="s",color=INK2,mfc=SURF,mew=1.8,lw=0,ms=7,label="Devbunova: scaffold on − off (minimal pairs)")]
header(fig, "Format pull on the purpose direction vs model size (no scaling law: layer-fragile, non-monotone)",
       "Decorrelated purpose probe (format balanced in training), projected onto items whose purpose is never stated. "
       "\nsd = stated-stratum OOF score sd; 95% cluster-bootstrap CIs. Dots = peak decorrelated-AUC layer "
       f"(8B L{PEAK[0]}, 70B L{PEAK[1]}, 405B L{PEAK[3]})\nThe 8B's peak layer sits in a local trough; at most depths its shift is +0.2 to +0.5 sd.", handles, 6)
fig.tight_layout(rect=(0,0,1,0.78)); fig.savefig(OUT / "09_format_pull_by_scale.png", dpi=200); plt.close(fig)

# summary table for the writeup
print(f"{'model':9s} {'ours nocue (peak)':>22s} {'hers scaffold Δ':>22s}")
hd = {s:(y,ci) for _,y,ci,_,s in hers}
for p,y,ci,c,s in ours:
    h = hd.get(s)
    print(f"{s:9s} {y:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]   " + (f"{h[0]:+.2f} [{h[1][0]:+.2f},{h[1][1]:+.2f}]" if h else "—"))
