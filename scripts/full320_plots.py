"""Plots for the v3 full320 freeze (320 blocks, 11,520 items, three models).

Counterparts of the v1/v2 plots 01-07/06a, computed from the v3 full-regime
sweeps (results/layer_sweep_full320-v3_*.json — same regimes and held-out
cue-vocabulary folds as v2, run by pilot80_sweep_full.py on the full320
stated stratum: 7,680 rows, 3,524 complete minimal pairs), plus a new plot
08 for the v3 surface-probe battery (results/surface_probes_v3_full320.json).

v2 -> v3 reading notes:
  - 4x the payload blocks (80 -> 320; every topic now 10 Grok + 10 Gemini),
    and the wrapper is no longer one template per format: 13 templates
    (7 benchmark, 6 casual), so "format" spans a template family.
  - Same three models and colors as v2 (8B blue, 3.1-70B violet,
    3.3-70B orange) so the plot sets are directly comparable, plus
    Llama-3.1-405B-Instruct (pink, extracted via NDIF/nnsight — see
    provenance/full320-v3/extraction_config_405b_ndif.json).
  - v2's endpoint plots 08-10 (generalization ladder, no-cue shift, cue
    locus) have no v3 counterpart yet: the v3 confirmation analysis has not
    been run. v3's stated stratum is all third-person; indirect/second-person
    strata do not exist in this freeze.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = str(ROOT / "results")
OUT = ROOT / "plots" / "full320 results 8-19-2026"
OUT.mkdir(parents=True, exist_ok=True)
OUT = str(OUT)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S71="#8a5cd6"; S70="#eb6834"; S405="#d6336c"; S3="#1baf7a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

MODELS = [  # (sweep json, short name, color)
    ("layer_sweep_full320-v3_llama31_8b.json",  "Llama-3.1-8B",  S8B),
    ("layer_sweep_full320-v3_llama31_70b.json", "Llama-3.1-70B", S71),
    ("layer_sweep_full320-v3_llama33_70b.json", "Llama-3.3-70B", S70),
    ("layer_sweep_full320-v3_llama31_405b_ndif.json", "Llama-3.1-405B", S405),
]
SHORT = ["8B", "3.1-70B", "3.3-70B", "405B"]
D = [json.load(open(f"{R}/{s}")) for s, _, _ in MODELS]
SURFACE = json.load(open(f"{R}/surface_probes_v3_full320.json"))
NAMES = [n for _, n, _ in MODELS]
COLS = [c for _, _, c in MODELS]
N_PAIRS = 3524  # complete (payload_block_id, purpose_pair_id, format) minimal pairs

def depth(d): n=len(d); return np.array([r["layer"]/(n-1)*100 for r in d])
def series(d,reg,key): return np.array([r[reg][key] for r in d])
X = [depth(d) for d in D]
PEAKS = [max(d,key=lambda r:r["decorrelated"]["auc"]) for d in D]

def header(fig,title,sub,handles,ncol):
    fig.text(0.008,0.975,title,fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
    fig.text(0.008,0.917,sub,fontsize=9.5,color=INK2,va="top",ha="left")
    if handles: fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(0.995,0.995),
                           ncol=ncol,frameon=False,fontsize=9.5,labelcolor=INK2,
                           handlelength=1.8,columnspacing=1.4)
def axchrome(ax,xlab,ylab):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)
def model_handles(lw=2):  # short labels: four full names overflow into the titles
    return [Line2D([],[],color=c,lw=lw,label=n) for n,c in zip(SHORT,COLS)]

# --- 01: purpose signal by depth (decorrelated solid, confounded dashed) ---
fig,ax=plt.subplots(figsize=(9.4,5.8))
for x,d,c in zip(X,D,COLS):
    ax.plot(x,series(d,"decorrelated","auc"),color=c,lw=2,zorder=3)
    ax.plot(x,series(d,"confounded","auc"),color=c,lw=2,ls=(0,(4,3)),alpha=.85,zorder=2)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
# all four peaks sit near 23-30% depth / AUC 0.99+, so stagger the labels
offsets=[(30,-58),(-40,14),(30,-32),(106,14)]
for p,name,c,x,(dx,dy) in zip(PEAKS,SHORT,COLS,X,offsets):
    xv=p["layer"]/(len(x)-1)*100; yv=p["decorrelated"]["auc"]
    ax.plot([xv],[yv],"o",ms=9,color=c,mec=SURF,mew=2,zorder=4)
    ax.annotate(f"{name} peak · L{p['layer']} · AUC {yv:.3f}",(xv,yv),textcoords="offset points",
                xytext=(dx,dy),color=c,fontsize=9.5,fontweight="bold",zorder=5)
axchrome(ax,"relative depth through the network (%)","purpose AUC (held-out vocabulary fold)")
ax.set_ylim(0.42,1.04)
header(fig,"Purpose decodes from unseen vocabulary",
       "Solid: decorrelated training, tested on a held-out cue-vocabulary fold.   Dashed: confounded training (purpose ⊗ format).",
       model_handles(),4)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/01_purpose_signal_by_depth.png",dpi=200); plt.close(fig)

# --- 02: confound signature at each model's peak layer ---
fig,axes=plt.subplots(1,4,figsize=(17.4,5.4),sharey=True)
cells=["evaluation::benchmark","evaluation::casual","deployment::benchmark","deployment::casual"]
lbl=["eval\nquiz","eval\ncasual","deploy\nquiz","deploy\ncasual"]
for ax,p,name,col in zip(axes,PEAKS,NAMES,COLS):
    i=np.arange(4); w=0.38
    cv=[p["confounded"]["cells"][c] for c in cells]; dv=[p["decorrelated"]["cells"][c] for c in cells]
    ax.bar(i-w/2-0.012,cv,w,color=MUTED,zorder=3)
    ax.bar(i+w/2+0.012,dv,w,color=col,zorder=3)
    for xi,v in zip(i-w/2-0.012,cv): ax.text(xi,v+2.5,f"{v:.0f}",ha="center",color=INK2,fontsize=9,fontweight="bold")
    for xi,v in zip(i+w/2+0.012,dv): ax.text(xi,v+2.5,f"{v:.0f}",ha="center",color=INK,fontsize=9,fontweight="bold")
    ax.set_xticks(i); ax.set_xticklabels(lbl,color=INK2,fontsize=9.5)
    ax.set_title(f"{name}  ·  layer {p['layer']}",color=INK,loc="left",fontsize=11,fontweight="bold",pad=10)
    ax.set_ylim(0,110); axchrome(ax,"",""); ax.grid(axis="x",visible=False)
axes[0].set_ylabel("% of prompts called “evaluation”")
header(fig,"Confounded training still answers by format; decorrelated answers by purpose","",
       [Line2D([],[],color=MUTED,lw=9,label="confounded training (only eval-quiz + deploy-casual)"),
        Line2D([],[],color=INK2,lw=9,label="decorrelated training (all four cells), model color")],1)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/02_confound_signature.png",dpi=200); plt.close(fig)

# --- 03: cross-format transfer by depth ---
fig,axes=plt.subplots(1,2,figsize=(11.4,5.2),sharey=True)
for axp,reg,title in ((axes[0],"b2c","trained on quiz  →  tested on casual"),
                      (axes[1],"c2b","trained on casual  →  tested on quiz")):
    for x,d,c in zip(X,D,COLS):
        axp.plot(x,series(d,reg,"auc"),color=c,lw=2,zorder=3)
    axp.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
    axp.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
    axp.set_title(title,color=INK,loc="left",fontsize=11,fontweight="bold",pad=10)
    axchrome(axp,"relative depth (%)",""); axp.set_ylim(0.42,1.0)
axes[0].set_ylabel("purpose AUC on the unseen format")
header(fig,"Transfer to an unseen format family (still on unseen vocabulary)","",model_handles(),4)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/03_transfer_by_depth.png",dpi=200); plt.close(fig)

# --- 04: ranking vs threshold at the peak layer ---
fig,ax=plt.subplots(figsize=(12.2,5.6))
groups=["decorrelated","quiz → casual","casual → quiz"]; regs=["decorrelated","b2c","c2b"]
i=np.arange(3); w=0.105
for k,(p,name,col) in enumerate(zip(PEAKS,SHORT,COLS)):
    au=[p[r]["auc"] for r in regs]; ba=[p[r]["bal_acc"] for r in regs]
    off=(k-1.5)*2.15*w
    ax.bar(i+off-w*0.52,au,w,color=col,zorder=3)
    ax.bar(i+off+w*0.52,ba,w,color=col,alpha=0.40,zorder=3)
    for xi,v in zip(i+off-w*0.52,au): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=7.5,color=INK,fontweight="bold")
    for xi,v in zip(i+off+w*0.52,ba): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=7.5,color=INK2)
    for g in range(3): ax.text(i[g]+off,-0.062,name,ha="center",color=INK2,fontsize=8,fontweight="bold")
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.set_xticks(i); ax.set_xticklabels(groups,color=INK,fontsize=11,fontweight="bold")
ax.tick_params(axis="x",pad=24,length=0); ax.set_ylim(0,1.06)
axchrome(ax,"","score"); ax.grid(axis="x",visible=False)
header(fig,"At full scale, ranking and threshold both survive the format change",
       "Solid bar: AUC (ranking quality).   Pale bar: balanced accuracy (fixed threshold at 0).   Each model at its own peak layer.",
       [Line2D([],[],color=MUTED,lw=9,label="AUC"),
        Line2D([],[],color=MUTED,lw=9,alpha=0.40,label="balanced accuracy")],2)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/04_ranking_vs_threshold.png",dpi=200); plt.close(fig)

# --- 05: paired consistency by depth ---
fig,ax=plt.subplots(figsize=(9.4,5.6))
ax.axhline(1.0,color=S3,lw=1.4,ls=(0,(4,3)),zorder=1)
for x,d,c in zip(X,D,COLS):
    ax.plot(x,series(d,"decorrelated","paired_pos"),color=c,lw=2,zorder=3)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
best_pp=max(max(series(d,"decorrelated","paired_pos")) for d in D)
ax.text(50,1.012,f"{int(round(best_pp*N_PAIRS)):,} / {N_PAIRS:,} matched pairs",ha="center",va="bottom",color=S3,fontsize=9.5,fontweight="bold")
axchrome(ax,"relative depth through the network (%)","fraction of pairs scored eval > deploy")
ax.set_ylim(0.45,1.05)
header(fig,"Minimal pairs: almost always ranked correctly",
       "Same payload, same wrapper, one sentence changed — scored by a probe that never saw the pair's vocabulary fold.  320 payload blocks.",
       model_handles(),4)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/05_paired_consistency_by_depth.png",dpi=200); plt.close(fig)

# --- 06: transferable vs format-bound signal ---
def curves(d):
    x=depth(d)
    dec=series(d,"decorrelated","auc")
    xf=(series(d,"b2c","auc")+series(d,"c2b","auc"))/2
    return x,dec,xf
fig,axes=plt.subplots(1,4,figsize=(17.4,5.5),sharey=True)
for ax,d,c,name in zip(axes,D,COLS,NAMES):
    x,dec,xf=curves(d)
    ax.fill_between(x,xf,dec,color=c,alpha=0.16,zorder=2,lw=0)
    ax.plot(x,dec,color=c,lw=2,zorder=3)
    ax.plot(x,xf,color=c,lw=2,ls=(0,(4,3)),zorder=3)
    ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
    ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
    ax.set_title(name,color=INK,loc="left",fontsize=11,fontweight="bold",pad=8)
    ax.set_xlabel("relative depth (%)"); ax.set_ylim(0.45,1.02)
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)
axes[0].set_ylabel("purpose AUC")
fig.text(0.007,0.98,"Purpose signal: transferable vs format-bound",
         fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
fig.text(0.007,0.925,"Solid: all readable purpose signal (decorrelated).   Dashed: the part surviving a format switch (mean of both directions).   Shaded: the format-bound difference.",
         fontsize=9.5,color=INK2,va="top",ha="left")
fig.legend(handles=[Line2D([],[],color=MUTED,lw=2,label="all signal"),
                    Line2D([],[],color=MUTED,lw=2,ls=(0,(4,3)),label="transfers"),
                    Patch(facecolor=MUTED,alpha=0.25,label="format-bound")],
           loc="upper right",bbox_to_anchor=(0.995,0.995),ncol=3,frameon=False,fontsize=9.5,
           labelcolor=INK2,handlelength=1.8,columnspacing=1.2)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/06_signal_split_by_depth.png",dpi=200); plt.close(fig)

# --- 06a: single-panel explainer (3.3-70B, matching the v1/v2 explainer model) ---
d70=D[2]; x,dec,xf=curves(d70)
fig,ax=plt.subplots(figsize=(9.6,5.6))
ax.plot(x,dec,color=S70,lw=2,zorder=3)
ax.plot(x,xf,color=S70,lw=2,ls=(0,(4,3)),zorder=3)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
ia=int(np.argmax(dec-xf))
ax.annotate("",xy=(x[ia],dec[ia]),xytext=(x[ia],xf[ia]),
            arrowprops=dict(arrowstyle="<->",color=INK2,lw=1.6),zorder=4)
ax.text(x[ia]+1.6,(dec[ia]+xf[ia])/2,"gap between the two\ntraining conditions",
        color=INK2,fontsize=9.5,ha="left",va="center")
ic=int(np.argmin(np.where(np.arange(len(x))>=8,dec-xf,np.inf)))
ax.plot([x[ic]],[dec[ic]],"o",ms=8,color=S70,mec=SURF,mew=2,zorder=4)
ax.plot([x[ic]],[xf[ic]],"o",ms=8,color=S70,mec=SURF,mew=2,zorder=4)
ax.annotate(f"layer {ic} · gap {dec[ic]-xf[ic]:.2f}\nnearly everything transfers",
            xy=(x[ic],xf[ic]-0.008),xytext=(32,0.82),
            arrowprops=dict(arrowstyle="-",color=INK2,lw=1.0),
            color=S70,fontsize=9.5,fontweight="bold",ha="left",va="center",zorder=5)
ax.text(40,dec[int(len(x)*0.45)]+0.015,"trained on both formats",color=S70,
        fontsize=9.5,fontweight="bold",ha="left",va="bottom")
ax.text(52,0.862,"trained on one format, tested on the other",
        color=S70,fontsize=9.5,fontweight="bold",ha="left",va="top")
axchrome(ax,"relative depth through the network (%)","purpose AUC (held-out vocabulary fold)")
ax.set_ylim(0.45,1.02)
fig.legend(handles=[Line2D([],[],color=S70,lw=2,label="trained on both formats"),
                    Line2D([],[],color=S70,lw=2,ls=(0,(4,3)),label="trained on one, tested on the other")],
           loc="upper right",bbox_to_anchor=(0.995,0.995),ncol=2,frameon=False,
           fontsize=9.5,labelcolor=INK2,handlelength=1.8,columnspacing=1.4)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/06a_signal_split_explainer.png",dpi=200); plt.close(fig)

# --- 07: format-bound gap ---
fig,ax=plt.subplots(figsize=(9.8,5.7))
for d,c in zip(D,COLS):
    x,dec,xf=curves(d)
    ax.plot(x,dec-xf,color=c,lw=2,zorder=3)
ax.axhline(0,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(99,0.004,"fully format-invariant",ha="right",va="bottom",color=MUTED,fontsize=9)
x,dec,xf=curves(D[2]); g=dec-xf; im=int(np.argmin(np.where(np.arange(len(x))>=8,g,np.inf)))
ax.plot([x[im]],[g[im]],"o",ms=9,color=S70,mec=SURF,mew=2,zorder=4)
ax.annotate(f"3.3-70B layer {im} · gap {g[im]:+.3f}",(x[im],g[im]),textcoords="offset points",
            xytext=(14,-18),color=S70,fontsize=9.5,fontweight="bold")
axchrome(ax,"relative depth through the network (%)","format-bound gap  (decorrelated − transfer AUC)")
ax.set_ylim(-0.02,0.38)
fig.legend(handles=model_handles(),loc="upper right",bbox_to_anchor=(0.995,0.995),
           ncol=1,frameon=False,fontsize=9.5,labelcolor=INK2,handlelength=1.8)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/07_format_bound_gap.png",dpi=200); plt.close(fig)

# --- 08 (new): surface-probe battery — the lexical control, upgraded ---
# TF-IDF n-gram probes on CUE-EXCISED prompt text (the recorded cue span is
# removed first). Purpose gates ~0.500 certify wrapper symmetry: after
# excision the eval/deploy twins are byte-identical, so nothing outside the
# cue sentence can carry purpose. Structure probes are expected decodable.
E = SURFACE["expectations"]
band = SURFACE["near_chance_band"]
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.6,5.4),width_ratios=[3,2])
gates=[("purpose_held_template_word","purpose\nword 1–2 grams"),
       ("purpose_held_template_char","purpose\nchar 2–5 grams"),
       ("purpose_held_template_x_fold_word","purpose\nword · template×fold"),
       ("purpose_held_template_x_fold_char","purpose\nchar · template×fold"),
       ("stratum_held_template_word","format\n(quiz vs casual)")]
i=np.arange(len(gates))
vals=[E[k]["auc"] for k,_ in gates]
cols=[S3]*4+[MUTED]
axL.axhspan(band[0],band[1],color=S3,alpha=0.07,zorder=0)
axL.bar(i,vals,0.56,color=cols,zorder=3)
for xi,v in zip(i,vals): axL.text(xi,v+.02,f"{v:.3f}",ha="center",fontsize=9,color=INK,fontweight="bold")
axL.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
axL.text(-0.45,band[1]-0.035,"near-chance band (hard gate)",ha="left",va="top",color=S3,fontsize=8.5)
axL.set_xticks(i); axL.set_xticklabels([l for _,l in gates],color=INK2,fontsize=8.5)
axL.set_ylim(0,1.06); axchrome(axL,"","AUC on cue-excised text"); axL.grid(axis="x",visible=False)
axL.set_title("purpose is NOT in the wrapper",color=INK,loc="left",fontsize=11,fontweight="bold",pad=10)
rep=[("template_id_held_block_word","template id\n(13 classes)"),
     ("contract_held_block_word","contract\n(5 classes)")]
j=np.arange(len(rep))
acc=[E[k]["accuracy"] for k,_ in rep]; ch=[E[k]["chance"] for k,_ in rep]
axR.bar(j,acc,0.44,color=MUTED,zorder=3)
for xi,v in zip(j,acc): axR.text(xi,v+.02,f"{v:.2f}",ha="center",fontsize=9,color=INK,fontweight="bold")
for xi,c in zip(j,ch):
    axR.plot([xi-0.31,xi+0.31],[c,c],color=INK2,lw=1.6,ls=(0,(3,2)),zorder=4)
    axR.text(xi,c+0.025,"chance",ha="center",va="bottom",color=INK2,fontsize=8)
axR.set_xticks(j); axR.set_xticklabels([l for _,l in rep],color=INK2,fontsize=8.5)
axR.set_xlim(-0.7,1.7)
axR.set_ylim(0,1.06); axchrome(axR,"","accuracy (held-out blocks)"); axR.grid(axis="x",visible=False)
axR.set_title("structure IS legible (as designed)",color=INK,loc="left",fontsize=11,fontweight="bold",pad=10)
header(fig,"Surface battery: nothing outside the cue sentence predicts purpose",
       "TF-IDF n-gram probes on cue-excised text, grouped splits.  Eval/deploy twins are byte-identical after excision — ≈0.500 certifies wrapper symmetry.",
       None,1)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/08_surface_battery.png",dpi=200); plt.close(fig)

print("wrote 9 plots to", OUT)
