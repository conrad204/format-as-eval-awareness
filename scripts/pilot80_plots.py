"""Plots for the v2 pilot (80 blocks, 1760 items, three models).

Counterparts of the v1 plots 01-07/06a where the design allows, computed from
the v2 full-regime sweeps (results/v2/pilot80_sweep_full_*.json — held-out
VOCABULARY folds, so "out-of-fold" is strictly stronger than v1), plus new
plots 08-10 for the v2-only endpoints (results/v2/pilot80_analysis_*.json).

v1 -> v2 reading notes:
  - "decorrelated" now also means vocabulary-disjoint: the tested fold's cue
    words were never in training. The permanent lexical control (E1 n-gram)
    sits at 0.500.
  - Three models: the two v1 models' colors are kept (8B blue, 3.3-70B
    orange) so old and new plots are directly comparable; 3.1-70B (new, same
    lineage as the 8B) gets violet.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = str(ROOT / "results")
OUT = ROOT / "plots" / "pilot results 8-16-2026"
OUT.mkdir(parents=True, exist_ok=True)
OUT = str(OUT)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S71="#8a5cd6"; S70="#eb6834"; S3="#1baf7a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

MODELS = [  # (sweep json, analysis json, short name, color)
    ("pilot80_sweep_full_llama31_8b.json",  "pilot80_analysis_8b_layer12.json",     "Llama-3.1-8B",  S8B),
    ("pilot80_sweep_full_llama31_70b.json", "pilot80_analysis_31_70b_layer31.json", "Llama-3.1-70B", S71),
    ("pilot80_sweep_full_llama33_70b.json", "pilot80_analysis_33_70b_layer17.json", "Llama-3.3-70B", S70),
]
D = [json.load(open(f"{R}/v2/{s}")) for s, _, _, _ in MODELS]
A = [json.load(open(f"{R}/v2/{a}"))["endpoints"] for _, a, _, _ in MODELS]
NAMES = [n for _, _, n, _ in MODELS]
COLS = [c for _, _, _, c in MODELS]

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
def model_handles(lw=2):
    return [Line2D([],[],color=c,lw=lw,label=n) for n,c in zip(NAMES,COLS)]

# --- 01: purpose signal by depth (decorrelated solid, confounded dashed) ---
fig,ax=plt.subplots(figsize=(9.4,5.8))
for x,d,c in zip(X,D,COLS):
    ax.plot(x,series(d,"decorrelated","auc"),color=c,lw=2,zorder=3)
    ax.plot(x,series(d,"confounded","auc"),color=c,lw=2,ls=(0,(4,3)),alpha=.85,zorder=2)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
offsets=[(10,-24),(10,14),(-115,14)]
for p,name,c,x,(dx,dy) in zip(PEAKS,["8B","3.1-70B","3.3-70B"],COLS,X,offsets):
    xv=p["layer"]/(len(x)-1)*100; yv=p["decorrelated"]["auc"]
    ax.plot([xv],[yv],"o",ms=9,color=c,mec=SURF,mew=2,zorder=4)
    ax.annotate(f"{name} peak · L{p['layer']} · AUC {yv:.3f}",(xv,yv),textcoords="offset points",
                xytext=(dx,dy),color=c,fontsize=9.5,fontweight="bold",zorder=5)
axchrome(ax,"relative depth through the network (%)","purpose AUC (held-out vocabulary fold)")
ax.set_ylim(0.42,1.04)
header(fig,"Purpose decodes from unseen vocabulary",
       "Solid: decorrelated training, tested on a held-out cue-vocabulary fold.   Dashed: confounded training (purpose ⊗ format).",
       model_handles(),3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/01_purpose_signal_by_depth.png",dpi=200); plt.close(fig)

# --- 02: confound signature at each model's peak layer ---
fig,axes=plt.subplots(1,3,figsize=(13.6,5.4),sharey=True)
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
header(fig,"Transfer to an unseen prompt format (still on unseen vocabulary)","",model_handles(),3)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/03_transfer_by_depth.png",dpi=200); plt.close(fig)

# --- 04: ranking vs threshold at the peak layer ---
fig,ax=plt.subplots(figsize=(10.6,5.6))
groups=["decorrelated","quiz → casual","casual → quiz"]; regs=["decorrelated","b2c","c2b"]
i=np.arange(3); w=0.14
for k,(p,name,col) in enumerate(zip(PEAKS,["8B","3.1-70B","3.3-70B"],COLS)):
    au=[p[r]["auc"] for r in regs]; ba=[p[r]["bal_acc"] for r in regs]
    off=(k-1)*2.1*w
    ax.bar(i+off-w*0.52,au,w,color=col,zorder=3)
    ax.bar(i+off+w*0.52,ba,w,color=col,alpha=0.40,zorder=3)
    for xi,v in zip(i+off-w*0.52,au): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=8,color=INK,fontweight="bold")
    for xi,v in zip(i+off+w*0.52,ba): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=8,color=INK2)
    for g in range(3): ax.text(i[g]+off,-0.062,name,ha="center",color=INK2,fontsize=8.5,fontweight="bold")
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.set_xticks(i); ax.set_xticklabels(groups,color=INK,fontsize=11,fontweight="bold")
ax.tick_params(axis="x",pad=24,length=0); ax.set_ylim(0,1.06)
axchrome(ax,"","score"); ax.grid(axis="x",visible=False)
header(fig,"Ranking survives a format change; the decision threshold suffers more",
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
n_pairs=480  # 960 stated rows = 480 (block, pair, format) minimal pairs
best_pp=max(max(series(d,"decorrelated","paired_pos")) for d in D)
ax.text(50,1.012,f"{int(round(best_pp*n_pairs))} / {n_pairs} matched pairs",ha="center",va="bottom",color=S3,fontsize=9.5,fontweight="bold")
axchrome(ax,"relative depth through the network (%)","fraction of pairs scored eval > deploy")
ax.set_ylim(0.45,1.05)
header(fig,"Minimal pairs: almost always ranked correctly",
       "Same payload, same wrapper, one sentence changed — scored by a probe that never saw the pair's vocabulary fold.",
       model_handles(),3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/05_paired_consistency_by_depth.png",dpi=200); plt.close(fig)

# --- 06: transferable vs format-bound signal ---
def curves(d):
    x=depth(d)
    dec=series(d,"decorrelated","auc")
    xf=(series(d,"b2c","auc")+series(d,"c2b","auc"))/2
    return x,dec,xf
fig,axes=plt.subplots(1,3,figsize=(13.6,5.5),sharey=True)
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

# --- 06a: single-panel explainer (3.3-70B, matching the v1 explainer model) ---
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
ax.text(x[ic]+2,0.93,f"layer {ic} · gap {dec[ic]-xf[ic]:.2f}\nnearly everything transfers",
        color=S70,fontsize=9.5,fontweight="bold",ha="left",va="center")
ax.text(40,dec[int(len(x)*0.45)]+0.015,"trained on both formats",color=S70,
        fontsize=9.5,fontweight="bold",ha="left",va="bottom")
ax.text(40,xf[int(len(x)*0.45)]-0.02,"trained on one format, tested on the other",
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

# --- 08 (new): held-out generalization ladder at the pinned layer ---
ENDPOINTS=[("E1_ngram_auc","n-gram control\n(cue words)"),
           ("E1_stated_auc","stated\n(held vocab)"),
           ("E7_payload_holdout","held payloads\n(seen vocab)"),
           ("E8_double_holdout","held vocab\n+ payloads"),
           ("E4_indirect_transfer","indirect cues\n(transfer)"),
           ("E5_second_transfer","second person\n(transfer)")]
fig,ax=plt.subplots(figsize=(10.8,5.8))
i=np.arange(len(ENDPOINTS)); w=0.24
for k,(a,name,col) in enumerate(zip(A,NAMES,COLS)):
    off=(k-1)*w
    pts=[a[e]["point"] for e,_ in ENDPOINTS]
    lo=[a[e]["point"]-a[e]["ci95"][0] for e,_ in ENDPOINTS]
    hi=[a[e]["ci95"][1]-a[e]["point"] for e,_ in ENDPOINTS]
    ax.bar(i+off,pts,w*0.86,color=col,zorder=3)
    ax.errorbar(i+off,pts,yerr=[lo,hi],fmt="none",ecolor=INK2,elinewidth=1.1,capsize=2.4,zorder=4)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(-0.42,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
ax.set_xticks(i); ax.set_xticklabels([l for _,l in ENDPOINTS],color=INK2,fontsize=9)
ax.set_ylim(0.4,1.04)
axchrome(ax,"","purpose AUC"); ax.grid(axis="x",visible=False)
header(fig,"The purpose read survives every holdout",
       "Pre-registered endpoints at each model's pilot argmax layer.  Whiskers: 95% cluster-bootstrap CIs over 80 payload blocks.",
       model_handles(9),3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/08_generalization_ladder.png",dpi=200); plt.close(fig)

# --- 09 (new): E6 no-cue format shift — the sign flip ---
fig,ax=plt.subplots(figsize=(9.6,4.6))
y=np.arange(len(MODELS))[::-1]
for k,(a,name,col) in enumerate(zip(A,NAMES,COLS)):
    e=a["E6_nocue_format_shift_sd"]; p=e["point"]; lo,hi=e["ci95"]
    ax.plot([lo,hi],[y[k],y[k]],color=col,lw=3,solid_capstyle="round",zorder=3)
    ax.plot([p],[y[k]],"o",ms=10,color=col,mec=SURF,mew=2,zorder=4)
    ax.text(p,y[k]+0.18,f"{p:+.2f} SD",ha="center",color=col,fontsize=10,fontweight="bold")
ax.axvline(0,color=BASE,lw=1.4,ls=":",zorder=1)
ax.set_yticks(y); ax.set_yticklabels(NAMES,color=INK,fontsize=10.5,fontweight="bold")
ax.text(-1.62,-0.62,"◀ benchmark format reads as “deployment”",color=INK2,fontsize=9.5,ha="left")
ax.text(0.42,-0.62,"reads as “evaluation” ▶",color=INK2,fontsize=9.5,ha="left")
axchrome(ax,"shift of purpose-neutral prompts along the purpose direction (SD units)","")
ax.set_xlim(-1.8,0.7); ax.set_ylim(-0.9,2.6); ax.grid(axis="y",visible=False)
header(fig,"Format alone moves purpose-neutral prompts — in a model-dependent direction",
       "No-cue items (no purpose sentence), benchmark − casual, projected on each model's purpose direction.  95% cluster-bootstrap CIs.",
       None,1)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/09_nocue_format_shift.png",dpi=200); plt.close(fig)

# --- 10 (new): cue-locus strata (E2) ---
LOCI=[("noun_only","noun only"),("verb_only","verb only"),("both","noun + verb")]
fig,ax=plt.subplots(figsize=(9.8,5.4))
i=np.arange(len(LOCI)); w=0.24
for k,(a,name,col) in enumerate(zip(A,NAMES,COLS)):
    off=(k-1)*w
    pts=[a["E2_locus"][l]["point"] for l,_ in LOCI]
    lo=[a["E2_locus"][l]["point"]-a["E2_locus"][l]["ci95"][0] for l,_ in LOCI]
    hi=[a["E2_locus"][l]["ci95"][1]-a["E2_locus"][l]["point"] for l,_ in LOCI]
    ax.bar(i+off,pts,w*0.86,color=col,zorder=3)
    ax.errorbar(i+off,pts,yerr=[lo,hi],fmt="none",ecolor=INK2,elinewidth=1.1,capsize=2.4,zorder=4)
    for xi,v in zip(i+off,pts): ax.text(xi,v+.02,f"{v:.2f}",ha="center",fontsize=8.5,color=INK)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(-0.42,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
ax.set_xticks(i); ax.set_xticklabels([l for _,l in LOCI],color=INK,fontsize=11,fontweight="bold")
ax.set_ylim(0.4,1.09)
axchrome(ax,"where the cue lives in the sentence","purpose AUC (held vocab)"); ax.grid(axis="x",visible=False)
header(fig,"A single swapped noun is enough",
       "Stated stratum by cue locus, 80 blocks each (v1 could not estimate noun-only: 3 pairs, one family).  95% cluster CIs.",
       model_handles(9),3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/10_cue_locus.png",dpi=200); plt.close(fig)

print("wrote 11 plots to", OUT)
