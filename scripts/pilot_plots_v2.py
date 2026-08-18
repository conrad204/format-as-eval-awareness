"""Plots for the v2 80-block pilot (exploratory, per docs/analysis_plan_v2.md).

Reads results/v2/pilot80_layer_sweep_*.json and results/v2/pilot80_analysis_*.json,
writes PNGs into plots/pilot results 8-16-2026/. Style matches make_plots.py.
"""

import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"; OUT=ROOT/"plots"/"pilot results 8-16-2026"; OUT.mkdir(parents=True,exist_ok=True)
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S70="#eb6834"; S3="#1baf7a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

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

MODELS=[  # (label, color, sweep file, analysis file, pinned layer)
    ("Llama-3.1-8B", S8B, "pilot80_layer_sweep_8b.json", "pilot80_analysis_8b_layer12.json", 12),
    ("Llama-3.1-70B", S3, "pilot80_layer_sweep_31_70b.json", "pilot80_analysis_31_70b_layer31.json", 31),
    ("Llama-3.3-70B", S70, "pilot80_layer_sweep_33_70b.json", "pilot80_analysis_33_70b_layer17.json", 17),
]
SW={n: json.load(open(R/"v2"/s)) for n,_,s,_,_ in MODELS}
AN={n: json.load(open(R/"v2"/a))["endpoints"] for n,_,_,a,_ in MODELS}
HANDLES=[Line2D([],[],color=c,lw=2,label=n) for n,c,_,_,_ in MODELS]

def ci_dot(ax,y,e,color,ms=7):
    lo,hi=e["ci95"]
    ax.plot([lo,hi],[y,y],color=color,lw=2,solid_capstyle="round",zorder=3)
    ax.plot([e["point"]],[y],"o",ms=ms,color=color,mec=SURF,mew=1.6,zorder=4)

# --- 1: stated held-fold AUC by depth ---
fig,ax=plt.subplots(figsize=(9.4,5.8))
for name,col,_,_,pin in MODELS:
    sweep=SW[name]["sweep"]; n=len(sweep)
    x=np.array([l/(n-1)*100 for l,_ in sweep]); y=np.array([a for _,a in sweep])
    ax.plot(x,y,color=col,lw=2,zorder=3)
    xv=pin/(n-1)*100; yv=dict(sweep)[pin]
    ax.plot([xv],[yv],"o",ms=9,color=col,mec=SURF,mew=2,zorder=4)
    dy={"Llama-3.1-8B":-24,"Llama-3.1-70B":14,"Llama-3.3-70B":-36}[name]
    ax.annotate(f"L{pin} · AUC {yv:.3f}",(xv,yv),textcoords="offset points",
                xytext=(10,dy),color=col,fontsize=9.5,fontweight="bold",zorder=5)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
ax.axhline(0.5,color=BASE,lw=0,zorder=0)
axchrome(ax,"relative depth through the network (%)","stated purpose AUC (held vocabulary fold)")
ax.set_ylim(0.42,1.06)
header(fig,"Purpose decodes from unseen vocabulary",
       "Leave-one-vocabulary-fold-out on the stated stratum, v2 pilot (80 blocks). Dots: per-model argmax layer.",
       HANDLES,3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(OUT/"01_stated_signal_by_depth.png",dpi=200); plt.close(fig)

# --- 2: endpoint forest ---
ROWS=[("E1 stated (held fold)","E1_stated_auc"),("E1 n-gram control","E1_ngram_auc"),
      ("E3 quiz → casual","E3_b2c"),("E3 casual → quiz","E3_c2b"),
      ("E7 payload holdout","E7_payload_holdout"),("E8 double holdout","E8_double_holdout")]
fig,ax=plt.subplots(figsize=(9.6,6.2))
yt=[]
for ri,(lab,key) in enumerate(ROWS):
    base=len(ROWS)-1-ri
    for mi,(name,col,_,_,_) in enumerate(MODELS):
        ci_dot(ax,base+0.26-0.26*mi,AN[name][key],col)
    yt.append((base,lab))
ax.axvline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(0.502,len(ROWS)-0.42,"chance",color=MUTED,fontsize=9,ha="left")
ax.set_yticks([y for y,_ in yt]); ax.set_yticklabels([l for _,l in yt],color=INK,fontsize=10)
axchrome(ax,"AUC (95% CI, cluster bootstrap over payload blocks)","")
ax.set_xlim(0.40,1.02); ax.set_ylim(-0.5,len(ROWS)-0.2)
header(fig,"Every generalization split clears chance",
       "v2 pilot, per-model argmax layer. E8 holds out vocabulary fold and payload topic group at once.",
       HANDLES,3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(OUT/"02_endpoint_forest.png",dpi=200); plt.close(fig)

# --- 3: strata — cue locus and cue style ---
ROWS3=[("noun-only (E2)",lambda E:E["E2_locus"]["noun_only"]),
       ("verb-only (E2)",lambda E:E["E2_locus"]["verb_only"]),
       ("noun+verb (E2)",lambda E:E["E2_locus"]["both"]),
       ("indirect, transfer (E4)",lambda E:E["E4_indirect_transfer"]),
       ("indirect, own CV (E4)",lambda E:E["E4_indirect_own_cv"]),
       ("2nd person, transfer (E5)",lambda E:E["E5_second_transfer"]),
       ("2nd person, own CV (E5)",lambda E:E["E5_second_own_cv"])]
fig,ax=plt.subplots(figsize=(9.6,6.6))
yt=[]
for ri,(lab,get) in enumerate(ROWS3):
    base=len(ROWS3)-1-ri
    for mi,(name,col,_,_,_) in enumerate(MODELS):
        ci_dot(ax,base+0.26-0.26*mi,get(AN[name]),col)
    yt.append((base,lab))
ax.axvline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.set_yticks([y for y,_ in yt]); ax.set_yticklabels([l for _,l in yt],color=INK,fontsize=10)
axchrome(ax,"AUC (95% CI, cluster bootstrap over payload blocks)","")
ax.set_xlim(0.40,1.02); ax.set_ylim(-0.5,len(ROWS3)-0.2)
header(fig,"v1's unestimable strata now decode",
       "Transfer rows score cue styles the probe never trained on; own-CV rows train within the stratum.",
       HANDLES,3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(OUT/"03_strata.png",dpi=200); plt.close(fig)

# --- 4: E6 no-cue format shift — the sign flip ---
fig,ax=plt.subplots(figsize=(9.4,4.6))
for mi,(name,col,_,_,_) in enumerate(MODELS):
    e=AN[name]["E6_nocue_format_shift_sd"]; y=len(MODELS)-1-mi
    lo,hi=e["ci95"]
    ax.plot([lo,hi],[y,y],color=col,lw=2.4,solid_capstyle="round",zorder=3)
    ax.plot([e["point"]],[y],"o",ms=10,color=col,mec=SURF,mew=2,zorder=4)
    ax.annotate(f"{e['point']:+.2f} SD",(e["point"],y),textcoords="offset points",
                xytext=(0,13),ha="center",color=col,fontsize=10,fontweight="bold")
ax.axvline(0,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(-0.05,len(MODELS)-0.42,"← reads as deployment",ha="right",color=MUTED,fontsize=9.5)
ax.text(0.05,len(MODELS)-0.42,"reads as evaluation →",ha="left",color=MUTED,fontsize=9.5)
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([n for n,_,_,_,_ in reversed(MODELS)],color=INK,fontsize=10)
axchrome(ax,"benchmark − casual shift of purpose-neutral prompts along the purpose direction (SD)","")
ax.set_ylim(-0.5,len(MODELS)-0.1)
header(fig,"Format alone moves purpose-neutral prompts — in a model-dependent direction",
       "No-cue items projected on held-fold purpose probes (E6). The 3.1 models flip sign against 3.3-70B.",
       None,1)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(OUT/"04_nocue_format_shift.png",dpi=200); plt.close(fig)

# --- 5: per-fold stated AUCs ---
FOLD_LBL={"fg1":"fg1\nreview /\npurchase","fg2":"fg2\nexam /\nclient","fg3":"fg3\naudit /\ntenant",
          "fg4":"fg4\ntest /\nsupport","fg5":"fg5\nverdict /\nhousehold"}
fig,ax=plt.subplots(figsize=(9.6,5.2))
folds=["fg1","fg2","fg3","fg4","fg5"]; i=np.arange(5)
for mi,(name,col,_,_,_) in enumerate(MODELS):
    xs=i+(mi-1)*0.22
    for xi,f in zip(xs,folds):
        e=AN[name]["per_fold_stated"][f]; lo,hi=e["ci95"]
        ax.plot([xi,xi],[lo,hi],color=col,lw=2,solid_capstyle="round",zorder=3)
        ax.plot([xi],[e["point"]],"o",ms=7,color=col,mec=SURF,mew=1.6,zorder=4)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.set_xticks(i); ax.set_xticklabels([FOLD_LBL[f] for f in folds],color=INK2,fontsize=9)
axchrome(ax,"","stated purpose AUC (held vocabulary fold)")
ax.set_ylim(0.45,1.04); ax.grid(axis="x",visible=False)
header(fig,"No vocabulary fold carries the result",
       "Each fold scored by a probe trained on the other four (E1 split), per item 12.",
       HANDLES,3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(OUT/"05_per_fold.png",dpi=200); plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
