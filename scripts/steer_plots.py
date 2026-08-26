"""Plots for the NDIF steering study (causal validation of the v3 probes).

Reads the analysis JSONs written by scripts/steer_analyze.py
(results/steering/analysis_<model>_<mode>.json) and writes PNGs in the repo
style. Nothing here touches activations or NDIF.

  01  verbalized eval-awareness: dose-response (effect vs baseline) per model,
      fmt / pur_dec / pur_conf lines against the 5-seed random null band
  02  direction specificity at |dose|=2: real directions vs the random band
  03  scale test: fmt and pur_dec |dose|=2 effect across 8B/70B/405B
  04  answer style (gen): fmt dose-response on answer-first rate and length
  05  mean-ablation: fmt / pur_dec / pur_conf vs random

Usage: steer_plots.py [results/steering] [out_dir]
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "steering"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "plots" / "steering 8-26-2026"
OUT.mkdir(parents=True, exist_ok=True)

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S71="#8a5cd6"; S405="#d6336c"
FMT="#2a78d6"; PDEC="#1baf7a"; PCONF="#eb6834"; RAND="#b9b7ad"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

MODELS = [("8b","Llama-3.1-8B",S8B),("70b","Llama-3.1-70B",S71),("405b","Llama-3.1-405B",S405)]
RANDS = ["rand0","rand1","rand2","rand3","rand4"]
DOSES = [-2,-1,1,2]

def load(model, mode):
    p = RES / f"analysis_{model}_{mode}.json"
    return json.load(open(p)) if p.exists() else None

def eff(cond, cid, key="effect_vs_base"):
    c = cond.get(cid, {}).get(key)
    return (c["mean"], c["ci95"][0], c["ci95"][1]) if c else (np.nan, np.nan, np.nan)

def header(fig,title,sub,handles=None,ncol=4):
    fig.text(0.008,0.975,title,fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
    if sub: fig.text(0.008,0.915,sub,fontsize=9.5,color=INK2,va="top",ha="left")
    if handles: fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(0.995,0.995),
                           ncol=ncol,frameon=False,fontsize=9.5,labelcolor=INK2,handlelength=1.8,columnspacing=1.4)
def axchrome(ax,xlab,ylab):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)

def rand_band(cond, cids):
    """min/max of the 5 random-seed effect means over the given dose cids."""
    out = {}
    for kk in DOSES:
        vals = [eff(cond, f"{r}{kk:+g}")[0] for r in RANDS]
        vals = [v for v in vals if not np.isnan(v)]
        out[kk] = (min(vals), max(vals)) if vals else (np.nan, np.nan)
    return out

LOG = {m: load(m,"logit") for m,_,_ in MODELS}
GEN = {m: load(m,"gen") for m,_,_ in MODELS}

def strength(m, dn):
    v = (LOG[m] or {}).get("verdicts", {}).get(dn, {})
    return v.get("strength_z")
def verdict_label(m, dn):
    return (LOG[m] or {}).get("verdicts", {}).get(dn, {}).get("verdict", "null")

# ---- 01 dose-response, eval-logodds, per model -----------------------------
have = [(m,n,c) for m,n,c in MODELS if LOG[m]]
if have:
    fig,axes=plt.subplots(1,len(have),figsize=(5.9*len(have),5.6),sharey=False)
    if len(have)==1: axes=[axes]
    for ax,(m,name,col) in zip(axes,have):
        cond=LOG[m]["conditions"]; x=np.array(DOSES)
        band=rand_band(cond,None)
        lo=[band[k][0] for k in DOSES]; hi=[band[k][1] for k in DOSES]
        ax.fill_between(x,lo,hi,color=RAND,alpha=0.55,zorder=1,label="random band (5 seeds)")
        for dname,dc,dl in [("fmt",FMT,"format"),("pur_dec",PDEC,"purpose (decorr)"),("pur_conf",PCONF,"purpose (confound)")]:
            ys=[eff(cond,f"{dname}{k:+g}")[0] for k in DOSES]
            los=[eff(cond,f"{dname}{k:+g}")[1] for k in DOSES]
            his=[eff(cond,f"{dname}{k:+g}")[2] for k in DOSES]
            ax.plot(x,ys,"-o",color=dc,lw=2,ms=5,zorder=3,label=dl)
            ax.fill_between(x,los,his,color=dc,alpha=0.14,zorder=2)
        ax.axhline(0,color=BASE,lw=1,ls=":")
        ax.axvline(0,color=GRID,lw=1)
        ax.set_title(name,color=INK,loc="left",fontsize=11,fontweight="bold",pad=8)
        axchrome(ax,"dose  (× Δ_fmt, matched norm)","Δ verbalized eval log-odds vs baseline")
        ax.set_xticks(DOSES)
    handles=[Line2D([],[],color=FMT,lw=2,marker="o",label="format"),
             Line2D([],[],color=PDEC,lw=2,marker="o",label="purpose (decorrelated)"),
             Line2D([],[],color=PCONF,lw=2,marker="o",label="purpose (confounded)"),
             Patch(fc=RAND,alpha=0.55,label="random band (5 seeds)")]
    header(fig,"Steering the format / purpose directions moves verbalized eval-awareness",
           "No-cue items (nothing stated). Effect = signed eval log-odds minus the same item's unsteered baseline; band = 320-block bootstrap 95%. All directions added at matched norm (±k·Δ_fmt).",
           handles,4)
    fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(OUT/"01_eval_dose_response.png",dpi=200); plt.close(fig)

# ---- 02 direction specificity: |z| vs random, per model --------------------
if have:
    fig,axes=plt.subplots(1,len(have),figsize=(5.9*len(have),5.4),sharey=True)
    if len(have)==1: axes=[axes]
    dirs=[("fmt","format",FMT),("pur_dec","purpose\ndecorr",PDEC),("pur_conf","purpose\nconfound",PCONF),
          ("fmt_dm","format\ndiff-means","#6a8caf"),("pur_dm","purpose\ndiff-means","#9a7bbf")]
    for ax,(m,name,col) in zip(axes,have):
        i=np.arange(len(dirs))
        for j,(dn,dl,dc) in enumerate(dirs):
            z=strength(m,dn)
            ax.bar(j,z if z is not None else 0,0.62,color=dc,zorder=3)
            if z is not None: ax.text(j,z+0.05,f"{z:.1f}",ha="center",color=INK2,fontsize=9,fontweight="bold")
        ax.axhline(2.0,color=INK,lw=1,ls="--",zorder=2); ax.text(len(dirs)-0.5,2.02,"strong",ha="right",va="bottom",color=INK2,fontsize=8)
        ax.axhline(1.0,color=BASE,lw=1,ls=":",zorder=2); ax.text(len(dirs)-0.5,1.02,"suggestive",ha="right",va="bottom",color=MUTED,fontsize=8)
        ax.set_xticks(i); ax.set_xticklabels([d[1] for d in dirs],fontsize=8.6,color=INK2)
        ax.set_title(name,color=INK,loc="left",fontsize=11,fontweight="bold",pad=8)
        axchrome(ax,"","causal strength  |z| vs 5 random directions"); ax.grid(axis="x",visible=False)
    header(fig,"Which directions causally move verbalized eval-awareness",
           "Sign-agnostic: |effect| in random-direction SDs, averaged over ±2 dose (a used direction may steer either sign). ≥2 clears ~95% of the random-norm null.",
           None,1)
    fig.tight_layout(rect=[0,0,1,0.88]); fig.savefig(OUT/"02_direction_specificity.png",dpi=200); plt.close(fig)

# ---- 03 scale test: causal strength across models --------------------------
if len([m for m,_,_ in MODELS if LOG[m]])>=2:
    fig,ax=plt.subplots(figsize=(9.4,5.8))
    ms=[m for m,_,_ in MODELS if LOG[m]]; labels=[n for m,n,_ in MODELS if LOG[m]]
    i=np.arange(len(ms)); w=0.26
    for off,dn,dc,dl in [(-w,"fmt",FMT,"format"),(0,"pur_dec",PDEC,"purpose (decorrelated)"),(w,"pur_conf",PCONF,"purpose (confounded)")]:
        zs=[strength(m,dn) or 0 for m in ms]
        ax.bar(i+off,zs,w,color=dc,zorder=3,label=dl)
        for xi,z in zip(i+off,zs): ax.text(xi,z+0.04,f"{z:.1f}",ha="center",color=INK2,fontsize=8.5,fontweight="bold")
    ax.axhline(2.0,color=INK,lw=1,ls="--"); ax.text(len(ms)-0.5,2.03,"strong",ha="right",va="bottom",color=INK2,fontsize=8.5)
    ax.axhline(1.0,color=BASE,lw=1,ls=":"); ax.text(len(ms)-0.5,1.03,"suggestive",ha="right",va="bottom",color=MUTED,fontsize=8.5)
    ax.set_xticks(i); ax.set_xticklabels(labels,fontsize=10,color=INK)
    axchrome(ax,"","causal strength  |z| vs random directions")
    ax.legend(loc="upper left",frameon=False,fontsize=9.5,labelcolor=INK2)
    ax.set_ylim(0,max(2.6, ax.get_ylim()[1]))
    header(fig,"Format steering is causally potent; format-orthogonal purpose steering is not",
           "Verbalized eval-awareness, |z| vs the random-norm null (≥2 strong, ≥1 suggestive).",
           None,1)
    fig.tight_layout(rect=[0,0,1,0.88]); fig.savefig(OUT/"03_scale_test.png",dpi=200); plt.close(fig)

# ---- 04 gen style: format +8/-8 effect on several features, vs random -------
haveg=[(m,n,c) for m,n,c in MODELS if GEN[m]]
GRANDS=["rand0","rand1"]  # gen ladder carries rand0/rand1 at +/-8
FEATS=[("answer_first","answer-first"),("has_markdown","markdown"),("meta_mention","meta-mention"),
       ("n_new","length (tok)"),("repetition","repetition")]
if haveg:
    fig,axes=plt.subplots(1,len(haveg),figsize=(5.9*len(haveg),5.6),sharey=False)
    if len(haveg)==1: axes=[axes]
    for ax,(m,name,col) in zip(axes,haveg):
        feats=GEN[m]["features"]; i=np.arange(len(FEATS)); w=0.36
        for off,dose,dc in [(-w/2,"fmt+8",FMT),(w/2,"fmt-8","#7bb0e8")]:
            mns,los,his=[],[],[]
            for fk,_ in FEATS:
                cnd=feats.get(fk,{}).get("conditions",{})
                mn,lo,hi=eff(cnd,dose)
                # normalize length to its baseline for comparability
                base=feats.get(fk,{}).get("baseline",{}).get("mean",1.0)
                sc=(1.0/base if fk=="n_new" and base else 1.0)
                mns.append(mn*sc); los.append(lo*sc); his.append(hi*sc)
            ax.bar(i+off,mns,w,color=dc,zorder=3,label=dose)
            ax.errorbar(i+off,mns,yerr=[np.array(mns)-np.array(los),np.array(his)-np.array(mns)],fmt="none",ecolor=INK,elinewidth=1,zorder=4)
        # random +8 band per feature (min/max of rand0,rand1)
        for j,(fk,_) in enumerate(FEATS):
            cnd=feats.get(fk,{}).get("conditions",{})
            base=feats.get(fk,{}).get("baseline",{}).get("mean",1.0)
            sc=(1.0/base if fk=="n_new" and base else 1.0)
            rv=[eff(cnd,f"{r}+8")[0]*sc for r in GRANDS if not np.isnan(eff(cnd,f"{r}+8")[0])]
            if rv: ax.axhspan(min(rv),max(rv),xmin=(j)/len(FEATS),xmax=(j+1)/len(FEATS),color=RAND,alpha=0.45,zorder=1)
        ax.axhline(0,color=BASE,lw=1,ls=":")
        ax.set_xticks(i); ax.set_xticklabels([f[1] for f in FEATS],fontsize=8.4,color=INK2,rotation=20,ha="right")
        ax.set_title(name,color=INK,loc="left",fontsize=11,fontweight="bold",pad=8)
        axchrome(ax,"","Δ feature vs baseline (length ÷ baseline)")
    header(fig,"Format steering changes answer formatting (8B); strong doses collapse style in the larger models",
           "Greedy no-cue generations, format ±8·Δ_fmt vs baseline (320-block 95% CI). Shaded: random-direction span at +8. The 8B markdown effect sign-flips beyond the random band; at ±8 the 70B/405B move with either sign (generic perturbation).",
           [Patch(fc=FMT,label="format +8"),Patch(fc="#7bb0e8",label="format -8"),Patch(fc=RAND,alpha=0.45,label="random +8")],3)
    fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(OUT/"04_gen_style.png",dpi=200); plt.close(fig)

# ---- 05 ablation -----------------------------------------------------------
if have:
    fig,ax=plt.subplots(figsize=(9.4,5.6))
    ms=[m for m,_,_ in MODELS if LOG[m]]; i=np.arange(len(ms)); w=0.2
    for off,dn,dc,dl in [(-1.5*w,"fmt",FMT,"format"),(-0.5*w,"pur_dec",PDEC,"purpose decorr"),
                         (0.5*w,"pur_conf",PCONF,"purpose confound"),(1.5*w,"rand0",RAND,"random0")]:
        mns=[eff(LOG[m]["conditions"],f"{dn}_abl")[0] for m in ms]
        los=[eff(LOG[m]["conditions"],f"{dn}_abl")[1] for m in ms]; his=[eff(LOG[m]["conditions"],f"{dn}_abl")[2] for m in ms]
        ax.bar(i+off,mns,w,color=dc,zorder=3,label=dl)
        ax.errorbar(i+off,mns,yerr=[np.array(mns)-np.array(los),np.array(his)-np.array(mns)],fmt="none",ecolor=INK,elinewidth=1.1,zorder=4)
    ax.axhline(0,color=BASE,lw=1,ls=":")
    ax.set_xticks(i); ax.set_xticklabels([n for m,n,_ in MODELS if LOG[m]],color=INK)
    axchrome(ax,"","Δ eval log-odds vs baseline (mean-ablation)")
    header(fig,"Mean-ablating each direction",
           "Component along the direction set to the no-cue mean at the probed positions.",
           [Patch(fc=FMT,label="format"),Patch(fc=PDEC,label="purpose decorr"),Patch(fc=PCONF,label="purpose confound"),Patch(fc=RAND,label="random")],4)
    fig.tight_layout(rect=[0,0,1,0.88]); fig.savefig(OUT/"05_ablation.png",dpi=200); plt.close(fig)

print(f"wrote plots to {OUT} ({len([p for p in OUT.glob('*.png')])} png)")
