"""Does the frozen eval-ness probe rank arena traffic above general use?

provenance_v3 carries two deployment-log corpora that share the label 0:

  WildChat-1M          general free ChatGPT use
  LMSYS-Chat-1M        Chatbot Arena, where users deliberately compare models

Both are "deployment" by corpus origin, but arena users are arguably engaged in a
mild evaluation activity. That yields a prediction the provenance label itself
cannot make: a probe tracking something eval-like should rank LMSYS above
WildChat, even though the label says both are 0.

This script reads the frozen-probe score arrays and reports, per layer:
  * AUC(eval vs WildChat) and AUC(eval vs LMSYS) separately;
  * AUC(LMSYS vs WildChat) inside the deployment class, where 0.5 is "the probe
    does not distinguish the two log corpora";
  * the same quantities within each response format.

Bootstrap resamples complete underlying-text pairs, stratified by source, with
one shared block sample across every layer and regime.

Caveat carried in the output: the two log corpora also differ in topic and
register, so a positive result is consistent with content differences and is not
by itself evidence of evaluation awareness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

WILDCHAT = "allenai/WildChat-1M"
LMSYS = "lmsys/lmsys-chat-1m"
SEED = 20260907


def auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def contrasts(score: np.ndarray, y: np.ndarray, source: np.ndarray,
              fmt: np.ndarray) -> dict:
    out = {}
    eval_mask = y == 1
    wild, lm = source == WILDCHAT, source == LMSYS
    for name, mask in (("all", np.ones(len(y), bool)),
                       ("benchmark", fmt == "benchmark"),
                       ("casual", fmt == "casual")):
        ev, w, arena = eval_mask & mask, wild & mask, lm & mask
        sel = ev | w
        out[f"{name}__auc_eval_vs_wildchat"] = auc(y[sel], score[sel])
        sel = ev | arena
        out[f"{name}__auc_eval_vs_lmsys"] = auc(y[sel], score[sel])
        # Inside the deployment class: is arena traffic ranked as more eval-like?
        sel = w | arena
        out[f"{name}__auc_lmsys_over_wildchat"] = auc(arena[sel].astype(int), score[sel])
        out[f"{name}__mean_score_wildchat"] = float(score[w].mean())
        out[f"{name}__mean_score_lmsys"] = float(score[arena].mean())
        out[f"{name}__mean_score_eval"] = float(score[ev].mean())
    return out


def bootstrap_samples(pair: np.ndarray, source: np.ndarray, n_boot: int) -> list[np.ndarray]:
    by_pair: dict[str, list[int]] = {}
    pair_source: dict[str, str] = {}
    for i, p in enumerate(pair):
        by_pair.setdefault(str(p), []).append(i)
        pair_source[str(p)] = str(source[i])
    groups: dict[str, list[str]] = {}
    for p, s in pair_source.items():
        groups.setdefault(s, []).append(p)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(n_boot):
        rows: list[int] = []
        for s in sorted(groups):
            ps = sorted(groups[s])
            for p in rng.choice(ps, len(ps), replace=True):
                rows.extend(by_pair[str(p)])
        samples.append(np.asarray(rows, dtype=np.int32))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="keep every Nth layer plus the last, to bound cost on deep models")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.scores, allow_pickle=False) as z:
        scores = z["scores"].astype(np.float64)
        layers = z["layers"].astype(int)
        regimes = [str(x) for x in z["regimes"]]
        y = z["y"].astype(np.int8)
        fmt = z["format"].astype(str)
        source = z["source"].astype(str)
        pair = z["pair_id"].astype(str)
    present = set(source.tolist())
    if not {WILDCHAT, LMSYS} <= present:
        raise RuntimeError(f"both log corpora required, found {sorted(present)}")

    if args.layer_stride > 1:
        keep = sorted(set(list(range(0, len(layers), args.layer_stride)) + [len(layers) - 1]))
        scores = scores[keep]
        layers = layers[keep]
        print(f"layer stride {args.layer_stride}: {len(layers)} of {len(keep)} kept", flush=True)

    samples = bootstrap_samples(pair, source, args.n_bootstrap)
    out_layers = []
    for li, layer in enumerate(layers):
        record = {"layer": int(layer), "regimes": {}}
        for ri, regime in enumerate(regimes):
            point = contrasts(scores[li, ri], y, source, fmt)
            boot = {k: [] for k in point}
            for idx in samples:
                b = contrasts(scores[li, ri, idx], y[idx], source[idx], fmt[idx])
                for k, v in b.items():
                    boot[k].append(v)
            record["regimes"][regime] = {
                "point": point,
                "block_bootstrap_ci95": {
                    k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
                    for k, v in boot.items()
                },
            }
        out_layers.append(record)
        print(f"layer {layer} contrasted", flush=True)

    joint = [r["regimes"]["joint"] for r in out_layers]
    key = "all__auc_lmsys_over_wildchat"
    vals = np.array([r["point"][key] for r in joint])
    above = int(np.sum([r["block_bootstrap_ci95"][key][0] > 0.5 for r in joint]))
    below = int(np.sum([r["block_bootstrap_ci95"][key][1] < 0.5 for r in joint]))
    result = {
        "schema_version": 1,
        "experiment": "provenance_v3_log_corpus_contrast",
        "model_key": args.model_key,
        "question": "does the frozen stated-purpose direction rank Chatbot Arena traffic as "
                    "more evaluation-like than general ChatGPT use, though both carry label 0",
        "positive_direction": "auc_lmsys_over_wildchat > 0.5 means arena traffic scores higher",
        "n_bootstrap": args.n_bootstrap,
        "inference_unit": "complete underlying-text pair, resampled within source corpus",
        "same_block_sample_across_layers_and_regimes": True,
        "caveat": "the two log corpora also differ in topic and register; a positive result is "
                  "consistent with content differences and is not by itself evidence of "
                  "evaluation awareness",
        "summary_joint_regime": {
            "auc_lmsys_over_wildchat_mean_over_layers": float(np.nanmean(vals)),
            "auc_lmsys_over_wildchat_median_over_layers": float(np.nanmedian(vals)),
            "auc_lmsys_over_wildchat_max": float(np.nanmax(vals)),
            "auc_lmsys_over_wildchat_min": float(np.nanmin(vals)),
            "layers_ci_above_half": above,
            "layers_ci_below_half": below,
            "n_layers": len(joint),
        },
        "layers": out_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result["summary_joint_regime"], indent=2))


if __name__ == "__main__":
    main()
