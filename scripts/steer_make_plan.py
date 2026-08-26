"""Build the steering plan (conditions, item subsets, doses) for one model.

Every direction is dosed at ONE common unit, Δ_fmt — the format probe's own
training-contrast gap in projection (benchmark-mean minus casual-mean on the
stated stratum, raw residual units). Additive dose k∈{-2,-1,+1,+2} means
α = k·Δ_fmt, so fmt, the purpose directions, the difference-of-means
directions, and the random controls are all perturbed at *identical norm* and
are directly comparable. This is the fix for an earlier design where random
directions were dosed at a larger gap and so dominated the readout for reasons
of norm, not direction. Each direction's own gap is still recorded for
interpretation ("+1Δ_fmt of pur_conf" vs "the eval/deploy separation").

Ablation ("_abl") is mean-ablation at the intervened positions: the component
along v is set to the no-cue mean projection m (norm-independent, so no dose).

Random control: rand0..rand4 (five seeds) at ±1,±2 · Δ_fmt define the null
band. A real direction "moves behavior" only if its effect exceeds this band.

Positions: "tail" = probed last prompt token and every later position (prefill
/ generated). "@all" (fmt only, logit) = every non-BOS prompt position too.

Item subsets (seeded, block-stratified over the 320 payload blocks):
  logit_items  4 per block (2 benchmark + 2 casual no-cue), 1280 (or --all-nocue)
  gen_items    2 per block (1 + 1), 640; subset of logit_items

Usage:
  steer_make_plan.py <model_key> <items.jsonl> <directions.json> <directions.npz> <out_plan.json>
                     [--gen-conditions core|extended] [--gen-items-per-block N]
                     [--logit-conditions core|extended] [--all-nocue] [--max-new-tokens N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steer_common import MODELS, PREFILLS, load_items, sha256_file  # noqa: E402

SEED = 0
DOSES = (-2.0, -1.0, 1.0, 2.0)
ADD_DIRS = ("fmt", "pur_dec", "pur_conf", "fmt_dm", "pur_dm", "rand0", "rand1", "rand2", "rand3", "rand4")
ABL_DIRS = ("fmt", "pur_dec", "pur_conf", "rand0")
RANDOMS = ("rand0", "rand1", "rand2", "rand3", "rand4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key", choices=sorted(MODELS))
    ap.add_argument("items")
    ap.add_argument("directions_json")
    ap.add_argument("directions_npz")
    ap.add_argument("out")
    ap.add_argument("--gen-conditions", choices=("core", "extended"), default="core")
    ap.add_argument("--logit-conditions", choices=("lean", "core", "extended"), default="lean")
    ap.add_argument("--gen-items-per-block", type=int, default=2)
    ap.add_argument("--logit-items-per-block", type=int, default=2)
    ap.add_argument("--all-nocue", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    spec = MODELS[args.model_key]
    info = json.loads(Path(args.directions_json).read_text())
    assert int(info["layer"]) == spec["layer"], (info["layer"], spec["layer"])
    D = info["directions"]
    for k in ADD_DIRS + ABL_DIRS:
        assert k in D, f"direction {k} missing from {args.directions_json}"

    unit = float(D["fmt"]["delta"])                 # Δ_fmt, the common dose unit
    assert unit > 0
    own_delta = {k: (D[k].get("delta") if D[k].get("delta") is not None else None) for k in ADD_DIRS}
    m_nocue = {k: float(D[k]["nocue_mean"]) for k in set(ADD_DIRS) | set(ABL_DIRS)}

    conditions = [dict(cid="base", dir=None, mode="none", alpha=0.0, m=0.0, positions="tail", k=0.0)]
    for k in ADD_DIRS:
        for kk in DOSES:
            conditions.append(dict(cid=f"{k}{kk:+g}", dir=k, mode="add", alpha=kk * unit, m=0.0,
                                   positions="tail", k=kk))
    for k in ABL_DIRS:
        conditions.append(dict(cid=f"{k}_abl", dir=k, mode="ablate", alpha=0.0, m=m_nocue[k], positions="tail"))
    for kk in (-2.0, 2.0):
        conditions.append(dict(cid=f"fmt{kk:+g}@all", dir="fmt", mode="add", alpha=kk * unit, m=0.0,
                               positions="all", k=kk))
    conditions.append(dict(cid="fmt_abl@all", dir="fmt", mode="ablate", alpha=0.0, m=m_nocue["fmt"], positions="all"))
    cids = [c["cid"] for c in conditions]
    assert len(set(cids)) == len(cids)

    logit_lean = (["base"]
                  + [f"{k}{kk:+g}" for k in ("fmt", "pur_dec", "pur_conf") for kk in DOSES]   # dose-response
                  + [f"{k}{kk:+g}" for k in ("fmt_dm", "pur_dm") for kk in (-2.0, 2.0)]        # diff-means endpoints
                  + [f"{k}{kk:+g}" for k in RANDOMS for kk in (-2.0, 2.0)]                     # 5-seed null band at +/-2
                  + [f"{k}_abl" for k in ABL_DIRS] + ["fmt-2@all", "fmt+2@all", "fmt_abl@all"])
    logit_core = (["base"] + [f"{k}{kk:+g}" for k in ("fmt", "pur_dec", "pur_conf", "rand0", "rand1")
                              for kk in DOSES]
                  + [f"{k}_abl" for k in ABL_DIRS] + ["fmt-2@all", "fmt+2@all", "fmt_abl@all"])
    logit_conditions = {"extended": cids, "core": logit_core, "lean": logit_lean}[args.logit_conditions]
    gen_core = ["base", "fmt-2", "fmt-1", "fmt+1", "fmt+2", "fmt_abl",
                "pur_dec-2", "pur_dec+2", "pur_dec_abl", "pur_conf-2", "pur_conf+2",
                "rand0-2", "rand0+2", "rand1-2", "rand1+2"]
    gen_extended = gen_core + ["pur_dec-1", "pur_dec+1", "fmt_dm-2", "fmt_dm+2", "pur_dm-2", "pur_dm+2",
                               "rand2-2", "rand2+2", "fmt-2@all", "fmt+2@all"]
    gen_conditions = gen_core if args.gen_conditions == "core" else gen_extended
    assert all(c in cids for c in gen_conditions) and all(c in cids for c in logit_conditions)

    rows = load_items(Path(args.items))
    nc = [r for r in rows if r["render_slot"] == "no_cue"]
    by_block: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"benchmark": [], "casual": []})
    for r in nc:
        by_block[r["payload_block_id"]][r["format"]].append(r["item_id"])
    rng = np.random.default_rng(SEED)
    gen_items, logit_items = [], []
    per_fmt_gen = args.gen_items_per_block // 2
    assert per_fmt_gen >= 1
    if args.logit_items_per_block == 1:
        # one no-cue item per block, format alternating by block index, so all
        # 320 blocks are covered with 160 benchmark + 160 casual items (half the
        # cost of 2/block; used for the expensive 405B). gen still 2/block.
        for bi, block in enumerate(sorted(by_block)):
            f = "benchmark" if bi % 2 == 0 else "casual"
            pool = sorted(by_block[block][f]) or sorted(by_block[block]["casual" if f == "benchmark" else "benchmark"])
            logit_items.append(pool[rng.permutation(len(pool))[0]])
            for ff in ("benchmark", "casual"):
                gp = sorted(by_block[block][ff])
                gen_items.extend([gp[i] for i in rng.permutation(len(gp))[:per_fmt_gen]])
    else:
        per_fmt_logit = args.logit_items_per_block // 2
        assert per_fmt_logit >= per_fmt_gen
        for block in sorted(by_block):
            for f in ("benchmark", "casual"):
                pool = sorted(by_block[block][f])
                pick = [pool[i] for i in rng.permutation(len(pool))[:per_fmt_logit]]
                logit_items.extend(pick)
                gen_items.extend(pick[:per_fmt_gen])
    if args.all_nocue:
        logit_items = [r["item_id"] for r in nc]

    plan = {
        "model_key": args.model_key, "hf_id": spec["hf_id"], "layer": spec["layer"], "hidden": spec["hidden"],
        "seed": SEED, "dose_unit": "delta_fmt", "dose_unit_value": unit, "doses": list(DOSES),
        "own_delta": own_delta, "nocue_mean": m_nocue,
        "residual_norm_nocue_mean": info["residual_norm"]["mean_nocue"],
        "items_sha256": sha256_file(Path(args.items)),
        "directions_npz": str(args.directions_npz), "directions_npz_sha256": sha256_file(Path(args.directions_npz)),
        "directions_json_sha256": sha256_file(Path(args.directions_json)),
        "conditions": conditions, "logit_conditions": logit_conditions, "gen_conditions": gen_conditions,
        "randoms": list(RANDOMS),
        "prefills": {k: {"text": v[0], "eval_letter": v[1], "genuine_letter": v[2]} for k, v in PREFILLS.items()},
        "gen": {"max_new_tokens": args.max_new_tokens, "do_sample": False},
        "gen_items": gen_items, "logit_items": logit_items,
        "positions_semantics": {
            "tail": "probed last prompt token and every later position (prefill / generated)",
            "all": "every non-BOS, non-pad position of the prompt plus every later position"},
    }
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    C = info["cosines"]
    print(f"{args.model_key}: L{spec['layer']}  unit Δ_fmt={unit:.3f}  |h| nocue={info['residual_norm']['mean_nocue']:.2f}  "
          f"own Δ: fmt={own_delta['fmt']:.3f} pur_dec={own_delta['pur_dec']:.3f} pur_conf={own_delta['pur_conf']:.3f}")
    print("  cos fmt·pur_dec=%.3f fmt·pur_conf=%.3f pur_dec·pur_conf=%.3f | fold-cos min fmt=%.3f pur_dec=%.3f" % (
        C["fmt"]["pur_dec"], C["fmt"]["pur_conf"], C["pur_dec"]["pur_conf"],
        info["fold_cosines"]["fmt"]["min"], info["fold_cosines"]["pur_dec"]["min"]))
    print(f"  {len(conditions)} conditions ({len(logit_conditions)} logit, {len(gen_conditions)} gen); "
          f"{len(gen_items)} gen items, {len(logit_items)} logit items")
    print(f"  logit rows {len(logit_items) * len(logit_conditions) * len(PREFILLS)}, "
          f"gen rows {len(gen_items) * len(gen_conditions)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
