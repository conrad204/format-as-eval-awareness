"""Same-environment bridge: STATED status -> INFERRED status, inside the hidden channel.

The full320 explicit probe failed to read the latent status, but it was fit in a
different environment on differently rendered items, so "different code" and
"cross-environment shift" were not separable. This test removes the environment
difference. Both row sets are hidden-channel **action** prompts from the same 32
episode families and the same 16 arithmetic items, in the same task world, with
the same prompt type. The only material difference:

  stated rows  (check bank)  the query states the route outright
  latent rows  (main bank)   the query states nothing; the status exists only in
                             the eight prior outcome episodes, mapping reversed

Protocol, fixed in advance:
  train    stated action rows, closed vs active, from 31 episode families
  test     latent action rows of the held-out 32nd family (map0/map1 only)
  probe    StandardScaler + LogisticRegression(C=0.1), fit on training rows only
  layer    given on the command line (8 for 8B, 18 for 70B); never selected
  sign     from the stated labels; never flipped using latent outcomes

Every latent row is therefore scored by a probe that saw neither its episode
family nor any inferred-status example. Within-domain stated OOF AUC is reported
in the same loop as a sanity bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROBE = dict(C=0.1, max_iter=3000, random_state=0)
LATENT_CONDITIONS = ("map0_K", "map0_M", "map1_K", "map1_M")
STATED_CONDITIONS = ("explicit_closed", "explicit_active")


def stored_index(meta: dict, layer: int) -> int:
    """Map an absolute layer index to its position in the stored tensor.

    Local extractions store every layer, so the index is the layer. Remote/partial
    extractions (e.g. the 405B runner) store a subset and record it in the meta as
    `stored_layers`; the requested layer must be one of them.
    """
    stored = meta.get("stored_layers")
    if not stored:
        return layer
    if layer not in stored:
        raise RuntimeError(f"layer {layer} was not stored; available: {stored}")
    return list(stored).index(layer)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(activations: Path, conditions: tuple[str, ...], layer: int, meta: dict):
    with np.load(activations, allow_pickle=False) as z:
        cond = z["condition"].astype(str)
        keep = np.isin(cond, conditions)
        out = {
            "ids": z["item_ids"].astype(str)[keep],
            "family": z["episode_family"].astype(str)[keep],
            "item": z["arithmetic_item"].astype(str)[keep],
            "condition": cond[keep],
            "status": z["query_status"].astype(str)[keep],
            "X": np.ascontiguousarray(z["X"][keep][:, stored_index(meta, layer), :]).astype(np.float32),
        }
    out["y"] = (out["status"] == "closed").astype(int)
    return out


def matched_units(scores, family, item, cond) -> tuple[np.ndarray, np.ndarray]:
    index = {(f, i, c): s for f, i, c, s in zip(family, item, cond, scores)}
    units, owner = [], []
    for f in sorted(set(family)):
        for i in sorted(set(item)):
            cells = {c: index[(f, i, c)] for c in LATENT_CONDITIONS}
            units.append(0.5 * ((cells["map0_K"] - cells["map0_M"])
                                + (cells["map1_M"] - cells["map1_K"])))
            owner.append(f)
    return np.array(units), np.array(owner)


def family_bootstrap(per_family, n, rng):
    draws = per_family[rng.integers(0, len(per_family), (n, len(per_family)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stated-activations", type=Path, required=True)
    ap.add_argument("--stated-meta", type=Path, required=True)
    ap.add_argument("--latent-activations", type=Path, required=True)
    ap.add_argument("--latent-meta", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--reverse", action="store_true",
                    help="train on the LATENT rows and score the STATED rows instead; tests "
                         "whether the inferred-status code contains the stated-status code")
    args = ap.parse_args()

    stated_meta = json.loads(args.stated_meta.read_text(encoding="utf-8"))
    latent_meta = json.loads(args.latent_meta.read_text(encoding="utf-8"))
    if stated_meta["model"] != latent_meta["model"]:
        raise RuntimeError("stated and latent activations come from different models")
    for meta, name in ((stated_meta, "stated"), (latent_meta, "latent")):
        if meta["position"] != "last_prompt_token_after_chat_template_before_generation":
            raise RuntimeError(f"{name} activations use an unexpected token position")

    stated = load(args.stated_activations, STATED_CONDITIONS, args.layer, stated_meta)
    latent = load(args.latent_activations, LATENT_CONDITIONS, args.layer, latent_meta)
    families = sorted(set(stated["family"]) & set(latent["family"]))
    if len(families) != 32:
        raise RuntimeError(f"expected 32 shared episode families, found {len(families)}")
    if len(stated["ids"]) != 1024 or len(latent["ids"]) != 2048:
        raise RuntimeError(f"unexpected row counts {len(stated['ids'])}/{len(latent['ids'])}")

    rng = np.random.default_rng(args.seed)
    # Forward: train stated, score latent. Reverse: train latent, score stated.
    src, dst = (latent, stated) if args.reverse else (stated, latent)
    transfer = np.full(len(dst["y"]), np.nan)
    source_oof = np.full(len(src["y"]), np.nan)
    for held in families:
        tr = src["family"] != held
        scaler = StandardScaler().fit(src["X"][tr])
        model = LogisticRegression(**PROBE).fit(scaler.transform(src["X"][tr]), src["y"][tr])
        te_src = ~tr
        source_oof[te_src] = model.decision_function(scaler.transform(src["X"][te_src]))
        te_dst = dst["family"] == held
        transfer[te_dst] = model.decision_function(scaler.transform(dst["X"][te_dst]))
    if np.isnan(transfer).any() or np.isnan(source_oof).any():
        raise RuntimeError("a row was never scored by a held-out-family probe")

    y = dst["y"]
    fam = dst["family"]
    fam_auc = np.array([roc_auc_score(y[fam == f], transfer[fam == f]) for f in families])
    if args.reverse:
        # Stated rows carry no mapping; the matched unit is (family, item) closed - active.
        idx = {(f, i, c): s for f, i, c, s in zip(fam, dst["item"], dst["condition"], transfer)}
        units = np.array([idx[(f, i, "explicit_closed")] - idx[(f, i, "explicit_active")]
                          for f in families for i in sorted(set(dst["item"]))])
        owner = np.array([f for f in families for _ in sorted(set(dst["item"]))])
        map0 = map1 = np.ones(len(y), dtype=bool)
    else:
        map0 = np.char.startswith(dst["condition"], "map0")
        map1 = ~map0
        units, owner = matched_units(transfer, fam, dst["item"], dst["condition"])
    per_family_unit = np.array([units[owner == f].mean() for f in families])
    source_fam_auc = np.array([roc_auc_score(src["y"][src["family"] == f],
                                             source_oof[src["family"] == f]) for f in families])

    result = {
        "schema_version": 1,
        "experiment": ("latent-status probe scored on the STATED action rows" if args.reverse
                       else "stated-status probe scored on the INFERRED-status action rows")
                      + " of held-out episode families, same hidden-channel environment",
        "direction": "latent_to_stated" if args.reverse else "stated_to_latent",
        "model": latent_meta["model"], "model_key": latent_meta["model_key"],
        "layer": args.layer, "layer_selected_from_latent_outcomes": False,
        "position": latent_meta["position"], "layer_semantics": latent_meta["layer_semantics"],
        "stated": {"activations": str(args.stated_activations),
                   "sha256": sha256(args.stated_activations),
                   "items_sha256": stated_meta["items_sha256"],
                   "conditions": list(STATED_CONDITIONS), "n_rows": int(len(stated["ids"]))},
        "latent": {"activations": str(args.latent_activations),
                   "items_sha256": latent_meta["items_sha256"],
                   "conditions": list(LATENT_CONDITIONS), "n_rows": int(len(latent["ids"]))},
        "environment_controls": "same 32 episode families, same 16 arithmetic items, same task "
                                "world, same action prompt type; only stated vs inferred differs",
        "probe": PROBE, "cross_validation": "leave-one-episode-family-out, 32 folds",
        "within_domain_source": {
            "pooled_auc": float(roc_auc_score(src["y"], source_oof)),
            "family_mean_auc": float(source_fam_auc.mean()),
            "family_ci95": family_bootstrap(source_fam_auc, args.n_bootstrap, rng),
            "families_above_half": int((source_fam_auc > 0.5).sum()),
        },
        "transfer": {
            "pooled_auc": float(roc_auc_score(y, transfer)),
            "family_mean_auc": float(fam_auc.mean()),
            "family_ci95": family_bootstrap(fam_auc, args.n_bootstrap, rng),
            "families_above_half": int((fam_auc > 0.5).sum()),
            "n_families": len(families),
            "auc_map0": float(roc_auc_score(y[map0], transfer[map0])),
            "auc_map1": float(roc_auc_score(y[map1], transfer[map1])),
            "matched_closed_minus_active": float(units.mean()),
            "matched_family_ci95": family_bootstrap(per_family_unit, args.n_bootstrap, rng),
            "per_family_auc": {f: float(a) for f, a in zip(families, fam_auc)},
        },
        "bootstrap": {"unit": "episode family", "n_bootstrap": args.n_bootstrap, "seed": args.seed},
        "evidence_class": "exploratory; environment held fixed, no selection on latent outcomes",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    w, t = result["within_domain_source"], result["transfer"]
    print(f"within-domain source: pooled {w['pooled_auc']:.4f} family-mean {w['family_mean_auc']:.4f} "
          f"CI {[round(x, 4) for x in w['family_ci95']]}")
    print(f"{result['direction']}: pooled {t['pooled_auc']:.4f} family-mean {t['family_mean_auc']:.4f} "
          f"CI {[round(x, 4) for x in t['family_ci95']]} | families>.5 "
          f"{t['families_above_half']}/{t['n_families']} | map0 {t['auc_map0']:.4f} "
          f"map1 {t['auc_map1']:.4f} | matched {t['matched_closed_minus_active']:+.4f} "
          f"{[round(x, 3) for x in t['matched_family_ci95']]}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
