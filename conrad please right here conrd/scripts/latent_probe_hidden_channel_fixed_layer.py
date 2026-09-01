"""Fresh latent-status probe on hidden-channel action rows at ONE fixed layer.

Companion to `explicit_ea_to_hidden_channel.py`. That script asks whether the
frozen explicit stated-purpose direction reads the latent status; this one asks
whether the latent status is linearly decodable at all from the same rows, so the
two numbers are directly comparable within model and layer.

Protocol, fixed in advance and identical across models:
  layer          given on the command line (8 for 8B, 18 for 70B), never selected
  rows           action prompts, conditions map0_K/map0_M/map1_K/map1_M
  target         closed (evaluation-like) vs active (deployment-like)
  cross-validation  leave ONE ENTIRE episode family out, 32 folds
  probe          StandardScaler + LogisticRegression(C=0.1, max_iter=3000), fit on
                 training families only, inside each fold
  null           status permuted within each family x mapping block, which
                 preserves the channel-letter and mapping marginals exactly

Also writes the direction fitted on ALL families at that layer, for reverse
transfer and geometry. That direction is explicitly secondary: it uses every
hidden-channel row, so it has no held-out status of its own.
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
COHERENT = ("map0_K", "map0_M", "map1_K", "map1_M")
STATUS_OF_CONDITION = {"map0_K": "closed", "map0_M": "active",
                       "map1_K": "active", "map1_M": "closed"}


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


def fit_score(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(**PROBE).fit(scaler.transform(Xtr), ytr)
    return model.decision_function(scaler.transform(Xte))


def leave_one_family_out(X, y, fam, families) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for held in families:
        te = fam == held
        oof[te] = fit_score(X[~te], y[~te], X[te])
    if np.isnan(oof).any():
        raise RuntimeError("some rows never appeared in a held-out family")
    return oof


def matched_units(scores, fam, item, cond) -> tuple[np.ndarray, np.ndarray]:
    """0.5 * [(map0_K - map0_M) + (map1_M - map1_K)] per (family, item)."""
    index = {(f, i, c): s for f, i, c, s in zip(fam, item, cond, scores)}
    units, owner = [], []
    for f in sorted(set(fam)):
        for i in sorted(set(item)):
            cells = {c: index[(f, i, c)] for c in COHERENT}
            units.append(0.5 * ((cells["map0_K"] - cells["map0_M"])
                                + (cells["map1_M"] - cells["map1_K"])))
            owner.append(f)
    return np.array(units), np.array(owner)


def family_bootstrap(per_family, n, rng):
    draws = per_family[rng.integers(0, len(per_family), (n, len(per_family)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def readout(oof, y, fam, item, cond, families, map0, map1, n_boot, rng) -> dict:
    fam_auc = np.array([roc_auc_score(y[fam == f], oof[fam == f]) for f in families])
    units, owner = matched_units(oof, fam, item, cond)
    per_family_unit = np.array([units[owner == f].mean() for f in families])
    return {
        "pooled_auc": float(roc_auc_score(y, oof)),
        "family_mean_auc": float(fam_auc.mean()),
        "family_ci95": family_bootstrap(fam_auc, n_boot, rng),
        "families_above_half": int((fam_auc > 0.5).sum()),
        "n_families": len(families),
        "auc_map0": float(roc_auc_score(y[map0], oof[map0])),
        "auc_map1": float(roc_auc_score(y[map1], oof[map1])),
        "matched_closed_minus_active": float(units.mean()),
        "matched_family_ci95": family_bootstrap(per_family_unit, n_boot, rng),
        "per_family_auc": {f: float(a) for f, a in zip(families, fam_auc)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--bank-items", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True, help="fixed, never selected")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--direction-out", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--n-permutation-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--skip-activation-hash", action="store_true")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    act_sha = None if args.skip_activation_hash else sha256(args.activations)
    if act_sha and act_sha != meta["activations_sha256"]:
        raise RuntimeError("activation hash mismatch")
    if sha256(args.bank_items) != meta["items_sha256"]:
        raise RuntimeError("hidden-channel bank hash mismatch against activation metadata")
    frozen = {}
    for line in args.bank_items.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["kind"] == "action" and row["condition"] in COHERENT:
            frozen[f"{row['episode_family']}::{row['arithmetic_item']}::{row['condition']}"] = row

    with np.load(args.activations, allow_pickle=False) as z:
        cond_all = z["condition"].astype(str)
        keep = np.isin(cond_all, COHERENT)
        ids = z["item_ids"].astype(str)[keep]
        fam = z["episode_family"].astype(str)[keep]
        item = z["arithmetic_item"].astype(str)[keep]
        cond = cond_all[keep]
        token = z["query_token"].astype(str)[keep]
        status = z["query_status"].astype(str)[keep]
        X = np.ascontiguousarray(z["X"][keep][:, stored_index(meta, args.layer), :]).astype(np.float32)

    families = sorted(set(fam))
    items = sorted(set(item))
    if (len(families), len(items)) != (32, 16) or len(ids) != 32 * 16 * 4:
        raise RuntimeError(f"unexpected eligible row set: {len(ids)} rows")
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate row ids")
    for row_id, f, i, c, s, t in zip(ids, fam, item, cond, status, token):
        if row_id != f"{f}::{i}::{c}" or row_id not in frozen:
            raise RuntimeError(f"row alignment failure at {row_id}")
        if frozen[row_id]["query_status"] != s or frozen[row_id]["query_token"] != t:
            raise RuntimeError(f"row {row_id} disagrees with the frozen bank")
        if STATUS_OF_CONDITION[c] != s:
            raise RuntimeError(f"condition/status mismatch at {row_id}")

    y = (status == "closed").astype(int)
    y_letter = (token == "K").astype(int)
    map0, map1 = np.char.startswith(cond, "map0"), np.char.startswith(cond, "map1")
    rng = np.random.default_rng(args.seed)

    oof = leave_one_family_out(X, y, fam, families)
    status_read = readout(oof, y, fam, item, cond, families, map0, map1, args.n_bootstrap, rng)
    print(f"status L{args.layer}: pooled {status_read['pooled_auc']:.4f} family-mean "
          f"{status_read['family_mean_auc']:.4f} CI {[round(x, 4) for x in status_read['family_ci95']]}",
          flush=True)

    # Positive control: the channel letter is trivially visible in the prompt.
    letter_oof = leave_one_family_out(X, y_letter, fam, families)
    letter_fam_auc = np.array([roc_auc_score(y_letter[fam == f], letter_oof[fam == f]) for f in families])
    print(f"letter control: pooled {roc_auc_score(y_letter, letter_oof):.4f}", flush=True)

    # Null: permute status inside each family x mapping block. Letter and mapping
    # marginals are preserved, so neither can become predictive of the null labels.
    null_pooled, null_family = [], []
    for repeat in range(args.n_permutation_repeats):
        y_perm = y.copy()
        for f in families:
            for sel in (map0, ~map0):
                idx = np.where((fam == f) & sel)[0]
                y_perm[idx] = rng.permutation(y[idx])
        perm_oof = leave_one_family_out(X, y_perm, fam, families)
        null_pooled.append(float(roc_auc_score(y_perm, perm_oof)))
        null_family.append(float(np.mean([roc_auc_score(y_perm[fam == f], perm_oof[fam == f])
                                          for f in families])))
        print(f"null repeat {repeat}: pooled {null_pooled[-1]:.4f} family-mean {null_family[-1]:.4f}",
              flush=True)

    result = {
        "schema_version": 1,
        "experiment": "fresh leave-one-episode-family-out latent-status probe, fixed layer",
        "model_key": meta["model_key"], "model": meta["model"], "precision": meta["precision"],
        "layer": args.layer, "layer_selected_from_performance": False,
        "position": meta["position"], "layer_semantics": meta["layer_semantics"],
        "activations": str(args.activations), "activations_sha256": act_sha,
        "meta_activations_sha256": meta["activations_sha256"], "items_sha256": meta["items_sha256"],
        "n_rows": int(len(ids)), "conditions": list(COHERENT),
        "probe": PROBE, "cross_validation": "leave-one-episode-family-out, 32 folds",
        "positive_class": "closed", "negative_class": "active",
        "status": status_read,
        "letter_control": {"pooled_auc": float(roc_auc_score(y_letter, letter_oof)),
                           "family_mean_auc": float(letter_fam_auc.mean())},
        "permutation_null": {"n_repeats": args.n_permutation_repeats,
                             "pooled_auc": null_pooled, "family_mean_auc": null_family,
                             "mean_family_mean_auc": float(np.mean(null_family)),
                             "convention": "status permuted within family x mapping block; "
                                           "letter and mapping marginals preserved"},
        "bootstrap": {"n_bootstrap": args.n_bootstrap, "unit": "episode family", "seed": args.seed},
        "evidence_class": "exploratory; fixed pre-specified layer, no selection",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)

    if args.direction_out:
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
        np.savez_compressed(
            args.direction_out,
            coef=model.coef_[0].astype(np.float32),
            intercept=np.asarray(model.intercept_[0], dtype=np.float32),
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
            layer=np.asarray(args.layer), model_key=np.asarray(meta["model_key"]),
            orientation=np.asarray("positive = closed (evaluation-like) hidden-channel status"),
            note=np.asarray("fit on ALL hidden-channel families; secondary use only"))
        print("wrote", args.direction_out)


if __name__ == "__main__":
    main()
