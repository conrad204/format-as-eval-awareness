"""Frozen analysis for the v2 crossed purpose-by-format design.

This script is the executable half of docs/analysis_plan_v2.md. It computes
every pre-registered endpoint (E1-E8) on a v2 rendered dataset plus its
activations, with cluster-bootstrap confidence intervals over payload blocks
(blocks are the independent unit; the renders inside a block are
near-duplicates, so resampling items would understate uncertainty).

Usage:
  python scripts/confirmation_analysis.py items.jsonl activations.npz out.json \
      --layer L [--bootstrap 10000] [--seed 0] [--confirmatory]

The npz must contain X with shape (n_items, n_layers, d) or (n_items, d) and
item_ids aligned to the jsonl row order. --confirmatory only relabels the
output mode: per the analysis plan it may be used exactly once, on the sealed
confirmation bank, at the pre-registered layer. Everything else is
exploratory/development by definition.

Probe recipe (frozen): StandardScaler + LogisticRegression(C=0.1,
max_iter=3000, random_state=0), leave-one-vocabulary-fold-out on the stated
third-person stratum; held-fold models are reused to score the indirect,
second-person, and no-cue strata so those items are always scored by a probe
that never saw their vocabulary fold (no-cue items, having no fold, are
scored by the mean of all five held-fold models).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_probe(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**PROBE).fit(scaler.transform(X), y)
    return lambda Z: model.decision_function(scaler.transform(Z))


def cluster_bootstrap(blocks, stat, *, n_boot, seed):
    """Percentile CI for stat(selected row indices), resampling blocks."""
    blocks = np.asarray(blocks)
    unique = np.unique(blocks)
    index_of = {b: np.flatnonzero(blocks == b) for b in unique}
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(blocks)))
    draws = []
    degenerate = 0
    for _ in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[b] for b in sample])
        value = stat(idx)
        if value is None:
            degenerate += 1
        else:
            draws.append(value)
    draws = np.asarray(draws, dtype=float)
    low, high = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    return {
        "point": None if point is None else float(point),
        "ci95": [float(low), float(high)],
        "n_boot": int(len(draws)),
        "degenerate_resamples": int(degenerate),
    }


def auc_stat(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)

    def stat(idx):
        y = labels[idx]
        if len(np.unique(y)) < 2:
            return None
        return roc_auc_score(y, scores[idx])

    return stat


def auc_delta_stat(scores_a, scores_b, labels):
    a_stat, b_stat = auc_stat(scores_a, labels), auc_stat(scores_b, labels)

    def stat(idx):
        a, b = a_stat(idx), b_stat(idx)
        return None if a is None or b is None else a - b

    return stat


def summarize(scores, labels, blocks, *, n_boot, seed):
    mask = ~np.isnan(scores)
    result = cluster_bootstrap(
        np.asarray(blocks)[mask], auc_stat(scores[mask], np.asarray(labels)[mask]),
        n_boot=n_boot, seed=seed,
    )
    result["n_items"] = int(mask.sum())
    result["n_blocks"] = int(len(np.unique(np.asarray(blocks)[mask])))
    return result


def ngram_oof(texts, labels, fold_ids):
    texts = np.asarray(texts, dtype=str)
    labels, fold_ids = np.asarray(labels, int), np.asarray(fold_ids)
    scores = np.full(len(texts), np.nan)
    for fold in sorted(set(fold_ids)):
        train, test = fold_ids != fold, fold_ids == fold
        vec = TfidfVectorizer(ngram_range=(1, 2))
        model = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
        model.fit(vec.fit_transform(texts[train]), labels[train])
        scores[test] = model.decision_function(vec.transform(texts[test]))
    return scores


def topic_groups(topics, n_groups=5):
    """Deterministic topic partition: sorted topics dealt round-robin."""
    grouping = {}
    for index, topic in enumerate(sorted(set(topics))):
        grouping[topic] = index % n_groups
    return np.asarray([grouping[t] for t in topics])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("items", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.items.read_text().splitlines() if l.strip()]
    npz = np.load(args.activations)
    assert [str(x) for x in npz["item_ids"]] == [r["item_id"] for r in rows], (
        "activations npz item_ids do not match items jsonl order"
    )
    X = npz["X"]
    if X.ndim == 3:
        if args.layer is None:
            raise SystemExit("--layer is required for all-layer activation files")
        X = X[:, args.layer, :]
    X = X.astype(float)

    slot = np.array([r["render_slot"] for r in rows])
    fam = np.array([r["purpose_family_id"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    blk = np.array([r["payload_block_id"] for r in rows])
    locus = np.array([r["cue_locus"] for r in rows])
    topic = np.array([r["topic_id"] for r in rows])
    label = np.array([-1 if r["label"] is None else int(r["label"]) for r in rows])

    stated = slot == "stated"
    boot = dict(n_boot=args.bootstrap, seed=args.seed)

    def held_fold_scores(train_mask, test_mask):
        """Leave-one-fold-out on stated rows; score test rows by their fold."""
        scores = np.full(len(rows), np.nan)
        models = {}
        for fold in FOLDS:
            train = stated & train_mask & (fam != fold)
            models[fold] = fit_probe(X[train], label[train])
            test = test_mask & (fam == fold)
            if test.any():
                scores[test] = models[fold](X[test])
        return scores, models

    # Core held-fold probes (decorrelated: trained on all four cells).
    cue_rows = slot != "no_cue"
    scores, models = held_fold_scores(np.ones(len(rows), bool), cue_rows)

    report = {"endpoints": {}}
    E = report["endpoints"]

    # E1: stated decorrelated AUC vs the lexical n-gram control.
    ngram = np.full(len(rows), np.nan)
    ngram[stated] = ngram_oof(
        [r["text"].partition("\n")[0] for r, s in zip(rows, stated) if s],
        label[stated], fam[stated],
    )
    E["E1_stated_auc"] = summarize(scores * np.where(stated, 1, np.nan), label, blk, **boot)
    E["E1_ngram_auc"] = summarize(ngram, label, blk, **boot)
    m = stated
    E["E1_auc_minus_ngram"] = cluster_bootstrap(
        blk[m], auc_delta_stat(scores[m], ngram[m], label[m]), **boot
    )

    # E2: cue-locus strata of the stated stratum.
    E["E2_locus"] = {
        loc: summarize(
            scores * np.where(stated & (locus == loc), 1, np.nan), label, blk, **boot
        )
        for loc in ("noun_only", "verb_only", "both")
    }

    # E3: cross-format transfer inside the stated stratum.
    for name, train_fmt, test_fmt in (("b2c", "benchmark", "casual"),
                                      ("c2b", "casual", "benchmark")):
        s, _ = held_fold_scores(fmt == train_fmt, stated & (fmt == test_fmt))
        E[f"E3_{name}"] = summarize(s, label, blk, **boot)

    # E4: indirect cues, scored by probes that never saw their fold's words.
    E["E4_indirect_transfer"] = summarize(
        scores * np.where(slot == "indirect", 1, np.nan), label, blk, **boot
    )
    ind = slot == "indirect"
    ind_scores = np.full(len(rows), np.nan)
    for fold in FOLDS:
        train, test = ind & (fam != fold), ind & (fam == fold)
        if train.any() and test.any():
            ind_scores[test] = fit_probe(X[train], label[train])(X[test])
    E["E4_indirect_own_cv"] = summarize(ind_scores, label, blk, **boot)

    # E5: second-person cues under third-person-trained probes, and own-CV.
    sec = slot == "second_person"
    E["E5_second_transfer"] = summarize(
        scores * np.where(sec, 1, np.nan), label, blk, **boot
    )
    E["E5_second_transfer_by_locus"] = {
        loc: summarize(
            scores * np.where(sec & (locus == loc), 1, np.nan), label, blk, **boot
        )
        for loc in ("noun_only", "context")
    }
    sec_scores = np.full(len(rows), np.nan)
    for fold in FOLDS:
        train, test = sec & (fam != fold), sec & (fam == fold)
        if train.any() and test.any():
            sec_scores[test] = fit_probe(X[train], label[train])(X[test])
    E["E5_second_own_cv"] = summarize(sec_scores, label, blk, **boot)

    # E6: no-cue projection - does benchmark formatting alone shift items
    # toward "evaluation" on the purpose direction? Mean of the five
    # held-fold models; positive delta = benchmark reads as more evaluative.
    nocue = slot == "no_cue"
    nocue_scores = np.mean([models[f](X[nocue]) for f in FOLDS], axis=0)
    nc_fmt = (fmt[nocue] == "benchmark").astype(int)
    nc_blk = blk[nocue]
    pooled_sd = float(np.std(nocue_scores, ddof=1))

    def nocue_delta(idx):
        s, f = nocue_scores[idx], nc_fmt[idx]
        if len(np.unique(f)) < 2:
            return None
        return (s[f == 1].mean() - s[f == 0].mean()) / pooled_sd

    E["E6_nocue_format_shift_sd"] = cluster_bootstrap(nc_blk, nocue_delta, **boot)
    E["E6_nocue_format_auc"] = cluster_bootstrap(
        nc_blk, auc_stat(nocue_scores, nc_fmt), **boot
    )

    # E7: payload holdout with frames seen (the split v1 could never make).
    groups = topic_groups(topic)
    e7 = np.full(len(rows), np.nan)
    for g in np.unique(groups):
        train, test = stated & (groups != g), stated & (groups == g)
        e7[test] = fit_probe(X[train], label[train])(X[test])
    E["E7_payload_holdout"] = summarize(e7, label, blk, **boot)

    # E8: doubly held out - new vocabulary fold AND new payload group.
    e8 = np.full(len(rows), np.nan)
    for fold in FOLDS:
        for g in np.unique(groups):
            train = stated & (fam != fold) & (groups != g)
            test = stated & (fam == fold) & (groups == g)
            if train.any() and test.any():
                e8[test] = fit_probe(X[train], label[train])(X[test])
    E["E8_double_holdout"] = summarize(e8, label, blk, **boot)

    # Per-fold detail for the stated stratum (clustered, per item 12).
    E["per_fold_stated"] = {
        fold: summarize(
            scores * np.where(stated & (fam == fold), 1, np.nan), label, blk, **boot
        )
        for fold in FOLDS
    }

    report["config"] = {
        "mode": "confirmatory" if args.confirmatory else "exploratory",
        "items": {"path": str(args.items), "sha256": sha256_file(args.items)},
        "activations": {
            "path": str(args.activations),
            "sha256": sha256_file(args.activations),
        },
        "layer": args.layer,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "probe": PROBE,
        "script_sha256": sha256_file(Path(__file__)),
        "n_items": len(rows),
        "n_blocks": int(len(np.unique(blk))),
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for name in ("E1_stated_auc", "E1_ngram_auc", "E4_indirect_transfer",
                 "E5_second_transfer", "E7_payload_holdout", "E8_double_holdout"):
        entry = E[name]
        print(f"{name}: {entry['point']:.3f} CI {entry['ci95'][0]:.3f}-{entry['ci95'][1]:.3f}"
              f" (n={entry['n_items']}, blocks={entry['n_blocks']})")
    shift = E["E6_nocue_format_shift_sd"]
    print(f"E6_nocue_format_shift: {shift['point']:+.3f} SD "
          f"CI {shift['ci95'][0]:+.3f}..{shift['ci95'][1]:+.3f}")
    print("ANALYSIS DONE", report["config"]["mode"])


if __name__ == "__main__":
    main()
