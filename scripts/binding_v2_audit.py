"""Audit the frozen binding_v2 bank before any activation exists.

Mirrors scripts/binding_v1_audit.py's contract and adds the checks that the two
new hardening axes require: template balance and template-held-out leakage
(front 1), and arm/hop balance plus per-arm and per-hop leakage (front 2).

Leakage CV holds out an ENTIRE relational family, and separately an ENTIRE
surface template, so a baseline cannot memorize either.

Usage: python scripts/binding_v2_audit.py
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROBE = dict(max_iter=3000, random_state=0)


def grouped_auc(texts, y, groups, vectorizer_kwargs) -> dict:
    """Leave-one-group-out AUC for a text baseline. Vectorizer is refit inside
    every fold on training rows only."""
    texts, y, groups = np.asarray(texts), np.asarray(y), np.asarray(groups)
    per_group = {}
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        vec = TfidfVectorizer(**vectorizer_kwargs)
        Xtr = vec.fit_transform(texts[tr])
        model = LogisticRegression(**PROBE).fit(Xtr, y[tr])
        per_group[str(g)] = float(roc_auc_score(y[te], model.decision_function(vec.transform(texts[te]))))
    vals = np.array(list(per_group.values()))
    return {"per_group_auc": per_group, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=ROOT / "results" / "binding_v2_frozen")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (args.bank / "items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest = json.loads((args.bank / "freeze_manifest.json").read_text())

    text = [r["text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    fam = np.array([r["binding_family"] for r in rows])
    tmpl = np.array([r["template_id"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    arm = np.array([r["binding_arm"] for r in rows])
    hop = np.array([r["binding_hop"] for r in rows])

    checks: dict = {}
    checks["1_counts"] = {
        "n_items": len(rows), "n_blocks": len({r["binding_block_id"] for r in rows}),
        "n_families": len(set(fam)), "n_templates": len(set(tmpl)),
        "label_counts": {str(k): int(v) for k, v in collections.Counter(y.tolist()).items()},
        "unique_item_ids": len({r["item_id"] for r in rows}),
    }
    checks["2_label_is_the_interaction"] = {
        "holds_for_all_items": all(
            r["label"] == int(r["binding_mapping"] * r["binding_selected"] == 1) for r in rows),
        "mapping_alone_label_mean": {str(m): float(y[np.array([r["binding_mapping"] for r in rows]) == m].mean())
                                     for m in (1, -1)},
        "selected_alone_label_mean": {str(s): float(y[np.array([r["binding_selected"] for r in rows]) == s].mean())
                                      for s in (1, -1)},
    }
    # Every nuisance factor must be label-balanced.
    balance = {}
    for name, arr in (("format", fmt), ("template", tmpl), ("arm", arm), ("hop", hop),
                      ("family", fam),
                      ("target_first", np.array([r["binding_target_first"] for r in rows])),
                      ("attr_flip", np.array([r["binding_attr_flip"] for r in rows])),
                      ("bind_order", np.array([r["binding_bind_order"] for r in rows])),
                      ("current_route_is_a", np.array([r["current_route_name"] == r["route_a_name"] for r in rows]))):
        balance[name] = {str(v): {"n": int((arr == v).sum()), "label_mean": float(y[arr == v].mean())}
                         for v in sorted(set(arr.tolist()))}
    checks["3_nuisance_label_balance"] = balance
    checks["4_template_balance"] = {
        "blocks_per_template_combination": sorted(collections.Counter(
            (m["benchmark_template"], m["casual_template"])
            for m in manifest["blocks"].values()).values()),
        "items_per_template": {str(k): int(v) for k, v in sorted(collections.Counter(tmpl.tolist()).items())},
        "template_orthogonal_to_label": all(
            abs(y[tmpl == t].mean() - 0.5) < 1e-12 for t in set(tmpl)),
    }

    # The core guarantee: within every (block, format), positives and negatives
    # carry identical bag-of-words multisets and identical length multisets.
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["binding_block_id"], r["format"])].append(r)
    bow_bad = len_bad = 0
    for group in cells.values():
        pos = sorted(tuple(sorted(collections.Counter(r["text"].lower().split()).items()))
                     for r in group if r["label"] == 1)
        neg = sorted(tuple(sorted(collections.Counter(r["text"].lower().split()).items()))
                     for r in group if r["label"] == 0)
        bow_bad += pos != neg
        if sorted(len(r["text"]) for r in group if r["label"] == 1) != \
           sorted(len(r["text"]) for r in group if r["label"] == 0):
            len_bad += 1
    L = np.array([len(t) for t in text], dtype=float)
    checks["5_within_block_invariance"] = {
        "blocks_checked": len(cells),
        "blocks_with_bagofwords_label_difference": int(bow_bad),
        "blocks_with_length_label_difference": int(len_bad),
        "mean_length_label1": float(L[y == 1].mean()),
        "mean_length_label0": float(L[y == 0].mean()),
        "mean_length_by_hop": {str(h): float(L[hop == h].mean()) for h in sorted(set(hop.tolist()))},
        "mean_length_by_arm": {str(a): float(L[arm == a].mean()) for a in sorted(set(arm.tolist()))},
    }

    # Text leakage baselines, held out by family and by template.
    word = dict(analyzer="word", ngram_range=(1, 2), min_df=2)
    char = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    leak = {
        "word_1_2gram_leave_one_family_out": grouped_auc(text, y, fam, word),
        "char_3_5gram_leave_one_family_out": grouped_auc(text, y, fam, char),
        "word_1_2gram_leave_one_template_out": grouped_auc(text, y, tmpl, word),
    }
    for a in sorted(set(arm.tolist())):
        m = arm == a
        leak[f"word_1_2gram_leave_one_family_out_arm_{a}"] = grouped_auc(
            np.asarray(text)[m], y[m], fam[m], word)
    for h in sorted(set(hop.tolist())):
        m = hop == h
        leak[f"word_1_2gram_leave_one_family_out_hop{h}"] = grouped_auc(
            np.asarray(text)[m], y[m], fam[m], word)
    checks["6_text_leakage_baselines"] = leak

    # Metadata-only baselines: each single factor must be uninformative. The
    # label is DEFINED as the interaction, so mapping and selection are jointly
    # predictive by construction; that is the design, not leakage.
    meta = {}
    for name, arr in (("mapping_only", np.array([r["binding_mapping"] for r in rows])),
                      ("selection_only", np.array([r["binding_selected"] for r in rows])),
                      ("format_only", fmt), ("template_only", tmpl), ("arm_only", arm),
                      ("hop_only", hop),
                      ("current_route_identity", np.array([r["current_route_name"] for r in rows])),
                      ("payload_identity", np.array([r["payload_block_id"] for r in rows]))):
        codes = {v: i for i, v in enumerate(sorted(set(arr.tolist())))}
        X = np.array([[codes[v]] for v in arr.tolist()], dtype=float)
        X = (X == np.arange(len(codes))).astype(float) if len(codes) > 2 else X
        per = {}
        for g in sorted(set(fam)):
            tr, te = fam != g, fam == g
            if len(set(y[tr])) < 2:
                continue
            model = LogisticRegression(**PROBE).fit(X[tr], y[tr])
            per[str(g)] = float(roc_auc_score(y[te], model.decision_function(X[te])))
        vals = np.array(list(per.values()))
        meta[name] = {"mean_auc": float(vals.mean()), "min_auc": float(vals.min()),
                      "max_auc": float(vals.max())}
    checks["7_metadata_only_baselines"] = meta

    # Neutral arm must contain no consequence vocabulary at all.
    banned = set(manifest["banned_consequence_words_neutral_arm"])
    neutral_hits = collections.Counter()
    for r in rows:
        if r["binding_arm"] == "neutral":
            setup = r["text"].split("\n\n")[0].lower()
            for w in banned:
                if f" {w} " in f" {setup} " or f" {w}." in setup:
                    neutral_hits[w] += 1
    checks["8_neutral_arm_consequence_vocabulary"] = {
        "hits": dict(neutral_hits), "total_hits": int(sum(neutral_hits.values())),
        "note": "counted over the Setup scaffolding only; the shared payload text is identical "
                "across arms and carries no role semantics",
    }

    report = {
        "schema_version": 1,
        "experiment": "binding_v2",
        "audited_before_activations": True,
        "bank": args.bank.relative_to(ROOT).as_posix(),
        "items_sha256": manifest["items_sha256"],
        "freeze_manifest_sha256_of_spec": manifest["manifest_sha256_of_spec"],
        "inference_unit": "binding_family",
        "text_baseline_cv": "leave-one-entire-relational-family-out and leave-one-entire-template-out",
        "checks": checks,
    }
    out = args.bank / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"items {checks['1_counts']['n_items']}  blocks {checks['1_counts']['n_blocks']}  "
          f"templates {checks['1_counts']['n_templates']}")
    print(f"label is the interaction: {checks['2_label_is_the_interaction']['holds_for_all_items']}")
    inv = checks["5_within_block_invariance"]
    print(f"within-block bag-of-words label differences: {inv['blocks_with_bagofwords_label_difference']}"
          f" / {inv['blocks_checked']}")
    print(f"within-block length label differences:       {inv['blocks_with_length_label_difference']}"
          f" / {inv['blocks_checked']}")
    for k, v in checks["6_text_leakage_baselines"].items():
        print(f"  {k}: mean {v['mean_auc']:.4f} (min {v['min_auc']:.4f}, max {v['max_auc']:.4f})")
    for k, v in checks["7_metadata_only_baselines"].items():
        print(f"  {k}: mean {v['mean_auc']:.4f}")
    print(f"neutral-arm consequence vocabulary hits: {checks['8_neutral_arm_consequence_vocabulary']['total_hits']}")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
