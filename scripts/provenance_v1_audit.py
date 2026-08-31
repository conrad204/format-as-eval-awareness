"""Audit the frozen provenance_v1 bank before any activation exists.

The decisive number is NOT whether format is predictable from text (it trivially
is -- format is a scaffold we add). It is how well surface text predicts
PROVENANCE, because that bounds how much of any later probe result could be
plain lexical content rather than a purpose representation. This script reports
that ceiling under CV that holds out whole source datasets and whole scaffolds,
and does not hide a high value.

Usage: python scripts/provenance_v1_audit.py
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


def grouped_auc(texts, y, groups, vec_kwargs) -> dict:
    texts, y, groups = np.asarray(texts), np.asarray(y), np.asarray(groups)
    per = {}
    for g in sorted(set(groups.tolist())):
        tr, te = groups != g, groups == g
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        vec = TfidfVectorizer(**vec_kwargs)
        model = LogisticRegression(**PROBE).fit(vec.fit_transform(texts[tr]), y[tr])
        per[str(g)] = float(roc_auc_score(y[te], model.decision_function(vec.transform(texts[te]))))
    if not per:
        return {"per_group_auc": {}, "note": "no fold had both classes"}
    vals = np.array(list(per.values()))
    return {"per_group_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max())}

def paired_group_auc(texts, y, groups, vec_kwargs) -> dict:
    """Hold out one whole group from each class in every fold."""
    texts, y, groups = np.asarray(texts), np.asarray(y), np.asarray(groups)
    classes = sorted(set(y.tolist()))
    if len(classes) != 2:
        raise ValueError(f"expected two classes, got {classes}")
    by_class = {
        c: sorted(set(groups[y == c].tolist()))
        for c in classes
    }
    per = {}
    for g0 in by_class[classes[0]]:
        for g1 in by_class[classes[1]]:
            te = ((y == classes[0]) & (groups == g0)) | (
                (y == classes[1]) & (groups == g1)
            )
            tr = ~te
            if len(set(y[tr].tolist())) < 2:
                continue
            vec = TfidfVectorizer(**vec_kwargs)
            model = LogisticRegression(**PROBE).fit(
                vec.fit_transform(texts[tr]), y[tr]
            )
            per[f"{g0}|{g1}"] = float(
                roc_auc_score(
                    y[te], model.decision_function(vec.transform(texts[te]))
                )
            )
    vals = np.array(list(per.values()))
    return {
        "per_group_pair_auc": per,
        "mean_auc": float(vals.mean()),
        "min_auc": float(vals.min()),
        "max_auc": float(vals.max()),
    }


def paired_group_numeric_auc(values, y, groups) -> dict:
    """Length-only counterpart of paired_group_auc."""
    values, y, groups = (
        np.asarray(values),
        np.asarray(y),
        np.asarray(groups),
    )
    classes = sorted(set(y.tolist()))
    by_class = {
        c: sorted(set(groups[y == c].tolist()))
        for c in classes
    }
    per = {}
    for g0 in by_class[classes[0]]:
        for g1 in by_class[classes[1]]:
            te = ((y == classes[0]) & (groups == g0)) | (
                (y == classes[1]) & (groups == g1)
            )
            tr = ~te
            model = LogisticRegression(**PROBE).fit(
                values[tr, None], y[tr]
            )
            per[f"{g0}|{g1}"] = float(
                roc_auc_score(
                    y[te], model.decision_function(values[te, None])
                )
            )
    vals = np.array(list(per.values()))
    return {
        "per_group_pair_auc": per,
        "mean_auc": float(vals.mean()),
        "min_auc": float(vals.min()),
        "max_auc": float(vals.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=ROOT / "results" / "provenance_v1_frozen")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (args.bank / "items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest = json.loads((args.bank / "freeze_manifest.json").read_text())

    text = np.array([r["text"] for r in rows])
    prov = np.array([r["provenance"] for r in rows])
    y_prov = np.array([r["label"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    y_fmt = (fmt == "benchmark").astype(int)
    tmpl = np.array([r["template_id"] for r in rows])
    src = np.array([r["source_dataset"] for r in rows])
    chars = np.array([r["prompt_chars"] for r in rows], dtype=float)

    checks: dict = {}
    checks["1_counts"] = {
        "n_rows": len(rows),
        "n_unique_texts": len({r["pair_id"] for r in rows}),
        "provenance_counts": {k: int(v) for k, v in collections.Counter(prov.tolist()).items()},
        "provenance_x_format": {f"{a}|{b}": int(v) for (a, b), v in
                                collections.Counter(zip(prov.tolist(), fmt.tolist())).items()},
        "source_counts": {k: int(v) for k, v in collections.Counter(src.tolist()).items()},
        "scaffold_counts": {k: int(v) for k, v in collections.Counter(tmpl.tolist()).items()},
    }
    checks["2_orthogonality"] = {
        "format_mean_by_provenance": {p: float(y_fmt[prov == p].mean()) for p in sorted(set(prov.tolist()))},
        "provenance_mean_by_format": {f: float(y_prov[fmt == f].mean()) for f in sorted(set(fmt.tolist()))},
        "provenance_mean_by_scaffold": {t: float(y_prov[tmpl == t].mean()) for t in sorted(set(tmpl.tolist()))},
        "every_text_appears_once_per_format": all(
            v == 2 for v in collections.Counter(r["pair_id"] for r in rows).values()),
    }
    ev, dp = chars[prov == "eval_origin"], chars[prov == "deployment_origin"]
    lo, hi = min(chars.min(), 0), chars.max()
    bins = np.linspace(lo, hi, 25)
    he, _ = np.histogram(ev, bins=bins, density=False)
    hd, _ = np.histogram(dp, bins=bins, density=False)
    overlap = float(np.minimum(he / he.sum(), hd / hd.sum()).sum())
    checks["3_length_match"] = {
        "eval_origin": {"mean": float(ev.mean()), "median": float(np.median(ev)),
                        "sd": float(ev.std(ddof=1)), "min": float(ev.min()), "max": float(ev.max())},
        "deployment_origin": {"mean": float(dp.mean()), "median": float(np.median(dp)),
                              "sd": float(dp.std(ddof=1)), "min": float(dp.min()), "max": float(dp.max())},
        "mean_difference_chars": float(ev.mean() - dp.mean()),
        "histogram_overlap_fraction": overlap,
        "note": "prompt_chars is the underlying text before scaffolding, so it is the length "
                "quantity that provenance could confound",
    }

    word = dict(analyzer="word", ngram_range=(1, 2), min_df=2)
    char = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    checks["4_surface_predicts_provenance"] = {
        "word_1_2gram_holdout_source_pair": paired_group_auc(
            text, y_prov, src, word
        ),
        "char_3_5gram_holdout_source_pair": paired_group_auc(
            text, y_prov, src, char
        ),
        "word_1_2gram_leave_one_scaffold_out": grouped_auc(
            text, y_prov, tmpl, word
        ),
        "length_only_holdout_source_pair": paired_group_numeric_auc(
            chars, y_prov, src
        ),
        "interpretation": "THIS IS THE CEILING THAT MATTERS. Real eval corpora and real user "
                          "turns differ lexically, so a high value here is expected and is a "
                          "property of natural data, not a defect of the bank. Any later "
                          "activation-probe result on provenance must be read against this "
                          "number: it cannot be claimed as evidence of a purpose representation "
                          "beyond what surface content already provides.",
    }
    # A single held-out source has one provenance class, making its AUC
    # undefined. Each source-held-out fold therefore holds out one evaluation
    # source and one deployment source together.

    checks["5_surface_predicts_format"] = {
        "word_1_2gram_holdout_scaffold_pair": paired_group_auc(
            text, y_fmt, tmpl, word
        ),
        "note": "expected to be near 1.0 and uninformative: format IS a scaffold we add. Reported "
                "for completeness only.",
    }

    report = {
        "schema_version": 1,
        "experiment": "provenance_v1",
        "audited_before_activations": True,
        "bank": args.bank.relative_to(ROOT).as_posix(),
        "items_sha256": manifest["items_sha256"],
        "freeze_manifest_sha256_of_spec": manifest["manifest_sha256_of_spec"],
        "known_limitation": manifest["known_limitation"],
        "checks": checks,
    }
    (args.bank / "audit_report.json").write_bytes(
        json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))

    print(f"rows {checks['1_counts']['n_rows']}  unique texts {checks['1_counts']['n_unique_texts']}")
    print(f"provenance x format: {checks['1_counts']['provenance_x_format']}")
    print(f"every text once per format: {checks['2_orthogonality']['every_text_appears_once_per_format']}")
    lm = checks["3_length_match"]
    print(f"prompt chars: eval {lm['eval_origin']['mean']:.1f} vs deployment "
          f"{lm['deployment_origin']['mean']:.1f} (diff {lm['mean_difference_chars']:+.1f}), "
          f"histogram overlap {lm['histogram_overlap_fraction']:.3f}")
    print("SURFACE -> PROVENANCE (the ceiling):")
    for k, v in checks["4_surface_predicts_provenance"].items():
        if isinstance(v, dict) and "mean_auc" in v:
            print(f"  {k}: mean {v['mean_auc']:.4f}")
    print("SURFACE -> FORMAT (expected ~1.0, uninformative):")
    v = checks["5_surface_predicts_format"][
        "word_1_2gram_holdout_scaffold_pair"
    ]
    print(f"  word_1_2gram_holdout_scaffold_pair: mean {v['mean_auc']:.4f}")
    print(f"wrote {(args.bank / 'audit_report.json').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
