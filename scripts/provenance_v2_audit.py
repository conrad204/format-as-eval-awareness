"""Pre-activation audit for provenance_v2.

Surface-content provenance baselines are diagnostics, not targets to minimize:
real evaluation corpora and real conversation logs naturally differ in topic and
wording. High AUC bounds the interpretation of later activation scoring.
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

ROOT = Path(__file__).resolve().parents[1]
PROBE = dict(max_iter=3000, random_state=0)


def fit_auc(text, y, train, test, vec_kwargs) -> float:
    vec = TfidfVectorizer(**vec_kwargs)
    model = LogisticRegression(**PROBE).fit(vec.fit_transform(text[train]), y[train])
    return float(roc_auc_score(y[test], model.decision_function(vec.transform(text[test]))))


def grouped_five_fold(text, y, pair, vec_kwargs) -> dict:
    fold = np.array([
        int(hashlib.sha256(str(p).encode()).hexdigest()[:8], 16) % 5 for p in pair
    ])
    per = {}
    for k in range(5):
        per[str(k)] = fit_auc(text, y, fold != k, fold == k, vec_kwargs)
    vals = np.array(list(per.values()))
    return {"per_fold_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max())}


def eval_source_swap(text, y, pair, source, vec_kwargs) -> dict:
    wild = source == "allenai/WildChat-1M"
    half = np.array([
        int(hashlib.sha256(str(p).encode()).hexdigest()[-8:], 16) % 2 for p in pair
    ])
    mmlu = source == "cais/mmlu"
    arc = source == "allenai/ai2_arc"
    folds = {
        "train_mmlu_wildchat_half0__test_arc_wildchat_half1": (
            mmlu | (wild & (half == 0)), arc | (wild & (half == 1))
        ),
        "train_arc_wildchat_half1__test_mmlu_wildchat_half0": (
            arc | (wild & (half == 1)), mmlu | (wild & (half == 0))
        ),
    }
    per = {name: fit_auc(text, y, train, test, vec_kwargs)
           for name, (train, test) in folds.items()}
    vals = np.array(list(per.values()))
    return {"per_swap_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max()),
            "note": "evaluation source and all held WildChat texts change between train/test; "
                    "WildChat remains the sole deployment-log corpus"}


def scaffold_pair_holdout(text, y, fmt, template, vec_kwargs) -> dict:
    classes = ("casual", "benchmark")
    groups = {f: sorted(set(template[fmt == f].tolist())) for f in classes}
    per = {}
    for casual in groups["casual"]:
        for benchmark in groups["benchmark"]:
            test = (template == casual) | (template == benchmark)
            per[f"{casual}|{benchmark}"] = fit_auc(
                text, y, ~test, test, vec_kwargs
            )
    vals = np.array(list(per.values()))
    return {"per_scaffold_pair_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=ROOT / "results" / "provenance_v2_frozen")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (args.bank / "items.jsonl").read_text(encoding="utf-8").splitlines() if x]
    manifest = json.loads((args.bank / "freeze_manifest.json").read_text(encoding="utf-8"))
    text = np.array([r["text"] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    template = np.array([r["template_id"] for r in rows])
    pair = np.array([r["pair_id"] for r in rows])
    source = np.array([r["source_dataset"] for r in rows])
    chars = np.array([r["prompt_chars"] for r in rows], dtype=float)
    word = dict(analyzer="word", ngram_range=(1, 2), min_df=2)
    char = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    cross = {f"{a}|{b}": int(np.sum((source == a) & (fmt == b)))
             for a in sorted(set(source)) for b in sorted(set(fmt))}
    pair_counts = {str(p): int(np.sum(pair == p)) for p in set(pair)}
    eval_chars, deploy_chars = chars[y == 1], chars[y == 0]
    checks = {
        "counts": {
            "n_rows": len(rows), "n_pairs": len(set(pair.tolist())),
            "source_x_format": cross,
            "every_pair_has_two_formats": all(v == 2 for v in pair_counts.values()),
            "provenance_mean_by_format": {f: float(y[fmt == f].mean()) for f in set(fmt)},
            "provenance_mean_by_template": {t: float(y[template == t].mean()) for t in set(template)},
        },
        "length_match": {
            "eval_mean": float(eval_chars.mean()),
            "deployment_mean": float(deploy_chars.mean()),
            "mean_difference": float(eval_chars.mean() - deploy_chars.mean()),
            "max_absolute_sorted_difference": float(
                np.max(np.abs(np.sort(eval_chars) - np.sort(deploy_chars)))
            ),
        },
        "surface_predicts_provenance": {
            "word_1_2gram_grouped_five_fold": grouped_five_fold(text, y, pair, word),
            "char_3_5gram_grouped_five_fold": grouped_five_fold(text, y, pair, char),
            "word_1_2gram_eval_source_swap": eval_source_swap(text, y, pair, source, word),
            "word_1_2gram_scaffold_pair_holdout": scaffold_pair_holdout(text, y, fmt, template, word),
            "interpretation": "upper bound on content confounding; not evidence of model purpose representation",
        },
        "length_only_grouped_five_fold": {},
    }
    fold = np.array([
        int(hashlib.sha256(str(p).encode()).hexdigest()[:8], 16) % 5 for p in pair
    ])
    per = {}
    for k in range(5):
        train, test = fold != k, fold == k
        model = LogisticRegression(**PROBE).fit(chars[train, None], y[train])
        per[str(k)] = float(roc_auc_score(y[test], model.decision_function(chars[test, None])))
    checks["length_only_grouped_five_fold"] = {
        "per_fold_auc": per, "mean_auc": float(np.mean(list(per.values())))
    }
    report = {
        "schema_version": 1,
        "experiment": "provenance_v2",
        "audited_before_activations": True,
        "items_sha256": manifest["items_sha256"],
        "freeze_manifest_sha256_of_spec": manifest["manifest_sha256_of_spec"],
        "known_limitations": manifest["known_limitations"],
        "checks": checks,
    }
    out = args.bank / "audit_report.json"
    out.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))
    print("rows", len(rows), "pairs", len(set(pair.tolist())))
    print("length difference", checks["length_match"]["mean_difference"],
          "max pair", checks["length_match"]["max_absolute_sorted_difference"])
    for key, value in checks["surface_predicts_provenance"].items():
        if isinstance(value, dict) and "mean_auc" in value:
            print(key, f"{value['mean_auc']:.4f}")
    print("length_only", f"{checks['length_only_grouped_five_fold']['mean_auc']:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
