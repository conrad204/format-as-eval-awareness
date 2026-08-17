"""Independent held-fold red-team lexical audit for a candidate bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("label") in (0, 1)]
    texts = np.asarray([str(row["text"]).split("\n", 1)[0] for row in rows], dtype=str)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    folds = np.asarray([str(row["purpose_family_id"]) for row in rows])
    results = {}
    configs = {
        "char_2_7_logistic": (TfidfVectorizer(analyzer="char", ngram_range=(2, 7), lowercase=True), LogisticRegression(C=1.0, max_iter=4000, random_state=17)),
        "char_2_7_svm": (TfidfVectorizer(analyzer="char", ngram_range=(2, 7), lowercase=True), LinearSVC(C=1.0, random_state=17)),
        "char_2_7_ridge": (TfidfVectorizer(analyzer="char", ngram_range=(2, 7), lowercase=True), RidgeClassifier(alpha=1.0)),
        "byte_2_7_logistic": (TfidfVectorizer(analyzer="char", ngram_range=(2, 7), lowercase=True, encoding="utf-8"), LogisticRegression(C=1.0, max_iter=4000, random_state=17)),
    }
    for name, (vectorizer, estimator) in configs.items():
        fold_auc = {}
        for fold in FOLDS:
            train, test = folds != fold, folds == fold
            model = make_pipeline(vectorizer, estimator).fit(texts[train], labels[train])
            scores = model.decision_function(texts[test])
            fold_auc[fold] = float(roc_auc_score(labels[test], scores))
        values = np.asarray(list(fold_auc.values()))
        results[name] = {"mean_auc": float(values.mean()), "max_auc": float(values.max()), "per_fold_auc": fold_auc}
    report = {"schema_version": 1, "items_sha256": sha256_file(args.items), "rows": len(rows), "scope": "cue_sentence_only", "results": results, "egregious": bool(any(v["max_auc"] > 0.70 or v["mean_auc"] > 0.60 for v in results.values()))}
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"egregious": report["egregious"], "items_sha256": report["items_sha256"]}))
    if report["egregious"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
