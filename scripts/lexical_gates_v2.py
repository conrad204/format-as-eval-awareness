"""Frozen construction gates with cue-scope and stratum breakdowns.

This does not modify or reuse the rejected bank. Every fit is leave-one-purpose-
family-out, with the predeclared mean/max AUC thresholds unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
THRESHOLD = {"mean_auc_max": 0.60, "max_fold_auc_max": 0.70}
PROBE = dict(C=1.0, max_iter=3000, random_state=0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("bank is empty")
    required = {"item_id", "text", "label", "purpose_family_id", "render_slot", "cue_locus", "format"}
    errors: list[str] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"row {index}: missing {missing}")
        item_id = str(row.get("item_id", ""))
        if item_id in ids:
            errors.append(f"row {index}: duplicate item_id {item_id}")
        ids.add(item_id)
        label = row.get("label")
        if row.get("render_slot") == "no_cue":
            if label is not None:
                errors.append(f"row {index}: no-cue row has a label")
        elif label not in (0, 1):
            errors.append(f"row {index}: labeled row has nonbinary label {label!r}")
    if errors:
        raise ValueError("bank schema errors: " + "; ".join(errors[:12]))
    return rows


def scope_text(row: dict, scope: str) -> str:
    text = str(row["text"])
    first, separator, rest = text.partition("\n")
    if scope in {"cue_sentence", "cue_span"}:
        return first
    if scope == "scaffold_without_cue":
        return rest if separator else ""
    if scope == "full_text":
        return text
    if scope == "first_sentence":
        match = re.search(r"[.!?](?:\s|$)", text)
        return text[:match.end()] if match else text
    if scope == "second_sentence":
        match = re.search(r"[.!?](?:\s|$)", text)
        return text[match.end():] if match else ""
    raise ValueError(scope)


def _stats(text: str) -> list[float]:
    chars = list(text)
    words = re.findall(r"(?u)\b\w+\b", text)
    punct = sum(not ch.isalnum() and not ch.isspace() for ch in chars)
    digits = sum(ch.isdigit() for ch in chars)
    whitespace = sum(ch.isspace() for ch in chars)
    return [float(len(text)), float(len(words)), float(punct), float(digits), float(whitespace), float(text.count("\n")), float(punct / max(1, len(text)))]


def fit_scores(kind: str, train_texts: np.ndarray, train_y: np.ndarray, test_texts: np.ndarray) -> np.ndarray:
    if kind.startswith("word_"):
        hi = int(kind.split("_")[-1])
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, hi), lowercase=True, token_pattern=r"(?u)\b\w+\b")
        model = LogisticRegression(**PROBE)
        return make_pipeline(vectorizer, model).fit(train_texts, train_y).decision_function(test_texts)
    if kind.startswith("char_"):
        hi = int(kind.split("_")[-1])
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, hi), lowercase=True)
        model = LogisticRegression(**PROBE)
        return make_pipeline(vectorizer, model).fit(train_texts, train_y).decision_function(test_texts)
    if kind == "length_punctuation":
        train_x = np.asarray([_stats(text) for text in train_texts], dtype=float)
        test_x = np.asarray([_stats(text) for text in test_texts], dtype=float)
        return make_pipeline(StandardScaler(), LogisticRegression(**PROBE)).fit(train_x, train_y).decision_function(test_x)
    raise ValueError(kind)


def evaluate(texts: np.ndarray, labels: np.ndarray, folds: np.ndarray, kind: str) -> dict:
    scores = np.full(len(texts), np.nan, dtype=float)
    fold_rows = {}
    for fold in FOLDS:
        train, test = folds != fold, folds == fold
        scores[test] = fit_scores(kind, texts[train], labels[train], texts[test])
        fold_rows[fold] = {
            "n": int(test.sum()),
            "positive": int(labels[test].sum()),
            "negative": int((labels[test] == 0).sum()),
            "auc": float(roc_auc_score(labels[test], scores[test])),
        }
    values = np.asarray([entry["auc"] for entry in fold_rows.values()])
    return {
        "mean_auc": float(values.mean()),
        "max_auc": float(values.max()),
        "per_fold": fold_rows,
        "pass": bool(values.mean() <= THRESHOLD["mean_auc_max"] and values.max() <= THRESHOLD["max_fold_auc_max"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.items)
    labeled = [row for row in rows if row["label"] in (0, 1)]
    strata: dict[str, list[dict]] = {"stated": [r for r in labeled if r["render_slot"] == "stated"], "indirect": [r for r in labeled if r["render_slot"] == "indirect"], "second_person": [r for r in labeled if r["render_slot"] == "second_person"]}
    for locus in ("noun_only", "verb_only", "both"):
        strata[f"stated_{locus}"] = [r for r in strata["stated"] if r["cue_locus"] == locus]
    scopes = ("full_text", "cue_sentence", "cue_span", "scaffold_without_cue", "first_sentence", "second_sentence")
    kinds = ("word_1", "word_2", "word_3", "char_5", "char_6", "length_punctuation")
    results = {}
    for stratum, selected in strata.items():
        labels = np.asarray([int(r["label"]) for r in selected], dtype=int)
        folds = np.asarray([str(r["purpose_family_id"]) for r in selected])
        if len(selected) == 0 or set(folds) != set(FOLDS):
            raise ValueError(f"stratum {stratum} lacks all folds")
        results[stratum] = {}
        for scope in scopes:
            texts = np.asarray([scope_text(r, scope) for r in selected], dtype=str)
            results[stratum][scope] = {kind: evaluate(texts, labels, folds, kind) for kind in kinds}
    report = {
        "schema_version": 2,
        "items_sha256": sha256_file(args.items),
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "folds": list(FOLDS),
        "threshold": THRESHOLD,
        "scopes": list(scopes),
        "strata": {key: len(value) for key, value in strata.items()},
        "results": results,
        "pass": bool(all(entry["pass"] for strata_result in results.values() for scope_result in strata_result.values() for entry in scope_result.values())),
        "diagnostic_note": "Cue scopes are reported separately because the frozen pilot controls use the first rendered line; full-text results are retained as a red-team view.",
    }
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "items_sha256": report["items_sha256"]}))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
