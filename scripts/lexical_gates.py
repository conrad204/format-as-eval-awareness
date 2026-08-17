"""Pre-extraction held-vocabulary lexical leakage gates for v2 banks.

The gate is intentionally independent of neural activations. It fits every
lexical classifier inside a leave-one-purpose-family-out fold and records the
mean, maximum, and per-fold AUC. Confirmation banks must pass the frozen
thresholds before activation extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
MEAN_MAX_THRESHOLD = 0.60
MAX_FOLD_THRESHOLD = 0.70
PROBE = dict(C=1.0, max_iter=3000, random_state=0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("bank is empty")
    required = {"text", "label", "purpose_family_id", "render_slot"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"bank is missing required v2 fields: {missing}")
    return rows


def _stated(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["render_slot"] == "stated"]
    folds = np.asarray([str(row["purpose_family_id"]) for row in selected])
    labels = np.asarray([int(row["label"]) for row in selected], dtype=int)
    texts = np.asarray([str(row["text"]) for row in selected], dtype=str)
    if tuple(sorted(set(folds))) != FOLDS:
        raise ValueError(f"expected folds {FOLDS}, got {tuple(sorted(set(folds)))}")
    if set(labels) != {0, 1}:
        raise ValueError("stated rows must contain both binary labels")
    return texts, labels, folds


def _auc_by_fold(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict:
    per_fold = {}
    for fold in FOLDS:
        mask = folds == fold
        if len(np.unique(labels[mask])) < 2:
            raise ValueError(f"held fold {fold} has one class")
        per_fold[fold] = float(roc_auc_score(labels[mask], scores[mask]))
    values = np.asarray(list(per_fold.values()), dtype=float)
    return {
        "mean_auc": float(values.mean()),
        "max_auc": float(values.max()),
        "per_fold_auc": per_fold,
        "pass": bool(values.mean() <= MEAN_MAX_THRESHOLD and values.max() <= MAX_FOLD_THRESHOLD),
    }


def _held_fold_model_scores(
    texts: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    fit_score: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    scores = np.full(len(texts), np.nan, dtype=float)
    for fold in FOLDS:
        train, test = folds != fold, folds == fold
        scores[test] = fit_score(texts[train], labels[train], texts[test])
    return scores


def _tfidf_scores(kind: str, train_texts: np.ndarray, train_y: np.ndarray, test_texts: np.ndarray) -> np.ndarray:
    if kind == "word_1gram":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), lowercase=True, token_pattern=r"(?u)\b\w+\b")
    elif kind == "word_1_2gram":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b")
    elif kind == "word_1_3gram":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), lowercase=True, token_pattern=r"(?u)\b\w+\b")
    elif kind == "char_3_5gram":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True)
    elif kind == "char_3_6gram":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), lowercase=True)
    else:
        raise ValueError(kind)
    matrix = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(**PROBE).fit(matrix, train_y)
    return model.decision_function(vectorizer.transform(test_texts))


def _stats(text: str) -> list[float]:
    chars = np.asarray(list(text), dtype=str)
    words = re.findall(r"(?u)\b\w+\b", text)
    punct = sum(not ch.isalnum() and not ch.isspace() for ch in chars)
    digits = sum(ch.isdigit() for ch in chars)
    whitespace = sum(ch.isspace() for ch in chars)
    newline = text.count("\n")
    half = max(1, len(text) // 2)
    left, right = text[:half], text[half:]
    return [
        float(len(text)), float(len(words)), float(punct), float(digits), float(whitespace), float(newline),
        float(len(left)), float(len(right)), float(punct / max(1, len(text))),
        float(right.count("\n") - left.count("\n")),
    ]


def _numeric_scores(train_texts: np.ndarray, train_y: np.ndarray, test_texts: np.ndarray) -> np.ndarray:
    train_x = np.asarray([_stats(text) for text in train_texts], dtype=float)
    test_x = np.asarray([_stats(text) for text in test_texts], dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(**PROBE)).fit(train_x, train_y)
    return model.decision_function(test_x)


def _half_scores(half: str, train_texts: np.ndarray, train_y: np.ndarray, test_texts: np.ndarray) -> np.ndarray:
    def cut(text: str) -> str:
        midpoint = max(1, len(text) // 2)
        return text[:midpoint] if half == "first_half" else text[midpoint:]
    return _tfidf_scores("word_1gram", np.asarray([cut(t) for t in train_texts]), train_y, np.asarray([cut(t) for t in test_texts]))


def run_gate(items: Path) -> dict:
    rows = _rows(items)
    texts, labels, folds = _stated(rows)
    results = {}
    for kind in ("word_1gram", "word_1_2gram", "word_1_3gram", "char_3_5gram", "char_3_6gram"):
        scores = _held_fold_model_scores(texts, labels, folds, lambda a, b, c, k=kind: _tfidf_scores(k, a, b, c))
        results[kind] = _auc_by_fold(scores, labels, folds)
    numeric = _held_fold_model_scores(texts, labels, folds, _numeric_scores)
    results["length_punctuation"] = _auc_by_fold(numeric, labels, folds)
    for half in ("first_half", "second_half"):
        scores = _held_fold_model_scores(texts, labels, folds, lambda a, b, c, h=half: _half_scores(h, a, b, c))
        results[half] = _auc_by_fold(scores, labels, folds)
    return {
        "schema_version": 1,
        "items_sha256": sha256_file(items),
        "rows": len(rows),
        "stated_rows": int(len(texts)),
        "folds": list(FOLDS),
        "threshold": {"mean_auc_max": MEAN_MAX_THRESHOLD, "max_fold_auc_max": MAX_FOLD_THRESHOLD},
        "features": {
            "word": {"lowercase": True, "token_pattern": r"(?u)\b\w+\b", "ranges": [[1, 1], [1, 2], [1, 3]]},
            "char": {"lowercase": True, "ranges": [[3, 5], [3, 6]]},
            "numeric": "length, word count, punctuation, digits, whitespace, newlines, half lengths and punctuation density",
            "halves": "word-unigram TF-IDF independently on first and second text halves",
        },
        "results": results,
        "pass": bool(all(result["pass"] for result in results.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = run_gate(args.items)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite lexical gate output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "items_sha256": report["items_sha256"]}))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
