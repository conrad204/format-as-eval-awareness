"""Diagnose held-fold lexical leakage in a development rendered bank.

This is exploratory and never consumes activations. It reports per-stratum,
per-scope held-fold AUCs and interpretable character-feature coefficients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
CHAR_KINDS = {"char_3_5gram": (3, 5), "char_3_6gram": (3, 6)}
PROBE = dict(C=1.0, max_iter=3000, random_state=0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cue_line(text: str) -> str:
    return text.split("\n", 1)[0]


def remove_cue(text: str) -> str:
    return text.split("\n", 1)[1] if "\n" in text else ""


def first_sentence(text: str) -> str:
    match = re.search(r"[.!?](?:\s|$)", text)
    return text[: match.end()] if match else text


def second_sentence(text: str) -> str:
    match = re.search(r"[.!?](?:\s|$)", text)
    return text[match.end() :] if match else ""


def scope_text(row: dict, scope: str) -> str:
    text = str(row["text"])
    if scope == "full_text":
        return text
    if scope == "cue_sentence_only":
        return cue_line(text)
    if scope == "cue_span_only":
        return cue_line(text) if row["render_slot"] != "no_cue" else ""
    if scope == "scaffold_without_cue":
        return remove_cue(text)
    if scope == "first_sentence":
        return first_sentence(text)
    if scope == "second_sentence":
        return second_sentence(text)
    raise ValueError(scope)


def strata(rows: Iterable[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"all": []}
    for index, row in enumerate(rows):
        groups["all"].append(index)
        slot = str(row["render_slot"])
        groups.setdefault(slot, []).append(index)
        if slot == "stated":
            groups.setdefault(f"stated_{row['cue_locus']}", []).append(index)
    return groups


def held_fold_scores(texts: np.ndarray, labels: np.ndarray, folds: np.ndarray, kind: str) -> np.ndarray:
    scores = np.full(len(texts), np.nan, dtype=float)
    lo, hi = CHAR_KINDS[kind]
    for fold in FOLDS:
        train = folds != fold
        test = folds == fold
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), lowercase=True)
        train_x = vectorizer.fit_transform(texts[train])
        model = LogisticRegression(**PROBE).fit(train_x, labels[train])
        scores[test] = model.decision_function(vectorizer.transform(texts[test]))
    return scores


def auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def metadata_counts(rows: list[dict], indices: list[int], present: np.ndarray) -> dict[str, dict[str, int]]:
    dimensions = {
        "purpose": lambda r: str(r.get("intended_purpose")),
        "render_slot": lambda r: str(r.get("render_slot")),
        "cue_locus": lambda r: str(r.get("cue_locus")),
        "format": lambda r: str(r.get("format")),
        "scaffold": lambda r: cue_line(str(r["text"])) if r.get("render_slot") != "no_cue" else "<no_cue>",
    }
    output: dict[str, dict[str, int]] = {}
    for name, getter in dimensions.items():
        output[name] = dict(sorted(Counter(getter(rows[i]) for i, hit in zip(indices, present) if hit).items()))
    return output


def coefficient_rows(rows: list[dict], labels: np.ndarray, folds: np.ndarray, scope: str, kind: str, top_n: int) -> list[dict]:
    texts = np.asarray([scope_text(row, scope) for row in rows], dtype=str)
    output: list[dict] = []
    for fold in FOLDS:
        train = folds != fold
        lo, hi = CHAR_KINDS[kind]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), lowercase=True)
        train_x = vectorizer.fit_transform(texts[train])
        model = LogisticRegression(**PROBE).fit(train_x, labels[train])
        coefficients = model.coef_[0]
        names = np.asarray(vectorizer.get_feature_names_out())
        order = np.argsort(-np.abs(coefficients))[:top_n]
        train_indices = np.flatnonzero(train).tolist()
        train_present = train_x != 0
        eval_mask = labels[train] == 1
        deploy_mask = labels[train] == 0
        for feature_index in order:
            feature = str(names[feature_index])
            present = np.asarray(train_present[:, feature_index].toarray()).ravel().astype(bool)
            counts = metadata_counts(rows, train_indices, present)
            eval_count = int(np.sum(present & eval_mask))
            deploy_count = int(np.sum(present & deploy_mask))
            output.append({
                "scope": scope,
                "kind": kind,
                "held_out_fold": fold,
                "feature": feature,
                "coefficient_eval_positive": float(coefficients[feature_index]),
                "abs_coefficient": float(abs(coefficients[feature_index])),
                "train_eval_count": eval_count,
                "train_deploy_count": deploy_count,
                "train_eval_rate": eval_count / max(1, int(eval_mask.sum())),
                "train_deploy_rate": deploy_count / max(1, int(deploy_mask.sum())),
                "train_asymmetry": eval_count / max(1, int(eval_mask.sum())) - deploy_count / max(1, int(deploy_mask.sum())),
                "frequency_by_purpose": json.dumps(counts["purpose"], sort_keys=True),
                "frequency_by_render_slot": json.dumps(counts["render_slot"], sort_keys=True),
                "frequency_by_cue_locus": json.dumps(counts["cue_locus"], sort_keys=True),
                "frequency_by_format": json.dumps(counts["format"], sort_keys=True),
                "frequency_by_scaffold": json.dumps(counts["scaffold"], sort_keys=True),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    rows = load_rows(args.items)
    stated = [row for row in rows if row["render_slot"] == "stated"]
    labels = np.asarray([int(row["label"]) for row in stated], dtype=int)
    folds = np.asarray([str(row["purpose_family_id"]) for row in stated])
    reports: list[dict] = []
    all_features: list[dict] = []
    scope_names = ("full_text", "cue_sentence_only", "cue_span_only", "scaffold_without_cue", "first_sentence", "second_sentence")
    for scope in scope_names:
        texts = np.asarray([scope_text(row, scope) for row in stated], dtype=str)
        groups = strata(stated)
        for kind in CHAR_KINDS:
            scores = held_fold_scores(texts, labels, folds, kind)
            for group, indices in groups.items():
                index_array = np.asarray(indices, dtype=int)
                for fold in FOLDS:
                    held = index_array[folds[index_array] == fold]
                    reports.append({
                        "scope": scope,
                        "kind": kind,
                        "stratum": group,
                        "held_out_fold": fold,
                        "n": int(len(held)),
                        "auc": auc_or_none(labels[held], scores[held]),
                    })
            all_features.extend(coefficient_rows(stated, labels, folds, scope, kind, args.top_n))
    report = {
        "schema_version": 1,
        "items_sha256": sha256_file(args.items),
        "rows": len(rows),
        "stated_rows": len(stated),
        "folds": list(FOLDS),
        "threshold": {"mean_auc_max": 0.60, "max_fold_auc_max": 0.70},
        "scopes": list(scope_names),
        "reports": reports,
        "interpretation": {
            "positive_coefficient": "feature increases held-fold evaluation score",
            "train_asymmetry": "presence rate in evaluation training rows minus deployment training rows",
            "diagnostic_only": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(all_features[0]) if all_features else []
    with args.features.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
    print(json.dumps({"items_sha256": report["items_sha256"], "report": str(args.report), "features": str(args.features), "feature_rows": len(all_features)}))


if __name__ == "__main__":
    main()
