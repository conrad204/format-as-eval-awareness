"""Development-only held-vocabulary text shortcut audit for pilot80."""
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
PROBE = dict(C=1.0, max_iter=3000, random_state=0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(text: str) -> list[float]:
    chars = list(text)
    words = re.findall(r"(?u)\b\w+\b", text)
    punct = sum(not c.isalnum() and not c.isspace() for c in chars)
    return [len(text), len(words), punct, sum(c.isdigit() for c in chars),
            sum(c.isspace() for c in chars), text.count("\n"),
            punct / max(1, len(text))]


def scope_text(text: str, scope: str) -> str:
    first, sep, rest = text.partition("\n")
    if scope == "full_prompt":
        return text
    if scope == "cue_only":
        return first
    if scope == "scaffold_without_cue":
        return rest if sep else ""
    if scope == "first_half":
        return text[: len(text) // 2]
    if scope == "second_half":
        return text[len(text) // 2 :]
    raise ValueError(scope)


def fit_scores(kind: str, train: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    if kind.startswith("word_"):
        hi = int(kind.split("_")[-1])
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, hi),
                              lowercase=True, token_pattern=r"(?u)\b\w+\b")
        return make_pipeline(vec, LogisticRegression(**PROBE)).fit(train, y).decision_function(test)
    if kind.startswith("char_"):
        lo, hi = map(int, kind.split("_")[1].split("-") )
        vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), lowercase=True)
        return make_pipeline(vec, LogisticRegression(**PROBE)).fit(train, y).decision_function(test)
    if kind == "length_punctuation":
        return make_pipeline(StandardScaler(), LogisticRegression(**PROBE)).fit(
            np.asarray([stats(x) for x in train]), y
        ).decision_function(np.asarray([stats(x) for x in test]))
    raise ValueError(kind)


def evaluate(rows: list[dict], selected: list[dict], scope: str, kind: str) -> dict:
    labels = np.asarray([int(r["label"]) for r in selected])
    folds = np.asarray([str(r["purpose_family_id"]) for r in selected])
    texts = np.asarray([scope_text(str(r["text"]), scope) for r in selected], dtype=str)
    scores = np.full(len(selected), np.nan)
    per_fold = {}
    for fold in FOLDS:
        train, test = folds != fold, folds == fold
        scores[test] = fit_scores(kind, texts[train], labels[train], texts[test])
        per_fold[fold] = {"auc": float(roc_auc_score(labels[test], scores[test])), "n": int(test.sum())}
    vals = np.asarray([x["auc"] for x in per_fold.values()])
    return {"mean_auc": float(vals.mean()), "max_fold_auc": float(vals.max()), "per_fold": per_fold}


def try_tokenizer() -> dict:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
        return {"status": "available", "name": str(tok.name_or_path), "vocab_size": int(tok.vocab_size)}
    except Exception as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__,
                "message": "Llama tokenizer unavailable locally; no alternate tokenizer substituted."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    labeled = [r for r in rows if r.get("label") in (0, 1)]
    strata = {
        "stated": [r for r in labeled if r["render_slot"] == "stated"],
        "indirect": [r for r in labeled if r["render_slot"] == "indirect"],
        "second_person": [r for r in labeled if r["render_slot"] == "second_person"],
    }
    for locus in ("noun_only", "verb_only", "both"):
        strata[f"stated_{locus}"] = [r for r in strata["stated"] if r["cue_locus"] == locus]
    # Full-prompt and cue-only are the primary scopes. Halves and scaffold are red-team views.
    scopes = ("full_prompt", "cue_only", "first_half", "second_half", "scaffold_without_cue")
    kinds = ("word_1", "word_2", "word_3", "char_2-5", "char_3-5", "char_3-6", "length_punctuation")
    results = {}
    for stratum, selected in strata.items():
        if not selected or set(r["purpose_family_id"] for r in selected) != set(FOLDS):
            raise ValueError(f"stratum missing folds: {stratum}")
        results[stratum] = {scope: {kind: evaluate(rows, selected, scope, kind)
                                    for kind in kinds} for scope in scopes}
    report = {
        "schema_version": 3,
        "mode": "development",
        "items": {"path": str(args.items), "sha256": sha256_file(args.items)},
        "rows": len(rows), "labeled_rows": len(labeled), "folds": list(FOLDS),
        "tokenizer_piece": {"status": "not_run", "reason": "No Llama tokenizer was available locally during this run."},
        "tokenizer_probe_attempt": try_tokenizer(),
        "kinds": list(kinds), "scopes": list(scopes), "strata_counts": {k: len(v) for k, v in strata.items()},
        "results": results,
        "note": "Text-only controls are descriptive development results; tokenizer-piece controls are not substituted with another tokenizer.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows), "strata": list(strata)}))


if __name__ == "__main__":
    main()
