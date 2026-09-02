"""Cross-fold ownership gates for purpose-asymmetric cue features.

A feature is allowed to be shared when its presence is purpose-symmetric. A
feature is rejected only when it carries a consistent eval/deploy asymmetry in
multiple vocabulary folds. Tokenizer checks are optional and require a local
Tokenizer path or model ID; no network access is attempted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

FOLDS = ("fg1", "fg2", "fg3", "fg4", "fg5")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_word(word: str) -> str:
    value = word.lower()
    for suffix in ("ization", "ation", "ments", "ment", "ingly", "edly", "ing", "ed", "ers", "er", "ies", "es", "s"):
        if len(value) > len(suffix) + 3 and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def words(text: str) -> list[str]:
    return re.findall(r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b", text.lower())


def features(text: str, tokenizer=None) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {"stem": set()}
    token_words = words(text)
    values["stem"] = {normalize_word(word) for word in token_words}
    for width in range(3, 7):
        values[f"char_{width}"] = {text.lower()[i : i + width] for i in range(len(text) - width + 1)}
    if tokenizer is not None:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        values["token_piece"] = {str(piece) for piece in token_ids}
        values["token_piece_1_2"] = {f"{token_ids[i]} {token_ids[i+1]}" for i in range(len(token_ids) - 1)}
    return values


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty bank")
    return rows


def run(rows: list[dict], tokenizer=None) -> dict:
    feature_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"eval": 0, "deploy": 0}))
    for row in rows:
        if row.get("label") not in (0, 1) or row.get("render_slot") == "no_cue":
            continue
        fold = str(row.get("purpose_family_id"))
        cue = str(row["text"]).split("\n", 1)[0]
        for kind, values in features(cue, tokenizer).items():
            for value in values:
                feature_stats[kind][value]["eval" if int(row["label"]) == 1 else "deploy"] += 1
    ownership: list[dict] = []
    for kind, values in feature_stats.items():
        # Recompute per fold because feature_stats above is global only for identifying candidates.
        per_feature: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"eval": 0, "deploy": 0}))
        for row in rows:
            if row.get("label") not in (0, 1) or row.get("render_slot") == "no_cue":
                continue
            fold = str(row["purpose_family_id"])
            cue = str(row["text"]).split("\n", 1)[0]
            for value in features(cue, tokenizer).get(kind, set()):
                per_feature[value][fold]["eval" if int(row["label"]) == 1 else "deploy"] += 1
        for value, by_fold in per_feature.items():
            asym_folds = []
            for fold in FOLDS:
                counts = by_fold.get(fold, {"eval": 0, "deploy": 0})
                total = counts["eval"] + counts["deploy"]
                asym = 0.0 if total == 0 else (counts["eval"] - counts["deploy"]) / total
                asym_folds.append({"fold": fold, "eval": counts["eval"], "deploy": counts["deploy"], "asymmetry": asym})
            signs = {1 if item["asymmetry"] > 0.2 else -1 if item["asymmetry"] < -0.2 else 0 for item in asym_folds}
            signs.discard(0)
            reusable = len(signs) == 1 and sum(item["asymmetry"] > 0.2 or item["asymmetry"] < -0.2 for item in asym_folds) >= 2
            ownership.append({"kind": kind, "feature": value, "folds": asym_folds, "reusable_asymmetric": reusable})
    failures = [item for item in ownership if item["reusable_asymmetric"]]
    return {
        "schema_version": 1,
        "folds": list(FOLDS),
        "feature_kinds": sorted(feature_stats),
        "ownership": ownership,
        "failure_count": len(failures),
        "pass": not failures,
        "rule": "A feature is rejected only when the same purpose-asymmetry sign occurs in at least two folds; purpose-symmetric shared features are allowed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = load_rows(args.items)
    report = run(rows, tokenizer)
    report["items_sha256"] = sha256_file(args.items)
    report["tokenizer"] = args.tokenizer
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "failure_count": report["failure_count"], "items_sha256": report["items_sha256"]}))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
