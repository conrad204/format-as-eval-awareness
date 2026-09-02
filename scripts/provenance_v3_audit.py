"""Pre-activation audit for provenance_v3.

Adds the two checks that a second deployment-log corpus makes possible:
  * leave-one-deployment-log-corpus-out surface baseline (genuine, not relabeled);
  * how separable the two log corpora are from each other, which quantifies how
    much "deployment" is really one population here.

As in v2, high surface-provenance AUC is expected and is reported as the ceiling
that bounds any later activation result, not treated as a defect.
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
WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=2)
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2)


def fit_auc(text, y, train, test, vec_kwargs) -> float:
    vec = TfidfVectorizer(**vec_kwargs)
    model = LogisticRegression(**PROBE).fit(vec.fit_transform(text[train]), y[train])
    return float(roc_auc_score(y[test], model.decision_function(vec.transform(text[test]))))


def pair_fold(pair, mod: int, tail: bool = False) -> np.ndarray:
    part = slice(-8, None) if tail else slice(0, 8)
    return np.array([
        int(hashlib.sha256(str(p).encode()).hexdigest()[part], 16) % mod for p in pair
    ])


def grouped_five_fold(text, y, pair, vec_kwargs) -> dict:
    fold = pair_fold(pair, 5)
    per = {str(k): fit_auc(text, y, fold != k, fold == k, vec_kwargs) for k in range(5)}
    vals = np.array(list(per.values()))
    return {"per_fold_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max())}


def source_pair_holdout(text, y, source, eval_sources, log_sources, vec_kwargs) -> dict:
    per = {}
    for es in eval_sources:
        for ds in log_sources:
            test = (source == es) | (source == ds)
            per[f"{es}|{ds}"] = fit_auc(text, y, ~test, test, vec_kwargs)
    vals = np.array(list(per.values()))
    return {"per_source_pair_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()), "max_auc": float(vals.max()),
            "note": "each fold holds out one whole evaluation source and one whole "
                    "deployment-log corpus simultaneously"}


def leave_one_log_corpus_out(text, y, source, log_sources, vec_kwargs) -> dict:
    per = {}
    for ds in log_sources:
        test = source == ds
        # Test set is single-class, so score it against the eval side held in the
        # same fold: hold out this log corpus AND a length-matched eval half.
        fold = pair_fold(np.arange(len(y)).astype(str), 2)
        test = test | ((y == 1) & (fold == log_sources.index(ds) % 2))
        per[ds] = fit_auc(text, y, ~test, test, vec_kwargs)
    vals = np.array(list(per.values()))
    return {"per_held_log_corpus_auc": per, "mean_auc": float(vals.mean()),
            "min_auc": float(vals.min()),
            "note": "the held deployment-log corpus is entirely unseen in training"}


def log_corpus_separability(text, source, log_sources, vec_kwargs) -> dict:
    mask = np.isin(source, log_sources)
    y = (source[mask] == log_sources[0]).astype(int)
    sub = text[mask]
    fold = pair_fold(np.arange(len(y)).astype(str), 5)
    per = {str(k): fit_auc(sub, y, fold != k, fold == k, vec_kwargs) for k in range(5)}
    vals = np.array(list(per.values()))
    return {"per_fold_auc": per, "mean_auc": float(vals.mean()),
            "interpretation": "WildChat vs LMSYS separability. Well above 0.5 means the two "
                              "log corpora are not two samples of one deployment population, "
                              "so they are reported separately downstream."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=ROOT / "results" / "provenance_v3_frozen")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (args.bank / "items.jsonl").read_text(encoding="utf-8").splitlines() if x]
    manifest = json.loads((args.bank / "freeze_manifest.json").read_text(encoding="utf-8"))
    grid = manifest["source_robustness_grid"]
    eval_sources, log_sources = grid["eval_sources"], grid["deployment_log_sources"]

    text = np.array([r["text"] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    fmt = np.array([r["format"] for r in rows])
    template = np.array([r["template_id"] for r in rows])
    pair = np.array([r["pair_id"] for r in rows])
    source = np.array([r["source_dataset"] for r in rows])
    chars = np.array([r["prompt_chars"] for r in rows], dtype=float)

    counts = {
        "n_rows": len(rows),
        "n_pairs": len(set(pair.tolist())),
        "source_x_format": {f"{a}|{b}": int(np.sum((source == a) & (fmt == b)))
                            for a in sorted(set(source.tolist())) for b in sorted(set(fmt.tolist()))},
        "every_pair_has_two_formats": all(
            v == 2 for v in __import__("collections").Counter(pair.tolist()).values()),
        "provenance_mean_by_format": {f: float(y[fmt == f].mean()) for f in sorted(set(fmt.tolist()))},
        "provenance_mean_by_template": {t: float(y[template == t].mean())
                                        for t in sorted(set(template.tolist()))},
    }
    per_source_len = {s: chars[source == s] for s in sorted(set(source.tolist()))}
    length = {
        "eval_mean": float(chars[y == 1].mean()),
        "deployment_mean": float(chars[y == 0].mean()),
        "mean_difference": float(chars[y == 1].mean() - chars[y == 0].mean()),
        "max_absolute_sorted_difference": float(
            np.max(np.abs(np.sort(chars[y == 1]) - np.sort(chars[y == 0])))),
        "per_source_mean": {s: float(v.mean()) for s, v in per_source_len.items()},
        "log_corpus_mean_difference": float(
            per_source_len[log_sources[0]].mean() - per_source_len[log_sources[1]].mean()),
    }
    checks = {
        "counts": counts,
        "length_match": length,
        "surface_predicts_provenance": {
            "word_1_2gram_grouped_five_fold": grouped_five_fold(text, y, pair, WORD),
            "char_3_5gram_grouped_five_fold": grouped_five_fold(text, y, pair, CHAR),
            "word_1_2gram_source_pair_holdout": source_pair_holdout(
                text, y, source, eval_sources, log_sources, WORD),
            "word_1_2gram_leave_one_log_corpus_out": leave_one_log_corpus_out(
                text, y, source, log_sources, WORD),
            "interpretation": "ceiling on content confounding; not evidence about the model",
        },
        "log_corpus_separability": log_corpus_separability(text, source, log_sources, WORD),
    }
    fold = pair_fold(pair, 5)
    per = {}
    for k in range(5):
        train, test = fold != k, fold == k
        model = LogisticRegression(**PROBE).fit(chars[train, None], y[train])
        per[str(k)] = float(roc_auc_score(y[test], model.decision_function(chars[test, None])))
    checks["length_only_grouped_five_fold"] = {
        "per_fold_auc": per, "mean_auc": float(np.mean(list(per.values())))}

    report = {
        "schema_version": 1,
        "experiment": "provenance_v3",
        "audited_before_activations": True,
        "items_sha256": manifest["items_sha256"],
        "freeze_manifest_sha256_of_spec": manifest["manifest_sha256_of_spec"],
        "known_limitations": manifest["known_limitations"],
        "log_corpora_are_not_interchangeable": manifest["log_corpora_are_not_interchangeable"],
        "checks": checks,
    }
    (args.bank / "audit_report.json").write_bytes(
        json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))

    print(f"rows {counts['n_rows']} pairs {counts['n_pairs']}")
    print(f"source x format {counts['source_x_format']}")
    print(f"length: eval {length['eval_mean']:.2f} vs deploy {length['deployment_mean']:.2f} "
          f"(diff {length['mean_difference']:+.3f}, max sorted {length['max_absolute_sorted_difference']:.1f}); "
          f"WildChat-LMSYS {length['log_corpus_mean_difference']:+.2f}")
    for k, v in checks["surface_predicts_provenance"].items():
        if isinstance(v, dict) and "mean_auc" in v:
            print(f"  {k}: {v['mean_auc']:.4f}")
    print(f"  length_only: {checks['length_only_grouped_five_fold']['mean_auc']:.4f}")
    print(f"WildChat vs LMSYS separability: {checks['log_corpus_separability']['mean_auc']:.4f}")
    print(f"wrote {(args.bank / 'audit_report.json')}")


if __name__ == "__main__":
    main()
