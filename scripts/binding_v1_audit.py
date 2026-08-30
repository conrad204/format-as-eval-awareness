"""Audit the frozen binding_v1 bank before any activation extraction.

Structural checks establish exact factorial balance; text and metadata
baselines use leave-one-entire-relational-family-out CV. No activations are
read and the frozen bank is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TOKEN_RE = re.compile(r"[A-Za-z]+")
PROBE = dict(C=0.1, max_iter=3000, random_state=0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grouped_auc(X, y: np.ndarray, family: np.ndarray, dense: bool) -> dict:
    per_family = {}
    for held in sorted(set(family)):
        train, test = family != held, family == held
        Xtr, Xte = X[train], X[test]
        if dense:
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        if len(set(y[train])) < 2:
            continue
        model = LogisticRegression(**PROBE).fit(Xtr, y[train])
        per_family[held] = float(roc_auc_score(y[test], model.decision_function(Xte)))
    values = np.array(list(per_family.values()))
    return {"per_family_auc": per_family, "mean_auc": float(values.mean()),
            "min_auc": float(values.min()), "max_auc": float(values.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    items = args.bank / "items.jsonl"
    manifest = json.loads((args.bank / "freeze_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines() if line.strip()]
    items_sha = sha256_file(items)
    if items_sha != manifest["items_sha256"] or manifest["status"] != "FROZEN_BEFORE_ACTIVATIONS":
        raise RuntimeError("frozen bank hash/status mismatch")

    y = np.array([r["label"] for r in rows], dtype=np.int8)
    family = np.array([r["binding_family"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    mapping = np.array([r["binding_mapping"] for r in rows])
    selected = np.array([r["binding_selected"] for r in rows])
    payload = np.array([r["payload_block_id"] for r in rows])
    checks: dict[str, object] = {}

    cells = Counter((r["binding_family"], r["payload_block_id"], r["binding_mapping"],
                     r["binding_selected"], r["format"]) for r in rows)
    checks["1_exact_factorial_balance"] = {
        "n_items": len(rows), "unique_cells": len(cells),
        "every_cell_exactly_once": set(cells.values()) == {1},
        "label_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
    }
    per_payload = Counter((r["binding_family"], r["payload_block_id"]) for r in rows)
    checks["2_payload_in_all_eight_cells"] = {
        "blocks": len(per_payload), "all_blocks_have_8_items": set(per_payload.values()) == {8},
    }

    def balance(keys: np.ndarray) -> dict:
        out = {}
        for key in sorted(set(keys.tolist())):
            m = keys == key
            out[str(key)] = {"n": int(m.sum()), "label_mean": float(y[m].mean())}
        return out

    name_keys = np.array([r["current_route_name"] for r in rows])
    name_bal = balance(name_keys)
    checks["3_nonce_route_purpose_balance"] = {
        "n_distinct_current_route_names": len(name_bal),
        "all_exactly_chance": all(abs(v["label_mean"] - 0.5) < 1e-12 for v in name_bal.values()),
        "max_abs_deviation": max(abs(v["label_mean"] - 0.5) for v in name_bal.values()),
    }
    checks["4_family_purpose_balance"] = balance(family)
    checks["5_format_purpose_balance"] = balance(fmt)
    checks["6_label_is_deterministic_interaction"] = {
        "holds_for_all_items": bool(np.all(y == (mapping * selected == 1).astype(np.int8))),
        "mapping_alone_label_mean": balance(mapping),
        "selected_alone_label_mean": balance(selected),
    }

    role_counter: dict[tuple[str, str], Counter] = {}
    for r, label in zip(rows, y.tolist()):
        head = r["text"].split("\n\n")[0]
        for role, name in (("measure", r["route_measure_name"]), ("operate", r["route_operate_name"])):
            sentence = next(s for s in head.split(". ") if name in s and "now being written" not in s)
            role_counter.setdefault((r["binding_family"], role), Counter())[label] += 1
            _ = sentence
    checks["7_role_description_label_balance"] = {
        f"{fam}::{role}": {"label0": int(c[0]), "label1": int(c[1]), "balanced": c[0] == c[1]}
        for (fam, role), c in sorted(role_counter.items())
    }
    banned = set(manifest["banned_old_cue_vocabulary"])
    scaffold_hits = Counter()
    for r in rows:
        head = r["text"].split("\n\n")[0]
        scaffold_hits.update({w.lower() for w in TOKEN_RE.findall(head)} & banned)
    checks["8_old_cue_vocabulary_in_scaffolding"] = {
        "hits": dict(scaffold_hits), "total_hits": int(sum(scaffold_hits.values())),
        "note": "reported for continuity; not the validity argument for binding_v1",
    }

    texts = [r["text"] for r in rows]
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000)
    checks["9a_word_1_2gram_leakage_leave_one_family_out"] = grouped_auc(
        word.fit_transform(texts), y, family, dense=False)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000)
    checks["9b_char_3_5gram_leakage_leave_one_family_out"] = grouped_auc(
        char.fit_transform(texts), y, family, dense=False)

    setups = [r["text"].split("\n\n")[0] for r in rows]
    setup_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2)
    checks["10_setup_only_word_1_3gram_leave_one_family_out"] = grouped_auc(
        setup_word.fit_transform(setups), y, family, dense=False)

    meta = {}
    all_names = sorted({r["current_route_name"] for r in rows}
                       | {r["route_measure_name"] for r in rows})
    name_index = {n: i for i, n in enumerate(all_names)}
    onehot = np.zeros((len(rows), len(all_names)), dtype=np.float32)
    for i, r in enumerate(rows):
        onehot[i, name_index[r["current_route_name"]]] = 1.0
        onehot[i, name_index[r["route_measure_name"]]] += 1.0
    meta["nonce_route_identity"] = grouped_auc(onehot, y, family, dense=False)
    meta["mapping_only"] = grouped_auc(mapping.reshape(-1, 1).astype(np.float32), y, family, dense=True)
    meta["selection_only"] = grouped_auc(selected.reshape(-1, 1).astype(np.float32), y, family, dense=True)
    meta["format_only"] = grouped_auc((fmt == "benchmark").reshape(-1, 1).astype(np.float32),
                                      y, family, dense=True)
    payload_names = sorted(set(payload.tolist()))
    payload_onehot = np.zeros((len(rows), len(payload_names)), dtype=np.float32)
    for i, p in enumerate(payload.tolist()):
        payload_onehot[i, payload_names.index(p)] = 1.0
    meta["payload_identity"] = grouped_auc(payload_onehot, y, family, dense=False)
    checks["11_metadata_only_baselines"] = meta

    report = {
        "schema_version": 1,
        "experiment": "binding_v1",
        "audited_before_activations": True,
        "bank": str(args.bank).replace("\\", "/"),
        "items_sha256": items_sha,
        "freeze_manifest_sha256": sha256_file(args.bank / "freeze_manifest.json"),
        "n_items": len(rows),
        "n_families": len(set(family.tolist())),
        "inference_unit": "binding_family",
        "text_baseline_cv": "leave-one-entire-relational-family-out",
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    summary = {
        "n_items": len(rows),
        "every_cell_exactly_once": checks["1_exact_factorial_balance"]["every_cell_exactly_once"],
        "all_blocks_have_8_items": checks["2_payload_in_all_eight_cells"]["all_blocks_have_8_items"],
        "label_is_interaction": checks["6_label_is_deterministic_interaction"]["holds_for_all_items"],
        "nonce_names_at_chance": checks["3_nonce_route_purpose_balance"]["all_exactly_chance"],
        "word_leakage_mean_auc": checks["9a_word_1_2gram_leakage_leave_one_family_out"]["mean_auc"],
        "char_leakage_mean_auc": checks["9b_char_3_5gram_leakage_leave_one_family_out"]["mean_auc"],
        "setup_only_mean_auc": checks["10_setup_only_word_1_3gram_leave_one_family_out"]["mean_auc"],
        "metadata_mean_aucs": {k: v["mean_auc"] for k, v in meta.items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
