"""F3 surface-control battery for v3 renders (validator plan section 4).

Probes run on cue-excised text: the recorded cue span is removed before
vectorization, so the check asks whether *anything other than the cue* — the
wrapper, contract, or their interaction with the body — predicts purpose.
Under the current single-turn catalog the eval/deploy twins of a pair become
byte-identical after excision, so the purpose gates read ~0.500 by
construction: like the v1 word-level control, they certify that wrapper
symmetry was achieved, and their teeth are for future catalogs (multi-turn
prior exchanges, noise lanes) where wrapper text could co-vary with purpose.
The template/stratum/contract probes are expected to be decodable and are
reported, not gated (Sclar et al.: formats are internally separable).

Splits are grouped, never random: held-template for the purpose gates (plus
the stricter held-template x held-fold variant), held-block for the nuisance
probes. Same probe family as frames_v2.ngram_holdout_check and the permanent
sweep control: TF-IDF + LogisticRegression(C=1.0, random_state=0), word 1-2
grams and char_wb 2-5 grams.
"""

from __future__ import annotations

import re
from typing import Any

from .io import MVPError


NEAR_CHANCE_BAND = (0.35, 0.65)
HELD_BLOCK_GROUPS = 5


def _sklearn():
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise MVPError(
            "surface probes need numpy+scikit-learn (pip install -e '.[dev]')"
        ) from exc
    return np, TfidfVectorizer, LogisticRegression, roc_auc_score


def cue_excised_text(row: dict[str, Any]) -> str:
    text = str(row["text"])
    start = row.get("cue_char_start")
    end = row.get("cue_char_end")
    if start is None or end is None:
        return text
    excised = text[: int(start)] + text[int(end) :]
    return re.sub(r"\n{3,}", "\n\n", excised).strip()


def _vectorizer(kind: str):
    _, TfidfVectorizer, _, _ = _sklearn()
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2))
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    raise MVPError(f"Unknown vectorizer kind {kind!r}")


def _grouped_oof_auc(
    texts: list[str], labels: list[int], groups: list[str], kind: str
) -> dict[str, Any]:
    np, _, LogisticRegression, roc_auc_score = _sklearn()
    text_array = np.array(texts)
    label_array = np.array(labels)
    group_array = np.array(groups)
    scores = np.full(len(texts), np.nan)
    for held in sorted(set(groups)):
        train, test = group_array != held, group_array == held
        if len(set(label_array[train].tolist())) < 2:
            raise MVPError(f"Surface probe: training split for {held!r} is one-class")
        vectorizer = _vectorizer(kind)
        x_train = vectorizer.fit_transform(text_array[train])
        x_test = vectorizer.transform(text_array[test])
        model = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
        model.fit(x_train, label_array[train])
        scores[test] = model.decision_function(x_test)
    auc = float(roc_auc_score(label_array, scores))
    return {
        "auc": auc,
        "n_texts": len(texts),
        "n_groups": len(set(groups)),
        "near_chance": bool(NEAR_CHANCE_BAND[0] <= auc <= NEAR_CHANCE_BAND[1]),
    }


def _double_split_auc(
    texts: list[str],
    labels: list[int],
    templates: list[str],
    folds: list[str],
    kind: str,
) -> dict[str, Any]:
    """Held-template x held-fold: train rows share neither the test rows'
    template nor their vocabulary fold."""
    np, _, LogisticRegression, roc_auc_score = _sklearn()
    text_array = np.array(texts)
    label_array = np.array(labels)
    template_array = np.array(templates)
    fold_array = np.array(folds)
    scores = np.full(len(texts), np.nan)
    for held_template in sorted(set(templates)):
        for held_fold in sorted(set(folds)):
            test = (template_array == held_template) & (fold_array == held_fold)
            if not test.any():
                continue
            train = (template_array != held_template) & (fold_array != held_fold)
            vectorizer = _vectorizer(kind)
            x_train = vectorizer.fit_transform(text_array[train])
            x_test = vectorizer.transform(text_array[test])
            model = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
            model.fit(x_train, label_array[train])
            scores[test] = model.decision_function(x_test)
    covered = ~np.isnan(scores)
    auc = float(roc_auc_score(label_array[covered], scores[covered]))
    return {
        "auc": auc,
        "n_texts": int(covered.sum()),
        "near_chance": bool(NEAR_CHANCE_BAND[0] <= auc <= NEAR_CHANCE_BAND[1]),
    }


def _grouped_oof_accuracy(
    texts: list[str], labels: list[str], groups: list[str], kind: str
) -> dict[str, Any]:
    np, _, LogisticRegression, _ = _sklearn()
    text_array = np.array(texts)
    label_array = np.array(labels)
    group_array = np.array(groups)
    predictions = np.empty(len(texts), dtype=object)
    for held in sorted(set(groups)):
        train, test = group_array != held, group_array == held
        vectorizer = _vectorizer(kind)
        x_train = vectorizer.fit_transform(text_array[train])
        x_test = vectorizer.transform(text_array[test])
        model = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
        model.fit(x_train, label_array[train])
        predictions[test] = model.predict(x_test)
    accuracy = float((predictions == label_array).mean())
    return {
        "accuracy": accuracy,
        "n_classes": len(set(labels)),
        "chance": 1.0 / len(set(labels)),
        "n_texts": len(texts),
    }


def surface_probe_battery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the pre-declared expectation table; hard-gate the purpose probes."""
    from .render_v2 import _digest  # local import to avoid cycles

    stated = [r for r in rows if r.get("render_slot") == "stated"]
    if not stated:
        raise MVPError("Surface battery needs stated (cued) rows")
    texts = [cue_excised_text(r) for r in stated]
    labels = [int(r["label"]) for r in stated]
    templates = [str(r["template_id"]) for r in stated]
    folds = [str(r["purpose_family_id"]) for r in stated]
    strata = [1 if r["format"] == "benchmark" else 0 for r in stated]
    block_groups = [
        f"bg{_digest(str(r['payload_block_id'])) % HELD_BLOCK_GROUPS}" for r in stated
    ]

    report: dict[str, Any] = {"expectations": {}}
    gates_failed: list[str] = []
    for kind in ("word", "char"):
        probe = _grouped_oof_auc(texts, labels, templates, kind)
        report["expectations"][f"purpose_held_template_{kind}"] = {
            **probe, "role": "hard_gate",
        }
        if not probe["near_chance"]:
            gates_failed.append(f"purpose_held_template_{kind} auc={probe['auc']:.3f}")
        double = _double_split_auc(texts, labels, templates, folds, kind)
        report["expectations"][f"purpose_held_template_x_fold_{kind}"] = {
            **double, "role": "hard_gate",
        }
        if not double["near_chance"]:
            gates_failed.append(
                f"purpose_held_template_x_fold_{kind} auc={double['auc']:.3f}"
            )

    report["expectations"]["template_id_held_block_word"] = {
        **_grouped_oof_accuracy(texts, templates, block_groups, "word"),
        "role": "report_only_expected_high",
    }
    report["expectations"]["stratum_held_template_word"] = {
        **_grouped_oof_auc(texts, strata, templates, "word"),
        "role": "report_only_expected_decodable",
    }
    contract_rows = [
        (t, str(r["contract_id"]), g)
        for t, r, g in zip(texts, stated, block_groups)
        if r.get("contract_id") is not None
    ]
    if contract_rows:
        report["expectations"]["contract_held_block_word"] = {
            **_grouped_oof_accuracy(
                [t for t, _, _ in contract_rows],
                [c for _, c, _ in contract_rows],
                [g for _, _, g in contract_rows],
                "word",
            ),
            "role": "report_only_expected_decodable",
        }

    report["near_chance_band"] = list(NEAR_CHANCE_BAND)
    report["gates_passed"] = not gates_failed
    if gates_failed:
        raise MVPError(
            "Surface battery purpose gates failed (wrapper text leaks purpose): "
            + "; ".join(gates_failed)
        )
    return report
