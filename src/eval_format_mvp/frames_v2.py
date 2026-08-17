"""Vocabulary-fold frame catalog (v2): loading and disjointness lint.

The v2 catalog groups purpose pairs into five vocabulary folds. Probing with
leave-one-fold-out splits only tests transfer to unseen purpose vocabulary if
the folds' cue lexicons are actually disjoint, so the lint enforces:

R1  Within every pair, the class-asymmetric content stems (tokens appearing on
    one side only) exactly match the declared cue_tokens (modulo a tiny
    closed-class stopword list). Nothing undeclared may separate the classes.
R2  Declared cue stems are disjoint across folds.
R3  A stem declared by one fold may appear in another fold's sentences only
    class-symmetrically (equal evaluation/deployment counts in that fold), so
    a probe trained on other folds can earn no weight for it.
R4  (functional, optional) A held-fold TF-IDF logistic-regression probe on the
    cue sentences alone must score near chance. This is the acceptance test
    that R1-R3 bookkeeping actually removed the lexical shortcut; it uses the
    same vectorizer/probe family as the permanent n-gram control in
    plot-pipeline sweep_full.py.

Stated pairs must additionally be minimal pairs: after removing declared cue
tokens, both sides reduce to the identical token sequence. Indirect and
second-person "context" pairs are exempt from the minimal-pair rule but not
from R1-R3, and must stay length-matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import MVPError, PROJECT_ROOT, load_yaml


FOLD_IDS = ("fg1", "fg2", "fg3", "fg4", "fg5")
CUE_STYLES = ("stated", "indirect")
CUE_PERSPECTIVES = ("third", "second")
CUE_LOCI = ("noun_only", "verb_only", "both", "context")

# Closed-class tokens that may differ between the two sides of a pair without
# being declared. Deliberately tiny: prepositions/particles (on, over, under,
# out, through, into, from, with, about, up) are NOT here — when asymmetric
# they carry purpose signal, so they must be declared and fold-owned.
STOPWORDS = frozenset(
    "a an the this that these those is are was were be being been will of and or to it its"
    " you your yours".split()
)

# Fold structure every fold must satisfy: (style, perspective, locus) -> count.
FOLD_COMPOSITION = {
    ("stated", "third", "both"): 2,
    ("stated", "third", "noun_only"): 2,
    ("stated", "third", "verb_only"): 2,
    ("indirect", "third", "context"): 2,
    ("stated", "second", "noun_only"): 1,
    ("stated", "second", "context"): 1,
}

CONTEXT_LENGTH_TOLERANCE = 3
SUBSTRING_ADVISORY_MIN_LENGTH = 4


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, single letters dropped (possessive 's')."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 1]


def stem(token: str) -> str:
    """Tiny deterministic suffix-stripper; only used for fold bookkeeping."""
    word = token
    for suffix, replacement in (
        ("ers", ""), ("ors", ""), ("ies", "y"), ("ing", ""), ("ed", ""),
        ("es", ""), ("er", ""), ("or", ""), ("s", ""),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: len(word) - len(suffix)] + replacement
            break
    if word.endswith("e") and len(word) >= 4:
        word = word[:-1]
    if word.endswith("y") and len(word) >= 4:
        word = word[:-1] + "i"
    return word


def _stems(tokens: list[str]) -> list[str]:
    return [stem(t) for t in tokens]


@dataclass(frozen=True)
class FramesV2:
    pairs: tuple[dict[str, Any], ...]
    folds: tuple[dict[str, Any], ...]

    @property
    def by_fold(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {f: [] for f in FOLD_IDS}
        for pair in self.pairs:
            grouped[str(pair["fold_id"])].append(pair)
        return grouped

    def pair(self, pair_id: str) -> dict[str, Any]:
        for row in self.pairs:
            if str(row["pair_id"]) == pair_id:
                return row
        raise MVPError(f"Unknown purpose pair {pair_id!r}")


def load_frames_v2(root: Path = PROJECT_ROOT) -> FramesV2:
    value = load_yaml(root / "catalogs" / "purpose_frames_v2.yaml")
    if value.get("schema_version") != 2:
        raise MVPError("purpose_frames_v2.yaml must declare schema_version 2")
    pairs = tuple(value.get("pairs", []))
    folds = tuple(value.get("folds", []))
    if {str(f["fold_id"]) for f in folds} != set(FOLD_IDS):
        raise MVPError(f"v2 catalog must describe folds {FOLD_IDS}")
    seen: set[str] = set()
    for pair in pairs:
        pair_id = str(pair.get("pair_id"))
        if not re.fullmatch(r"vp[0-9]{2}", pair_id):
            raise MVPError(f"Bad pair_id {pair_id!r}")
        if pair_id in seen:
            raise MVPError(f"Duplicate pair_id {pair_id}")
        seen.add(pair_id)
        if str(pair.get("fold_id")) not in FOLD_IDS:
            raise MVPError(f"{pair_id}: unknown fold_id {pair.get('fold_id')!r}")
        if str(pair.get("cue_style")) not in CUE_STYLES:
            raise MVPError(f"{pair_id}: bad cue_style")
        if str(pair.get("cue_perspective")) not in CUE_PERSPECTIVES:
            raise MVPError(f"{pair_id}: bad cue_perspective")
        if str(pair.get("cue_locus")) not in CUE_LOCI:
            raise MVPError(f"{pair_id}: bad cue_locus")
        if set(pair.get("en", {})) != {"evaluation", "deployment"}:
            raise MVPError(f"{pair_id}: en must contain evaluation and deployment")
        tokens = pair.get("cue_tokens", {})
        if set(tokens) != {"evaluation", "deployment"}:
            raise MVPError(f"{pair_id}: cue_tokens must cover both classes")
    for fold_id, group in FramesV2(pairs, folds).by_fold.items():
        counts: dict[tuple[str, str, str], int] = {}
        for pair in group:
            key = (str(pair["cue_style"]), str(pair["cue_perspective"]), str(pair["cue_locus"]))
            counts[key] = counts.get(key, 0) + 1
        if counts != FOLD_COMPOSITION:
            raise MVPError(
                f"Fold {fold_id} composition {counts} != required {FOLD_COMPOSITION}"
            )
    return FramesV2(pairs, folds)


def _pair_asymmetry(pair: dict[str, Any]) -> dict[str, set[str]]:
    """Class-asymmetric stems per side, stopwords removed."""
    sides = {}
    for purpose in ("evaluation", "deployment"):
        sides[purpose] = [
            t for t in tokenize(str(pair["en"][purpose])) if t not in STOPWORDS
        ]
    eval_stems = set(_stems(sides["evaluation"]))
    deploy_stems = set(_stems(sides["deployment"]))
    return {
        "evaluation": eval_stems - deploy_stems,
        "deployment": deploy_stems - eval_stems,
    }


def lint_frames_v2(frames: FramesV2) -> dict[str, Any]:
    """Enforce R1-R3 plus the minimal-pair rule; return a report."""
    declared_by_fold: dict[str, set[str]] = {f: set() for f in FOLD_IDS}
    problems: list[str] = []

    # R1 + minimal-pair structure + declared bookkeeping.
    for pair in frames.pairs:
        pair_id = str(pair["pair_id"])
        declared = {
            purpose: set(_stems([str(t).lower() for t in pair["cue_tokens"][purpose]]))
            for purpose in ("evaluation", "deployment")
        }
        asymmetric = _pair_asymmetry(pair)
        for purpose in ("evaluation", "deployment"):
            if asymmetric[purpose] != declared[purpose]:
                problems.append(
                    f"R1 {pair_id} {purpose}: asymmetric stems "
                    f"{sorted(asymmetric[purpose])} != declared {sorted(declared[purpose])}"
                )
            declared_by_fold[str(pair["fold_id"])] |= declared[purpose]
        eval_tokens = tokenize(str(pair["en"]["evaluation"]))
        deploy_tokens = tokenize(str(pair["en"]["deployment"]))
        cue_stems = declared["evaluation"] | declared["deployment"]
        if str(pair["cue_locus"]) == "context":
            if abs(len(eval_tokens) - len(deploy_tokens)) > CONTEXT_LENGTH_TOLERANCE:
                problems.append(
                    f"length {pair_id}: context pair sides differ by more than "
                    f"{CONTEXT_LENGTH_TOLERANCE} tokens"
                )
        else:
            scaffold_eval = [
                t for t in eval_tokens if stem(t) not in cue_stems and t not in STOPWORDS
            ]
            scaffold_deploy = [
                t for t in deploy_tokens if stem(t) not in cue_stems and t not in STOPWORDS
            ]
            if scaffold_eval != scaffold_deploy:
                problems.append(
                    f"minimal-pair {pair_id}: scaffolds diverge "
                    f"{scaffold_eval} != {scaffold_deploy}"
                )

    # R2: declared stems disjoint across folds.
    for i, fold_a in enumerate(FOLD_IDS):
        for fold_b in FOLD_IDS[i + 1 :]:
            shared = declared_by_fold[fold_a] & declared_by_fold[fold_b]
            if shared:
                problems.append(f"R2 {fold_a}/{fold_b} share cue stems {sorted(shared)}")

    # R3: a fold's declared stems must be class-symmetric inside other folds.
    fold_class_counts: dict[str, dict[str, dict[str, int]]] = {}
    for fold_id, group in frames.by_fold.items():
        counts: dict[str, dict[str, int]] = {}
        for pair in group:
            for purpose in ("evaluation", "deployment"):
                for token_stem in _stems(tokenize(str(pair["en"][purpose]))):
                    entry = counts.setdefault(token_stem, {"evaluation": 0, "deployment": 0})
                    entry[purpose] += 1
        fold_class_counts[fold_id] = counts
    for owner in FOLD_IDS:
        for token_stem in sorted(declared_by_fold[owner]):
            for other in FOLD_IDS:
                if other == owner:
                    continue
                entry = fold_class_counts[other].get(token_stem)
                if entry and entry["evaluation"] != entry["deployment"]:
                    problems.append(
                        f"R3 stem {token_stem!r} owned by {owner} appears "
                        f"class-asymmetrically in {other} ({entry})"
                    )

    # Advisory: cross-fold substring relations that a subword tokenizer could
    # exploit even though the word-level stems are disjoint.
    warnings: list[str] = []
    for i, fold_a in enumerate(FOLD_IDS):
        for fold_b in FOLD_IDS[i + 1 :]:
            for stem_a in sorted(declared_by_fold[fold_a]):
                for stem_b in sorted(declared_by_fold[fold_b]):
                    short, long_ = sorted((stem_a, stem_b), key=len)
                    if len(short) >= SUBSTRING_ADVISORY_MIN_LENGTH and short in long_:
                        warnings.append(
                            f"substring: {fold_a}:{stem_a!r} vs {fold_b}:{stem_b!r}"
                        )

    if problems:
        raise MVPError(
            "frames-v2 lint failed:\n" + "\n".join(f"  {p}" for p in problems)
        )
    return {
        "status": "passed",
        "pairs": len(frames.pairs),
        "declared_stems_per_fold": {
            fold: sorted(stems) for fold, stems in declared_by_fold.items()
        },
        "warnings": warnings,
    }


def ngram_holdout_check(
    frames: FramesV2, *, texts_and_meta: list[tuple[str, str, int]] | None = None
) -> dict[str, Any]:
    """R4: held-fold TF-IDF logistic regression on cue sentences ~ chance.

    By default runs on the catalog's own 100 sentences (fold = CV group).
    Pass texts_and_meta = [(text, fold_id, label), ...] to run the same check
    on rendered rows. Mirrors the permanent n-gram control in sweep_full.py
    (word TF-IDF, 1-2 grams, LogisticRegression C=1.0, random_state=0).
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise MVPError(
            "R4 n-gram check needs numpy+scikit-learn (pip install -e '.[dev]')"
        ) from exc

    if texts_and_meta is None:
        texts_and_meta = [
            (str(pair["en"][purpose]), str(pair["fold_id"]), 1 if purpose == "evaluation" else 0)
            for pair in frames.pairs
            for purpose in ("evaluation", "deployment")
        ]
    texts = np.array([row[0] for row in texts_and_meta])
    fold = np.array([row[1] for row in texts_and_meta])
    labels = np.array([int(row[2]) for row in texts_and_meta])
    scores = np.full(len(texts), np.nan)
    for held in sorted(set(fold)):
        train, test = fold != held, fold == held
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        x_train = vectorizer.fit_transform(texts[train])
        x_test = vectorizer.transform(texts[test])
        model = LogisticRegression(C=1.0, max_iter=3000, random_state=0)
        model.fit(x_train, labels[train])
        scores[test] = model.decision_function(x_test)
    auc = float(roc_auc_score(labels, scores))
    return {
        "auc": auc,
        "n_texts": int(len(texts)),
        "near_chance": bool(0.35 <= auc <= 0.65),
    }
