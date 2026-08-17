"""Semantic near-duplicate detection for canonical payloads.

Exact-hash checks only catch byte-identical payloads; this module catches
paraphrases (e.g. five renderings of "classify the C-E-G triad") with a
dependency-free TF-IDF cosine over word 1-2-grams of each payload's
context, question, and correct answer. Deterministic by construction.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Calibrated on data/payloads.grok43-medium_gpt54.full.audited.jsonl: at 0.42
# the five near-identical t035 triad payloads form one connected cluster
# (top pairwise similarities 0.437-0.547) while the maximum similarity between
# payloads of different topics is 0.186.
DEFAULT_NEAR_DUP_THRESHOLD = 0.42


def payload_dedup_text(payload: dict[str, Any]) -> str:
    return " ".join(
        str(payload.get(key, "")) for key in ("context", "question", "correct_answer")
    )


def _terms(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    """L2-normalized TF-IDF vectors (smoothed IDF, matching sklearn's default)."""
    term_lists = [_terms(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for terms in term_lists:
        document_frequency.update(set(terms))
    n = len(texts)
    idf = {
        term: math.log((1 + n) / (1 + count)) + 1.0
        for term, count in document_frequency.items()
    }
    vectors: list[dict[str, float]] = []
    for terms in term_lists:
        weighted = {term: count * idf[term] for term, count in Counter(terms).items()}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        vectors.append(
            {term: value / norm for term, value in weighted.items()} if norm else {}
        )
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def near_duplicate_pairs(
    payloads: list[dict[str, Any]], threshold: float = DEFAULT_NEAR_DUP_THRESHOLD
) -> list[dict[str, Any]]:
    identifiers = [str(row["payload_block_id"]) for row in payloads]
    vectors = tfidf_vectors([payload_dedup_text(row) for row in payloads])
    pairs: list[dict[str, Any]] = []
    for i in range(len(payloads)):
        for j in range(i + 1, len(payloads)):
            similarity = cosine(vectors[i], vectors[j])
            if similarity >= threshold:
                pairs.append(
                    {
                        "a": identifiers[i],
                        "b": identifiers[j],
                        "similarity": round(similarity, 4),
                    }
                )
    pairs.sort(key=lambda row: (-row["similarity"], row["a"], row["b"]))
    return pairs


def near_duplicate_clusters(
    payloads: list[dict[str, Any]], threshold: float = DEFAULT_NEAR_DUP_THRESHOLD
) -> list[list[str]]:
    """Connected components of the near-duplicate pair graph, largest first."""
    pairs = near_duplicate_pairs(payloads, threshold)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        parent[find(pair["a"])] = find(pair["b"])
    members: dict[str, set[str]] = {}
    for pair in pairs:
        for key in (pair["a"], pair["b"]):
            members.setdefault(find(key), set()).add(key)
    clusters = [sorted(group) for group in members.values()]
    clusters.sort(key=lambda group: (-len(group), group[0]))
    return clusters


def near_duplicate_report(
    payloads: list[dict[str, Any]], threshold: float = DEFAULT_NEAR_DUP_THRESHOLD
) -> dict[str, Any]:
    pairs = near_duplicate_pairs(payloads, threshold)
    return {
        "method": "tfidf_cosine_word_1_2gram_v1",
        "threshold": threshold,
        "payloads": len(payloads),
        "flagged_pairs": pairs,
        "clusters": near_duplicate_clusters(payloads, threshold),
    }


def most_similar_payload(
    candidate: dict[str, Any], accepted: list[dict[str, Any]]
) -> tuple[str, float]:
    """The accepted payload most similar to the candidate, as (id, similarity)."""
    if not accepted:
        return "", 0.0
    vectors = tfidf_vectors(
        [payload_dedup_text(candidate)] + [payload_dedup_text(row) for row in accepted]
    )
    best_id, best_similarity = "", -1.0
    for row, vector in zip(accepted, vectors[1:]):
        similarity = cosine(vectors[0], vector)
        if similarity > best_similarity:
            best_id, best_similarity = str(row["payload_block_id"]), similarity
    return best_id, best_similarity
