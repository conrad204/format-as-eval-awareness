"""Uniform read-out for every frozen forced-choice recognition bank.

One code path for the evaluative latent bank, the explicit-statement check, the
incongruent check, the dose-16 supplement and the arbitrary-contingency control,
so no bank gets a bespoke analysis. For each run it reports accuracy with a
two-way crossed bootstrap over episode families and question families, the split
by mapping (a constant channel-letter prior scores 0.5 pooled), the recency split
(is the queried channel the one named in the final episode outcome), per-question
accuracy, and family-level consistency.

A run file may carry several independent banks (the supplement file holds both
`incongruent_*` and `dose16_*` conditions). Pooling those would average two
different experiments, so `--group PREFIX` restricts the identical estimator to
the conditions starting with PREFIX and reports one record per group instead of a
pooled number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"\b([KMQR])\b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def analyze(rec: list[dict], get_indices, last_token: dict[str, str],
            query_token: dict[str, str]) -> dict:
    """Crossed-bootstrap read-out for one bank's recognition rows."""
    families = sorted({r["episode_family"] for r in rec})
    questions = sorted({r["question_family"] for r in rec})
    conditions = sorted({r["condition"] for r in rec})
    index: dict[tuple[str, str], list[float]] = {}
    for r in rec:
        index.setdefault((r["episode_family"], r["question_family"]), []).append(
            float(r["target_logit_margin"] > 0))
    missing = [(f, q) for f in families for q in questions if (f, q) not in index]
    if missing:
        raise RuntimeError(f"ragged crossing: {len(missing)} empty cells, e.g. {missing[0]}")
    counts = {len(v) for v in index.values()}
    if len(counts) != 1:
        raise RuntimeError(f"ragged crossing: cell sizes {counts}")
    grid = np.array([[np.mean(index[(f, q)]) for q in questions] for f in families])
    ih, iq = get_indices(len(families), len(questions))
    draws = np.array([grid[np.ix_(a, b)].mean() for a, b in zip(ih, iq)])

    by_condition = {c: float(np.mean([float(r["target_logit_margin"] > 0)
                                      for r in rec if r["condition"] == c])) for c in conditions}
    mapping_split = {}
    for prefix in ("map0", "map1", "scrambled", "dose16_map0", "dose16_map1",
                   "incongruent", "explicit"):
        sel = [float(r["target_logit_margin"] > 0) for r in rec if r["condition"].startswith(prefix)]
        if sel:
            mapping_split[prefix] = float(np.mean(sel))
    recency = {}
    if last_token:
        same = [float(r["target_logit_margin"] > 0) for r in rec
                if last_token.get(r["item_id"]) and query_token.get(r["item_id"]) == last_token.get(r["item_id"])]
        diff = [float(r["target_logit_margin"] > 0) for r in rec
                if last_token.get(r["item_id"]) and query_token.get(r["item_id"]) != last_token.get(r["item_id"])]
        if same and diff:
            recency = {"queried_channel_is_last_episode_channel": float(np.mean(same)),
                       "n_same": len(same),
                       "queried_channel_differs": float(np.mean(diff)), "n_diff": len(diff)}
    return {
        "n_recognition_rows": len(rec), "n_episode_families": len(families),
        "n_question_families": len(questions), "conditions": conditions,
        "accuracy": float(grid.mean()), "accuracy_crossed_ci95": ci(draws),
        "above_chance": bool(ci(draws)[0] > 0.5),
        "accuracy_by_condition": by_condition,
        "accuracy_by_mapping_group": mapping_split,
        "recency_split": recency,
        "accuracy_by_question": {q: float(grid[:, j].mean()) for j, q in enumerate(questions)},
        "families_above_chance": int((grid.mean(axis=1) > 0.5).sum()),
        "questions_above_chance": int((grid.mean(axis=0) > 0.5).sum()),
    }


def report(label: str, key: str, rec: dict) -> None:
    print(f"[{label}] {key}: acc {rec['accuracy']:.4f} CI {[round(x, 4) for x in rec['accuracy_crossed_ci95']]}"
          f" | above chance {rec['above_chance']}"
          f" | families {rec['families_above_chance']}/{rec['n_episode_families']}"
          f" | questions {rec['questions_above_chance']}/{rec['n_question_families']}", flush=True)
    if rec["accuracy_by_mapping_group"]:
        print("   mapping groups:", {k: round(v, 4) for k, v in rec["accuracy_by_mapping_group"].items()})
    if rec["recency_split"]:
        print("   recency:", {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in rec["recency_split"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--bank-items", type=Path, default=None,
                    help="frozen items file, needed for the recency split")
    ap.add_argument("--label", required=True)
    ap.add_argument("--group", action="append", default=None,
                    help="condition prefix to analyze separately; repeatable. "
                         "Without it the whole run is pooled into one bank.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    last_token: dict[str, str] = {}
    query_token: dict[str, str] = {}
    if args.bank_items:
        for line in args.bank_items.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "recognition":
                continue
            outcome = row["messages"][-1]["content"].split("\n\n")[0]
            found = TOKEN_RE.search(outcome)
            if found:
                last_token[row["item_id"]] = found.group(1)
            query_token[row["item_id"]] = row.get("query_token", "")

    rng = np.random.default_rng(args.seed)
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def get_indices(n_families: int, n_questions: int) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap indices shared by every bank of the same shape."""
        shape = (n_families, n_questions)
        if shape not in cache:
            cache[shape] = (rng.integers(0, n_families, (args.n_bootstrap, n_families)),
                            rng.integers(0, n_questions, (args.n_bootstrap, n_questions)))
        return cache[shape]

    models = {}
    for path in args.runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = [r for r in data["rows"] if r["kind"] == "recognition"]
        if not rec:
            raise RuntimeError(f"{path} has no recognition rows")
        record = {"source": str(path), "source_sha256": sha256(path),
                  "precision": data.get("precision"), "load_in_8bit": data.get("load_in_8bit", False)}
        if args.group:
            groups = {}
            for prefix in args.group:
                subset = [r for r in rec if r["condition"].startswith(prefix)]
                if not subset:
                    raise RuntimeError(f"{path} has no condition starting with {prefix!r}")
                groups[prefix] = analyze(subset, get_indices, last_token, query_token)
            covered = sum(g["n_recognition_rows"] for g in groups.values())
            if covered != len(rec):
                raise RuntimeError(f"{path}: groups cover {covered} of {len(rec)} recognition rows")
            record["groups"] = groups
        else:
            record.update(analyze(rec, get_indices, last_token, query_token))
        models[data["model_key"]] = record

    result = {"schema_version": 1, "label": args.label, "n_bootstrap": args.n_bootstrap,
              "seed": args.seed, "groups": args.group,
              "inference": "two-way crossed episode-family x question-family bootstrap, shared indices",
              "models": models}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, rec in models.items():
        if "groups" in rec:
            for prefix, group in rec["groups"].items():
                report(f"{args.label}:{prefix}", key, group)
        else:
            report(args.label, key, rec)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
