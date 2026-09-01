"""Uniform read-out for every frozen forced-choice recognition bank.

One code path for the evaluative latent bank, the explicit-statement check, the
incongruent check, the dose-16 supplement and the arbitrary-contingency control,
so no bank gets a bespoke analysis. For each run it reports accuracy with a
two-way crossed bootstrap over episode families and question families, the split
by mapping (a constant channel-letter prior scores 0.5 pooled), the recency split
(is the queried channel the one named in the final episode outcome), per-question
accuracy, and family-level consistency.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--bank-items", type=Path, default=None,
                    help="frozen items file, needed for the recency split")
    ap.add_argument("--label", required=True)
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
    indices = None
    models = {}
    for path in args.runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = [r for r in data["rows"] if r["kind"] == "recognition"]
        if not rec:
            raise RuntimeError(f"{path} has no recognition rows")
        families = sorted({r["episode_family"] for r in rec})
        questions = sorted({r["question_family"] for r in rec})
        conditions = sorted({r["condition"] for r in rec})
        index: dict[tuple[str, str], list[float]] = {}
        for r in rec:
            index.setdefault((r["episode_family"], r["question_family"]), []).append(
                float(r["target_logit_margin"] > 0))
        counts = {len(v) for v in index.values()}
        if len(counts) != 1:
            raise RuntimeError(f"ragged crossing in {path}: cell sizes {counts}")
        grid = np.array([[np.mean(index[(f, q)]) for q in questions] for f in families])
        if indices is None:
            indices = (rng.integers(0, len(families), (args.n_bootstrap, len(families))),
                       rng.integers(0, len(questions), (args.n_bootstrap, len(questions))))
        ih, iq = indices
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
        models[data["model_key"]] = {
            "source": str(path), "source_sha256": sha256(path),
            "precision": data.get("precision"), "load_in_8bit": data.get("load_in_8bit", False),
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
    result = {"schema_version": 1, "label": args.label, "n_bootstrap": args.n_bootstrap,
              "seed": args.seed,
              "inference": "two-way crossed episode-family x question-family bootstrap, shared indices",
              "models": models}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, rec in models.items():
        print(f"[{args.label}] {key}: acc {rec['accuracy']:.4f} CI {[round(x, 4) for x in rec['accuracy_crossed_ci95']]}"
              f" | above chance {rec['above_chance']}"
              f" | families {rec['families_above_chance']}/{rec['n_episode_families']}"
              f" | questions {rec['questions_above_chance']}/{rec['n_question_families']}", flush=True)
        if rec["accuracy_by_mapping_group"]:
            print("   mapping groups:", {k: round(v, 4) for k, v in rec["accuracy_by_mapping_group"].items()})
        if rec["recency_split"]:
            print("   recency:", {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in rec["recency_split"].items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
