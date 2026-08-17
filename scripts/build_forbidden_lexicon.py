"""Derive a conservative purpose-asymmetric lexicon from spent artifacts.

This is a pre-confirmation bookkeeping tool. It never reads credentials and it
never consumes confirmation outcomes. The v2 development stems documented in
pilot_design_v2.md are included explicitly because the raw v2 catalog is not
in this checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

V2_DOCUMENTED_STEMS = {
    "reviewer", "critique", "go", "accuracy", "qa", "customer", "act", "purchasing", "crew", "buyer", "po",
    "grader", "mark", "exam", "panel", "rubric", "points", "client", "heed", "caller", "invoice", "call",
    "auditor", "inspect", "comb", "through", "spot", "check", "tenant", "apply", "carry", "out", "work", "order", "repair",
    "scorer", "rate", "score", "table", "item", "test", "user", "employ", "forward", "ticket", "support", "queue",
    "judge", "appraise", "vet", "verdict", "shopper", "adopt", "put", "into", "practice", "grocery",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stem(token: str) -> str:
    token = token.lower().strip("'-")
    for suffix in ("ingly", "edly", "ing", "ers", "er", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=Path("data/rendered_items.jsonl"))
    parser.add_argument("--docs", type=Path, default=Path("docs/pilot_design_v2.md"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/dev_forbidden_lexicon.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.v1.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(dict)
    for row in rows:
        key = (row["payload_block_id"], row["purpose_pair_id"], row["format"])
        purpose = row["intended_purpose"]
        groups[key][purpose] = str(row["text"]).split("\n", 1)[0]
    counts: Counter[tuple[str, str]] = Counter()
    for pair in groups.values():
        if set(pair) != {"evaluation", "deployment"}:
            continue
        eval_tokens = {token.lower() for token in TOKEN_RE.findall(pair["evaluation"])}
        deploy_tokens = {token.lower() for token in TOKEN_RE.findall(pair["deployment"])}
        for token in sorted(eval_tokens ^ deploy_tokens):
            counts[(token, "evaluation" if token in eval_tokens else "deployment")] += 1
    records = []
    for (token, label), count in sorted(counts.items()):
        records.append({
            "token": token,
            "stem": stem(token),
            "source_artifacts": [str(args.v1)],
            "label_asymmetry": label,
            "count": count,
            "extraction_method": "first-line cue sentence symmetric difference within matched v1 payload/pair/format groups",
        })
    for token in sorted(V2_DOCUMENTED_STEMS):
        records.append({
            "token": token,
            "stem": stem(token),
            "source_artifacts": [str(args.docs)],
            "label_asymmetry": "development_catalog_declared_asymmetric",
            "count": 1,
            "extraction_method": "conservative explicit stem list transcribed from pilot_design_v2.md fold theme table",
        })
    unique = {}
    for record in records:
        key = (record["stem"], record["label_asymmetry"])
        if key not in unique:
            unique[key] = record
        else:
            unique[key]["count"] += record["count"]
            unique[key]["source_artifacts"] = sorted(set(unique[key]["source_artifacts"] + record["source_artifacts"]))
    output = {
        "schema_version": 1,
        "status": "development_forbidden_lexicon",
        "generated_from": {
            "v1_path": str(args.v1),
            "v1_sha256": sha256(args.v1),
            "docs_path": str(args.docs),
            "docs_sha256": sha256(args.docs),
        },
        "extraction_method": "conservative union of matched-v1 asymmetric cue stems and all explicitly documented v2 development stems; uncertain stems are retained",
        "records": sorted(unique.values(), key=lambda row: (row["stem"], row["label_asymmetry"])),
        "confirmation_use": "reject purpose-asymmetric confirmation stems; symmetric occurrences require separate pair-level validation",
    }
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(output["records"]), "sha256": sha256(args.out)}))


if __name__ == "__main__":
    main()
