"""Reasoning-allowed variant of the latent hidden-channel recognition bank.

The primary read-out forces a single letter immediately after ~700 tokens of
episodes, which is the hardest possible way to ask for an inference that requires
integrating eight prior outcomes. This bank changes exactly one thing: the model
may reason briefly before committing. Episodes, mapping reversal, channel letters,
questions, option order and family structure are byte-identical to the frozen
latent bank; only the final instruction line differs.

Any accuracy gain is therefore attributable to allowing inference to be computed
in tokens rather than in a single forward pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "behavioral_ea_hidden_channel_frozen"
OUT = ROOT / "results" / "behavioral_ea_hidden_channel_cot_frozen"
OLD_TAIL = "\n\nReply with only A or B."
NEW_TAIL = ("\n\nThink it through in at most 50 words, using what the earlier results showed about "
            "each channel, then finish with a final line of exactly 'Answer: A' or 'Answer: B'.")


def main() -> None:
    manifest_src = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((SRC / "items.jsonl").read_bytes()).hexdigest() != manifest_src["items_sha256"]:
        raise RuntimeError("source bank hash mismatch")
    rows = []
    for line in (SRC / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["kind"] != "recognition":
            continue
        final = row["messages"][-1]["content"]
        if not final.endswith(OLD_TAIL):
            raise RuntimeError(f"unexpected instruction tail in {row['item_id']}")
        messages = [dict(m) for m in row["messages"]]
        messages[-1] = {"role": "user", "content": final[: -len(OLD_TAIL)] + NEW_TAIL}
        rows.append({**row, "kind": "reasoning", "messages": messages,
                     "item_id": row["item_id"] + "::cot",
                     "unit_id": row["unit_id"] + "::cot"})
    if len(rows) != 1024:
        raise RuntimeError(f"expected 1024 reasoning prompts, built {len(rows)}")
    rendered = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    items = OUT / "items.jsonl"
    if items.exists() and items.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("refusing to overwrite a non-identical CoT bank")
    items.write_text(rendered, encoding="utf-8")
    manifest = {
        "status": "FROZEN_BEFORE_COT_OUTCOMES",
        "derived_from": manifest_src["items_sha256"],
        "single_manipulation": "final instruction permits <=50 words of reasoning before 'Answer: A/B'",
        "items_sha256": hashlib.sha256(items.read_bytes()).hexdigest(),
        "n_items": len(rows), "n_recognition_items": 0, "n_reasoning_items": len(rows),
        "n_episode_families": 32, "n_question_families": 8,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "FROZEN.sha256").write_text(
        f"items.jsonl {manifest['items_sha256']}\n"
        f"manifest.json {hashlib.sha256(manifest_path.read_bytes()).hexdigest()}\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
