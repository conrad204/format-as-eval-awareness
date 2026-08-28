"""Build development-only no-cue format battery from existing pilot payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = {}
    for row in rows:
        if row["render_slot"] == "no_cue" and row["format"] == "benchmark":
            source.setdefault(row["payload_block_id"], row)
    if len(source) != 80:
        raise ValueError(f"expected 80 no-cue payload blocks, got {len(source)}")
    families = (
        ("mcq_letter", "benchmark"),
        ("numbered_options", "benchmark"),
        ("json_schema", "benchmark"),
        ("exact_label", "benchmark"),
        ("conversational", "casual"),
        ("email_style", "casual"),
        ("chat_explanation", "casual"),
        ("bullet_answer", "casual"),
    )
    output = []
    for block_id, row in sorted(source.items()):
        text = row["text"]
        if not text.startswith("Question:\n") or "\n\nOptions:\n" not in text:
            raise ValueError(f"unexpected benchmark rendering for {block_id}")
        core, options = text[len("Question:\n"):].split("\n\nOptions:\n", 1)
        option_lines = options.strip().splitlines()
        options_text = "\n".join(option_lines)
        for family, polarity in families:
            if family == "mcq_letter":
                rendered = f"Question:\n{core}\n\nOptions:\n{options_text}"
            elif family == "numbered_options":
                numbered = []
                for index, line in enumerate(option_lines, 1):
                    numbered.append(f"{index}. {line[3:] if len(line) > 2 and line[1:3] == '. ' else line}")
                rendered = f"Question:\n{core}\n\nChoices:\n" + "\n".join(numbered)
            elif family == "json_schema":
                rendered = f"Question:\n{core}\n\nChoices:\n{options_text}\n\nRespond as JSON with exactly one field: {{\"answer\": \"your choice\"}}."
            elif family == "exact_label":
                rendered = f"Question:\n{core}\n\nChoices:\n{options_text}\n\nReply with exactly one choice label."
            elif family == "conversational":
                rendered = f"I'm trying to figure something out:\n{core}"
            elif family == "email_style":
                rendered = f"Hi, could you help me with this?\n\n{core}\n\nThanks."
            elif family == "chat_explanation":
                rendered = f"Please answer this for a user:\n{core}"
            elif family == "bullet_answer":
                rendered = f"Please give the answer in one bullet:\n- {core}"
            else:
                raise AssertionError(family)
            item = dict(row)
            item.update({"item_id": f"{row['item_id']}::family::{family}", "text": rendered, "format": polarity, "format_family_id": family, "format_family_polarity": polarity, "render_slot": "no_cue", "cue_style": "none", "cue_perspective": "none", "intended_purpose": "none", "label": None, "cue_text": None, "cue_char_start": None, "cue_char_end": None})
            output.append(item)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in output), encoding="utf-8")
    print(json.dumps({"rows": len(output), "blocks": len(source), "families": [x[0] for x in families], "topics": len({x["topic_id"] for x in output})}))


if __name__ == "__main__":
    main()
