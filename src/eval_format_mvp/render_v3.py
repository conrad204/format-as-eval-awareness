"""Format-catalog v2 rendering (scheme "crossed_formats_v3").

Implements format_catalog_v2_plan.md section 6 and the validator plan:
purpose (eval / deploy / no-cue) x template x contract, payload bodies and
the v2 vocabulary-fold BIBD reused unchanged from render_v2 so every block
keeps its pilot80 fold triple.

Per block: 3 of 6 benchmark templates + 3 of 5 deployment templates
(topic-local hash rotation, insertion-stable) + 1 anchor (alternating);
per (block, template): one stated third-person cue pair -> {eval, deploy}
renders, plus one no-cue render => 7 x 3 = 21 items.

Design notes (validator plan sections 5-8):
- The structural validator derives every expected quantity from
  build_dataset_plan_v3 — the same function the renderer uses — and then
  requires byte-identical reconstruction. Constants cannot drift from code.
- Answer keys (item 17b) come from a pure per-(block, template) hash: uniform
  in expectation over A-D, topic-local, and uncorrelated with the template
  schedule. Rank-based rotation was rejected: per-(topic, template) groups
  hold only 2-3 blocks, which would starve the last key position.
- Anchors keep the v1 option shuffle (render._options) byte-for-byte so
  anchor renders bridge to pilot80; they carry no contract slot.
- deploy_json_v2 is built via json.dumps, never string templating.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .formats_v2 import (
    LABELED_OPTION_STYLES,
    FormatsV2,
    lint_formats_v2,
    wrapper_lines,
)
from .frames_v2 import FOLD_IDS, FramesV2, lint_frames_v2
from .io import (
    MVPError,
    PROJECT_ROOT,
    load_json,
    sha256_file,
    sha256_object,
    sha256_text,
    write_json,
    write_jsonl,
)
from .render import PURPOSES, _cue_annotations, _options
from .render_v2 import _digest, _pick, assign_block_folds
from .schema import validate_against_schema


SCHEME_V3 = "crossed_formats_v3"
SCHEME_V3_FULL = "crossed_formats_v3_full"
# schedule name -> (scheme id, benchmark templates per block, deployment
# templates per block; None = all). "rotation" is the budget lane (3+3 of
# 6+5, topic-local hash rotation); "full" crosses every rendered template
# with every block, making template contrasts within-block paired.
SCHEDULES = {"rotation": SCHEME_V3, "full": SCHEME_V3_FULL}
SCHEME_TO_SCHEDULE = {scheme: name for name, scheme in SCHEDULES.items()}
ITEMS_PER_BLOCK_V3 = 21
ITEMS_PER_BLOCK_FULL = 36
BENCH_PER_BLOCK = 3
DEPLOY_PER_BLOCK = 3
B4_TEMPLATE_ID = "bench_qa_record_v2"
JSON_TEMPLATE_ID = "deploy_json_v2"
SLOT_PATTERN = re.compile(r"\{(purpose|body|options|contract)\}")
# Hard-check answer-key coverage only for templates with at least this many
# distinct option arrangements; smaller runs (tests, smokes) report only.
KEY_COVERAGE_MIN_ARRANGEMENTS = 16


def _collapse(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _substitute(template: str, mapping: dict[str, str]) -> str:
    """Single-pass slot substitution; inserted text is never re-scanned."""
    return SLOT_PATTERN.sub(lambda match: mapping[match.group(1)], template)


def _ordered_blocks_by_topic(
    payloads: list[dict[str, Any]],
) -> dict[str, list[str]]:
    by_topic: dict[str, list[str]] = {}
    for payload in payloads:
        by_topic.setdefault(str(payload["topic_id"]), []).append(
            str(payload["payload_block_id"])
        )
    return {
        topic: sorted(blocks, key=sha256_text) for topic, blocks in by_topic.items()
    }


def _stated_third(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        pair
        for pair in pairs
        if pair["cue_style"] == "stated" and pair["cue_perspective"] == "third"
    ]


def template_options_present(template: dict[str, Any], mode: str | None) -> bool:
    if str(template["dimensions"]["option_style"]) == "absent":
        return False
    if str(template["template_id"]) == B4_TEMPLATE_ID and mode == "gsm8k":
        return False
    return True


def compatible_contracts(
    catalog: FormatsV2, template: dict[str, Any], mode: str | None
) -> list[dict[str, Any]]:
    """C2 requires an option-bearing render whose style yields labels."""
    options_present = template_options_present(template, mode)
    labeled = (
        str(template["dimensions"]["option_style"]) in LABELED_OPTION_STYLES
        and options_present
    )
    return [
        contract
        for contract in catalog.contracts
        if not contract["requires_labels"] or labeled
    ]


def build_dataset_plan_v3(
    payloads: list[dict[str, Any]],
    frames: FramesV2,
    catalog: FormatsV2,
    schedule: str = "rotation",
) -> dict[str, list[dict[str, Any]]]:
    """block_id -> ordered slot plans. Deterministic and topic-local."""
    if schedule not in SCHEDULES:
        raise MVPError(f"Unknown schedule {schedule!r}; expected one of {sorted(SCHEDULES)}")
    fold_assignment = assign_block_folds(payloads)
    by_topic = _ordered_blocks_by_topic(payloads)
    by_fold = frames.by_fold
    bench = sorted(
        (t for t in catalog.rendered_templates if t["format"] == "benchmark"),
        key=lambda t: str(t["template_id"]),
    )
    deploy = sorted(
        (t for t in catalog.rendered_templates if t["format"] == "casual"),
        key=lambda t: str(t["template_id"]),
    )
    anchors = sorted(catalog.anchors, key=lambda t: str(t["template_id"]))

    plans: dict[str, list[dict[str, Any]]] = {}
    for topic_id, block_ids in by_topic.items():
        bench_offset = _digest(f"{topic_id}::bench")
        deploy_offset = _digest(f"{topic_id}::deploy")
        anchor_offset = _digest(f"{topic_id}::anchor")
        b4_offset = _digest(f"{topic_id}::b4mode")
        # Contract rotation is keyed on how often this template has already
        # appeared in this topic (not the raw block index): the template is
        # scheduled on only a subset of blocks, and rotating over appearances
        # cycles cleanly through the compatible contracts.
        appearances: dict[str, int] = {}
        for index, block_id in enumerate(block_ids):
            if schedule == "full":
                templates = (
                    list(bench)
                    + list(deploy)
                    + [anchors[(anchor_offset + index) % len(anchors)]]
                )
            else:
                bench_start = (bench_offset + index) % len(bench)
                deploy_start = (deploy_offset + index) % len(deploy)
                templates = (
                    [
                        bench[(bench_start + k) % len(bench)]
                        for k in range(BENCH_PER_BLOCK)
                    ]
                    + [
                        deploy[(deploy_start + k) % len(deploy)]
                        for k in range(DEPLOY_PER_BLOCK)
                    ]
                    + [anchors[(anchor_offset + index) % len(anchors)]]
                )
            folds = fold_assignment[block_id]
            fold_rotation = _digest(f"{block_id}::foldrot") % len(folds)
            slots: list[dict[str, Any]] = []
            for slot_index, template in enumerate(templates):
                template_id = str(template["template_id"])
                fold_id = folds[(slot_index + fold_rotation) % len(folds)]
                pair = _pick(
                    _stated_third(by_fold[fold_id]),
                    f"{block_id}::{template_id}::pair",
                )
                mode = (
                    ["choice", "gsm8k"][(b4_offset + index) % 2]
                    if template_id == B4_TEMPLATE_ID
                    else None
                )
                appearance = appearances.get(template_id, 0)
                appearances[template_id] = appearance + 1
                if template["legacy"]:
                    contract = None
                else:
                    compatible = compatible_contracts(catalog, template, mode)
                    contract_index = (
                        _digest(f"{topic_id}::{template_id}::contract") + appearance
                    ) % len(compatible)
                    contract = compatible[contract_index]
                key_position = (
                    _digest(f"{block_id}::{template_id}::key") % 4
                    if template_options_present(template, mode) and not template["legacy"]
                    else None
                )
                slots.append(
                    {
                        "slot_index": slot_index,
                        "template": template,
                        "fold_id": fold_id,
                        "pair": pair,
                        "mode": mode,
                        "contract": contract,
                        "key_position": key_position,
                    }
                )
            plans[block_id] = slots
    return plans


def _arranged_values(
    payload: dict[str, Any], block_id: str, template_id: str, key_position: int
) -> tuple[list[str], int]:
    distractors = sorted(
        (str(d) for d in payload["distractors"]),
        key=lambda d: sha256_text(f"{block_id}::{template_id}::{d}"),
    )
    values = list(distractors)
    values.insert(key_position, str(payload["correct_answer"]))
    return values, key_position


def _render_option_block(style: str, values: list[str]) -> str:
    if style == "letter_dot":
        return "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", values))
    if style == "letter_paren":
        return "\n".join(f"{letter}) {value}" for letter, value in zip("ABCD", values))
    if style == "letter_paren_inline":
        return " ".join(f"({letter}) {value}" for letter, value in zip("abcd", values))
    if style == "choice_record":
        return "\n".join(f"choice: {value}" for value in values)
    if style == "bullet":
        return "\n".join(f"- {value}" for value in values)
    if style == "numbered":
        return "\n".join(f"{number}. {value}" for number, value in zip("1234", values))
    raise MVPError(f"No block renderer for option style {style!r}")


def _render_text(
    template: dict[str, Any],
    *,
    purpose_text: str,
    body: str,
    options_block: str,
    option_values: list[str] | None,
    contract_text: str,
) -> str:
    if str(template["template_id"]) == JSON_TEMPLATE_ID:
        payload_object: dict[str, Any] = {"input": body}
        payload_object["choices"] = option_values if option_values is not None else []
        if contract_text:
            payload_object["response_instruction"] = contract_text
        line = json.dumps(payload_object, ensure_ascii=False)
        prefix = f"{purpose_text}\n\n" if purpose_text else ""
        return _collapse(prefix + line)
    return _collapse(
        _substitute(
            str(template["template"]),
            {
                "purpose": purpose_text,
                "body": body,
                "options": options_block,
                "contract": contract_text,
            },
        )
    )


def render_payload_v3(
    payload: dict[str, Any],
    slots: list[dict[str, Any]],
    scheme: str = SCHEME_V3,
) -> list[dict[str, Any]]:
    payload_hash = sha256_object(payload)
    block_id = str(payload["payload_block_id"])
    body = f"{str(payload['context']).strip()}\n\n{str(payload['question']).strip()}"
    rows: list[dict[str, Any]] = []

    for slot in slots:
        template = slot["template"]
        template_id = str(template["template_id"])
        dimensions = template["dimensions"]
        style = str(dimensions["option_style"])
        mode = slot["mode"]
        contract = slot["contract"]
        contract_id = None if contract is None else str(contract["contract_id"])
        contract_text = "" if contract is None else str(contract["text"])

        options_present = template_options_present(template, mode)
        effective_style = style if options_present else "absent"
        option_values: list[str] | None = None
        options_block = ""
        answer_key: str | None = None
        if template["legacy"]:
            if effective_style != "absent":
                options_block, answer_key = _options(payload)
        elif options_present:
            option_values, key_position = _arranged_values(
                payload, block_id, template_id, int(slot["key_position"])
            )
            if effective_style != "json_array":
                options_block = _render_option_block(effective_style, option_values)
            labels = LABELED_OPTION_STYLES.get(effective_style)
            answer_key = labels[key_position] if labels else None
        effective_mode = mode if template_id == B4_TEMPLATE_ID else None

        for purpose in (*PURPOSES, "none"):
            no_cue = purpose == "none"
            pair = slot["pair"]
            purpose_text = "" if no_cue else str(pair["en"][purpose]).strip()
            text = _render_text(
                template,
                purpose_text=purpose_text,
                body=body,
                options_block=options_block,
                option_values=option_values,
                contract_text=contract_text,
            )
            purpose_tag = {"evaluation": "ev", "deployment": "dp", "none": "nc"}[purpose]
            item_id = (
                f"item::{block_id}::{'none' if no_cue else pair['pair_id']}::"
                f"{purpose_tag}::{template_id}::{contract_id or 'none'}"
            )
            row: dict[str, Any] = {
                "schema_version": 3,
                "item_id": item_id,
                "payload_block_id": block_id,
                "topic_id": payload["topic_id"],
                "task_family": payload["task_family"],
                "intended_purpose": purpose,
                "label": None if no_cue else (1 if purpose == "evaluation" else 0),
                "format": template["format"],
                "template_id": template_id,
                "contract_id": contract_id,
                "template_mode": effective_mode,
                "render_slot": "no_cue" if no_cue else "stated",
                "purpose_pair_id": "none" if no_cue else str(pair["pair_id"]),
                "purpose_family_id": "none" if no_cue else str(slot["fold_id"]),
                "cue_style": "none" if no_cue else "stated",
                "cue_perspective": "none" if no_cue else "third",
                "cue_locus": "none" if no_cue else str(pair["cue_locus"]),
                "heading_style": str(dimensions["heading_style"]),
                "option_style": effective_style,
                "serialization": str(dimensions["serialization"]),
                "register": str(dimensions["register"]),
                "terminal_field": dimensions["terminal_field"],
                "legacy": bool(template["legacy"]),
                "text": text,
                "correct_answer": str(payload["correct_answer"]),
                "answer_key": answer_key,
                "canonical_payload_sha256": payload_hash,
                "rendered_text_sha256": sha256_text(text),
                "split_group_id": block_id,
                "frame_assignment_scheme": scheme,
            }
            if no_cue:
                row.update(
                    {
                        "cue_text": None,
                        "cue_char_start": None,
                        "cue_char_end": None,
                        "cue_position": None,
                        "message_role": "user",
                    }
                )
            else:
                row.update(_cue_annotations(text, purpose_text, item_id))
            validate_against_schema(row, "rendered_item_v3.schema.json", label=item_id)
            rows.append(row)
    return rows


def render_dataset_v3(
    payloads: list[dict[str, Any]],
    frames: FramesV2,
    catalog: FormatsV2,
    schedule: str = "rotation",
) -> list[dict[str, Any]]:
    plans = build_dataset_plan_v3(payloads, frames, catalog, schedule=schedule)
    scheme = SCHEDULES[schedule]
    rows: list[dict[str, Any]] = []
    for payload in sorted(payloads, key=lambda row: str(row["payload_block_id"])):
        rows.extend(
            render_payload_v3(
                payload, plans[str(payload["payload_block_id"])], scheme=scheme
            )
        )
    return rows


OPTION_LINE_PATTERNS = {
    "letter_dot": re.compile(r"^[A-D]\. ", re.MULTILINE),
    "letter_paren": re.compile(r"^[A-D]\) ", re.MULTILINE),
    "choice_record": re.compile(r"^choice: ", re.MULTILINE),
    "bullet": re.compile(r"^- ", re.MULTILINE),
    "numbered": re.compile(r"^[1-4]\. ", re.MULTILINE),
}


def _check_row_fidelity(
    row: dict[str, Any], template: dict[str, Any], catalog: FormatsV2
) -> list[str]:
    """F4: declared dimensions must match the rendered text."""
    problems: list[str] = []
    item_id = str(row["item_id"])
    text = str(row["text"])
    style = str(row["option_style"])
    template_id = str(row["template_id"])

    if template_id == JSON_TEMPLATE_ID:
        last_line = text.splitlines()[-1]
        try:
            payload_object = json.loads(last_line)
        except json.JSONDecodeError:
            problems.append(f"F4 {item_id}: JSON line does not parse")
            return problems
        if not isinstance(payload_object.get("input"), str) or not isinstance(
            payload_object.get("choices"), list
        ):
            problems.append(f"F4 {item_id}: JSON object missing input/choices")
        if row["contract_id"] not in (None, "C0") and "response_instruction" not in payload_object:
            problems.append(f"F4 {item_id}: JSON object missing response_instruction")
        if len(payload_object.get("choices", [])) != 4:
            problems.append(f"F4 {item_id}: JSON choices must hold 4 values")
    else:
        # Every constant wrapper-line residue must appear in the render.
        for raw_line in str(template["template"]).splitlines():
            residue = SLOT_PATTERN.sub("", raw_line).strip()
            if residue and residue not in text:
                problems.append(f"F4 {item_id}: wrapper residue {residue!r} missing")
        pattern = OPTION_LINE_PATTERNS.get(style)
        if pattern and len(pattern.findall(text)) < 4:
            problems.append(f"F4 {item_id}: option style {style!r} lines missing")

    terminal = row["terminal_field"]
    if terminal == "json_close":
        if not text.rstrip().endswith("}"):
            problems.append(f"F4 {item_id}: JSON render must end with '}}'")
    elif terminal is not None and not text.endswith(str(terminal)):
        problems.append(f"F4 {item_id}: text does not end with {terminal!r}")

    labels = LABELED_OPTION_STYLES.get(style)
    key = row["answer_key"]
    if labels and not row["legacy"]:
        if key not in labels:
            problems.append(f"F4 {item_id}: answer_key {key!r} outside label space")
    elif labels is None and key is not None and not row["legacy"]:
        problems.append(f"F4 {item_id}: unlabeled style {style!r} cannot carry a key")
    if key is not None and style == "letter_dot":
        if f"{key}. {row['correct_answer']}" not in text:
            problems.append(f"F4 {item_id}: answer_key does not point at correct_answer")
    return problems


def validate_rendered_v3(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    frames: FramesV2,
    catalog: FormatsV2,
) -> dict[str, Any]:
    if not rows:
        raise MVPError("Rendered v3 dataset is empty")
    item_ids: set[str] = set()
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        validate_against_schema(
            row, "rendered_item_v3.schema.json", label=str(row.get("item_id"))
        )
        item_id = str(row["item_id"])
        if item_id in item_ids:
            raise MVPError(f"Duplicate rendered item_id {item_id}")
        item_ids.add(item_id)
        by_block.setdefault(str(row["payload_block_id"]), []).append(row)

    payload_ids = {str(row["payload_block_id"]) for row in payloads}
    if set(by_block) != payload_ids:
        raise MVPError("Rendered v3 blocks differ from canonical payloads")

    # The rows themselves declare which schedule built them.
    schemes = {str(row["frame_assignment_scheme"]) for row in rows}
    if len(schemes) != 1:
        raise MVPError(f"Rendered rows mix schemes: {sorted(schemes)}")
    scheme = schemes.pop()
    if scheme not in SCHEME_TO_SCHEDULE:
        raise MVPError(f"Unknown scheme {scheme!r} in rendered rows")
    schedule = SCHEME_TO_SCHEDULE[scheme]
    expected_items = (
        ITEMS_PER_BLOCK_FULL if schedule == "full" else ITEMS_PER_BLOCK_V3
    )

    # Structural checks derived from the single-source-of-truth plan.
    plans = build_dataset_plan_v3(payloads, frames, catalog, schedule=schedule)
    reserved_ids = {str(t["template_id"]) for t in catalog.reserved}
    problems: list[str] = []
    for block_id, block in sorted(by_block.items()):
        if len(block) != expected_items:
            problems.append(
                f"block {block_id}: {len(block)} items, expected {expected_items}"
            )
            continue
        plan_by_template = {
            str(slot["template"]["template_id"]): slot for slot in plans[block_id]
        }
        rows_by_template: dict[str, list[dict[str, Any]]] = {}
        for row in block:
            rows_by_template.setdefault(str(row["template_id"]), []).append(row)
        if set(rows_by_template) != set(plan_by_template):
            problems.append(
                f"block {block_id}: templates {sorted(rows_by_template)} != plan "
                f"{sorted(plan_by_template)}"
            )
            continue
        for template_id, template_rows in rows_by_template.items():
            slot = plan_by_template[template_id]
            purposes = sorted(str(r["intended_purpose"]) for r in template_rows)
            if purposes != ["deployment", "evaluation", "none"]:
                problems.append(
                    f"block {block_id} {template_id}: purposes {purposes} != "
                    "one eval + one deploy + one no-cue"
                )
            expected_contract = (
                None if slot["contract"] is None else str(slot["contract"]["contract_id"])
            )
            for row in template_rows:
                if row["template_id"] in reserved_ids:
                    problems.append(f"block {block_id}: reserved template rendered")
                if row["contract_id"] != expected_contract:
                    problems.append(
                        f"block {block_id} {template_id}: contract "
                        f"{row['contract_id']} != plan {expected_contract}"
                    )
                if row["render_slot"] == "stated" and (
                    row["purpose_pair_id"] != str(slot["pair"]["pair_id"])
                    or row["purpose_family_id"] != str(slot["fold_id"])
                ):
                    problems.append(
                        f"block {block_id} {template_id}: pair/fold differs from plan"
                    )
                problems.extend(
                    _check_row_fidelity(row, catalog.template(template_id), catalog)
                )
    if problems:
        raise MVPError(
            "Rendered v3 structural validation failed:\n"
            + "\n".join(f"  {p}" for p in problems[:20])
        )

    # Deterministic reconstruction (scheme-locked; render_v2 is untouched).
    expected = render_dataset_v3(payloads, frames, catalog, schedule=schedule)
    expected_by_id = {str(r["item_id"]): r for r in expected}
    actual_by_id = {str(r["item_id"]): r for r in rows}
    if actual_by_id != expected_by_id:
        missing = sorted(set(expected_by_id) - set(actual_by_id))[:3]
        extra = sorted(set(actual_by_id) - set(expected_by_id))[:3]
        changed = sorted(
            i
            for i in set(actual_by_id) & set(expected_by_id)
            if actual_by_id[i] != expected_by_id[i]
        )[:3]
        raise MVPError(
            "Rendered v3 rows do not reproduce deterministically: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    # Aggregate design checks and report.
    stated_rows = [r for r in rows if r["render_slot"] == "stated"]
    topics = sorted({str(p["topic_id"]) for p in payloads})
    topic_fold = {(r["topic_id"], r["purpose_family_id"]) for r in stated_rows}
    coverage_holes = [
        (topic, fold)
        for topic in topics
        for fold in FOLD_IDS
        if (topic, fold) not in topic_fold
    ]
    if coverage_holes:
        raise MVPError(f"Topic x fold coverage holes: {coverage_holes[:10]}")

    contract_cells: dict[str, dict[str, int]] = {}
    key_distribution: dict[str, dict[str, int]] = {}
    arrangements: dict[str, set[str]] = {}
    for row in rows:
        template_id = str(row["template_id"])
        if row["contract_id"] is not None:
            cells = contract_cells.setdefault(template_id, {})
            cells[str(row["contract_id"])] = cells.get(str(row["contract_id"]), 0) + 1
        if row["answer_key"] is not None and not row["legacy"]:
            keys = key_distribution.setdefault(template_id, {})
            keys[str(row["answer_key"])] = keys.get(str(row["answer_key"]), 0) + 1
            arrangements.setdefault(template_id, set()).add(
                str(row["payload_block_id"])
            )

    aggregate_problems: list[str] = []
    for template in catalog.rendered_templates:
        template_id = str(template["template_id"])
        if template_id not in contract_cells:
            continue
        modes = (
            ("choice", "gsm8k") if template_id == B4_TEMPLATE_ID else (None,)
        )
        compatible_ids = {
            str(c["contract_id"])
            for mode in modes
            for c in compatible_contracts(catalog, template, mode)
        }
        rendered_ids = set(contract_cells[template_id])
        if not rendered_ids <= compatible_ids:
            aggregate_problems.append(
                f"{template_id}: incompatible contracts rendered "
                f"{sorted(rendered_ids - compatible_ids)}"
            )
        template_blocks = sum(contract_cells[template_id].values()) // 3
        if template_blocks >= 3 * len(compatible_ids) and rendered_ids != compatible_ids:
            aggregate_problems.append(
                f"{template_id}: compatible contracts never rendered "
                f"{sorted(compatible_ids - rendered_ids)}"
            )
    deployment_c2 = any(
        "C2" in cells
        and str(catalog.template(template_id)["format"]) == "casual"
        for template_id, cells in contract_cells.items()
    )
    if len(by_block) >= 10 and not deployment_c2:
        aggregate_problems.append(
            "C2 never rendered in a deployment-shaped wrapper (plan section 4)"
        )
    for template_id, keys in key_distribution.items():
        labels = LABELED_OPTION_STYLES[
            str(catalog.template(template_id)["dimensions"]["option_style"])
        ]
        if len(arrangements[template_id]) >= KEY_COVERAGE_MIN_ARRANGEMENTS:
            absent = [label for label in labels if label not in keys]
            if absent:
                aggregate_problems.append(
                    f"{template_id}: answer keys never land on {absent}"
                )
    if aggregate_problems:
        raise MVPError(
            "Rendered v3 aggregate checks failed:\n"
            + "\n".join(f"  {p}" for p in aggregate_problems)
        )

    template_counts: dict[str, int] = {}
    for row in rows:
        template_counts[str(row["template_id"])] = (
            template_counts.get(str(row["template_id"]), 0) + 1
        )
    return {
        "status": "passed",
        "payload_blocks": len(by_block),
        "rendered_items": len(rows),
        "items_per_template": dict(sorted(template_counts.items())),
        "contract_cells": {k: dict(sorted(v.items())) for k, v in sorted(contract_cells.items())},
        "answer_key_distribution": {
            k: dict(sorted(v.items())) for k, v in sorted(key_distribution.items())
        },
        "b4_modes": dict(
            sorted(
                {
                    mode: sum(
                        1
                        for r in rows
                        if r["template_mode"] == mode and r["render_slot"] == "stated"
                    )
                    for mode in ("choice", "gsm8k")
                }.items()
            )
        ),
        "deployment_c2_present": deployment_c2,
    }


def write_rendered_dataset_v3(
    *,
    payload_path: Path,
    output_path: Path,
    manifest_path: Path,
    payloads: list[dict[str, Any]],
    frames: FramesV2,
    catalog: FormatsV2,
    schedule: str = "rotation",
) -> dict[str, Any]:
    frames_lint = lint_frames_v2(frames)
    formats_lint = lint_formats_v2(catalog, frames)
    rows = render_dataset_v3(payloads, frames, catalog, schedule=schedule)
    validation = validate_rendered_v3(rows, payloads, frames, catalog)
    write_jsonl(output_path, rows)
    catalog_inputs = {
        name: {
            "path": str(PROJECT_ROOT / "catalogs" / f"{name}.yaml"),
            "sha256": sha256_file(PROJECT_ROOT / "catalogs" / f"{name}.yaml"),
        }
        for name in ("purpose_frames_v2", "formats_v2", "formats", "topics")
    }
    manifest = {
        "schema_version": 3,
        "manifest_kind": "render_v3",
        "status": "development_unlocked",
        "hash_policy": "enforced",
        "frame_assignment": SCHEDULES[schedule],
        "inputs": {
            "payloads": {"path": str(payload_path), "sha256": sha256_file(payload_path)},
            **catalog_inputs,
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "frames_lint": frames_lint,
        "formats_lint": formats_lint,
        "validation": validation,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_render_manifest_v3(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("manifest_kind") != "render_v3":
        raise MVPError(f"{manifest_path} is not a render_v3 manifest")
    mismatches: list[dict[str, str]] = []
    checked: list[str] = []
    entries = dict(manifest["inputs"])
    entries["output"] = manifest["output"]
    for name, entry in entries.items():
        checked.append(name)
        path = Path(entry["path"])
        if not path.exists():
            mismatches.append({"artifact": name, "path": str(path), "error": "missing file"})
            continue
        actual = sha256_file(path)
        if actual != str(entry["sha256"]):
            mismatches.append(
                {
                    "artifact": name,
                    "path": str(path),
                    "expected_sha256": str(entry["sha256"]),
                    "actual_sha256": actual,
                }
            )
    if mismatches:
        details = "; ".join(
            f"{row['artifact']} ({row.get('error', 'sha256 mismatch')})"
            for row in mismatches
        )
        raise MVPError(f"Render v3 manifest hash verification failed: {details}")
    return {
        "manifest": str(manifest_path),
        "hash_policy": "enforced",
        "checked": checked,
        "status": "passed",
    }
