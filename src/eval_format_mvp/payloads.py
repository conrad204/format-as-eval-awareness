"""Canonical-payload request construction and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalogs import Catalogs
from .io import MVPError, PROJECT_ROOT, load_jsonl, write_jsonl
from .schema import validate_against_schema


def validate_payload(row: dict[str, Any], catalogs: Catalogs) -> None:
    payload_id = str(row.get("payload_block_id", "<missing>"))
    validate_against_schema(row, "payload.schema.json", label=payload_id)
    if row["topic_id"] not in catalogs.topic_ids:
        raise MVPError(f"Payload {payload_id} uses unknown topic {row['topic_id']}")
    topic = next(item for item in catalogs.topics if item["topic_id"] == row["topic_id"])
    if row["task_family"] != topic["task_family"]:
        raise MVPError(
            f"Payload {payload_id} task family {row['task_family']} does not match "
            f"topic catalog value {topic['task_family']}"
        )
    answers = [str(row["correct_answer"]), *map(str, row["distractors"])]
    folded = [answer.strip().casefold() for answer in answers]
    if len(set(folded)) != 4:
        raise MVPError(f"Payload {payload_id} answer and distractors must be distinct")


def validate_payloads(rows: list[dict[str, Any]], catalogs: Catalogs) -> None:
    if not rows:
        raise MVPError("Payload file is empty")
    identifiers: set[str] = set()
    for row in rows:
        validate_payload(row, catalogs)
        payload_id = str(row["payload_block_id"])
        if payload_id in identifiers:
            raise MVPError(f"Duplicate payload_block_id {payload_id}")
        identifiers.add(payload_id)


def load_and_validate_payloads(path: Path, catalogs: Catalogs) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    validate_payloads(rows, catalogs)
    return rows


def _load_generation_prompt_assets(prompt_path: Path) -> tuple[str, dict[str, Any]]:
    prompt = prompt_path.read_text(encoding="utf-8")
    response_schema = __import__("json").loads(
        (PROJECT_ROOT / "schemas" / "generation_response.schema.json").read_text(
            encoding="utf-8"
        )
    )
    return prompt, response_schema


def _request_row(
    topic: dict[str, Any],
    sample_number: int,
    prompt: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    payload_id = f"pb_{topic['topic_id']}_{sample_number:02d}"
    return {
        "request_id": f"request::{payload_id}",
        "payload_block_id": payload_id,
        "topic_id": topic["topic_id"],
        "task_family": topic["task_family"],
        "system": "You create controlled research stimuli and return valid JSON.",
        "user": prompt.format(
            topic_display_name=topic["display_name"],
            topic_id=topic["topic_id"],
            task_family=topic["task_family"],
            sample_number=sample_number,
        ).strip(),
        "response_schema": response_schema,
    }


def build_generation_requests(
    catalogs: Catalogs,
    *,
    per_topic: int = 5,
    prompt_path: Path = PROJECT_ROOT / "prompts" / "payload_generation.txt",
) -> list[dict[str, Any]]:
    if per_topic < 1:
        raise MVPError("per_topic must be positive")
    prompt, response_schema = _load_generation_prompt_assets(prompt_path)
    rows: list[dict[str, Any]] = []
    for topic in catalogs.topics:
        for sample_number in range(1, per_topic + 1):
            rows.append(_request_row(topic, sample_number, prompt, response_schema))
    return rows


def build_refill_requests(
    existing_payloads: list[dict[str, Any]],
    catalogs: Catalogs,
    *,
    per_topic: int = 5,
    reserved_ids: frozenset[str] | set[str] = frozenset(),
    prompt_path: Path = PROJECT_ROOT / "prompts" / "payload_generation.txt",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Targeted requests that top every topic back up to its quota.

    Slot IDs already present in existing_payloads or reserved_ids (e.g. the
    pre-audit file, so audit-removed IDs are never reused for different
    content) are skipped; fresh sample numbers are assigned instead.
    """
    if per_topic < 1:
        raise MVPError("per_topic must be positive")
    prompt, response_schema = _load_generation_prompt_assets(prompt_path)
    existing_ids = {str(row["payload_block_id"]) for row in existing_payloads}
    unusable = existing_ids | {str(value) for value in reserved_ids}
    per_topic_counts: dict[str, int] = {}
    for row in existing_payloads:
        topic_id = str(row["topic_id"])
        per_topic_counts[topic_id] = per_topic_counts.get(topic_id, 0) + 1

    rows: list[dict[str, Any]] = []
    topics_report: dict[str, dict[str, Any]] = {}
    task_family_counts: dict[str, int] = {}
    for topic in catalogs.topics:
        topic_id = str(topic["topic_id"])
        have = per_topic_counts.get(topic_id, 0)
        task_family = str(topic["task_family"])
        task_family_counts[task_family] = task_family_counts.get(task_family, 0) + have
        missing = per_topic - have
        if missing <= 0:
            continue
        new_ids: list[str] = []
        sample_number = 1
        while len(new_ids) < missing:
            payload_id = f"pb_{topic_id}_{sample_number:02d}"
            if payload_id not in unusable:
                rows.append(_request_row(topic, sample_number, prompt, response_schema))
                new_ids.append(payload_id)
            sample_number += 1
        topics_report[topic_id] = {
            "have": have,
            "missing": missing,
            "new_payload_block_ids": new_ids,
        }

    report = {
        "schema_version": 1,
        "per_topic_quota": per_topic,
        "existing_payloads": len(existing_payloads),
        "total_missing": sum(entry["missing"] for entry in topics_report.values()),
        "topics_below_quota": topics_report,
        "task_family_counts": dict(sorted(task_family_counts.items())),
    }
    return report, rows


def write_generation_requests(path: Path, catalogs: Catalogs, per_topic: int) -> int:
    rows = build_generation_requests(catalogs, per_topic=per_topic)
    write_jsonl(path, rows)
    return len(rows)
