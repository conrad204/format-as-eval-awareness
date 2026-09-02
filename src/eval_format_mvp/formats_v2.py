"""Format catalog v2: loading and the F1/F2 wrapper lints.

Implements the lint contract of format_catalog_validator_v2_plan.md:

F1  Purpose symmetry, two tiers. No wrapper or contract token whose stem is
    purpose-connoting may appear. Banned = every declared frame cue stem
    (catalogs/purpose_frames_v2.yaml) union a manual open-class list. A
    fold-owned stem may be admitted through an in-catalog exception entry
    (token + scope + justification) only if the lint verifies the stem
    appears class-symmetrically in every fold that does not own it — the R3
    criterion — so a probe trained on other folds can earn no weight for it.
    `legacy: true` templates (the v1 anchors) are report-only: they predate
    the rule and are excluded from the factorial analysis.

F2  No watermark collision. A normalized constant wrapper line may belong to
    at most one template, except lines on the generic exemption list
    (terminal fields shared across real harness surfaces). A whole-wrapper
    TF-IDF cosine (dedup.py machinery) is advisory only — structural
    siblings like B1/B5 are intentional.

Both lints run on the catalog alone (no renders needed), so they gate
authoring before any rendering or GPU spend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dedup import cosine, tfidf_vectors
from .frames_v2 import STOPWORDS, FramesV2, stem, tokenize
from .io import MVPError, PROJECT_ROOT, load_yaml


SCHEMA_VERSION = 3
STRATA = ("benchmark", "casual")
SLOT_PATTERN = re.compile(r"\{(purpose|body|options|contract)\}")
CONTRACT_IDS = ("C0", "C1", "C2", "C3", "C4")
OPTION_STYLES = (
    "letter_dot",
    "letter_paren",
    "letter_paren_inline",
    "choice_record",
    "bullet",
    "numbered",
    "json_array",
    "absent",
)
# Option styles whose rendered options carry an addressable label (drives
# both C2 compatibility and the answer_key label space).
LABELED_OPTION_STYLES = {
    "letter_dot": ("A", "B", "C", "D"),
    "letter_paren": ("A", "B", "C", "D"),
    "letter_paren_inline": ("a", "b", "c", "d"),
    "numbered": ("1", "2", "3", "4"),
}
NEAR_DUP_ADVISORY_THRESHOLD = 0.9
MIN_DISTINCTIVE_LINE_CHARS = 3


@dataclass(frozen=True)
class FormatsV2:
    templates: tuple[dict[str, Any], ...]
    contracts: tuple[dict[str, Any], ...]
    f1_exceptions: tuple[dict[str, Any], ...]
    banned_stem_manual: tuple[str, ...]
    f2_generic_lines: frozenset[str]

    def template(self, template_id: str) -> dict[str, Any]:
        for row in self.templates:
            if str(row["template_id"]) == template_id:
                return row
        raise MVPError(f"Unknown template {template_id!r}")

    def contract(self, contract_id: str) -> dict[str, Any]:
        for row in self.contracts:
            if str(row["contract_id"]) == contract_id:
                return row
        raise MVPError(f"Unknown contract {contract_id!r}")

    @property
    def rendered_templates(self) -> tuple[dict[str, Any], ...]:
        """Development templates: not reserved, not legacy anchors."""
        return tuple(
            t for t in self.templates if not t["reserved"] and not t["legacy"]
        )

    @property
    def anchors(self) -> tuple[dict[str, Any], ...]:
        return tuple(t for t in self.templates if t["legacy"])

    @property
    def reserved(self) -> tuple[dict[str, Any], ...]:
        return tuple(t for t in self.templates if t["reserved"])


def wrapper_text(template: dict[str, Any]) -> str:
    """The template's constant text with all slot markers removed."""
    return SLOT_PATTERN.sub(" ", str(template["template"]))


def wrapper_lines(template: dict[str, Any]) -> list[str]:
    """Normalized non-empty constant lines (lowercased, whitespace-collapsed)."""
    lines = []
    for raw_line in str(template["template"]).splitlines():
        residue = SLOT_PATTERN.sub(" ", raw_line)
        normalized = re.sub(r"\s+", " ", residue).strip().lower()
        if normalized:
            lines.append(normalized)
    return lines


def _check_slot_consistency(template: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    template_id = str(template["template_id"])
    text = str(template["template"])
    slots = set(SLOT_PATTERN.findall(text))
    if "purpose" not in slots or "body" not in slots:
        problems.append(f"{template_id}: template must carry {{purpose}} and {{body}}")
    option_style = str(template["dimensions"]["option_style"])
    has_options_slot = "options" in slots
    if (option_style != "absent") != has_options_slot:
        problems.append(
            f"{template_id}: option_style {option_style!r} inconsistent with "
            f"{{options}} slot presence ({has_options_slot})"
        )
    if not template["legacy"] and "contract" not in slots:
        problems.append(f"{template_id}: non-legacy template must carry {{contract}}")
    if template["legacy"] and "contract" in slots:
        problems.append(f"{template_id}: legacy anchor must keep its embedded contract")
    return problems


def load_formats_v2(root: Path = PROJECT_ROOT) -> FormatsV2:
    value = load_yaml(root / "catalogs" / "formats_v2.yaml")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise MVPError(f"formats_v2.yaml must declare schema_version {SCHEMA_VERSION}")
    templates = tuple(value.get("templates", []))
    contracts = tuple(value.get("contracts", []))
    exceptions = tuple(value.get("f1_exceptions", []))
    manual = tuple(str(t) for t in value.get("banned_stem_manual", []))
    generic = frozenset(str(t).lower() for t in value.get("f2_generic_lines", []))

    if tuple(str(c["contract_id"]) for c in contracts) != CONTRACT_IDS:
        raise MVPError(f"Contract bank must be exactly {CONTRACT_IDS}, in order")
    problems: list[str] = []
    seen: set[str] = set()
    for template in templates:
        template_id = str(template.get("template_id", ""))
        if not template_id or template_id in seen:
            problems.append(f"missing or duplicate template_id {template_id!r}")
            continue
        seen.add(template_id)
        for key in ("format", "family_id", "legacy", "reserved", "dimensions", "template"):
            if key not in template:
                problems.append(f"{template_id}: missing key {key!r}")
        if str(template.get("format")) not in STRATA:
            problems.append(f"{template_id}: format must be one of {STRATA}")
        dimensions = template.get("dimensions", {})
        expected_dims = {
            "heading_style", "option_style", "serialization", "register",
            "terminal_field",
        }
        if set(dimensions) != expected_dims:
            problems.append(f"{template_id}: dimensions must be exactly {sorted(expected_dims)}")
        elif str(dimensions["option_style"]) not in OPTION_STYLES:
            problems.append(f"{template_id}: unknown option_style {dimensions['option_style']!r}")
        if template.get("legacy") and template.get("reserved"):
            problems.append(f"{template_id}: cannot be both legacy and reserved")
        problems.extend(_check_slot_consistency(template))
    catalog = FormatsV2(templates, contracts, exceptions, manual, generic)

    counts = {
        "benchmark_rendered": sum(
            1 for t in catalog.rendered_templates if t["format"] == "benchmark"
        ),
        "casual_rendered": sum(
            1 for t in catalog.rendered_templates if t["format"] == "casual"
        ),
        "reserved": len(catalog.reserved),
        "anchors": len(catalog.anchors),
    }
    if counts != {
        "benchmark_rendered": 6, "casual_rendered": 5, "reserved": 2, "anchors": 2,
    }:
        problems.append(f"catalog composition {counts} != plan section 5 (6+5 rendered, 2 reserved, 2 anchors)")
    if {str(t["format"]) for t in catalog.reserved} != set(STRATA):
        problems.append("reserved templates must cover one per stratum")

    # Continuity anchors must stay byte-identical to the v1 catalog.
    v1_path = root / "catalogs" / "formats.yaml"
    if v1_path.exists():
        v1_rows = {
            str(row["format_variant_id"]): str(row["template"])
            for row in load_yaml(v1_path).get("formats", [])
        }
        for anchor in catalog.anchors:
            anchor_id = str(anchor["template_id"])
            if anchor_id not in v1_rows:
                problems.append(f"anchor {anchor_id} not found in formats.yaml")
            elif str(anchor["template"]) != v1_rows[anchor_id]:
                problems.append(f"anchor {anchor_id} template drifted from formats.yaml")
    if problems:
        raise MVPError(
            "formats_v2 catalog invalid:\n" + "\n".join(f"  {p}" for p in problems)
        )
    return catalog


def _exception_index(
    catalog: FormatsV2,
) -> dict[str, set[str]]:
    """token -> set of scope ids ('*' means everywhere)."""
    index: dict[str, set[str]] = {}
    for entry in catalog.f1_exceptions:
        token = str(entry["token"]).lower()
        scope = entry.get("scope", [])
        scopes = {"*"} if scope == "*" else {str(s) for s in scope}
        index.setdefault(token, set()).update(scopes)
    return index


def _fold_class_counts(frames: FramesV2) -> dict[str, dict[str, dict[str, int]]]:
    """fold -> stem -> {evaluation, deployment} counts over the frame sentences."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for fold_id, group in frames.by_fold.items():
        fold_counts: dict[str, dict[str, int]] = {}
        for pair in group:
            for purpose in ("evaluation", "deployment"):
                for token in tokenize(str(pair["en"][purpose])):
                    entry = fold_counts.setdefault(
                        stem(token), {"evaluation": 0, "deployment": 0}
                    )
                    entry[purpose] += 1
        counts[fold_id] = fold_counts
    return counts


def _declared_cue_stems(frames: FramesV2) -> dict[str, set[str]]:
    """stem -> owning fold ids."""
    owners: dict[str, set[str]] = {}
    for pair in frames.pairs:
        for purpose in ("evaluation", "deployment"):
            for token in pair["cue_tokens"][purpose]:
                owners.setdefault(stem(str(token).lower()), set()).add(
                    str(pair["fold_id"])
                )
    return owners


def lint_formats_v2(catalog: FormatsV2, frames: FramesV2) -> dict[str, Any]:
    """Run F1 + F2; raise MVPError on any hard violation, return a report."""
    owners = _declared_cue_stems(frames)
    banned = set(owners) | {stem(t.lower()) for t in catalog.banned_stem_manual}
    exceptions = _exception_index(catalog)
    class_counts = _fold_class_counts(frames)

    problems: list[str] = []
    warnings: list[str] = []
    exceptions_used: list[dict[str, str]] = []

    # Tier-2 precondition: every exception token's stem must be verifiably
    # class-symmetric in all non-owning folds (the R3 criterion).
    for entry in catalog.f1_exceptions:
        token = str(entry["token"]).lower()
        token_stem = stem(token)
        if not str(entry.get("justification", "")).strip():
            problems.append(f"F1 exception {token!r}: missing justification")
        if token_stem not in owners:
            warnings.append(
                f"F1 exception {token!r}: stem {token_stem!r} is not a declared "
                "cue stem (dead exception)"
            )
            continue
        for fold_id, fold_counts in class_counts.items():
            if fold_id in owners[token_stem]:
                continue
            entry_counts = fold_counts.get(token_stem)
            if entry_counts and entry_counts["evaluation"] != entry_counts["deployment"]:
                problems.append(
                    f"F1 exception {token!r}: stem {token_stem!r} is class-"
                    f"asymmetric in non-owning fold {fold_id} ({entry_counts})"
                )

    # F1 over templates and contracts.
    def _f1_scan(text: str, scope_id: str, legacy: bool) -> None:
        for token in tokenize(text):
            if token in STOPWORDS:
                continue
            token_stem = stem(token)
            if token_stem not in banned:
                continue
            scopes = exceptions.get(token, set())
            if "*" in scopes or scope_id in scopes:
                if not legacy:
                    exceptions_used.append({"token": token, "scope": scope_id})
                continue
            message = (
                f"F1 {scope_id}: token {token!r} (stem {token_stem!r}, "
                f"owned by {sorted(owners.get(token_stem, {'manual'}))})"
            )
            if legacy:
                warnings.append(f"legacy {message}")
            else:
                problems.append(message)

    for template in catalog.templates:
        _f1_scan(
            wrapper_text(template),
            str(template["template_id"]),
            bool(template["legacy"]),
        )
    for contract in catalog.contracts:
        _f1_scan(str(contract["text"]), str(contract["contract_id"]), legacy=False)

    # F2: distinctive-line uniqueness across templates.
    line_owners: dict[str, list[str]] = {}
    for template in catalog.templates:
        template_id = str(template["template_id"])
        for line in set(wrapper_lines(template)):
            if line in catalog.f2_generic_lines:
                continue
            if len(line) < MIN_DISTINCTIVE_LINE_CHARS:
                continue
            line_owners.setdefault(line, []).append(template_id)
    for line, template_ids in sorted(line_owners.items()):
        if len(template_ids) < 2:
            continue
        non_legacy = [
            t for t in template_ids if not catalog.template(t)["legacy"]
        ]
        message = f"F2 line {line!r} shared by {sorted(template_ids)}"
        if len(non_legacy) >= 2:
            problems.append(message)
        else:
            warnings.append(message)

    # F2 advisory: whole-wrapper TF-IDF cosine (empty wrappers excluded).
    wrapper_by_id = {
        str(t["template_id"]): wrapper_text(t).strip() for t in catalog.templates
    }
    nonempty = [(tid, text) for tid, text in wrapper_by_id.items() if text]
    vectors = tfidf_vectors([text for _, text in nonempty])
    for i in range(len(nonempty)):
        for j in range(i + 1, len(nonempty)):
            similarity = cosine(vectors[i], vectors[j])
            if similarity >= NEAR_DUP_ADVISORY_THRESHOLD:
                warnings.append(
                    f"F2 advisory: wrappers {nonempty[i][0]} / {nonempty[j][0]} "
                    f"cosine {similarity:.3f}"
                )

    if problems:
        raise MVPError(
            "formats-v2 lint failed:\n" + "\n".join(f"  {p}" for p in problems)
        )
    return {
        "status": "passed",
        "templates": len(catalog.templates),
        "contracts": len(catalog.contracts),
        "banned_stems": len(banned),
        "exceptions_used": sorted(
            {f"{row['scope']}::{row['token']}" for row in exceptions_used}
        ),
        "warnings": warnings,
    }
