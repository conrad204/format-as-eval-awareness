"""Command-line interface for the English purpose-by-format MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_activations
from .catalogs import load_catalogs
from .dedup import DEFAULT_NEAR_DUP_THRESHOLD, near_duplicate_report
from .extract import extract_activations
from .generation import generate_validated_payloads
from .io import MVPError, PROJECT_ROOT, load_jsonl, write_json, write_jsonl
from .payloads import (
    build_refill_requests,
    load_and_validate_payloads,
    write_generation_requests,
)
from .render import (
    DEFAULT_FRAME_ASSIGNMENT,
    DEFAULT_HASH_POLICY,
    FRAME_ASSIGNMENT_SCHEMES,
    HASH_POLICIES,
    verify_render_manifest,
    write_rendered_dataset,
)
from .frames_v2 import lint_frames_v2, load_frames_v2, ngram_holdout_check
from .render_v2 import (
    validate_rendered_v2,
    verify_render_manifest_v2,
    write_rendered_dataset_v2,
)
from .annotate_tokens import annotate_tokens
from .formats_v2 import lint_formats_v2, load_formats_v2
from .render_v3 import (
    validate_rendered_v3,
    verify_render_manifest_v3,
    write_rendered_dataset_v3,
)
from .surface_v3 import surface_probe_battery
from .validate import validate_rendered_dataset


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-format-mvp",
        description="English matched-payload purpose x format probing MVP",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    requests = commands.add_parser("requests", help="write offline generation requests")
    requests.add_argument("--out", type=_path, required=True)
    requests.add_argument("--per-topic", type=int, default=5)

    generate = commands.add_parser(
        "generate",
        help="generate with one or more models (OpenRouter or Vertex) and "
        "validate with another",
    )
    generate.add_argument(
        "--generator-profile",
        type=_path,
        required=True,
        action="append",
        dest="generator_profiles",
        help="generator profile YAML; repeat the flag to hash-split payload "
        "slots across several generators (e.g. Grok + Gemini)",
    )
    generate.add_argument("--validator-profile", type=_path, required=True)
    generate.add_argument("--out", type=_path, required=True)
    generate.add_argument("--run-dir", type=_path, required=True)
    generate.add_argument("--per-topic", type=int, default=5)
    generate.add_argument("--max-generation-attempts", type=int, default=6)
    generate.add_argument("--max-semantic-attempts", type=int, default=3)
    generate.add_argument("--max-validator-attempts", type=int, default=2)
    generate.add_argument(
        "--max-attempts",
        type=int,
        help="legacy alias for --max-semantic-attempts; generator budget remains separate",
    )
    generate.add_argument("--limit", type=int)
    generate.add_argument(
        "--near-dup-threshold",
        type=float,
        default=DEFAULT_NEAR_DUP_THRESHOLD,
        help="reject candidates this cosine-similar to an accepted payload; 0 disables",
    )
    generate.add_argument(
        "--requests",
        type=_path,
        help="targeted request JSONL (e.g. from refill-requests) instead of the full grid",
    )

    payloads = commands.add_parser("validate-payloads", help="validate payload JSONL")
    payloads.add_argument("--payloads", type=_path, required=True)
    payloads.add_argument(
        "--near-dup-threshold",
        type=float,
        default=DEFAULT_NEAR_DUP_THRESHOLD,
        help="report payload pairs at or above this TF-IDF cosine similarity; 0 disables",
    )
    payloads.add_argument(
        "--fail-on-near-dups",
        action="store_true",
        help="exit with an error when any near-duplicate pair is flagged",
    )

    refill = commands.add_parser(
        "refill-requests",
        help="write targeted generation requests for topics below quota",
    )
    refill.add_argument("--payloads", type=_path, required=True)
    refill.add_argument("--out", type=_path, required=True)
    refill.add_argument("--per-topic", type=int, default=5)
    refill.add_argument(
        "--reserve",
        type=_path,
        action="append",
        default=[],
        help="payload JSONL whose payload_block_ids must never be reused "
        "(e.g. the pre-audit file containing audit-removed slots); repeatable",
    )
    refill.add_argument("--report", type=_path, help="also write the shortfall report JSON")

    render = commands.add_parser("render", help="render the complete English 2x2")
    render.add_argument("--payloads", type=_path, required=True)
    render.add_argument("--out", type=_path, required=True)
    render.add_argument("--manifest", type=_path, required=True)
    render.add_argument(
        "--frame-assignment",
        choices=FRAME_ASSIGNMENT_SCHEMES,
        default=DEFAULT_FRAME_ASSIGNMENT,
        help="index_mod_v1 reproduces legacy assignment but reshuffles frames on "
        "payload insertion/deletion; stable_hash_v1 keys on payload_block_id",
    )
    render.add_argument(
        "--hash-policy",
        choices=HASH_POLICIES,
        default=DEFAULT_HASH_POLICY,
        help="enforced makes verify-manifest abort on any checksum mismatch",
    )

    lint_v2 = commands.add_parser(
        "lint-frames-v2",
        help="check the v2 frame catalog's vocabulary-fold disjointness",
    )
    lint_v2.add_argument(
        "--skip-ngram",
        action="store_true",
        help="skip the functional held-fold n-gram probe (needs scikit-learn)",
    )

    render_v2 = commands.add_parser(
        "render-v2", help="render the crossed frames-by-payloads v2 dataset"
    )
    render_v2.add_argument("--payloads", type=_path, required=True)
    render_v2.add_argument("--out", type=_path, required=True)
    render_v2.add_argument("--manifest", type=_path, required=True)

    verify_v2 = commands.add_parser(
        "verify-manifest-v2", help="recompute every hash in a v2 render manifest"
    )
    verify_v2.add_argument("--manifest", type=_path, required=True)

    validate_v2 = commands.add_parser(
        "validate-rendered-v2", help="reconstruct and validate v2 rendered JSONL"
    )
    validate_v2.add_argument("--payloads", type=_path, required=True)
    validate_v2.add_argument("--items", type=_path, required=True)
    validate_v2.add_argument("--report", type=_path)

    lint_formats = commands.add_parser(
        "lint-formats-v2",
        help="run the F1/F2 wrapper lints on the v2 format catalog",
    )
    lint_formats.add_argument("--report", type=_path)

    render_v3 = commands.add_parser(
        "render-v3", help="render the purpose x template x contract v3 dataset"
    )
    render_v3.add_argument("--payloads", type=_path, required=True)
    render_v3.add_argument("--out", type=_path, required=True)
    render_v3.add_argument("--manifest", type=_path, required=True)
    render_v3.add_argument(
        "--schedule",
        choices=("rotation", "full"),
        default="rotation",
        help="rotation = 3+3 of 11 templates per block (21 items); full = "
        "every rendered template on every block (36 items, within-block "
        "paired template contrasts)",
    )

    verify_v3 = commands.add_parser(
        "verify-manifest-v3", help="recompute every hash in a v3 render manifest"
    )
    verify_v3.add_argument("--manifest", type=_path, required=True)

    validate_v3 = commands.add_parser(
        "validate-rendered-v3", help="reconstruct and validate v3 rendered JSONL"
    )
    validate_v3.add_argument("--payloads", type=_path, required=True)
    validate_v3.add_argument("--items", type=_path, required=True)
    validate_v3.add_argument("--report", type=_path)

    surface = commands.add_parser(
        "surface-probes-v3",
        help="run the F3 cue-excised surface battery (word+char TF-IDF)",
    )
    surface.add_argument("--items", type=_path, required=True)
    surface.add_argument("--report", type=_path)

    annotate = commands.add_parser(
        "annotate-tokens",
        help="write the tokenizer sidecar (lengths + terminal token ids)",
    )
    annotate.add_argument("--items", type=_path, required=True)
    annotate.add_argument("--out", type=_path, required=True)
    annotate.add_argument("--manifest", type=_path, required=True)
    annotate.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    annotate.add_argument("--revision")

    verify = commands.add_parser(
        "verify-manifest", help="recompute and check every hash in a render manifest"
    )
    verify.add_argument("--manifest", type=_path, required=True)
    verify.add_argument(
        "--strict",
        action="store_true",
        help="abort on mismatch even if the manifest's hash policy is observational",
    )

    rendered = commands.add_parser(
        "validate-rendered", help="reconstruct and validate rendered JSONL"
    )
    rendered.add_argument("--payloads", type=_path, required=True)
    rendered.add_argument("--items", type=_path, required=True)
    rendered.add_argument("--report", type=_path)

    extract = commands.add_parser("extract", help="extract fixed model activations")
    extract.add_argument("--items", type=_path, required=True)
    extract.add_argument("--out-dir", type=_path, required=True)
    extract.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    extract.add_argument("--revision")
    extract.add_argument("--layer", type=int, default=24)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument(
        "--all-layers",
        action="store_true",
        help="capture every transformer block in the same forward pass",
    )
    extract.add_argument(
        "--device-map",
        help="accelerate device map; use 'auto' to shard weights across GPUs",
    )
    extract.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="maximum allowed prompt tokens; longer prompts fail rather than truncate",
    )

    analyze = commands.add_parser("analyze", help="run grouped probe comparisons")
    analyze.add_argument("--items", type=_path, required=True)
    analyze.add_argument("--activations", type=_path, required=True)
    analyze.add_argument("--out", type=_path, required=True)
    analyze.add_argument("--C", type=float, default=0.1)
    return parser


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        catalogs = load_catalogs(PROJECT_ROOT)
        if args.command == "requests":
            count = write_generation_requests(args.out, catalogs, args.per_topic)
            _print({"status": "written", "requests": count, "path": str(args.out)})
        elif args.command == "generate":
            semantic_attempts = (
                args.max_attempts
                if args.max_attempts is not None
                else args.max_semantic_attempts
            )
            _print(
                generate_validated_payloads(
                    catalogs=catalogs,
                    generator_profile_paths=args.generator_profiles,
                    validator_profile_path=args.validator_profile,
                    output_path=args.out,
                    run_dir=args.run_dir,
                    per_topic=args.per_topic,
                    max_generation_attempts=args.max_generation_attempts,
                    max_semantic_attempts=semantic_attempts,
                    max_validator_attempts=args.max_validator_attempts,
                    limit=args.limit,
                    near_dup_threshold=args.near_dup_threshold,
                    requests_path=args.requests,
                )
            )
        elif args.command == "validate-payloads":
            rows = load_and_validate_payloads(args.payloads, catalogs)
            result = {
                "status": "passed",
                "payloads": len(rows),
                "topics": len({row["topic_id"] for row in rows}),
            }
            if args.near_dup_threshold > 0:
                result["near_duplicates"] = near_duplicate_report(
                    rows, args.near_dup_threshold
                )
                if args.fail_on_near_dups and result["near_duplicates"]["flagged_pairs"]:
                    _print(result)
                    raise MVPError(
                        f"{len(result['near_duplicates']['flagged_pairs'])} near-duplicate "
                        "payload pairs at or above threshold "
                        f"{args.near_dup_threshold}"
                    )
            _print(result)
        elif args.command == "refill-requests":
            rows = load_and_validate_payloads(args.payloads, catalogs)
            reserved: set[str] = set()
            for reserve_path in args.reserve:
                reserved.update(
                    str(row["payload_block_id"]) for row in load_jsonl(reserve_path)
                )
            report, request_rows = build_refill_requests(
                rows,
                catalogs,
                per_topic=args.per_topic,
                reserved_ids=reserved,
            )
            write_jsonl(args.out, request_rows)
            report["requests_path"] = str(args.out)
            report["requests_written"] = len(request_rows)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "render":
            rows = load_and_validate_payloads(args.payloads, catalogs)
            manifest = write_rendered_dataset(
                payload_path=args.payloads,
                output_path=args.out,
                manifest_path=args.manifest,
                payloads=rows,
                catalogs=catalogs,
                frame_assignment=args.frame_assignment,
                hash_policy=args.hash_policy,
            )
            rendered_rows = load_jsonl(args.out)
            manifest["validation"] = validate_rendered_dataset(
                rendered_rows, rows, catalogs
            )
            write_json(args.manifest, manifest)
            _print(manifest)
        elif args.command == "lint-frames-v2":
            frames = load_frames_v2()
            report = {"lint": lint_frames_v2(frames)}
            if not args.skip_ngram:
                report["ngram_holdout"] = ngram_holdout_check(frames)
                if not report["ngram_holdout"]["near_chance"]:
                    _print(report)
                    raise MVPError(
                        "Held-fold n-gram probe on cue sentences is not near "
                        f"chance (AUC {report['ngram_holdout']['auc']:.3f}); "
                        "the fold lexicons leak"
                    )
            _print(report)
        elif args.command == "render-v2":
            frames = load_frames_v2()
            rows = load_and_validate_payloads(args.payloads, catalogs)
            _print(
                write_rendered_dataset_v2(
                    payload_path=args.payloads,
                    output_path=args.out,
                    manifest_path=args.manifest,
                    payloads=rows,
                    frames=frames,
                    catalogs=catalogs,
                )
            )
        elif args.command == "verify-manifest-v2":
            _print(verify_render_manifest_v2(args.manifest))
        elif args.command == "lint-formats-v2":
            frames = load_frames_v2()
            catalog = load_formats_v2()
            report = lint_formats_v2(catalog, frames)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "render-v3":
            frames = load_frames_v2()
            catalog = load_formats_v2()
            rows = load_and_validate_payloads(args.payloads, catalogs)
            _print(
                write_rendered_dataset_v3(
                    payload_path=args.payloads,
                    output_path=args.out,
                    manifest_path=args.manifest,
                    payloads=rows,
                    frames=frames,
                    catalog=catalog,
                    schedule=args.schedule,
                )
            )
        elif args.command == "verify-manifest-v3":
            _print(verify_render_manifest_v3(args.manifest))
        elif args.command == "validate-rendered-v3":
            frames = load_frames_v2()
            catalog = load_formats_v2()
            payload_rows = load_and_validate_payloads(args.payloads, catalogs)
            rendered_rows = load_jsonl(args.items)
            report = validate_rendered_v3(rendered_rows, payload_rows, frames, catalog)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "surface-probes-v3":
            rendered_rows = load_jsonl(args.items)
            report = surface_probe_battery(rendered_rows)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "annotate-tokens":
            _print(
                annotate_tokens(
                    items_path=args.items,
                    output_path=args.out,
                    manifest_path=args.manifest,
                    model_name=args.model,
                    revision=args.revision,
                )
            )
        elif args.command == "validate-rendered-v2":
            frames = load_frames_v2()
            payload_rows = load_and_validate_payloads(args.payloads, catalogs)
            rendered_rows = load_jsonl(args.items)
            report = validate_rendered_v2(rendered_rows, payload_rows, frames, catalogs)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "verify-manifest":
            _print(
                verify_render_manifest(
                    args.manifest, strict=True if args.strict else None
                )
            )
        elif args.command == "validate-rendered":
            payload_rows = load_and_validate_payloads(args.payloads, catalogs)
            rendered_rows = load_jsonl(args.items)
            report = validate_rendered_dataset(rendered_rows, payload_rows, catalogs)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "extract":
            _print(
                extract_activations(
                    items_path=args.items,
                    output_dir=args.out_dir,
                    model_name=args.model,
                    model_revision=args.revision,
                    layer=args.layer,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device_map=args.device_map,
                    all_layers=args.all_layers,
                )
            )
        elif args.command == "analyze":
            _print(
                analyze_activations(
                    items_path=args.items,
                    activations_path=args.activations,
                    output_path=args.out,
                    c=args.C,
                )
            )
        else:  # pragma: no cover
            parser.error(f"Unknown command {args.command}")
    except MVPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
