"""Thin CLI for GPU activation extraction (replaces the full eval-format-mvp CLI,
which drags in catalogs and the OpenRouter generation stack this repo omits)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_format_mvp.extract import extract_activations  # noqa: E402
from eval_format_mvp.io import MVPError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="extract fixed model activations")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--all-layers", action="store_true",
                        help="capture every transformer block in the same forward pass")
    parser.add_argument("--device-map",
                        help="accelerate device map; use 'auto' to shard weights across GPUs")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="maximum allowed prompt tokens; longer prompts fail rather than truncate")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="tokenize raw prompt text (BOS only, no assistant header) — "
                             "Devbunova's regime; the probed last token is the final "
                             "content token instead of the constant header token")
    args = parser.parse_args()
    try:
        config = extract_activations(
            items_path=args.items,
            output_dir=args.out_dir,
            model_name=args.model,
            model_revision=args.revision,
            layer=args.layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_map=args.device_map,
            all_layers=args.all_layers,
            chat_template=not args.no_chat_template,
        )
    except MVPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
