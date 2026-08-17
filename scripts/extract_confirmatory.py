from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval_format_mvp.extract import extract_activations

MODELS = {
    "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", 12),
    "llama31_70b": ("meta-llama/Llama-3.1-70B-Instruct", "1605565b47bb9346c5515c34102e054115b4f98b", 31),
    "llama33_70b": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", 17),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODELS), required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args()
    model, revision, layer = MODELS[args.model_key]
    config = extract_activations(items_path=args.items, output_dir=args.output, model_name=model, model_revision=revision, layer=layer, batch_size=args.batch_size, max_length=args.max_length, device_map=args.device_map, all_layers=False)
    print(json.dumps(config, sort_keys=True))


if __name__ == "__main__":
    main()
