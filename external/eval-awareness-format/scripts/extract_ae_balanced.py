#!/usr/bin/env python3
"""
Extract AE vectors from BALANCED datasets (1076 each).

Uses:
- Bench-Eval: 1076 sampled examples (formatted prompts)
- Casual-Deploy Matched: 1076 examples (length-matched)

Both datasets have similar length distributions.
"""

import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
VECTORS_DIR = BASE_DIR / "data" / "vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed sample size
SAMPLE_SIZE = 1076
random.seed(42)
np.random.seed(42)


def format_bench_eval_prompt(item):
    """Format bench_eval item as prompt."""
    source = item.get("source", "unknown")

    if source == "mmlu":
        q = item.get("question", "")
        choices = item.get("choices", [])
        return f"Question: {q}\nChoices: {', '.join(choices)}\nAnswer:"
    elif source == "arc":
        q = item.get("question", "")
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choice_str = ", ".join([f"{l}) {t}" for l, t in zip(labels, texts)])
        else:
            choice_str = str(choices)
        return f"Question: {q}\nOptions: {choice_str}\nAnswer:"
    elif source == "gsm8k":
        q = item.get("question", "")
        return f"Problem: {q}\nSolution:"

    return str(item.get("question", item))


def load_balanced_datasets() -> Tuple[List[str], List[str]]:
    """Load balanced datasets (1076 each)."""

    # Load and sample Bench-Eval
    print("Loading Bench-Eval...")
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval_raw = json.load(f)

    bench_sampled = random.sample(bench_eval_raw, SAMPLE_SIZE)
    bench_prompts = [format_bench_eval_prompt(item)[:500] for item in bench_sampled]

    # Load Casual-Deploy Matched (already 1076)
    print("Loading Casual-Deploy Matched...")
    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        casual_matched = json.load(f)

    casual_prompts = [item['first_turn'][:500] for item in casual_matched[:SAMPLE_SIZE]]

    # Stats
    bench_lens = [len(p) for p in bench_prompts]
    casual_lens = [len(p) for p in casual_prompts]

    print(f"\nDataset Statistics:")
    print(f"  Bench-Eval:    n={len(bench_prompts)}, mean_len={np.mean(bench_lens):.0f}, median={np.median(bench_lens):.0f}")
    print(f"  Casual-Deploy: n={len(casual_prompts)}, mean_len={np.mean(casual_lens):.0f}, median={np.median(casual_lens):.0f}")

    return bench_prompts, casual_prompts


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        raise ValueError("Cannot find layers")


def extract_activations(model, tokenizer, prompts, layer_idx, device="cuda", batch_size=8, max_length=256):
    """Extract last token activations from a specific layer."""
    model.eval()
    all_activations = []
    layers = get_model_layers(model)

    activations_cache = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations_cache.append(hidden.detach())

    hook = layers[layer_idx].register_forward_hook(hook_fn)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Layer {layer_idx}", leave=False):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(device)
            activations_cache.clear()

            with torch.no_grad():
                model(**inputs)

            hidden = activations_cache[0]
            seq_lens = inputs.attention_mask.sum(dim=1)

            for j, seq_len in enumerate(seq_lens):
                all_activations.append(hidden[j, seq_len - 1, :].cpu())
    finally:
        hook.remove()

    return torch.stack(all_activations)


def evaluate_layer(bench_acts, casual_acts, n_cv=5):
    """Evaluate separation quality using linear probe."""
    X = torch.cat([bench_acts, casual_acts], dim=0).float().numpy()
    y = np.array([0] * len(bench_acts) + [1] * len(casual_acts))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')

    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    # Compute difference vector
    bench_mean = bench_acts.float().mean(dim=0)
    casual_mean = casual_acts.float().mean(dim=0)
    diff_vector = casual_mean - bench_mean

    return {
        "accuracy": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "auc": float(auc),
        "vector_norm": float(diff_vector.norm().item()),
        "diff_vector": diff_vector
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/my_projects/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=str, default="8,12,16,20,22,24,26,28,30,31")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print("AE Vector Extraction: BALANCED Datasets (1076 each)")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    model_path = Path(args.model).expanduser()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    n_layers = len(get_model_layers(model))
    print(f"Model loaded: {n_layers} layers")
    print(f"Target layers: {target_layers}")

    # Load datasets
    print("\n" + "=" * 60)
    bench_prompts, casual_prompts = load_balanced_datasets()

    # Extract activations
    print(f"\nExtracting activations from {len(target_layers)} layers...")
    all_bench_acts = {}
    all_casual_acts = {}

    for layer_idx in tqdm(target_layers, desc="Layers"):
        bench_acts = extract_activations(model, tokenizer, bench_prompts, layer_idx)
        casual_acts = extract_activations(model, tokenizer, casual_prompts, layer_idx)
        all_bench_acts[layer_idx] = bench_acts
        all_casual_acts[layer_idx] = casual_acts

    # Evaluate single layers
    print("\n" + "=" * 60)
    print("SINGLE LAYER EVALUATION")
    print("=" * 60)

    results = {}
    for layer_idx in target_layers:
        res = evaluate_layer(all_bench_acts[layer_idx], all_casual_acts[layer_idx])
        results[layer_idx] = res
        print(f"Layer {layer_idx:2d}: Acc={res['accuracy']:.3f}±{res['accuracy_std']:.3f}, "
              f"AUC={res['auc']:.3f}, ||v||={res['vector_norm']:.1f}")

    # Find best layers
    sorted_layers = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    top3_layers = [layer for layer, _ in sorted_layers[:3]]
    print(f"\n🏆 Top-3 layers: {top3_layers}")

    # Save results
    output_path = args.output or VECTORS_DIR / "ae_balanced_1076.pt"

    save_data = {
        "config": {
            "model": str(args.model),
            "n_samples": SAMPLE_SIZE,
            "contrast": "bench_eval vs casual_deploy_matched (balanced, length-matched)",
            "target_layers": target_layers
        },
        "single_layer_results": {k: {key: val for key, val in v.items() if key != "diff_vector"}
                                  for k, v in results.items()},
        "top3_layers": top3_layers,
        "vectors": {layer_idx: results[layer_idx]["diff_vector"] for layer_idx in target_layers}
    }

    torch.save(save_data, output_path)
    print(f"\n✅ Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_layer, best_res = sorted_layers[0]
    print(f"Best layer: {best_layer} (Acc={best_res['accuracy']:.3f}, AUC={best_res['auc']:.3f})")
    print(f"Samples: {SAMPLE_SIZE} bench vs {SAMPLE_SIZE} casual (balanced, length-matched)")


if __name__ == "__main__":
    main()

