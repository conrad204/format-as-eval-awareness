#!/usr/bin/env python3
"""
Cross-condition probe evaluation with CACHED activations.

This is the main experimental script for testing evaluation awareness vs format sensitivity.

Purpose:
--------
Tests whether models distinguish between evaluation and deployment contexts,
or simply respond to surface-level format cues (MCQ structure).

Experimental Design:
--------------------
Training: Linear probe on Bench-Eval (MCQ) vs Casual-Deploy (free-form)
Testing: 2×2 factorial design varying Format (MCQ/free) × Style (casual/formal)

Training data:
- Class 0 (Eval): Bench-Eval (A) - benchmark prompts with MCQ format
- Class 1 (Deploy): Casual-Deploy (B) - casual user queries

Test data (2×2 design):
- Casual-Eval (C) - free-form + casual style (baseline)
- Bench-Deploy (D) - MCQ format + casual style (rewritten)
- Casual-Deploy-Formal (E) - free-form + formal style (rewritten)
- Bench-Deploy-Formal (F) - MCQ format + formal style (rewritten)

Key Finding:
------------
Probe overwhelmingly responds to MCQ format (~94%), not style (~0%).
This suggests "evaluation awareness" may actually be "format sensitivity".

Technical Features:
-------------------
- Caching: Activations and probes cached independently for fast re-runs
- No truncation: Full input context preserved
- Proper normalization: Test data scaled using training scaler
- GPU selection: Specify which GPUs to use
- Detailed logging: Immediate output with flush=True

Usage:
------
python evaluate_probe_cached.py \\
    --model ~/models/Llama-3.1-8B-Instruct \\
    --layers "8,16,24,30,31" \\
    --gpus "0,1" \\
    --force_recompute  # Optional: clear cache

Output:
-------
- Results JSON: data/results/cross_condition_eval_cached.json
- Cached activations: data/cache/activations_{dataset}_layer{N}.pt
- Cached probes: data/cache/probe_layer{N}.pkl
"""

import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from tqdm import tqdm
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
CACHE_DIR = BASE_DIR / "data" / "cache"
RESULTS_DIR = BASE_DIR / "data" / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed seed
random.seed(42)
np.random.seed(42)

SAMPLE_SIZE = 1076


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


def load_dataset(name: str) -> Optional[List[str]]:
    """Load a dataset by name, return None if not found."""

    if name == "bench_eval":
        path = PROCESSED_DIR / "bench_eval.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        sampled = random.sample(data, min(SAMPLE_SIZE, len(data)))
        return [format_bench_eval_prompt(item) for item in sampled]

    elif name == "casual_deploy":
        path = PROCESSED_DIR / "casual_deploy_matched.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return [item['first_turn'] for item in data[:SAMPLE_SIZE]]

    elif name == "casual_eval":
        path = PROCESSED_DIR / "casual_eval_mutual.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return [item['first_turn'] for item in data[:SAMPLE_SIZE]]

    elif name == "bench_deploy":
        path = PROCESSED_DIR / "bench_deploy_rewritten.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return [item['rewritten'] for item in data if item.get('rewritten')]

    elif name == "casual_deploy_formal":
        path = PROCESSED_DIR / "casual_deploy_formal.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return [item['rewritten'] for item in data if item.get('rewritten')]

    elif name == "bench_deploy_formal":
        path = PROCESSED_DIR / "bench_deploy_formal.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return [item['rewritten'] for item in data if item.get('rewritten')]

    return None


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        raise ValueError("Cannot find layers")


def extract_activations(model, tokenizer, prompts, layer_idx, device="cuda", batch_size=8, max_length=None):
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
            tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
            if max_length is not None:
                tokenizer_kwargs["truncation"] = True
                tokenizer_kwargs["max_length"] = max_length
            inputs = tokenizer(batch, **tokenizer_kwargs).to(device)
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


def get_cache_path(dataset_name: str, layer_idx: int) -> Path:
    """Get cache file path for a dataset/layer combination.
    
    Args:
        dataset_name: Name of dataset (e.g., 'bench_eval', 'casual_deploy')
        layer_idx: Layer index (0-31 for 32-layer model)
        
    Returns:
        Path to cache file
    """
    return CACHE_DIR / f"activations_{dataset_name}_layer{layer_idx}.pt"


def load_or_extract_activations(
    model, tokenizer, datasets: Dict[str, List[str]],
    target_layers: List[int], force_recompute: bool = False
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load cached activations or extract new ones."""

    all_activations = {name: {} for name in datasets.keys()}

    for name, prompts in datasets.items():
        if prompts is None or len(prompts) == 0:
            continue

        for layer_idx in target_layers:
            cache_path = get_cache_path(name, layer_idx)

            if cache_path.exists() and not force_recompute:
                # Load from cache
                print(f"  Loading cached: {name} layer {layer_idx}", flush=True)
                all_activations[name][layer_idx] = torch.load(cache_path)
            else:
                # Extract and cache
                print(f"  Extracting: {name} layer {layer_idx} ({len(prompts)} prompts)", flush=True)
                acts = extract_activations(model, tokenizer, prompts, layer_idx)
                all_activations[name][layer_idx] = acts
                torch.save(acts, cache_path)
                print(f"    Cached to {cache_path.name}", flush=True)

    return all_activations


def load_or_train_probe(
    all_activations: Dict[str, Dict[int, torch.Tensor]],
    layer_idx: int,
    force_recompute: bool = False
) -> Tuple[LogisticRegression, StandardScaler]:
    """Load cached probe (classifier + scaler) or train new one.
    
    Trains on bench_eval (class 0) vs casual_deploy (class 1).
    Caches both the trained classifier and the fitted scaler for reproducibility.
    
    Args:
        all_activations: Nested dict {dataset_name: {layer_idx: activations}}
        layer_idx: Layer index to train probe on
        force_recompute: If True, ignore cache and retrain
        
    Returns:
        Tuple of (trained classifier, fitted scaler)
        
    Cache Location:
        data/cache/probe_layer{layer_idx}.pkl
    """
    cache_path = CACHE_DIR / f"probe_layer{layer_idx}.pkl"

    if not force_recompute and cache_path.exists():
        print(f"    ✓ Found cached probe for layer {layer_idx} -> Loading...", flush=True)
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            print(f"    ✓ Loaded probe for layer {layer_idx}", flush=True)
            return cache_data['clf'], cache_data['scaler']

    # Training
    print(f"    ✗ No cached probe for layer {layer_idx} -> Training new probe...", flush=True)
    X_train = torch.cat([
        all_activations['bench_eval'][layer_idx],
        all_activations['casual_deploy'][layer_idx]
    ], dim=0).float().numpy()

    y_train = np.array(
        [0] * len(all_activations['bench_eval'][layer_idx]) +
        [1] * len(all_activations['casual_deploy'][layer_idx])
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Cache the trained probe
    cache_data = {'clf': clf, 'scaler': scaler}
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"    ✓ Trained and cached probe to {cache_path.name}", flush=True)

    return clf, scaler


def evaluate_dataset(
    clf: LogisticRegression,
    scaler: StandardScaler,
    activations: Dict[int, torch.Tensor],
    layer_idx: int,
    name: str
) -> Optional[Dict[str, float]]:
    """Evaluate a single dataset with the trained classifier.
    
    Args:
        clf: Trained LogisticRegression classifier
        scaler: Fitted StandardScaler (from training data)
        activations: Dict mapping layer_idx to activation tensors
        layer_idx: Layer to evaluate
        name: Dataset name (for logging)
        
    Returns:
        Dict with classification metrics:
        - pct_eval: % classified as class 0 (eval/bench)
        - pct_deploy: % classified as class 1 (deploy/casual)
        - mean_prob: Mean probability of class 1
        - std_prob: Std of probability of class 1
        
        Returns None if activations not found.
    """
    if layer_idx not in activations or len(activations[layer_idx]) == 0:
        return None

    X = activations[layer_idx].float().numpy()
    X_scaled = scaler.transform(X)

    pred = clf.predict(X_scaled)
    prob = clf.predict_proba(X_scaled)[:, 1]

    return {
        'pct_eval': float((pred == 0).mean()),
        'pct_deploy': float((pred == 1).mean()),
        'mean_prob': float(prob.mean()),
        'std_prob': float(prob.std())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/my_projects/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=str, default="8,16,24,30,31", help="0-based layer indices, e.g. 24 = 25th layer (model.model.layers[24])")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force_recompute", action="store_true", help="Force recompute all activations")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g., '3,4')")
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    print("=" * 80)
    print("CROSS-CONDITION PROBE EVALUATION (with caching)")
    print("=" * 80)

    # Define all datasets
    DATASET_NAMES = [
        "bench_eval",           # A - training class 0
        "casual_deploy",        # B - training class 1
        "casual_eval",          # C - test
        "bench_deploy",         # D - test (MCQ + casual)
        "casual_deploy_formal", # E - test (no MCQ + formal)
        "bench_deploy_formal",  # F - test (MCQ + formal)
    ]

    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    for name in DATASET_NAMES:
        data = load_dataset(name)
        if data is not None:
            datasets[name] = data
            print(f"  ✓ {name}: {len(data)} prompts")
        else:
            print(f"  ✗ {name}: not found")

    # Check required datasets
    if "bench_eval" not in datasets or "casual_deploy" not in datasets:
        print("\nERROR: Missing required training datasets (bench_eval, casual_deploy)")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    model_path = Path(args.model).expanduser()

    # Set GPU visibility if specified
    if args.gpus:
        import os
        gpu_list = args.gpus.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        print(f"Using GPUs: {args.gpus} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

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

    # Load or extract activations
    print("\n" + "=" * 80)
    print("Loading/extracting activations (cached)...")
    print("=" * 80)

    all_activations = load_or_extract_activations(
        model, tokenizer, datasets, target_layers,
        force_recompute=args.force_recompute
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    results = {}
    test_datasets = ["casual_eval", "bench_deploy", "casual_deploy_formal", "bench_deploy_formal"]

    total_layers = len(target_layers)
    for layer_num, layer_idx in enumerate(target_layers, 1):
        print(f"\n{'=' * 80}", flush=True)
        print(f"Layer {layer_idx} ({layer_num}/{total_layers})", flush=True)
        print(f"{'=' * 80}", flush=True)

        # Load or train probe (classifier + scaler)
        clf, scaler = load_or_train_probe(
            all_activations, layer_idx,
            force_recompute=args.force_recompute
        )

        # Calculate training accuracy for reporting
        X_train = torch.cat([
            all_activations['bench_eval'][layer_idx],
            all_activations['casual_deploy'][layer_idx]
        ], dim=0).float().numpy()
        X_train_scaled = scaler.transform(X_train)
        train_acc = clf.score(X_train_scaled, np.array(
            [0] * len(all_activations['bench_eval'][layer_idx]) +
            [1] * len(all_activations['casual_deploy'][layer_idx])
        ))
        print(f"  Training (A vs B): Acc={train_acc:.3f}", flush=True)

        layer_results = {'train_acc': float(train_acc)}

        # Evaluate each test dataset
        print(f"  → Testing probe on {len(test_datasets)} test datasets...", flush=True)
        for ds_name in test_datasets:
            if ds_name in all_activations and layer_idx in all_activations[ds_name]:
                res = evaluate_dataset(clf, scaler, all_activations[ds_name], layer_idx, ds_name)
                if res:
                    layer_results[ds_name] = res
                    short_name = ds_name.replace("casual_", "").replace("bench_", "").replace("_", "-")[:10]
                    print(f"  {ds_name:20s}: {res['pct_eval']*100:5.1f}% → eval, {res['pct_deploy']*100:5.1f}% → deploy")

        results[layer_idx] = layer_results

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Header
    header = "Layer |"
    for ds in test_datasets:
        if ds in datasets:
            short = ds.replace("casual_", "C-").replace("bench_", "B-").replace("_formal", "-F").replace("_deploy", "-D").replace("_eval", "-E")
            header += f" {short[:8]:^8s} |"
    print(header)
    print("-" * len(header))

    # Rows
    for layer_idx in target_layers:
        row = f"  {layer_idx:2d}  |"
        for ds_name in test_datasets:
            if ds_name in results[layer_idx]:
                pct_eval = results[layer_idx][ds_name]['pct_eval'] * 100
                row += f" {pct_eval:5.1f}%E |"
            elif ds_name in datasets:
                row += "   N/A   |"
        print(row)

    print("\nLegend: %E = % classified as Eval/Bench (class 0)")
    print("        Higher %E = more formal/benchmark-like")
    print("        Lower %E = more casual-like")

    # Save results
    output_path = args.output or RESULTS_DIR / "cross_condition_eval_cached.json"

    save_data = {
        'config': {
            'model': str(args.model),
            'layers': target_layers,
            'datasets': {name: len(prompts) for name, prompts in datasets.items()}
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")
    print(f"📁 Activations cached in {CACHE_DIR}")


if __name__ == "__main__":
    main()


