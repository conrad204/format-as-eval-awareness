#!/usr/bin/env python3
"""
Extract AE vectors from LENGTH-MATCHED datasets.

Uses:
- Bench-Eval: sampled to match size
- Casual-Deploy Matched: filtered to match Bench-Eval length distribution

This controls for prompt length as a confound.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
import random
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

random.seed(42)
np.random.seed(42)


def load_matched_datasets(n_samples: int):
    """Load length-matched datasets."""
    
    # Load casual_deploy_matched
    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        casual_matched = json.load(f)
    
    casual_prompts = [item["first_turn"][:500] for item in casual_matched[:n_samples]]
    
    # Load bench_eval and sample same amount
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval = json.load(f)
    
    # Sample randomly
    bench_sampled = random.sample(bench_eval, min(n_samples, len(bench_eval)))
    
    bench_prompts = []
    for item in bench_sampled:
        source = item.get("source", "unknown")
        if source == "mmlu":
            q = item.get("question", "")
            choices = item.get("choices", [])
            prompt = f"Question: {q}\nChoices: {', '.join(choices)}\nAnswer:"
        elif source == "arc":
            q = item.get("question", "")
            choices = item.get("choices", {})
            if isinstance(choices, dict):
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                choice_str = ", ".join([f"{l}) {t}" for l, t in zip(labels, texts)])
            else:
                choice_str = str(choices)
            prompt = f"Question: {q}\nOptions: {choice_str}\nAnswer:"
        elif source == "gsm8k":
            q = item.get("question", "")
            prompt = f"Problem: {q}\nSolution:"
        else:
            prompt = str(item.get("question", item))
        bench_prompts.append(prompt[:500])
    
    print(f"Loaded {len(bench_prompts)} bench_eval (sampled)")
    print(f"Loaded {len(casual_prompts)} casual_deploy (matched)")
    
    # Print length stats
    bench_lens = [len(p) for p in bench_prompts]
    casual_lens = [len(p) for p in casual_prompts]
    print(f"Bench lengths: mean={np.mean(bench_lens):.0f}, median={np.median(bench_lens):.0f}")
    print(f"Casual lengths: mean={np.mean(casual_lens):.0f}, median={np.median(casual_lens):.0f}")
    
    return bench_prompts, casual_prompts


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        raise ValueError("Cannot find layers")


def extract_activations(model, tokenizer, prompts, layer_idx, device="cuda", batch_size=8, max_length=256):
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
        for i in range(0, len(prompts), batch_size):
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
    X = torch.cat([bench_acts, casual_acts], dim=0).float().numpy()
    y = np.array([0] * len(bench_acts) + [1] * len(casual_acts))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')
    
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    
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


def evaluate_multi_layer(all_bench_acts, all_casual_acts, layer_indices, n_cv=5):
    bench_concat = torch.cat([all_bench_acts[i] for i in layer_indices], dim=1)
    casual_concat = torch.cat([all_casual_acts[i] for i in layer_indices], dim=1)
    
    X = torch.cat([bench_concat, casual_concat], dim=0).float().numpy()
    y = np.array([0] * len(bench_concat) + [1] * len(casual_concat))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')
    
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    
    return {
        "layers": layer_indices,
        "accuracy": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "auc": float(auc)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/my_projects/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_samples", type=int, default=817)
    parser.add_argument("--layers", type=str, default="8,12,16,20,24,28,31")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    target_layers = [int(x) for x in args.layers.split(",")]
    
    print("=" * 60)
    print("AE Vector Extraction: LENGTH-MATCHED Datasets")
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
    
    # Load datasets
    print("\nLoading length-matched datasets...")
    bench_prompts, casual_prompts = load_matched_datasets(args.n_samples)
    
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
        print(f"Layer {layer_idx:2d}: Acc={res['accuracy']:.3f}±{res['accuracy_std']:.3f}, AUC={res['auc']:.3f}, ||v||={res['vector_norm']:.1f}")
    
    # Find top-3 layers
    sorted_layers = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    top3_layers = [layer for layer, _ in sorted_layers[:3]]
    print(f"\n🏆 Top-3 layers: {top3_layers}")
    
    # Multi-layer evaluation
    print("\n" + "=" * 60)
    print("MULTI-LAYER EVALUATION")
    print("=" * 60)
    
    multi_results = []
    
    # Top-3 combined
    res3 = evaluate_multi_layer(all_bench_acts, all_casual_acts, top3_layers)
    multi_results.append(res3)
    print(f"Top-3 {top3_layers}: Acc={res3['accuracy']:.3f}±{res3['accuracy_std']:.3f}, AUC={res3['auc']:.3f}")
    
    # Top-2 combined
    top2_layers = top3_layers[:2]
    res2 = evaluate_multi_layer(all_bench_acts, all_casual_acts, top2_layers)
    multi_results.append(res2)
    print(f"Top-2 {top2_layers}: Acc={res2['accuracy']:.3f}±{res2['accuracy_std']:.3f}, AUC={res2['auc']:.3f}")
    
    # Save results
    output_path = args.output or VECTORS_DIR / "ae_matched_length.pt"
    
    save_data = {
        "config": {
            "model": str(args.model),
            "n_samples": len(bench_prompts),
            "contrast": "bench_eval vs casual_deploy (length-matched)",
            "target_layers": target_layers
        },
        "single_layer_results": {k: {key: val for key, val in v.items() if key != "diff_vector"}
                                  for k, v in results.items()},
        "multi_layer_results": multi_results,
        "top3_layers": top3_layers,
        "vectors": {layer_idx: results[layer_idx]["diff_vector"] for layer_idx in target_layers}
    }
    
    torch.save(save_data, output_path)
    print(f"\n✅ Results saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_single = sorted_layers[0]
    print(f"Best single layer: {best_single[0]} (Acc={best_single[1]['accuracy']:.3f})")
    print(f"Top-3 combined: {top3_layers} (Acc={res3['accuracy']:.3f})")
    print(f"Samples: {len(bench_prompts)} bench vs {len(casual_prompts)} casual (length-matched)")


if __name__ == "__main__":
    main()


