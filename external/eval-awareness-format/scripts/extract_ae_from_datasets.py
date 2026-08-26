#!/usr/bin/env python3
"""
Extract Activation Engineering (AE) vectors from Bench-Eval vs Casual-Deploy.

Purpose:
--------
Identifies which layers best distinguish between benchmark (eval) and casual (deploy)
contexts, and extracts steering vectors for potential interventions.

Method:
-------
1. Extract last-token activations from target layers
2. Train linear probes (LogisticRegression) on each layer
3. Evaluate separation quality (accuracy, AUC)
4. Compute mean difference vectors: casual_mean - bench_mean
5. Test single-layer and multi-layer combinations

Output:
-------
Saves to data/vectors/ae_bench_vs_casual_llama.pt:
- Difference vectors for each layer
- Linear probe weights and metrics
- Best single-layer and multi-layer combinations

Usage:
------
python extract_ae_from_datasets.py \\
    --model ~/models/Llama-3.1-8B-Instruct \\
    --n_samples 200 \\
    --output data/vectors/ae_vectors_custom.pt

Key Metrics:
------------
- Accuracy: Cross-validated classification accuracy
- AUC: Area under ROC curve
- Vector norm: Magnitude of difference vector

Target Layers:
--------------
Default: [8, 12, 16, 20, 22, 24, 26, 28, 30, 31]
Based on prior work showing middle-to-late layers encode semantic features.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
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


@dataclass
class Config:
    model_path: str = "~/my_projects/models/Llama-3.1-8B-Instruct"
    n_samples: int = 200  # samples per class
    max_length: int = 256
    batch_size: int = 8
    device: str = "cuda"
    # Layers to analyze (based on paper: every 4th layer + final layers)
    target_layers: list = None  # Will be set based on model

    def get_target_layers(self, n_layers: int) -> list:
        """Get target layers for analysis."""
        if self.target_layers:
            return [l for l in self.target_layers if l < n_layers]
        # Default: [8, 12, 16, 20, 22, 24, 26, 28, 30, 31] for 32-layer model
        default_layers = [8, 12, 16, 20, 22, 24, 26, 28, 30, 31]
        return [l for l in default_layers if l < n_layers]


def load_datasets(n_samples: int) -> Tuple[List[str], List[str]]:
    """Load and format samples from bench_eval and casual_deploy."""

    # Load bench_eval
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval = json.load(f)

    # Load casual_deploy
    with open(PROCESSED_DIR / "casual_deploy_sharegpt4.json") as f:
        casual_deploy = json.load(f)

    # Format bench_eval prompts (benchmark style)
    bench_prompts = []
    for item in bench_eval[:n_samples]:
        if item.get("source") == "mmlu":
            q = item.get("question", "")
            choices = item.get("choices", [])
            prompt = f"Question: {q}\nChoices: {', '.join(choices)}\nAnswer:"
        elif item.get("source") == "arc":
            q = item.get("question", "")
            choices = item.get("choices", {})
            if isinstance(choices, dict):
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                choice_str = ", ".join([f"{l}) {t}" for l, t in zip(labels, texts)])
            else:
                choice_str = str(choices)
            prompt = f"Question: {q}\nOptions: {choice_str}\nAnswer:"
        elif item.get("source") == "gsm8k":
            q = item.get("question", "")
            prompt = f"Problem: {q}\nSolution:"
        else:
            prompt = str(item.get("question", item))
        bench_prompts.append(prompt[:500])  # truncate

    # Format casual_deploy prompts (first user query only)
    casual_prompts = []
    for item in casual_deploy[:n_samples]:
        convs = item.get("conversations", [])
        # Find first human/user message (handle both formats: "from" or "user" field)
        for turn in convs:
            sender = turn.get("from", turn.get("user", "")).lower()
            if sender in ["human", "user"]:
                # Get text (handle both "value" and "text" fields)
                text = turn.get("value", turn.get("text", ""))[:500]
                casual_prompts.append(text)
                break

    print(f"Loaded {len(bench_prompts)} bench_eval, {len(casual_prompts)} casual_deploy")
    return bench_prompts, casual_prompts


def get_model_layers(model):
    """Get the layers module from model (handles different architectures)."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("Cannot find layers in model architecture")


def extract_activations(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    config: Config
) -> torch.Tensor:
    """Extract activations from a specific layer for all prompts."""

    model.eval()
    all_activations = []
    layers = get_model_layers(model)

    # Register hook to capture activations
    activations_cache = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations_cache.append(hidden.detach())

    hook = layers[layer_idx].register_forward_hook(hook_fn)

    try:
        for i in range(0, len(prompts), config.batch_size):
            batch = prompts[i:i + config.batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length
            ).to(config.device)

            activations_cache.clear()

            with torch.no_grad():
                model(**inputs)

            # Get last token activation for each sequence
            hidden = activations_cache[0]
            seq_lens = inputs.attention_mask.sum(dim=1)

            batch_acts = []
            for j, seq_len in enumerate(seq_lens):
                batch_acts.append(hidden[j, seq_len - 1, :].cpu())

            all_activations.extend(batch_acts)
    finally:
        hook.remove()

    return torch.stack(all_activations)


def evaluate_layer(
    bench_acts: torch.Tensor,
    casual_acts: torch.Tensor,
    n_cv: int = 5
) -> Dict[str, float]:
    """Evaluate separation quality using linear probe."""

    # Prepare data - convert to float32 first
    X = torch.cat([bench_acts, casual_acts], dim=0).float().numpy()
    y = np.array([0] * len(bench_acts) + [1] * len(casual_acts))

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')

    # Full fit for AUC
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    # Compute mean difference vector (convert to float32)
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


def evaluate_multi_layer(
    all_bench_acts: Dict[int, torch.Tensor],
    all_casual_acts: Dict[int, torch.Tensor],
    layer_indices: List[int],
    n_cv: int = 5
) -> Dict[str, float]:
    """Evaluate separation using concatenated activations from multiple layers."""

    bench_concat = torch.cat([all_bench_acts[i] for i in layer_indices], dim=1)
    casual_concat = torch.cat([all_casual_acts[i] for i in layer_indices], dim=1)

    # Convert to float32
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
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = Config(
        model_path=args.model,
        n_samples=args.n_samples
    )

    print("=" * 60)
    print("AE Vector Extraction: Bench-Eval vs Casual-Deploy")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {config.model_path}")
    model_path = Path(config.model_path).expanduser()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    n_layers = len(get_model_layers(model))
    target_layers = config.get_target_layers(n_layers)
    print(f"Model loaded: {n_layers} layers")
    print(f"Target layers: {target_layers}")

    # Load datasets
    print("\nLoading datasets...")
    bench_prompts, casual_prompts = load_datasets(config.n_samples)

    # Extract activations for target layers only
    print(f"\nExtracting activations from {len(target_layers)} target layers...")
    all_bench_acts = {}
    all_casual_acts = {}

    for layer_idx in tqdm(target_layers, desc="Layers"):
        bench_acts = extract_activations(model, tokenizer, bench_prompts, layer_idx, config)
        casual_acts = extract_activations(model, tokenizer, casual_prompts, layer_idx, config)
        all_bench_acts[layer_idx] = bench_acts
        all_casual_acts[layer_idx] = casual_acts

    # Evaluate single layers
    print("\n" + "=" * 60)
    print("SINGLE LAYER EVALUATION")
    print("=" * 60)

    results = {}
    best_layer = -1
    best_acc = 0

    for layer_idx in target_layers:
        res = evaluate_layer(all_bench_acts[layer_idx], all_casual_acts[layer_idx])
        results[layer_idx] = res

        if res["accuracy"] > best_acc:
            best_acc = res["accuracy"]
            best_layer = layer_idx

        print(f"Layer {layer_idx:2d}: Acc={res['accuracy']:.3f}±{res['accuracy_std']:.3f}, AUC={res['auc']:.3f}, ||v||={res['vector_norm']:.1f}")

    print(f"\n🏆 Best single layer: {best_layer} (Acc={best_acc:.3f})")

    # Find top-3 layers by accuracy
    sorted_layers = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    top3_layers = [layer for layer, _ in sorted_layers[:3]]
    print(f"\n📊 Top-3 layers: {top3_layers}")
    
    # Evaluate multi-layer combinations
    print("\n" + "=" * 60)
    print("MULTI-LAYER EVALUATION (Top-3 combinations)")
    print("=" * 60)
    
    multi_results = []
    
    # Test top-3 combined
    print(f"\nTesting combination: {top3_layers}")
    res = evaluate_multi_layer(all_bench_acts, all_casual_acts, top3_layers)
    multi_results.append(res)
    print(f"  Layers {top3_layers}: Acc={res['accuracy']:.3f}±{res['accuracy_std']:.3f}, AUC={res['auc']:.3f}")
    
    # Test top-2 combined
    top2_layers = top3_layers[:2]
    print(f"\nTesting combination: {top2_layers}")
    res2 = evaluate_multi_layer(all_bench_acts, all_casual_acts, top2_layers)
    multi_results.append(res2)
    print(f"  Layers {top2_layers}: Acc={res2['accuracy']:.3f}±{res2['accuracy_std']:.3f}, AUC={res2['auc']:.3f}")
    
    # Find best multi-layer result
    best_multi = max(multi_results, key=lambda x: x["accuracy"])

    # Save vectors
    output_path = args.output or VECTORS_DIR / "ae_bench_vs_casual_llama.pt"

    save_data = {
        "config": {
            "model": str(config.model_path),
            "n_samples": config.n_samples,
            "contrast": "bench_eval vs casual_deploy",
            "target_layers": target_layers
        },
        "single_layer_results": {k: {key: val for key, val in v.items() if key != "diff_vector"}
                                  for k, v in results.items()},
        "multi_layer_results": [{"layers": r["layers"], "accuracy": r["accuracy"],
                                  "accuracy_std": r["accuracy_std"], "auc": r["auc"]}
                                 for r in multi_results],
        "best_single_layer": best_layer,
        "best_multi_layers": best_multi["layers"],
        "vectors": {
            layer_idx: results[layer_idx]["diff_vector"]
            for layer_idx in target_layers
        }
    }

    torch.save(save_data, output_path)
    print(f"\n✅ Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Analyzed layers: {target_layers}")
    print(f"Best single layer: {best_layer} (Acc={best_acc:.3f})")
    print(f"Best multi-layer: {best_multi['layers']} (Acc={best_multi['accuracy']:.3f})")
    print(f"Vectors saved for all {len(target_layers)} target layers")


if __name__ == "__main__":
    main()

