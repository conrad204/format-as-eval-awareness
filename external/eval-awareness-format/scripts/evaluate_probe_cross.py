#!/usr/bin/env python3
"""
Cross-condition evaluation of the AE probe.

Training data:
- Class 0: Bench-Eval (A) - benchmark prompts in evaluation context
- Class 1: Casual-Deploy (B) - casual prompts in deployment context

Test data (WITHOUT retraining):
- Casual-Eval (C) - casual prompts in evaluation context (MuTual)
- Bench-Deploy (D) - benchmark-style prompts rewritten for deployment

Key diagnostic question:
When evaluation context and prompt genre are partially decoupled,
what does the probe actually track?

If probe tracks EVAL vs DEPLOY context:
  - Casual-Eval (C) → should predict "eval" (class 0)
  - Bench-Deploy (D) → should predict "deploy" (class 1)

If probe tracks BENCH vs CASUAL style:
  - Casual-Eval (C) → should predict "casual" (class 1)
  - Bench-Deploy (D) → should predict "bench" (class 0)
"""

import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
VECTORS_DIR = BASE_DIR / "data" / "vectors"
RESULTS_DIR = BASE_DIR / "data" / "results"
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


def load_all_datasets():
    """Load all 4 datasets for 2x2 matrix."""

    # A: Bench-Eval (training class 0)
    print("Loading Bench-Eval (A)...")
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval_raw = json.load(f)
    bench_eval_sampled = random.sample(bench_eval_raw, SAMPLE_SIZE)
    bench_eval_prompts = [format_bench_eval_prompt(item) for item in bench_eval_sampled]

    # B: Casual-Deploy (training class 1)
    print("Loading Casual-Deploy Matched (B)...")
    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        casual_deploy = json.load(f)
    casual_deploy_prompts = [item['first_turn'] for item in casual_deploy[:SAMPLE_SIZE]]

    # C: Casual-Eval (test - MuTual dialogues)
    print("Loading Casual-Eval MuTual (C)...")
    with open(PROCESSED_DIR / "casual_eval_mutual.json") as f:
        casual_eval = json.load(f)
    casual_eval_prompts = [item['first_turn'] for item in casual_eval[:SAMPLE_SIZE]]

    # D: Bench-Deploy Rewritten (test) - MCQ + casual style
    print("Loading Bench-Deploy Rewritten (D)...")
    with open(PROCESSED_DIR / "bench_deploy_rewritten.json") as f:
        bench_deploy = json.load(f)
    bench_deploy_prompts = [item['rewritten'] for item in bench_deploy]

    # E: Casual-Deploy Formal (test) - no MCQ + formal style
    print("Loading Casual-Deploy Formal (E)...")
    casual_deploy_formal_prompts = []
    formal_file = PROCESSED_DIR / "casual_deploy_formal.json"
    if formal_file.exists():
        with open(formal_file) as f:
            casual_deploy_formal = json.load(f)
        casual_deploy_formal_prompts = [item['rewritten'] for item in casual_deploy_formal if item.get('rewritten')]
    else:
        print("  WARNING: casual_deploy_formal.json not found, skipping")

    datasets = {
        'bench_eval': bench_eval_prompts,           # A - training class 0
        'casual_deploy': casual_deploy_prompts,     # B - training class 1
        'casual_eval': casual_eval_prompts,         # C - test
        'bench_deploy': bench_deploy_prompts,       # D - test (MCQ + casual)
        'casual_deploy_formal': casual_deploy_formal_prompts,  # E - test (no MCQ + formal)
    }

    print(f"\nDataset sizes:")
    for name, prompts in datasets.items():
        lens = [len(p) for p in prompts]
        print(f"  {name}: n={len(prompts)}, mean_len={np.mean(lens):.0f}")

    return datasets


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/my_projects/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=str, default="8,16,24,30,31", help="0-based layer indices, e.g. 24 = 25th layer (model.model.layers[24])")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("CROSS-CONDITION PROBE EVALUATION")
    print("=" * 70)
    print("\n2x2 Matrix:")
    print("              |   EVAL context   |  DEPLOY context")
    print("  ------------|------------------|------------------")
    print("  BENCH style |  A: Bench-Eval   |  D: Bench-Deploy")
    print("  CASUAL style|  C: Casual-Eval  |  B: Casual-Deploy")
    print()
    print("Training: A (class 0) vs B (class 1)")
    print("Testing:  C and D (without retraining)")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    datasets = load_all_datasets()

    # Extract activations for all datasets
    print("\n" + "=" * 70)
    print("Extracting activations...")
    print("=" * 70)

    all_activations = {name: {} for name in datasets.keys()}

    for layer_idx in tqdm(target_layers, desc="Layers"):
        for name, prompts in datasets.items():
            acts = extract_activations(model, tokenizer, prompts, layer_idx)
            all_activations[name][layer_idx] = acts

    # Evaluate for each layer
    print("\n" + "=" * 70)
    print("RESULTS: Cross-condition probe evaluation")
    print("=" * 70)

    results = {}

    for layer_idx in target_layers:
        print(f"\n--- Layer {layer_idx} ---")

        # Training data
        X_train = torch.cat([
            all_activations['bench_eval'][layer_idx],
            all_activations['casual_deploy'][layer_idx]
        ], dim=0).float().numpy()

        y_train = np.array(
            [0] * len(all_activations['bench_eval'][layer_idx]) +
            [1] * len(all_activations['casual_deploy'][layer_idx])
        )

        # Fit scaler and classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Training accuracy
        train_acc = clf.score(X_train_scaled, y_train)
        print(f"  Training (A vs B): Acc={train_acc:.3f}")

        # Test on Casual-Eval (C)
        X_casual_eval = all_activations['casual_eval'][layer_idx].float().numpy()
        X_casual_eval_scaled = scaler.transform(X_casual_eval)

        pred_c = clf.predict(X_casual_eval_scaled)
        prob_c = clf.predict_proba(X_casual_eval_scaled)[:, 1]

        pct_eval_c = (pred_c == 0).mean()  # % predicted as "eval" (class 0)
        pct_deploy_c = (pred_c == 1).mean()  # % predicted as "deploy" (class 1)

        print(f"  Casual-Eval (C):   {pct_eval_c*100:.1f}% → eval (0), {pct_deploy_c*100:.1f}% → deploy (1), mean_prob={prob_c.mean():.3f}")

        # Test on Bench-Deploy (D) - MCQ + casual style
        X_bench_deploy = all_activations['bench_deploy'][layer_idx].float().numpy()
        X_bench_deploy_scaled = scaler.transform(X_bench_deploy)

        pred_d = clf.predict(X_bench_deploy_scaled)
        prob_d = clf.predict_proba(X_bench_deploy_scaled)[:, 1]

        pct_eval_d = (pred_d == 0).mean()
        pct_deploy_d = (pred_d == 1).mean()

        print(f"  Bench-Deploy (D):  {pct_eval_d*100:.1f}% → eval (0), {pct_deploy_d*100:.1f}% → deploy (1), mean_prob={prob_d.mean():.3f}")

        # Test on Casual-Deploy-Formal (E) - no MCQ + formal style
        pct_eval_e, pct_deploy_e, prob_e_mean, prob_e_std = 0, 0, 0, 0
        if len(all_activations.get('casual_deploy_formal', {}).get(layer_idx, [])) > 0:
            X_formal = all_activations['casual_deploy_formal'][layer_idx].float().numpy()
            X_formal_scaled = scaler.transform(X_formal)

            pred_e = clf.predict(X_formal_scaled)
            prob_e = clf.predict_proba(X_formal_scaled)[:, 1]

            pct_eval_e = (pred_e == 0).mean()
            pct_deploy_e = (pred_e == 1).mean()
            prob_e_mean = prob_e.mean()
            prob_e_std = prob_e.std()

            print(f"  Casual-Formal (E): {pct_eval_e*100:.1f}% → eval (0), {pct_deploy_e*100:.1f}% → deploy (1), mean_prob={prob_e_mean:.3f}")

        # Interpretation
        print(f"\n  📊 Interpretation:")
        if pct_eval_c > 0.5 and pct_deploy_d > 0.5:
            print(f"     → Probe tracks CONTEXT (eval vs deploy)")
        elif pct_deploy_c > 0.5 and pct_eval_d > 0.5:
            print(f"     → Probe tracks STYLE (bench vs casual)")
        else:
            print(f"     → Mixed/unclear signal")

        if pct_eval_e > 0.5:
            print(f"     → Casual-Formal (E) classified as FORMAL/BENCH (supports STYLE hypothesis)")
        elif pct_deploy_e > 0.5:
            print(f"     → Casual-Formal (E) classified as CASUAL (supports CONTENT hypothesis)")

        results[layer_idx] = {
            'train_acc': float(train_acc),
            'casual_eval': {
                'pct_eval': float(pct_eval_c),
                'pct_deploy': float(pct_deploy_c),
                'mean_prob': float(prob_c.mean()),
                'std_prob': float(prob_c.std())
            },
            'bench_deploy': {
                'pct_eval': float(pct_eval_d),
                'pct_deploy': float(pct_deploy_d),
                'mean_prob': float(prob_d.mean()),
                'std_prob': float(prob_d.std())
            },
            'casual_deploy_formal': {
                'pct_eval': float(pct_eval_e),
                'pct_deploy': float(pct_deploy_e),
                'mean_prob': float(prob_e_mean),
                'std_prob': float(prob_e_std)
            }
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nIf probe tracks CONTEXT (eval vs deploy):")
    print("  → Casual-Eval (C) should be classified as 'eval' (class 0)")
    print("  → Bench-Deploy (D) should be classified as 'deploy' (class 1)")
    print("\nIf probe tracks STYLE (bench vs casual):")
    print("  → Casual-Eval (C) should be classified as 'casual' (class 1)")
    print("  → Bench-Deploy (D) should be classified as 'bench' (class 0)")

    print("\n" + "-" * 90)
    print("Layer | C→eval | C→deploy | D→eval | D→deploy | E→eval | E→deploy | Interpretation")
    print("-" * 90)

    for layer_idx in target_layers:
        r = results[layer_idx]
        c_eval = r['casual_eval']['pct_eval'] * 100
        c_deploy = r['casual_eval']['pct_deploy'] * 100
        d_eval = r['bench_deploy']['pct_eval'] * 100
        d_deploy = r['bench_deploy']['pct_deploy'] * 100
        e_eval = r['casual_deploy_formal']['pct_eval'] * 100
        e_deploy = r['casual_deploy_formal']['pct_deploy'] * 100

        if c_eval > 50 and d_deploy > 50:
            interp = "CONTEXT"
        elif c_deploy > 50 and d_eval > 50:
            interp = "STYLE"
        else:
            interp = "MIXED"

        print(f"  {layer_idx:2d}  | {c_eval:5.1f}% | {c_deploy:6.1f}%  | {d_eval:5.1f}% | {d_deploy:6.1f}%  | {e_eval:5.1f}% | {e_deploy:6.1f}%  | {interp}")

    # Save results
    output_path = args.output or RESULTS_DIR / "cross_condition_eval.json"

    save_data = {
        'config': {
            'model': str(args.model),
            'layers': target_layers,
            'sample_size': SAMPLE_SIZE,
            'datasets': {
                'bench_eval': len(datasets['bench_eval']),
                'casual_deploy': len(datasets['casual_deploy']),
                'casual_eval': len(datasets['casual_eval']),
                'bench_deploy': len(datasets['bench_deploy']),
                'casual_deploy_formal': len(datasets.get('casual_deploy_formal', []))
            }
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()

