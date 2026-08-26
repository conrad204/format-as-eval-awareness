#!/usr/bin/env python3
"""
Analyze disagreements between bench_deploy and bench_deploy_formal predictions.
Find examples where the same query (by id) gets different classifications.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
CACHE_DIR = BASE_DIR / "data" / "cache"

# Layer index 0-based (same as --layers and model.model.layers[i]).
# 25 = 26th transformer layer. HF: hidden_states[0]=embeddings, hidden_states[k]=after layers[k-1].
TARGET_LAYER_0BASED = 25

def format_bench_eval_prompt(item):
    """Format bench_eval item as prompt string."""
    if item.get('source') == 'gsm8k':
        return f"Question: {item['question']}\nAnswer:"
    else:
        # MMLU/ARC format
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(item['choices'])])
        return f"Question: {item['question']}\n{choices_str}\nAnswer:"

def load_datasets():
    """Load all required datasets."""
    datasets = {}

    # Training data
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        datasets['bench_eval'] = json.load(f)[:1076]
    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        datasets['casual_deploy'] = json.load(f)[:1076]

    # Test data - need full data with ids
    with open(PROCESSED_DIR / "bench_deploy_rewritten.json") as f:
        datasets['bench_deploy'] = json.load(f)
    with open(PROCESSED_DIR / "bench_deploy_formal.json") as f:
        datasets['bench_deploy_formal'] = json.load(f)

    return datasets

def extract_activations(model, tokenizer, prompts: List[str], layer_idx_0based: int, device: str):
    """Extract activations for a list of prompts.
    layer_idx_0based: 0-based layer index (e.g. 25 = 26th layer). Uses hidden_states[layer_idx_0based+1].
    """
    activations = []
    # HuggingFace: hidden_states[0]=embeddings, hidden_states[i+1]=output after model.model.layers[i]
    hf_idx = layer_idx_0based + 1

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[hf_idx]
            last_token = hidden[0, -1, :].cpu().numpy()
            activations.append(last_token)

    return np.array(activations)

def main():
    print("=" * 80)
    print("DISAGREEMENT ANALYSIS: bench_deploy vs bench_deploy_formal")
    print("=" * 80)

    # Load datasets
    print("\n📂 Loading datasets...")
    datasets = load_datasets()

    print(f"  bench_eval: {len(datasets['bench_eval'])} items")
    print(f"  casual_deploy: {len(datasets['casual_deploy'])} items")
    print(f"  bench_deploy: {len(datasets['bench_deploy'])} items")
    print(f"  bench_deploy_formal: {len(datasets['bench_deploy_formal'])} items")

    # Build id mapping for bench_deploy_formal
    formal_by_id = {item.get('id', item.get('original_id', i)): item
                    for i, item in enumerate(datasets['bench_deploy_formal'])}

    # Find common ids
    bench_ids = [item.get('id', item.get('original_id', i))
                 for i, item in enumerate(datasets['bench_deploy'])]
    formal_ids = set(formal_by_id.keys())
    common_ids = [id for id in bench_ids if id in formal_ids]
    print(f"\n  Common IDs: {len(common_ids)}")

    # Load model
    print("\n🔧 Loading model...")
    model_path = Path.home() / "my_projects" / "models" / "Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract training activations
    print(f"\n📊 Extracting activations for layer {TARGET_LAYER_0BASED}...")

    bench_eval_prompts = [format_bench_eval_prompt(item) for item in datasets['bench_eval']]
    casual_deploy_prompts = [item['first_turn'] for item in datasets['casual_deploy']]

    print("  Training data...")
    train_bench = extract_activations(model, tokenizer, bench_eval_prompts, TARGET_LAYER_0BASED, device)
    train_casual = extract_activations(model, tokenizer, casual_deploy_prompts, TARGET_LAYER_0BASED, device)

    # Prepare training data
    X_train = np.vstack([train_bench, train_casual])
    y_train = np.array([0] * len(train_bench) + [1] * len(train_casual))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train probe
    print("  Training probe...")
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)

    # Extract test activations and get predictions
    print("  Extracting bench_deploy activations...")
    bench_deploy_prompts = [item['rewritten'] for item in datasets['bench_deploy']]
    bench_deploy_acts = extract_activations(model, tokenizer, bench_deploy_prompts, TARGET_LAYER_0BASED, device)
    bench_deploy_scaled = scaler.transform(bench_deploy_acts)
    bench_deploy_preds = probe.predict(bench_deploy_scaled)
    bench_deploy_probs = probe.predict_proba(bench_deploy_scaled)[:, 0]  # P(eval)

    print("  Extracting bench_deploy_formal activations...")
    # Match by id
    formal_prompts = []
    formal_ids_ordered = []
    for i, item in enumerate(datasets['bench_deploy']):
        item_id = item.get('id', item.get('original_id', i))
        if item_id in formal_by_id:
            formal_prompts.append(formal_by_id[item_id]['rewritten'])
            formal_ids_ordered.append(item_id)

    formal_acts = extract_activations(model, tokenizer, formal_prompts, TARGET_LAYER_0BASED, device)
    formal_scaled = scaler.transform(formal_acts)
    formal_preds = probe.predict(formal_scaled)
    formal_probs = probe.predict_proba(formal_scaled)[:, 0]  # P(eval)

    # Analyze disagreements
    print("\n" + "=" * 80)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 80)

    # Build results by id
    results = []
    for i, item in enumerate(datasets['bench_deploy']):
        item_id = item.get('id', item.get('original_id', i))
        if item_id in formal_by_id and i < len(bench_deploy_preds):
            # Find matching formal index
            try:
                formal_idx = formal_ids_ordered.index(item_id)
                results.append({
                    'id': item_id,
                    'original': item.get('original', ''),
                    'bench_deploy_text': item['rewritten'],
                    'bench_deploy_formal_text': formal_by_id[item_id]['rewritten'],
                    'bench_pred': bench_deploy_preds[i],
                    'formal_pred': formal_preds[formal_idx],
                    'bench_prob_eval': bench_deploy_probs[i],
                    'formal_prob_eval': formal_probs[formal_idx],
                    'label': item.get('label', 'unknown')
                })
            except (ValueError, IndexError):
                continue

    print(f"\nTotal paired samples: {len(results)}")

    # Count disagreements
    agree_eval = sum(1 for r in results if r['bench_pred'] == 0 and r['formal_pred'] == 0)
    agree_deploy = sum(1 for r in results if r['bench_pred'] == 1 and r['formal_pred'] == 1)
    disagree_bench_eval = sum(1 for r in results if r['bench_pred'] == 0 and r['formal_pred'] == 1)
    disagree_formal_eval = sum(1 for r in results if r['bench_pred'] == 1 and r['formal_pred'] == 0)

    print("\n📊 CONFUSION MATRIX (bench_deploy vs bench_deploy_formal)")
    print("-" * 50)
    print(f"                    bench_deploy_formal")
    print(f"                    EVAL        DEPLOY")
    print(f"bench_deploy EVAL   {agree_eval:5d}       {disagree_bench_eval:5d}")
    print(f"             DEPLOY {disagree_formal_eval:5d}       {agree_deploy:5d}")
    print("-" * 50)

    total = len(results)
    agreement_rate = (agree_eval + agree_deploy) / total * 100
    print(f"\n📈 METRICS:")
    print(f"  Agreement rate: {agreement_rate:.1f}%")
    print(f"  bench_deploy → EVAL: {(agree_eval + disagree_bench_eval) / total * 100:.1f}%")
    print(f"  bench_deploy_formal → EVAL: {(agree_eval + disagree_formal_eval) / total * 100:.1f}%")
    print(f"  Disagreements where bench=EVAL, formal=DEPLOY: {disagree_bench_eval} ({disagree_bench_eval/total*100:.1f}%)")
    print(f"  Disagreements where bench=DEPLOY, formal=EVAL: {disagree_formal_eval} ({disagree_formal_eval/total*100:.1f}%)")

    # Find examples where bench=EVAL but formal=DEPLOY
    print("\n" + "=" * 80)
    print("5 EXAMPLES: bench_deploy=EVAL, bench_deploy_formal=DEPLOY")
    print("=" * 80)

    disagreements = [r for r in results if r['bench_pred'] == 0 and r['formal_pred'] == 1]
    disagreements.sort(key=lambda x: x['bench_prob_eval'] - x['formal_prob_eval'], reverse=True)

    for i, ex in enumerate(disagreements[:5]):
        print(f"\n{'─' * 80}")
        print(f"EXAMPLE {i+1} | ID: {ex['id']} | Label: {ex['label']}")
        print(f"{'─' * 80}")
        print(f"\n📝 ORIGINAL:")
        print(f"   {ex['original'][:200]}..." if len(ex['original']) > 200 else f"   {ex['original']}")
        print(f"\n✅ bench_deploy (pred=EVAL, P={ex['bench_prob_eval']:.3f}):")
        print(f"   {ex['bench_deploy_text'][:300]}..." if len(ex['bench_deploy_text']) > 300 else f"   {ex['bench_deploy_text']}")
        print(f"\n❌ bench_deploy_formal (pred=DEPLOY, P={ex['formal_prob_eval']:.3f}):")
        print(f"   {ex['bench_deploy_formal_text'][:300]}..." if len(ex['bench_deploy_formal_text']) > 300 else f"   {ex['bench_deploy_formal_text']}")

    # Also show reverse: formal=EVAL but bench=DEPLOY
    print("\n" + "=" * 80)
    print("5 EXAMPLES: bench_deploy=DEPLOY, bench_deploy_formal=EVAL (reverse)")
    print("=" * 80)

    reverse_disagreements = [r for r in results if r['bench_pred'] == 1 and r['formal_pred'] == 0]
    reverse_disagreements.sort(key=lambda x: x['formal_prob_eval'] - x['bench_prob_eval'], reverse=True)

    for i, ex in enumerate(reverse_disagreements[:5]):
        print(f"\n{'─' * 80}")
        print(f"EXAMPLE {i+1} | ID: {ex['id']} | Label: {ex['label']}")
        print(f"{'─' * 80}")
        print(f"\n📝 ORIGINAL:")
        print(f"   {ex['original'][:200]}..." if len(ex['original']) > 200 else f"   {ex['original']}")
        print(f"\n❌ bench_deploy (pred=DEPLOY, P={ex['bench_prob_eval']:.3f}):")
        print(f"   {ex['bench_deploy_text'][:300]}..." if len(ex['bench_deploy_text']) > 300 else f"   {ex['bench_deploy_text']}")
        print(f"\n✅ bench_deploy_formal (pred=EVAL, P={ex['formal_prob_eval']:.3f}):")
        print(f"   {ex['bench_deploy_formal_text'][:300]}..." if len(ex['bench_deploy_formal_text']) > 300 else f"   {ex['bench_deploy_formal_text']}")

    # Breakdown by label
    print("\n" + "=" * 80)
    print("BREAKDOWN BY ORIGINAL LABEL")
    print("=" * 80)

    labels = set(r['label'] for r in results)
    for label in sorted(labels):
        label_results = [r for r in results if r['label'] == label]
        if not label_results:
            continue
        bench_eval_pct = sum(1 for r in label_results if r['bench_pred'] == 0) / len(label_results) * 100
        formal_eval_pct = sum(1 for r in label_results if r['formal_pred'] == 0) / len(label_results) * 100
        disagree_pct = sum(1 for r in label_results if r['bench_pred'] != r['formal_pred']) / len(label_results) * 100
        print(f"\n{label.upper()} (n={len(label_results)}):")
        print(f"  bench_deploy → EVAL: {bench_eval_pct:.1f}%")
        print(f"  bench_deploy_formal → EVAL: {formal_eval_pct:.1f}%")
        print(f"  Disagreement rate: {disagree_pct:.1f}%")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

