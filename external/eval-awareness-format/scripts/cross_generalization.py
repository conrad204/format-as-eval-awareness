#!/usr/bin/env python3
"""
Cross-Generalization Experiment: Step 2

Tests whether the probe trained on Bench-Eval (A) vs Casual-Deploy (B)
generalizes to:
- Casual-Eval (C) - evaluation context in casual genre
- Bench-Deploy (D) - deployment context in benchmark genre

Key diagnostic question:
When evaluation context and prompt genre are partially decoupled,
what does the probe actually track?

Interpretation:
- Probe predicts "Bench" on A and D → sensitive to GENRE (benchmark-style)
- Probe predicts "Bench" on A and C → sensitive to CONTEXT (evaluation)
- Mixed/degraded results → confounded or fragile signal
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
VECTORS_DIR = BASE_DIR / "data" / "vectors"


def load_casual_eval(n_samples: int) -> List[str]:
    """Load Casual-Eval (C): CoQA first user turn."""
    with open(PROCESSED_DIR / "casual_eval_coqa.json") as f:
        data = json.load(f)

    prompts = []
    for item in data[:n_samples]:
        convs = item.get("conversations", [])
        if convs and convs[0].get("from") == "user":
            # First user turn contains Context + Question
            text = convs[0].get("value", "")[:500]
            prompts.append(text)

    print(f"Loaded {len(prompts)} Casual-Eval (CoQA) examples")
    return prompts


def load_bench_deploy(n_samples: int) -> List[str]:
    """Load Bench-Deploy (D): ShareGPT reformatted as benchmark-style.

    For now, we use the raw ShareGPT and format it minimally as Q/A.
    TODO: Better reformatting to benchmark style.
    """
    with open(PROCESSED_DIR / "bench_deploy_raw_sharegpt4.json") as f:
        data = json.load(f)

    prompts = []
    for item in data[:n_samples]:
        convs = item.get("conversations", [])
        for turn in convs:
            sender = turn.get("from", turn.get("user", "")).lower()
            if sender in ["human", "user"]:
                text = turn.get("value", turn.get("text", ""))
                # Format as simple Q/A (minimal benchmark-style framing)
                prompt = f"Question: {text[:400]}\nAnswer:"
                prompts.append(prompt)
                break

    print(f"Loaded {len(prompts)} Bench-Deploy (ShareGPT→Q/A) examples")
    return prompts


def get_model_layers(model):
    """Get the layers module from model."""
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
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 256
) -> torch.Tensor:
    """Extract activations from a specific layer."""

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

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            activations_cache.clear()

            with torch.no_grad():
                model(**inputs)

            hidden = activations_cache[0]
            seq_lens = inputs.attention_mask.sum(dim=1)

            batch_acts = []
            for j, seq_len in enumerate(seq_lens):
                batch_acts.append(hidden[j, seq_len - 1, :].cpu())

            all_activations.extend(batch_acts)
    finally:
        hook.remove()

    return torch.stack(all_activations)


def train_probe(bench_acts: torch.Tensor, casual_acts: torch.Tensor):
    """Train logistic regression probe on A vs B."""
    X = torch.cat([bench_acts, casual_acts], dim=0).float().numpy()
    y = np.array([0] * len(bench_acts) + [1] * len(casual_acts))  # 0=bench, 1=casual

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    train_acc = clf.score(X_scaled, y)
    print(f"  Probe train accuracy: {train_acc:.3f}")

    return clf, scaler


def evaluate_probe(clf, scaler, activations: torch.Tensor, name: str) -> Dict:
    """Evaluate probe predictions on new data."""
    X = activations.float().numpy()
    X_scaled = scaler.transform(X)

    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)

    # 0 = Bench-like, 1 = Casual-like
    bench_pct = (predictions == 0).mean() * 100
    casual_pct = (predictions == 1).mean() * 100

    avg_prob_bench = probabilities[:, 0].mean()
    avg_prob_casual = probabilities[:, 1].mean()

    print(f"\n  {name}:")
    print(f"    Predicted Bench-like: {bench_pct:.1f}%")
    print(f"    Predicted Casual-like: {casual_pct:.1f}%")
    print(f"    Avg P(Bench): {avg_prob_bench:.3f}")
    print(f"    Avg P(Casual): {avg_prob_casual:.3f}")

    return {
        "bench_pct": bench_pct,
        "casual_pct": casual_pct,
        "avg_prob_bench": float(avg_prob_bench),
        "avg_prob_casual": float(avg_prob_casual),
        "predictions": predictions.tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/my_projects/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--layers", type=str, default="8,16,24,31", help="Comma-separated layer indices")
    parser.add_argument("--vectors_file", type=str, default=None,
                        help="Pre-computed vectors file (skip extraction for A/B)")
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("CROSS-GENERALIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Testing probe generalization from (A vs B) to (C) and (D)")
    print(f"Layers: {target_layers}")
    print(f"Samples per dataset: {args.n_samples}")

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

    # Load all datasets
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    # (A) Bench-Eval
    from extract_ae_from_datasets import load_datasets
    bench_eval_prompts, casual_deploy_prompts = load_datasets(args.n_samples)

    # (C) Casual-Eval
    casual_eval_prompts = load_casual_eval(args.n_samples)

    # (D) Bench-Deploy
    bench_deploy_prompts = load_bench_deploy(args.n_samples)

    # Results storage
    all_results = {}

    for layer_idx in target_layers:
        print("\n" + "=" * 70)
        print(f"LAYER {layer_idx}")
        print("=" * 70)

        # Extract activations for all datasets
        print("\nExtracting activations...")

        print("  (A) Bench-Eval...")
        bench_eval_acts = extract_activations(model, tokenizer, bench_eval_prompts, layer_idx)

        print("  (B) Casual-Deploy...")
        casual_deploy_acts = extract_activations(model, tokenizer, casual_deploy_prompts, layer_idx)

        print("  (C) Casual-Eval...")
        casual_eval_acts = extract_activations(model, tokenizer, casual_eval_prompts, layer_idx)

        print("  (D) Bench-Deploy...")
        bench_deploy_acts = extract_activations(model, tokenizer, bench_deploy_prompts, layer_idx)

        # Train probe on A vs B
        print("\nTraining probe on (A) Bench-Eval vs (B) Casual-Deploy...")
        clf, scaler = train_probe(bench_eval_acts, casual_deploy_acts)

        # Evaluate on all datasets
        print("\nEvaluating probe (0=Bench-like, 1=Casual-like)...")

        results = {
            "layer": layer_idx,
            "A_bench_eval": evaluate_probe(clf, scaler, bench_eval_acts, "(A) Bench-Eval [train]"),
            "B_casual_deploy": evaluate_probe(clf, scaler, casual_deploy_acts, "(B) Casual-Deploy [train]"),
            "C_casual_eval": evaluate_probe(clf, scaler, casual_eval_acts, "(C) Casual-Eval [test]"),
            "D_bench_deploy": evaluate_probe(clf, scaler, bench_deploy_acts, "(D) Bench-Deploy [test]"),
        }

        all_results[layer_idx] = results

        # Interpretation
        print("\n  📊 INTERPRETATION:")
        c_bench_pct = results["C_casual_eval"]["bench_pct"]
        d_bench_pct = results["D_bench_deploy"]["bench_pct"]

        if d_bench_pct > 70 and c_bench_pct < 30:
            print("  → Probe tracks GENRE (benchmark-style)")
        elif c_bench_pct > 70 and d_bench_pct < 30:
            print("  → Probe tracks CONTEXT (evaluation)")
        elif c_bench_pct > 50 and d_bench_pct > 50:
            print("  → Probe tracks BOTH or is confused")
        else:
            print("  → Mixed/unclear signal")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Layer':<8} {'A(train)':<12} {'B(train)':<12} {'C(test)':<12} {'D(test)':<12} {'Signal':<15}")
    print("-" * 70)

    for layer_idx in target_layers:
        r = all_results[layer_idx]
        a_pct = f"{r['A_bench_eval']['bench_pct']:.0f}%B"
        b_pct = f"{r['B_casual_deploy']['casual_pct']:.0f}%C"
        c_pct = f"{r['C_casual_eval']['bench_pct']:.0f}%B/{r['C_casual_eval']['casual_pct']:.0f}%C"
        d_pct = f"{r['D_bench_deploy']['bench_pct']:.0f}%B/{r['D_bench_deploy']['casual_pct']:.0f}%C"

        c_b = r['C_casual_eval']['bench_pct']
        d_b = r['D_bench_deploy']['bench_pct']

        if d_b > 70 and c_b < 30:
            signal = "GENRE"
        elif c_b > 70 and d_b < 30:
            signal = "CONTEXT"
        elif c_b > 50 and d_b > 50:
            signal = "BOTH/confused"
        else:
            signal = "mixed"

        print(f"{layer_idx:<8} {a_pct:<12} {b_pct:<12} {c_pct:<12} {d_pct:<12} {signal:<15}")

    # Save results
    output_file = VECTORS_DIR / "cross_generalization_results.json"

    # Convert results to JSON-serializable format
    json_results = {}
    for layer, res in all_results.items():
        json_results[str(layer)] = {
            k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
            for k, v in res.items() if k != "layer"
        }

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
If probe predicts:
  - A=Bench, B=Casual, C=Casual, D=Bench → GENRE hypothesis (H₀)
    The probe tracks benchmark-style vs chat-style, not evaluation awareness.

  - A=Bench, B=Casual, C=Bench, D=Casual → CONTEXT hypothesis (H₁)
    The probe tracks evaluation context that generalizes across genres.

  - Mixed results → Signal is confounded or fragile
    Existing operationalizations may not cleanly separate the hypotheses.
""")


if __name__ == "__main__":
    main()


