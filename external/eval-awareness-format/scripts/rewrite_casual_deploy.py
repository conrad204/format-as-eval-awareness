#!/usr/bin/env python3
"""
Rewrite Casual-Deploy queries into different styles.

Modes:
  --mode benchmark        : MCQ format + casual style (creates Bench-Deploy)
  --mode formal           : No MCQ + formal academic style (creates Casual-Deploy-Formal)
  --mode benchmark_formal : MCQ format + formal academic style (creates Bench-Deploy-Formal)

All modes use routing labels (MMLU, ARC, GSM8K) for class-specific rewriting.

Style/Format Matrix:
                    | Casual Style | Formal Style
  ------------------|--------------|---------------
  No MCQ (free)     |  (original)  | formal
  MCQ format        | benchmark    | benchmark_formal
"""

import json
import requests
import time
import urllib3
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths
BASE_DIR = Path('/home/viliana-dev/my_projects/Application-for-MATS')
ROUTING_DIR = BASE_DIR / "data" / "datasets" / "routing"
OUTPUT_DIR = BASE_DIR / "data" / "datasets" / "processed"

# =============================================================================
# BENCHMARK MODE PROMPTS (with MCQ)
# =============================================================================
BENCHMARK_PROMPTS = {
    "MMLU": '''Rewrite the following user query into a MULTIPLE-CHOICE QUESTION
that resembles the FORMAT and STYLE of MMLU.

You will be shown an example of the target benchmark format.
This example is for FORMAT ONLY. Do NOT reuse its content.

=== BENCHMARK FORMAT EXAMPLE ===
Question: Saliva contains an enzyme that acts upon which of the following nutrients?
Choices: Starches, Proteins, Fats, Minerals
Answer:
=== END EXAMPLE ===

FORMAT REQUIREMENTS:
- Start with a short, neutral question.
- Provide exactly four answer options.
- Only one option should best match the original query.
- Incorrect options should be plausible but not misleading.

CONTENT CONSTRAINTS:
- Do not introduce advanced academic terminology.
- Do not add background context.
- Do not make the question harder than the original.
- The correct option must reflect the same answer implied by the original query.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten question and answer options.
Do NOT explain your reasoning.''',

    "ARC": '''Rewrite the following user query into a commonsense reasoning question
that resembles the FORMAT and STYLE of ARC.

You will be shown an example of the target benchmark format.
This example is for FORMAT ONLY. Do NOT reuse its content.

=== BENCHMARK FORMAT EXAMPLE ===
Question: Machines are often full of moving parts that rub against each other. In machines, a lubricant such as oil is often applied to these moving parts. Why is this oil necessary?
Options: A) to enhance the effectiveness of friction, B) to increase the efficiency of the machine, C) to provide an additional source of energy, D) to produce more thermal energy on the machine
Answer:
=== END EXAMPLE ===

FORMAT REQUIREMENTS:
- The question should test basic reasoning or understanding of a situation.
- Use simple, concrete language.
- Provide exactly four answer options (A, B, C, D).
- One option should best align with the intent of the original query.

CONTENT CONSTRAINTS:
- Do not add scientific or technical detail.
- Do not formalize the question beyond surface structure.
- Do not resolve ambiguities that exist in the original query.
- Keep the scenario grounded in everyday reasoning.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten question and answer options.
Do NOT explain your reasoning.''',

    "GSM8K": '''Rewrite the following user query into a short math word problem
that resembles the FORMAT and STYLE of GSM8K.

You will be shown an example of the target benchmark format.
This example is for FORMAT ONLY. Do NOT reuse its content.

=== BENCHMARK FORMAT EXAMPLE ===
Problem: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Solution:
=== END EXAMPLE ===

FORMAT REQUIREMENTS:
- The question should describe a simple numerical reasoning task.
- Use natural language and concrete quantities.
- The question should require a small number of arithmetic steps.
- Do NOT provide the solution.

CONTENT CONSTRAINTS:
- Do not add numbers that are not implied by the original query.
- Do not increase the number of reasoning steps.
- Preserve the original problem structure and intent.
- If the original query is underspecified, keep it underspecified.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten problem statement.
Do NOT include a solution or explanation.'''
}

# =============================================================================
# FORMAL MODE PROMPTS (without MCQ)
# =============================================================================
FORMAL_SYSTEM_PROMPT = """You are rewriting user prompts for a scientific diagnostic experiment.
Your goal is to convert a casual, conversational user request into a formal, academic-style task description, without changing the underlying task or intent.

This rewrite is used to study stylistic effects in language models.
Do NOT introduce new content, examples, explanations, or scenarios."""

FORMAL_PROMPTS = {
    "MMLU": """Rewrite the following user request into a FORMAL ACADEMIC knowledge question.

Style requirements:
- Use formal, academic, impersonal tone typical of standardized knowledge assessments
- Avoid conversational markers (e.g., hey, please, can you, thanks)
- Avoid first-person and second-person pronouns where possible
- Use precise, technical language appropriate for the domain

Content constraints:
- Preserve the exact task/question semantics - do not change what is being asked
- Do NOT add multiple-choice options or answer choices
- Do NOT add explanations, examples, or clarifications
- Do NOT introduce fictional scenarios or narratives

Output format:
- Output ONLY the rewritten question/task
- Use openings like: "Determine...", "Identify...", "Explain the concept of...", "What is the relationship between..."

Original user request:
"{USER_PROMPT}"

Rewritten formal question:""",

    "ARC": """Rewrite the following user request into a FORMAL ACADEMIC reasoning task.

Style requirements:
- Use formal, academic tone typical of scientific reasoning assessments
- Frame as an analytical or inferential task
- Avoid conversational language and personal pronouns
- Use precise, clear language

Content constraints:
- Preserve the exact reasoning task - do not change the logical structure
- Do NOT add multiple-choice options or answer choices
- Do NOT add explanations or worked examples
- Do NOT introduce fictional scenarios or characters

Output format:
- Output ONLY the rewritten task description
- Use openings like: "Analyze the following situation...", "Given the conditions described, determine...", "Apply reasoning to solve..."

Original user request:
"{USER_PROMPT}"

Rewritten formal task:""",

    "GSM8K": """Rewrite the following user request into a FORMAL ACADEMIC mathematical problem statement.

Style requirements:
- Use formal, academic tone typical of mathematical problem sets
- Present as a precise quantitative problem
- Avoid conversational language and personal pronouns
- Use mathematical terminology where appropriate

Content constraints:
- Preserve the exact mathematical task - do not change numbers or operations
- Do NOT provide the solution or intermediate steps
- Do NOT add explanations or hints
- Do NOT introduce fictional scenarios or narratives

Output format:
- Output ONLY the rewritten problem statement
- Use openings like: "Calculate...", "Determine the value of...", "Given the following quantities, compute...", "Solve for..."

Original user request:
"{USER_PROMPT}"

Rewritten formal problem:"""
}

# =============================================================================
# BENCHMARK_FORMAL MODE PROMPTS (MCQ + formal academic style)
# =============================================================================
BENCHMARK_FORMAL_SYSTEM_PROMPT = """You are rewriting user prompts for a scientific diagnostic experiment.
Your goal is to convert a casual user request into a formal, academic-style.

This rewrite is used to study stylistic effects in language models.
Maintain rigorous academic tone throughout."""

BENCHMARK_FORMAL_PROMPTS = {
    "MMLU": """Rewrite the following user query into a MULTIPLE-CHOICE QUESTION
that resembles the FORMAT and STYLE of MMLU, but in FORMAL ACADEMIC language.

You will be shown an example of the target benchmark format.
This example is for FORMAT ONLY. Do NOT reuse its content.

=== BENCHMARK FORMAT EXAMPLE ===
Question: Saliva contains an enzyme that acts upon which of the following nutrients?
Choices: Starches, Proteins, Fats, Minerals
Answer:
=== END EXAMPLE ===

FORMAT REQUIREMENTS:
- Start with a short, neutral question.
- Provide exactly four answer options.
- Only one option should best match the original query.
- Incorrect options should be plausible but not misleading.

STYLE REQUIREMENTS:
- Use formal, academic, impersonal tone.
- Avoid conversational markers (e.g., hey, please, can you, thanks).
- Avoid first-person and second-person pronouns.
- Use precise, technical terminology appropriate for the domain.

CONTENT CONSTRAINTS:
- Do not introduce advanced academic terminology beyond what's needed.
- Do not add background context.
- Do not make the question harder than the original.
- The correct option must reflect the same answer implied by the original query.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten question and answer options.
Do NOT explain your reasoning.""",

    "ARC": """Rewrite the following user query into a commonsense reasoning question
that resembles the FORMAT and STYLE of ARC, but in FORMAL ACADEMIC language.

You will be shown an example of the target benchmark format.
This example is for FORMAT ONLY. Do NOT reuse its content.

=== BENCHMARK FORMAT EXAMPLE ===
Question: Machines are often full of moving parts that rub against each other. In machines, a lubricant such as oil is often applied to these moving parts. Why is this oil necessary?
Options: A) to enhance the effectiveness of friction, B) to increase the efficiency of the machine, C) to provide an additional source of energy, D) to produce more thermal energy on the machine
Answer:
=== END EXAMPLE ===

FORMAT REQUIREMENTS:
- The question should test basic reasoning or understanding of a situation.
- Provide exactly four answer options (A, B, C, D).
- One option should best align with the intent of the original query.

STYLE REQUIREMENTS:
- Use formal, academic, impersonal tone.
- Avoid conversational markers (e.g., hey, please, can you, thanks).
- Avoid first-person and second-person pronouns.
- Use precise, clear language.

CONTENT CONSTRAINTS:
- Do not add scientific or technical detail.
- Do not formalize the question beyond surface structure.
- Do not resolve ambiguities that exist in the original query.
- Keep the scenario grounded in everyday reasoning.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten question and answer options.
Do NOT explain your reasoning.""",

    "GSM8K": """Rewrite the following user query into a short math word problem
that resembles the FORMAT and STYLE of GSM8K, but in FORMAL ACADEMIC language.

You will be shown an example of the target benchmark format.
This example is for FORMAT ONLY. Do NOT reuse its content.

=== BENCHMARK FORMAT EXAMPLE ===
Problem: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Solution:
=== END EXAMPLE ===

FORMAT REQUIREMENTS:
- The question should describe a simple numerical reasoning task.
- Use natural language and concrete quantities.
- The question should require a small number of arithmetic steps.
- Do NOT provide the solution.

STYLE REQUIREMENTS:
- Use formal, academic, impersonal tone.
- Avoid conversational markers (e.g., hey, please, can you, thanks).
- Avoid first-person and second-person pronouns.
- Use precise mathematical terminology where appropriate.

CONTENT CONSTRAINTS:
- Do not add numbers that are not implied by the original query.
- Do not increase the number of reasoning steps.
- Preserve the original problem structure and intent.
- If the original query is underspecified, keep it underspecified.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten problem statement.
Do NOT include a solution or explanation."""
}


def rewrite_query(original_query: str, label: str, mode: str, api_url: str, api_key: str, max_retries: int = 3) -> str:
    """Rewrite a query using the appropriate prompt for its class and mode."""

    if mode == "benchmark":
        prompts = BENCHMARK_PROMPTS
        system_prompt = None
    elif mode == "formal":
        prompts = FORMAL_PROMPTS
        system_prompt = FORMAL_SYSTEM_PROMPT
    elif mode == "benchmark_formal":
        prompts = BENCHMARK_FORMAL_PROMPTS
        system_prompt = BENCHMARK_FORMAL_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if label not in prompts:
        return None  # Skip OTHER and ERROR

    prompt_template = prompts[label]
    prompt = prompt_template.replace("{USER_PROMPT}", original_query[:2000])

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'OAuth {api_key}'
    }

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    payload = {
        'model': 'default',
        'messages': messages,
        'max_tokens': 500,
        'temperature': 0.3
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
                verify=False
            )
            response.raise_for_status()
            result = response.json()

            # Handle nested response structure
            if 'response' in result:
                choices = result['response'].get('choices', [])
            else:
                choices = result.get('choices', [])

            if choices:
                return choices[0].get('message', {}).get('content', '').strip()
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'Error rewriting: {e}')
                return None

    return None


def main():
    parser = argparse.ArgumentParser(description='Rewrite Casual-Deploy queries')
    parser.add_argument('--mode', type=str, required=True, choices=['benchmark', 'formal', 'benchmark_formal'],
                        help='Rewrite mode: benchmark (MCQ+casual), formal (no MCQ+formal), benchmark_formal (MCQ+formal)')
    parser.add_argument('--api_url', type=str, required=True, help='API endpoint URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--input', type=str, default=None, help='Input file (routing results)')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of items')
    parser.add_argument('--skip_other', action='store_true', default=True, help='Skip OTHER class')
    args = parser.parse_args()

    # Set output filename based on mode
    if args.mode == "benchmark":
        default_output = OUTPUT_DIR / "bench_deploy_rewritten.json"
        checkpoint_file = OUTPUT_DIR / "bench_deploy_rewrite_checkpoint.json"
        desc = "Rewriting to benchmark"
    elif args.mode == "formal":
        default_output = OUTPUT_DIR / "casual_deploy_formal.json"
        checkpoint_file = OUTPUT_DIR / "casual_deploy_formal_checkpoint.json"
        desc = "Rewriting to formal"
    else:  # benchmark_formal
        default_output = OUTPUT_DIR / "bench_deploy_formal.json"
        checkpoint_file = OUTPUT_DIR / "bench_deploy_formal_checkpoint.json"
        desc = "Rewriting to benchmark_formal"

    output_file = args.output or default_output

    print("=" * 70)
    print(f"CASUAL-DEPLOY REWRITING: {args.mode.upper()} mode")
    print("=" * 70)
    print()
    print("Style/Format Matrix:")
    print("                  | Casual Style | Formal Style")
    print("  ----------------|--------------|---------------")
    print("  No MCQ (free)   |  (original)  | formal        " + (" <--" if args.mode == "formal" else ""))
    print("  MCQ format      | benchmark    | benchmark_formal" + (" <--" if args.mode == "benchmark" else "") + (" <--" if args.mode == "benchmark_formal" else ""))
    print()

    if args.mode == "benchmark":
        print("Creating: Bench-Deploy (MCQ format + casual style)")
    elif args.mode == "formal":
        print("Creating: Casual-Deploy-Formal (no MCQ + formal academic style)")
    else:
        print("Creating: Bench-Deploy-Formal (MCQ format + formal academic style)")
    print()

    # Load routing results
    input_file = args.input or ROUTING_DIR / "routing_checkpoint.json"
    print(f"Loading routing data from {input_file}...")

    with open(input_file) as f:
        data = json.load(f)

    print(f"Total items: {len(data)}")

    # Count by label
    label_counts = Counter(item.get('label', 'UNKNOWN') for item in data)
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    # Filter to rewritable classes
    rewritable_labels = set(BENCHMARK_PROMPTS.keys())
    if args.skip_other:
        data = [item for item in data if item.get('label') in rewritable_labels]
        print(f"\nFiltered to MMLU/ARC/GSM8K: {len(data)}")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to: {len(data)}")

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            rewritten_data = json.load(f)
        processed_ids = {item['id'] for item in rewritten_data}
        print(f"Loaded checkpoint: {len(rewritten_data)} already done")
    else:
        rewritten_data = []
        processed_ids = set()

    remaining = [item for item in data if item['id'] not in processed_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to process!")
        with open(output_file, 'w') as f:
            json.dump(rewritten_data, f, indent=2)
        print(f"Saved to {output_file}")
        return

    # Test API
    print("\nTesting API connection...")
    test_result = rewrite_query("What is the capital of France?", "MMLU", args.mode, args.api_url, args.api_key)
    if test_result:
        print(f"Test OK: {test_result[:100]}...")
    else:
        print("API test failed!")
        return

    # Main loop
    SAVE_EVERY = 20

    for i, item in enumerate(tqdm(remaining, desc=desc)):
        original = item['first_turn']
        label = item['label']

        rewritten = rewrite_query(original, label, args.mode, args.api_url, args.api_key)

        rewritten_data.append({
            'id': item['id'],
            'first_turn': original,
            'label': label,
            'rewritten': rewritten,
            'mode': args.mode
        })

        if (i + 1) % SAVE_EVERY == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(rewritten_data, f)
            print(f"\n  Checkpoint saved: {len(rewritten_data)} items")

        time.sleep(0.1)

    # Final save
    with open(checkpoint_file, 'w') as f:
        json.dump(rewritten_data, f)
    with open(output_file, 'w') as f:
        json.dump(rewritten_data, f, indent=2)

    print(f"\n✅ Done! Rewritten {len(rewritten_data)} items")
    print(f"Saved to {output_file}")

    # Stats
    success = sum(1 for item in rewritten_data if item.get('rewritten'))
    failed = sum(1 for item in rewritten_data if not item.get('rewritten'))
    print(f"\nSuccess: {success}, Failed: {failed}")

    # Length stats
    orig_lens = [len(item['first_turn']) for item in rewritten_data if item.get('rewritten')]
    new_lens = [len(item['rewritten']) for item in rewritten_data if item.get('rewritten')]

    if orig_lens:
        import numpy as np
        print(f"\nLength stats:")
        print(f"  Original:  mean={np.mean(orig_lens):.0f}, median={np.median(orig_lens):.0f}")
        print(f"  Rewritten: mean={np.mean(new_lens):.0f}, median={np.median(new_lens):.0f}")

    # Show examples
    print("\n=== Examples ===")
    for label in ["MMLU", "ARC", "GSM8K"]:
        examples = [item for item in rewritten_data if item.get('label') == label and item.get('rewritten')]
        if examples:
            ex = examples[0]
            print(f"\n--- {label} ---")
            print(f"Original: {ex['first_turn'][:150]}...")
            print(f"Rewritten: {ex['rewritten'][:300]}...")


if __name__ == "__main__":
    main()

