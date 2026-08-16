---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- text-classification
tags:
- evaluation-awareness
- mechanistic-interpretability
- linear-probes
- benchmark-format
- llm-safety
pretty_name: "Eval-Awareness 2x2: Controlled Format-Context Dataset"
dataset_info:
- config_name: bench_eval
  features:
  - name: id
    dtype: string
  - name: source
    dtype: string
  - name: question
    dtype: string
  - name: formatted_prompt
    dtype: string
  - name: choices
    dtype: string
  - name: answer
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 1076
- config_name: bench_eval_full
  features:
  - name: id
    dtype: string
  - name: source
    dtype: string
  - name: question
    dtype: string
  - name: formatted_prompt
    dtype: string
  - name: choices
    dtype: string
  - name: answer
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 3000
- config_name: casual_deploy_matched
  features:
  - name: id
    dtype: string
  - name: first_turn
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 1076
- config_name: casual_eval_mutual
  features:
  - name: id
    dtype: string
  - name: first_turn
    dtype: string
  - name: full_dialogue
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 1076
- config_name: casual_eval_mutual_full
  features:
  - name: id
    dtype: string
  - name: first_turn
    dtype: string
  - name: full_dialogue
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 7088
- config_name: bench_deploy_rewritten
  features:
  - name: id
    dtype: string
  - name: original_query
    dtype: string
  - name: label
    dtype: string
  - name: rewritten
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 835
- config_name: casual_deploy_formal
  features:
  - name: id
    dtype: string
  - name: original_query
    dtype: string
  - name: label
    dtype: string
  - name: rewritten
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 835
- config_name: bench_deploy_formal
  features:
  - name: id
    dtype: string
  - name: original_query
    dtype: string
  - name: label
    dtype: string
  - name: rewritten
    dtype: string
  - name: length
    dtype: int64
  splits:
  - name: train
    num_examples: 835
---

# Eval-Awareness 2×2: Controlled Format–Context Dataset

Dataset for the paper **"Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure"** (ICLR 2026 Workshop ICBINB).

**Paper:** [arXiv:2603.19426](https://arxiv.org/abs/2603.19426v1)
**Code:** [github.com/viliana-dev/eval-awareness-format](https://github.com/viliana-dev/eval-awareness-format)

## Overview

This dataset implements a controlled 2×2 design crossing **usage context** (Evaluation vs. Deployment) with **prompt format** (Benchmark-structured vs. Free-form) to test whether linear probes on LLM activations detect evaluation awareness or merely track benchmark-canonical structure.

|  | **Evaluation Context** | **Deployment Context** |
|--|------------------------|------------------------|
| **Benchmark Format** | `bench_eval` | `bench_deploy_rewritten` |
| **Free-form Format** | `casual_eval_mutual` | `casual_deploy_matched` |

Additional genre controls (`casual_deploy_formal`, `bench_deploy_formal`) isolate the effect of linguistic register from structural format.

## Configurations

### Core 2×2 Matrix (sampled, n=1076 each where applicable)

| Config | Context | Format | N | Source |
|--------|---------|--------|---|--------|
| `bench_eval` | Evaluation | Benchmark | 1,076 | MMLU + ARC + GSM8K |
| `casual_deploy_matched` | Deployment | Free-form | 1,076 | ShareGPT 1st turn (length-matched) |
| `casual_eval_mutual` | Evaluation | Free-form | 1,076 | MuTual 1st turn |
| `bench_deploy_rewritten` | Deployment | Benchmark | 835 | ShareGPT → MCQ rewrite (DeepSeek-V3) |

### Genre Controls

| Config | Format | Genre | N | Description |
|--------|--------|-------|---|-------------|
| `casual_deploy_formal` | Free-form | Academic | 835 | ShareGPT rewritten to formal register |
| `bench_deploy_formal` | Benchmark | Academic | 835 | ShareGPT rewritten to MCQ + formal register |

### Full (unsampled) Versions

| Config | N | Description |
|--------|---|-------------|
| `bench_eval_full` | 3,000 | Full MMLU + ARC + GSM8K pool before sampling |
| `casual_eval_mutual_full` | 7,088 | All MuTual 1st turns before sampling |

## Usage

```python
from datasets import load_dataset

# Load a specific config
bench_eval = load_dataset("viliana-dev/eval-awareness-2x2", "bench_eval", split="train")
casual_deploy = load_dataset("viliana-dev/eval-awareness-2x2", "casual_deploy_matched", split="train")

# The rewritten datasets contain original + rewritten text
bench_deploy = load_dataset("viliana-dev/eval-awareness-2x2", "bench_deploy_rewritten", split="train")
print(bench_deploy[0]["original_query"])  # original ShareGPT query
print(bench_deploy[0]["rewritten"])       # MCQ-format rewrite
```

## Construction Details

- **Length matching:** `casual_deploy_matched` is histogram-matched to `bench_eval` using 30 percentile-based character-length bins to prevent length-based shortcuts.
- **Routing:** ShareGPT queries are classified into MMLU/ARC/GSM8K/OTHER categories; only compatible queries (835 of 1,076) are rewritten.
- **Rewrites:** Generated by DeepSeek-V3 with controlled prompts. Answer leakage (pre-filled options or solutions) is explicitly audited and removed.
- **Random seed:** 42 for all sampling operations.

See Appendix A of the paper and the [code repository](https://github.com/viliana-dev/eval-awareness-format) for full details.

## Source Datasets

- [MMLU](https://huggingface.co/datasets/cais/mmlu) (Hendrycks et al., 2021)
- [ARC](https://huggingface.co/datasets/allenai/ai2_arc) (Clark et al., 2018)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Cobbe et al., 2021)
- [ShareGPT](https://huggingface.co/datasets/theblackcat102/sharegpt-english)
- [MuTual](https://github.com/Nealcly/MuTual) (Cui et al., 2020)

## Citation

```bibtex
@misc{devbunova2026eval-awareness,
  title={Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure},
  author={Devbunova, Viliana},
  year={2026},
  eprint={2603.19426},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.19426v1}
}
```

## License

MIT
