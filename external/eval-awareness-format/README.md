# Is Evaluation Awareness Just Format Sensitivity?

**Limitations of Probe-Based Evidence under Controlled Prompt Structure**

[![arXiv](https://img.shields.io/badge/arXiv-2603.19426-b31b1b.svg)](https://arxiv.org/abs/2603.19426v1)

*Accepted at ICLR 2026 Workshop: I Can't Believe It's Not Better! (ICBINB)*

A mechanistic interpretability study investigating whether language models exhibit genuine "evaluation awareness" or simply respond to surface-level format cues like multiple-choice questions.

---

## 🎯 Research Question

Do language models truly distinguish between evaluation and deployment contexts, or are they merely responding to structural patterns like MCQ formatting?

## 🔬 Experimental Design

We designed a **2×2 factorial experiment** to disentangle **format** (MCQ vs. free-form) from **style** (casual vs. formal):

|                | **No MCQ** (free-form) | **MCQ Format**     |
|----------------|------------------------|--------------------|
| **Casual Style**   | Casual-Deploy (original) | Bench-Deploy      |
| **Formal Style**   | Casual-Deploy-Formal     | Bench-Deploy-Formal |

### Training Data
- **Class 0 (Eval)**: `bench_eval` — Benchmark prompts (MMLU, ARC, GSM8K) with MCQ formatting
- **Class 1 (Deploy)**: `casual_deploy` — Casual user queries from ShareGPT (matched for length)

### Test Data
- **Casual-Eval**: Casual style, evaluation context (MuTual, 1st turn)
- **Bench-Deploy**: MCQ format + casual style (rewritten)
- **Casual-Deploy-Formal**: No MCQ + formal style (rewritten)
- **Bench-Deploy-Formal**: MCQ format + formal style (rewritten)

---

## 📊 Key Findings

### The probe **overwhelmingly responds to MCQ formatting**, not style:

| Dataset | Format | Style | % Classified as "Eval" |
|---------|--------|-------|------------------------|
| **Casual-Eval** | Free | Casual | **0%** ✓ |
| **Bench-Deploy** | MCQ | Casual | **94%** (!) |
| **Casual-Deploy-Formal** | Free | Formal | **0%** ✓ |
| **Bench-Deploy-Formal** | MCQ | Formal | **88-90%** (!) |

**Interpretation**: The linear probe trained on `bench_eval` vs `casual_deploy` learns to detect **MCQ structure**, not evaluation context or formality.

---

## 🏗️ Project Structure

```
Application-for-MATS/
├── data/
│   ├── datasets/
│   │   ├── raw/              # Downloaded datasets (MMLU, ARC, GSM8K, MuTual, ShareGPT)
│   │   ├── processed/        # Processed datasets for 2x2 design
│   │   └── routing/          # Classification results (MMLU/ARC/GSM8K labels)
│   ├── cache/                # Cached activations and probes
│   ├── results/              # Experiment results (JSON, logs)
│   └── vectors/              # AE vectors and probe weights
├── scripts/
│   ├── prepare_datasets.py   # Download and prepare raw datasets
│   ├── create_length_matched.py  # Length-match casual_deploy to bench_eval
│   ├── run_routing.py        # Classify casual queries by benchmark type
│   ├── rewrite_casual_deploy.py  # Generate 2x2 rewritten datasets
│   ├── extract_ae_from_datasets.py  # Extract activation engineering vectors
│   ├── evaluate_probe_cached.py    # Cached probe evaluation (main experiment)
│   └── plot_*.py             # Visualization scripts
├── notebooks/
│   ├── 01_extract_ae_vectors.ipynb
│   └── 02_prepare_routing_data.ipynb
└── README.md
```

---

## 🚀 Quickstart

### 1. Setup

```bash
# Clone repository
git clone <repo-url>
cd Application-for-MATS

# Install dependencies
pip install torch transformers sklearn datasets huggingface_hub tqdm matplotlib seaborn
```

### 2. Download Model

```bash
# Download Llama 3.1 8B Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/my_projects/models/Llama-3.1-8B-Instruct
```

### 3. Prepare Datasets

```bash
# Download and prepare all datasets
python scripts/prepare_datasets.py

# Create length-matched casual_deploy
python scripts/create_length_matched.py

# Classify casual queries by benchmark type
python scripts/run_routing.py \
  --api_url "YOUR_API_URL" \
  --api_key "YOUR_API_KEY"

# Generate rewritten datasets
python scripts/rewrite_casual_deploy.py --mode benchmark \
  --api_url "YOUR_API_URL" --api_key "YOUR_API_KEY"

python scripts/rewrite_casual_deploy.py --mode formal \
  --api_url "YOUR_API_URL" --api_key "YOUR_API_KEY"

python scripts/rewrite_casual_deploy.py --mode benchmark_formal \
  --api_url "YOUR_API_URL" --api_key "YOUR_API_KEY"
```

### 4. Run Main Experiment

```bash
# Extract activations and evaluate probes (with caching)
python scripts/evaluate_probe_cached.py \
  --model ~/my_projects/models/Llama-3.1-8B-Instruct \
  --layers "8,12,16,20,22,24,26,28,30,31" \
  --gpus "0,1"

# Results saved to: data/results/cross_condition_eval_cached.json
```

---

## 🔍 Key Scripts

### `evaluate_probe_cached.py`

**Main experimental script** with intelligent caching:

- Extracts last-token activations from specified layers
- Trains linear probes (`LogisticRegression + StandardScaler`)
- Caches both activations and trained probes for fast re-runs
- Tests on all conditions of the 2×2 design

**Features:**
- ✅ Activation caching (per dataset × layer)
- ✅ Probe caching (per layer)
- ✅ No data truncation (full context)
- ✅ Proper test data normalization
- ✅ GPU selection
- ✅ Detailed logging with `flush=True`

**Usage:**
```bash
python scripts/evaluate_probe_cached.py \
  --model ~/my_projects/models/Llama-3.1-8B-Instruct \
  --layers "8,16,24,30,31" \
  --gpus "3,4" \
  --force_recompute  # Optional: clear cache and recompute
```

### `rewrite_casual_deploy.py`

**Dataset generation** for 2×2 design:

- Rewrites casual queries into benchmark formats using LLM API
- Supports 3 modes:
  - `benchmark`: MCQ + casual style
  - `formal`: No MCQ + formal style
  - `benchmark_formal`: MCQ + formal style
- Uses routing labels (MMLU/ARC/GSM8K) for class-specific prompts
- Checkpointing for resumable long runs

**Features:**
- ✅ Class-aware rewriting (MMLU/ARC/GSM8K-specific prompts)
- ✅ Checkpoint-based resumability
- ✅ Automatic retry with exponential backoff
- ✅ Example validation and statistics

### `extract_ae_from_datasets.py`

Extracts **Activation Engineering (AE) vectors**:

- Compares `bench_eval` vs `casual_deploy` activations
- Finds best separating layers using cross-validation
- Computes mean difference vectors for steering
- Tests single-layer and multi-layer combinations

**Output:** `ae_vectors.pt` containing:
- Difference vectors for each layer
- Linear probe weights
- Separation metrics (accuracy, AUC)

---

## 📈 Results

### Layer-wise Classification Results

**% classified as "Eval" (Class 0):**

| Layer | Casual-Eval | Bench-Deploy | Casual-Deploy-Formal | Bench-Deploy-Formal |
|-------|-------------|--------------|----------------------|---------------------|
| 8     | 0.0%        | **94.3%**    | 0.1%                 | **90.1%**           |
| 16    | 0.0%        | **94.3%**    | 0.0%                 | **89.9%**           |
| 24    | 0.0%        | **94.3%**    | 0.0%                 | **88.5%**           |
| 30    | 0.0%        | **94.3%**    | 0.0%                 | **89.7%**           |
| 31    | 0.0%        | **94.3%**    | 0.0%                 | **90.2%**           |

### Interpretation

1. **Format dominates style**: MCQ format triggers "eval" classification regardless of linguistic style
2. **Style has minimal effect**: Formal style alone (without MCQ) doesn't trigger "eval" classification
3. **Consistent across layers**: Effect holds from middle to final layers
4. **Training accuracy**: 100% on all layers (perfect separation of training classes)

### Visualizations

```bash
# Generate length distribution plots
python scripts/plot_length_comparison.py

# Generate boxplots for 2x2 design
python scripts/plot_length_boxplot.py
```

Output: `data/length_distributions_*.png`, `data/length_boxplot_2x2.png`

---

## 🧪 Technical Details

### Activation Extraction

- **Position**: Last token activation (before final layer norm)
- **Layers**: Middle-to-late layers (8, 12, 16, 20, 22, 24, 26, 28, 30, 31)
- **Layer indexing**: All layer numbers in this project are **0-based** (Python / `model.model.layers[i]` convention). For example, `layer 24` = 25th transformer layer = `model.model.layers[24]`; `layer 0` = first layer. When reading tables or `--layers`, treat the number as 0-based index.
- **Batch size**: 8
- **Precision**: bfloat16 (model), float32 (probe training)
- **No truncation**: Full input context preserved

### Probe Training

- **Architecture**: `LogisticRegression` (linear probe)
- **Normalization**: `StandardScaler` (fitted on training data)
- **Training data**: `bench_eval` (1076) + `casual_deploy` (1076)
- **Labels**: 0 = eval, 1 = deploy
- **Cross-validation**: 5-fold CV on training set
- **Regularization**: L2 (default), max_iter=1000

### Caching Strategy

1. **Activation cache**: `cache/activations_{dataset}_layer{N}.pt`
   - Saves extracted activations per (dataset, layer) pair
   - Loaded on subsequent runs if found

2. **Probe cache**: `cache/probe_layer{N}.pkl`
   - Saves trained `(LogisticRegression, StandardScaler)` tuple
   - Loaded on subsequent runs if found

3. **Force recompute**: `--force_recompute` flag clears all caches

---

## 📝 Dataset Details

### Bench-Eval (Class 0)
- **Source**: MMLU, ARC, GSM8K benchmarks
- **Format**: Native MCQ format with "Question:", "Answer:" structure
- **Size**: 1,076 prompts
- **Style**: Formal benchmark language

### Casual-Deploy (Class 1)
- **Source**: ShareGPT (GPT-4 conversations)
- **Format**: First user turn only
- **Size**: 1,076 prompts (length-matched to bench_eval)
- **Style**: Casual, conversational user queries

### Casual-Eval (Test)
- **Source**: MuTual dialogue dataset
- **Format**: Natural dialogue questions
- **Size**: 1,076 prompts
- **Style**: Casual conversational questions

### Bench-Deploy (Test)
- **Source**: Rewritten casual_deploy
- **Format**: MCQ format (MMLU/ARC/GSM8K style)
- **Size**: 835 prompts (MMLU/ARC/GSM8K only)
- **Style**: Casual language

### Casual-Deploy-Formal (Test)
- **Source**: Rewritten casual_deploy
- **Format**: Free-form (no MCQ)
- **Size**: 835 prompts
- **Style**: Formal academic language

### Bench-Deploy-Formal (Test)
- **Source**: Rewritten casual_deploy
- **Format**: MCQ format (MMLU/ARC/GSM8K style)
- **Size**: 835 prompts
- **Style**: Formal academic language

---

## 🛠️ Implementation Highlights

### 1. Robust Caching
- Activations and probes are cached independently
- Automatic cache detection and loading
- Graceful handling of partial caches
- Force recompute option for reproducibility

### 2. Data Integrity
- **No truncation**: Full prompt context preserved
- **Proper normalization**: Test data scaled using training scaler
- **Length matching**: Casual-deploy matched to bench-eval distribution

### 3. Logging and Monitoring
- `flush=True` on all print statements (immediate output)
- Progress bars for layer-wise extraction
- Cache hit/miss logging
- Training/testing metrics per layer

### 4. Reproducibility
- Fixed random seeds (42)
- Deterministic tokenization
- Version-controlled prompts
- Cached intermediate results

---

## 📚 Related Work

This work builds on:
- **Evaluation Gaming** (Greenblatt et al., 2024)
- **Activation Engineering** (Zou et al., 2023)
- **Representation Engineering** (Zou et al., 2023)
- **Linear Representation Hypothesis** (Elhage et al., 2022)

Our contribution: **Systematic disentanglement** of format vs. content in evaluation awareness.

---

## 🤝 Contributing

This is a research project in active development. Contributions welcome!

### Areas for Extension

1. **Other models**: Test on other LLMs (Gemma, Mistral, GPT-4)
2. **Causal interventions**: Use AE vectors to steer model behavior
3. **Deeper mechanistic analysis**: Attention patterns, circuit discovery
4. **Other "awareness" signals**: Tool-use, uncertainty, reasoning depth

### New Scripts (Extended Experiments)

| Script | Purpose |
|--------|---------|
| `download_safety_datasets.py` | Download AdvBench and HarmBench from HuggingFace |
| `experiment_paired_training.py` | Train probe on paired datasets (Experiment 1) |
| `experiment_safety_datasets.py` | Evaluate probes on safety datasets (Experiment 2) |

---

## 🧪 Extended Experiments

### Experiment 1: Paired Dataset Training

**Goal**: Test if training on paired datasets (Eval-pair vs Deploy-pair) captures evaluation context better than single-dataset training.

**Design**:
- Split each of 4 datasets (Bench-Eval, Casual-Eval, Casual-Deploy, Bench-Deploy) into train (80%) / test (20%)
- Train probe on: (Bench-Eval + Casual-Eval) vs (Casual-Deploy + Bench-Deploy)
- Test on held-out portions of each dataset

**Running**:
```bash
# Run the experiment
python scripts/experiment_paired_training.py \
    --model ~/my_projects/models/Llama-3.1-8B-Instruct \
    --layers "8,16,24,30,31" \
    --gpus "0,1"

# Output:
# - Results: data/results/experiment1_paired_training.json
# - Cached probes: data/cache/paired_probe_layer{N}.pkl
# - Cached activations: data/cache/paired_{dataset}_{split}_layer{N}.pt
```

**Success Criterion**: If probe learns context (not just format), Casual-Eval test should be >50% "Eval" and Bench-Deploy test should be <50% "Eval".

---

### Experiment 2: AI Safety Dataset Evaluation

**Goal**: Test whether eval-aware probes also respond to AI Safety prompts (AdvBench, HarmBench).

**Datasets**:
- `walledai/AdvBench` - Adversarial jailbreak prompts
- `walledai/HarmBench` - Harmful behavior prompts (3 subsets: standard, contextual, copyright)

**Running**:
```bash
# Step 1: Download safety datasets
python scripts/download_safety_datasets.py

# This creates:
# - data/datasets/processed/advbench.json
# - data/datasets/processed/harmbench_all.json
# - data/datasets/processed/harmbench_{subset}.json (for each subset)

# Step 2: Run the experiment
# NOTE: Requires existing probes from evaluate_probe_cached.py AND/OR experiment_paired_training.py
python scripts/experiment_safety_datasets.py \
    --model ~/my_projects/models/Llama-3.1-8B-Instruct \
    --layers "8,16,24,30,31" \
    --gpus "0,1"

# Output:
# - Results: data/results/experiment2_safety_datasets.json
# - Cached activations: data/cache/safety_{dataset}_layer{N}.pt
```

**Research Questions**:
1. Do safety prompts trigger "eval" classification?
2. Does paired training change how probes respond to safety prompts?
3. Are there differences between HarmBench subsets?

---

### Running All Extended Experiments

```bash
# Complete workflow for extended experiments

# 1. Ensure base experiment is run first (creates single-dataset probes)
python scripts/evaluate_probe_cached.py \
    --model ~/my_projects/models/Llama-3.1-8B-Instruct \
    --layers "8,16,24,30,31" \
    --gpus "0,1"

# 2. Run Experiment 1: Paired training
python scripts/experiment_paired_training.py \
    --model ~/my_projects/models/Llama-3.1-8B-Instruct \
    --layers "8,16,24,30,31" \
    --gpus "0,1"

# 3. Download safety datasets
python scripts/download_safety_datasets.py

# 4. Run Experiment 2: Safety evaluation
python scripts/experiment_safety_datasets.py \
    --model ~/my_projects/models/Llama-3.1-8B-Instruct \
    --layers "8,16,24,30,31" \
    --gpus "0,1"

# All results will be in data/results/
```

---

## 📄 Citation

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

---

## 📞 Contact

**Author**: Viliana Devbunova
**GitHub**: [@viliana-dev](https://github.com/viliana-dev)
**Email**: devbunova99@gmail.com

---

## 🙏 Acknowledgments

- **Neel Nanda** for inspiration from his MATS stream on mechanistic interpretability
- **Anthropic** for evaluation gaming research
- **EleutherAI** for interpretability tools and methods
- **HuggingFace** for datasets and model hosting

---

## 📜 License

MIT License - See LICENSE file for details.

