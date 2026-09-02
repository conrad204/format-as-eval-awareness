# Quick Start Guide

Get your experiment running in 5 minutes!

---

## 🚀 Installation

```bash
# Clone repo
git clone https://github.com/viliana-dev/eval-awareness-format.git
cd eval-awareness-format

# Install dependencies
pip install -r requirements.txt

# Download model (one-time, ~16GB)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/models/Llama-3.1-8B-Instruct
```

---

## 🏃 Run Main Experiment

### Option 1: Use Pre-processed Data (Fast)

If you have access to the pre-processed datasets:

```bash
python scripts/evaluate_probe_cached.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --layers "8,16,24,30,31" \
  --gpus "0,1"
```

**Time**: ~30 minutes (first run), ~5 minutes (cached)

---

### Option 2: Full Pipeline (From Scratch)

#### Step 1: Download Raw Datasets
```bash
python scripts/prepare_datasets.py
```
**Time**: ~15 minutes
**Output**: `data/datasets/raw/` (MMLU, ARC, GSM8K, CoQA, ShareGPT)

---

#### Step 2: Create Length-Matched Casual-Deploy
```bash
python scripts/create_length_matched.py
```
**Time**: ~2 minutes
**Output**: `data/datasets/processed/casual_deploy_matched.json`

---

#### Step 3: Classify Queries by Benchmark Type
```bash
python scripts/run_routing.py \
  --api_url "https://your-api-endpoint.com/v1/chat/completions" \
  --api_key "YOUR_API_KEY"
```
**Time**: ~2 hours (1,076 queries × 6 seconds/query)
**Output**: `data/datasets/routing/routing_checkpoint.json`

---

#### Step 4: Generate Rewritten Datasets

**Benchmark mode** (MCQ + casual):
```bash
python scripts/rewrite_casual_deploy.py \
  --mode benchmark \
  --api_url "https://your-api-endpoint.com/v1/chat/completions" \
  --api_key "YOUR_API_KEY"
```

**Formal mode** (no MCQ + formal):
```bash
python scripts/rewrite_casual_deploy.py \
  --mode formal \
  --api_url "https://your-api-endpoint.com/v1/chat/completions" \
  --api_key "YOUR_API_KEY"
```

**Benchmark-Formal mode** (MCQ + formal):
```bash
python scripts/rewrite_casual_deploy.py \
  --mode benchmark_formal \
  --api_url "https://your-api-endpoint.com/v1/chat/completions" \
  --api_key "YOUR_API_KEY"
```

**Time per mode**: ~2 hours
**Total time**: ~6 hours for all 3 modes
**Output**:
- `data/datasets/processed/bench_deploy_rewritten.json`
- `data/datasets/processed/casual_deploy_formal.json`
- `data/datasets/processed/bench_deploy_formal.json`

---

#### Step 5: Run Main Experiment
```bash
python scripts/evaluate_probe_cached.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --layers "8,16,24,30,31" \
  --gpus "0,1"
```

**Time**: ~30 minutes (first run)
**Output**: `data/results/cross_condition_eval_cached.json`

---

## 📊 View Results

### Command-line Summary
```bash
# View final results
python -c "
import json
with open('data/results/cross_condition_eval_cached.json') as f:
    results = json.load(f)

print('Layer 24 Results:')
for ds, metrics in results['results']['24'].items():
    if isinstance(metrics, dict) and 'pct_eval' in metrics:
        print(f'  {ds:25s}: {metrics[\"pct_eval\"]*100:5.1f}% → eval')
"
```

### Generate Visualizations
```bash
# Length distributions
python scripts/plot_length_comparison.py

# Boxplots for 2x2 design
python scripts/plot_length_boxplot.py
```

**Output**: `data/length_*.png`

---

## 🔧 Common Issues

### Issue: "Model not found"
```bash
# Check model path
ls ~/models/Llama-3.1-8B-Instruct/

# If empty, re-download
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/models/Llama-3.1-8B-Instruct
```

---

### Issue: "CUDA out of memory"
```bash
# Use smaller batch size (edit scripts/evaluate_probe_cached.py)
# Change: batch_size=8 → batch_size=4

# Or use fewer GPUs
python scripts/evaluate_probe_cached.py --gpus "0"
```

---

### Issue: "API timeout" during rewriting
```bash
# Resume from checkpoint (automatic)
python scripts/rewrite_casual_deploy.py --mode benchmark \
  --api_url "..." --api_key "..."

# The script automatically loads checkpoint and continues
```

---

### Issue: "Cache inconsistency"
```bash
# Clear all caches and recompute
python scripts/evaluate_probe_cached.py \
  --force_recompute \
  --model ~/models/Llama-3.1-8B-Instruct
```

---

## 📈 Expected Results

After running the main experiment, you should see:

```
Layer 24 (5/5)
  Training (A vs B): Acc=1.000
  → Testing probe on 4 test datasets...
  casual_eval         :   0.0% → eval,  100.0% → deploy
  bench_deploy        :  94.3% → eval,    5.7% → deploy  ← Format effect!
  casual_deploy_formal:   0.0% → eval,  100.0% → deploy  ← No style effect
  bench_deploy_formal :  88.5% → eval,   11.5% → deploy  ← Format dominates
```

**Key Finding**: MCQ format causes 94% "eval" classification, regardless of style!

---

## 🎓 Advanced Usage

### Extract AE Vectors for Steering
```bash
python scripts/extract_ae_from_datasets.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --n_samples 200 \
  --output data/vectors/ae_custom.pt
```

### Test on Different Layers
```bash
python scripts/evaluate_probe_cached.py \
  --layers "0,4,8,12,16,20,24,28,31"
```

### Test with Larger Sample Size
Edit `SAMPLE_SIZE` in `scripts/evaluate_probe_cached.py`:
```python
SAMPLE_SIZE = 2000  # Default: 1076
```

---

## 🆘 Getting Help

1. **Check logs**: `data/results/*.log`
2. **Check cache**: `ls data/cache/`
3. **GitHub Issues**: https://github.com/viliana-dev/eval-awareness-format/issues
4. **Email**: devbunova99@gmail.com

---

## ⚡ Performance Tips

**Speed up activation extraction**:
- Use multiple GPUs: `--gpus "0,1,2,3"`
- Cache is preserved across runs (only extract once)

**Speed up rewriting**:
- Use faster API endpoint
- Increase `max_workers` in concurrent requests
- Resume from checkpoint if interrupted

**Reduce memory usage**:
- Use smaller batch size
- Test fewer layers
- Use model quantization (int8)

---

## 📝 Citation

```bibtex
@misc{devbunova2025eval-awareness,
  title={Is It Evaluation Awareness or Just Format Sensitivity?},
  author={Devbunova, Viliana},
  year={2025},
  url={https://github.com/viliana-dev/eval-awareness-format}
}
```

---

## ✨ Next Steps

- Read `README.md` for detailed documentation
- Check `RESULTS.md` for complete experimental results
- See `MATS_APPLICATION.md` for research reflection
- Explore `notebooks/` for interactive analysis

**Happy researching! 🚀**

