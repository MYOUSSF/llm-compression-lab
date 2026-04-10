# LLM Compression Lab

A research pipeline for compressing and fine-tuning transformer models, applied to **DistilBERT** on sentiment classification (SST-2). The lab covers the full spectrum of compression techniques: post-training quantization, sensitivity-guided mixed precision, magnitude pruning, LoRA fine-tuning, and knowledge distillation.

---

## Results at a Glance

| Method | Task | Accuracy | Size | Speedup |
|---|---|---|---|---|
| FP32 baseline | SST-2 | 91.06% | 267.9 MB | — |
| PTQ INT8 | SST-2 | 89.68% | 138.7 MB | 1.30× |
| Mixed-precision | SST-2 | 89.68% | 138.7 MB | 1.25× |
| Sparse 30% + INT8 | SST-2 | 90.60% | 138.7 MB | 1.30× |
| Sparse 50% + INT8 | SST-2 | 87.84% | 138.7 MB | 1.29× |
| LoRA FP32 (r=8) | MNLI | 66.02% | 270.2 MB | — |
| LoRA + QAT INT8 | MNLI | 64.40% | 139.3 MB | — |
| Distillation (T=4, α=0.7) | SST-2 | 90.94% | 270.9 MB | — |

**Best accuracy–compression trade-off:** Sparse 30% + INT8 — only −0.46% accuracy loss at 1.93× compression and 1.30× speedup.  
**Best accuracy recovery:** Knowledge distillation at T=4 recovers to 90.94%, just 0.12% below FP32.

---

## Project Structure

```
llm-compression-lab/
├── src/
│   ├── __init__.py            # Public API for all modules
│   ├── sensitivity.py         # Layerwise quantization sensitivity scoring
│   ├── mixed_precision.py     # Mixed-precision policy builder
│   ├── benchmark.py           # Latency / throughput / size benchmarking
│   └── distillation.py        # Knowledge distillation trainer + sweeps
├── notebook_01_compression.ipynb   # PTQ, sensitivity, pruning, ONNX export
├── notebook_02_finetuning.ipynb    # LoRA, QAT, distillation, sweeps
└── results/
    ├── benchmark_table.csv
    ├── benchmark_table_nb02.csv
    ├── sensitivity_chart.png
    ├── activation_histograms.png
    ├── benchmark_comparison.png
    ├── benchmark_comparison_nb02.png
    ├── lora_training_curves.png
    ├── gradient_statistics.png
    ├── distillation_history.png
    ├── temperature_sweep.png
    └── alpha_sweep.png
```

---

## Notebooks

### Notebook 01 — Compression Pipeline

**Model:** `distilbert-base-uncased-finetuned-sst-2-english` · **Task:** SST-2 binary sentiment · **Hardware:** T4 GPU (training) + CPU (quantized inference)

| Section | Content |
|---|---|
| §2 | FP32 baseline (91.06%, 267.9 MB) |
| §3 | Dynamic PTQ INT8 (89.68%, 138.7 MB, 1.30× faster) |
| §4 | Layerwise sensitivity scoring |
| §5 | Mixed-precision quantization policy |
| §6 | Magnitude pruning + sparse INT8 |
| §7 | Activation distribution analysis (KL divergence) |
| §8 | ONNX export + ONNXRuntime benchmark |
| §9 | Consolidated latency / throughput / size table |

### Notebook 02 — Fine-Tuning Pipeline

**Model:** DistilBERT · **Tasks:** MNLI (LoRA) and SST-2 (distillation) · **Hardware:** T4 GPU

| Section | Content |
|---|---|
| §2 | FP32 baseline |
| §3 | LoRA fine-tuning on MNLI (r=8, alpha=16) |
| §4 | QAT INT8 on the LoRA-adapted model |
| §5 | Knowledge distillation (FP32 teacher → LoRA student) |
| §6 | Temperature sweep (T = 2, 4, 8) |
| §7 | Alpha sweep (soft-loss weight = 0.3, 0.5, 0.7, 0.9) |
| §8 | Final benchmark — all model variants |

---

## Key Findings

### 1 · Post-Training Quantization (PTQ INT8)
Dynamic INT8 quantization (weights only) gives a clean **1.93× size reduction** with only **−1.38% accuracy loss**. Activations remain FP32 at runtime, which is the correct approach for BERT-family models that lack `QuantStub`/`DeQuantStub` in their forward pass.

### 2 · Layerwise Sensitivity Analysis
The quantize-then-restore method reveals that DistilBERT's sensitivity is highly concentrated:
- **`layer.2.ffn.lin2`** — highest sensitivity: Δ = +1.95% when restored to FP32
- **`layer.1.attention.q_lin`** — Δ = +1.56%
- Most layers (84%) show Δ ≈ 0% and are safe INT4 candidates

The resulting mixed-precision policy assigns: **4 layers → FP32** (10.5%) · **2 layers → INT8** (5.3%) · **32 layers → INT4** (84.2%).

### 3 · Magnitude Pruning
L1 unstructured pruning before INT8 quantization yields the best accuracy–compression trade-off:
- **30% sparsity:** 90.60% accuracy (only −0.46% vs FP32), 1.30× speedup
- **50% sparsity:** 87.84% accuracy (−3.21% vs FP32) — diminishing returns

Zeroed weights map cleanly to the INT8 zero-point, causing no additional quantization error.

### 4 · Activation Distributions
KL divergence analysis across 6 attention and FFN layers shows all KL values < 0.01 (range: 0.0003–0.0017), well below the meaningful-shift threshold. Dynamic INT8 preserves the activation distribution almost perfectly.

### 5 · LoRA Fine-Tuning (MNLI)
LoRA (r=8, alpha=16) adapts DistilBERT to MNLI with only **739,586 trainable parameters** (1.1% of total). The model converges to 66% accuracy in 3 epochs. Gradient statistics confirm healthy training: all norms between 1e−6 and 10, no vanishing or exploding gradients.

### 6 · Knowledge Distillation
Distillation from an FP32 teacher to a LoRA student (T=4, α=0.7) recovers accuracy to **90.94%** — within 0.12% of the FP32 baseline.

**Temperature sweep results:**
| T | Best Accuracy | Recovery vs Baseline |
|---|---|---|
| **4.0** | **91.17%** | **+0.11%** |
| 8.0 | 91.06% | 0.00% |
| 2.0 | 90.71% | −0.34% |

**Alpha sweep results (T=4.0):**
| α | Best Accuracy |
|---|---|
| 0.5 | 91.06% |
| 0.7 | 90.94% |
| 0.9 | 90.83% |
| 0.3 | 90.71% |

T=4 is the optimal temperature: sharp enough to carry meaningful class-probability information, soft enough to provide useful inter-class signals beyond the argmax.

---

## Source Modules

### `src/sensitivity.py` — `SensitivityAnalyzer`

Quantize-then-restore layerwise sensitivity scoring.

```python
from src.sensitivity import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(fp32_model, quantized_model, eval_loader, device)
scores   = analyzer.run()              # {layer_name: accuracy_delta}
df       = analyzer.to_dataframe()     # sorted DataFrame
fig      = analyzer.plot(top_k=20)     # matplotlib Figure
policy   = analyzer.recommend_policy(top_k=5)  # names of sensitive layers
```

### `src/mixed_precision.py` — `MixedPrecisionPolicy`

Sensitivity-driven precision assignment with configurable thresholds.

```python
from src.mixed_precision import MixedPrecisionPolicy

policy = MixedPrecisionPolicy(
    sensitivity_scores  = scores,
    fp32_threshold      = 1.0,   # Δ > 1.0% → FP32
    int8_threshold      = 0.3,   # Δ > 0.3% → INT8, else INT4
    force_fp32          = ['embeddings', 'classifier', 'pooler'],
)
policy.print_summary()
quantized_model = policy.apply(fp32_model, calibration_loader, device)
```

### `src/benchmark.py` — `ModelBenchmark`

Latency, throughput, and model-size benchmarking with warm-up runs.

### `src/distillation.py` — `DistillationTrainer`

Knowledge distillation with configurable temperature and alpha, plus grid-search utilities.

```python
from src.distillation import DistillationTrainer, temperature_sweep

trainer = DistillationTrainer(teacher, student, train_loader, val_loader, device)
history = trainer.train(epochs=3, T=4.0, alpha=0.7)

results = temperature_sweep(teacher, student, train_loader, val_loader, device,
                            temperatures=[2.0, 4.0, 8.0])
```

---

## Setup

```bash
pip install transformers>=4.38.0 datasets>=2.18.0 accelerate>=0.27.0 \
            onnx>=1.15.0 onnxruntime>=1.17.0 \
            bitsandbytes>=0.43.0 peft>=0.9.0 \
            pandas>=2.0.0 matplotlib>=3.7.0
```

Run notebooks in order:
1. `notebook_01_compression.ipynb` — quantization and pruning pipeline
2. `notebook_02_finetuning.ipynb` — LoRA and distillation pipeline

---

## Hardware

Experiments ran on Kaggle with a **Tesla T4 GPU (15.6 GB VRAM)**. Quantized inference was benchmarked on **CPU** (standard deployment target for compressed models). Benchmark figures: 100 timed runs with 10 warm-up runs.
