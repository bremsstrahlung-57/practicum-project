# CNN Compression for Edge Deployment
### ResNet-18 on CIFAR-10 — Empirical Compression Benchmarking

---

## Overview

Systematic evaluation of how far ResNet-18 can be compressed before accuracy collapses — using pruning, quantization, and knowledge distillation — targeting CPU-only edge inference.

**Dataset:** CIFAR-10 (32×32, 10 classes, 60k images)  
**Model:** ResNet-18 (CIFAR-adapted: 3×3 stride-1 conv, no maxpool)  
**Tracking:** Weights & Biases (WandB)  
**Environment:** Google Colab (T4 GPU for training), CPU-only for benchmarking

---

## Phase 1 — Baseline

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **95.34%** |
| Model Size | 44.77 MB |
| Inference Latency (CPU) | ~28.2 ms |
| Parameters | ~11M |
| Architecture | ResNet-18 (CIFAR-adapted) |

Training from scratch on CIFAR-10. Checkpoint saved to GitHub for persistence across Colab sessions.

---

## Phase 2 — Pruning (Complete)

**Method:** Magnitude-based unstructured pruning via `torch.nn.utils.prune`  
**Fine-tuning:** 15 epochs, LR 1e-4, SGD + cosine annealing  
**Sparsity levels tested:** 10%, 30%, 50%, 70%, 90%

| Sparsity | Accuracy | Size | Latency |
|----------|----------|------|---------|
| 10% | ~95.3% | 44.77 MB | ~28 ms |
| 30% | ~95.2% | 44.77 MB | ~28 ms |
| 50% | ~95.1% | 44.77 MB | ~28 ms |
| 70% | ~94.9% | 44.77 MB | ~28 ms |
| 90% | ~42.0% | 44.77 MB | ~28 ms |

**Key findings:**
- Accuracy stable through 70% sparsity — only ~0.4% total degradation
- Sharp collapse at 90% — drops to ~42%, clearly unusable
- Size and latency **completely flat** across all sparsity levels
- Unstructured pruning masks weights in-place; no weights are removed, so no real FLOPs or memory reduction occurs
- This is expected, not a failure — it's the canonical limitation of unstructured pruning and directly motivates quantization

> Unstructured sparsity ≠ hardware speedup. Clean empirical confirmation of a known theoretical limitation.

---

## Phase 3 — Quantization (Complete)

**Method:** Post-Training Static Quantization (PTQ) via `torch.ao.quantization` FX graph mode  
**Backend:** fbgemm (x86 CPU)  
**Calibration:** 100 batches from test set (no augmentation)

### Baseline + Static INT8

| Metric | FP32 | INT8 |
|--------|------|------|
| Accuracy | 95.24% | 95.25% |
| Model Size | 44.77 MB | 11.30 MB |
| Latency (CPU) | 39.22 ms | 13.58 ms |
| Size Reduction | — | **3.96x** |
| Speedup | — | **2.89x** |

### Pruned + Static INT8 (selected sparsity levels)

| Sparsity | FP32 Acc | INT8 Acc | Size | Speedup |
|----------|----------|----------|------|---------|
| 10% | ~95.3% | ~95.3% | 11.30 MB | ~2.3x |
| 30% | ~95.2% | ~95.2% | 11.30 MB | ~2.1x |
| 50% | ~95.1% | ~95.1% | 11.30 MB | ~1.9x |
| 70% | ~94.9% | ~94.9% | 11.30 MB | ~2.6x |
| 90% | ~42.0% | ~10.0% | 11.30 MB | — |

**Key findings:**
- Static PTQ delivers ~4x size reduction and ~2.9x latency speedup with near-zero accuracy loss
- Accuracy delta across all valid sparsity levels is negligible — quantization is essentially free on this task
- Baseline + INT8 outperforms pruned + INT8 on speedup — pruning before quantization adds no deployment benefit
- Compression gain is driven entirely by quantization; unstructured pruning contributes nothing to size or latency
- 90% pruned model collapses further under quantization (10% accuracy) — broken weights compound under INT8 conversion
- Dynamic quantization was tested first but yielded no measurable gains on this conv-heavy architecture; static PTQ with calibrated activation ranges is the correct approach for ResNet-class models

---

## Phase 4 — Knowledge Distillation (Planned)

**Teacher:** ResNet-34 or ResNet-50  
**Student:** ResNet-18  
**Goal:** Evaluate whether distillation beats pruning on the accuracy/size trade-off

*Not yet started.*

---

## Metrics Pipeline

| Metric | Method |
|--------|--------|
| Accuracy | PyTorch eval loop on CIFAR-10 test set |
| Inference latency | `time.perf_counter`, 100-run average, CPU-only |
| Peak RAM | `tracemalloc` |
| Model size | Checkpoint file size on disk |
| Sparsity | Weight mask inspection via `torch.nn.utils.prune` |

---

## Theoretical Anchors

- **Lottery Ticket Hypothesis** — Frankle & Carlin (2019): motivates pruning; the 70% stability result is consistent with the existence of sparse trainable subnetworks
- **Deep Compression** — Han et al. (2016): this project reproduces the pruning + quantization stages in a controlled, reproducible setup

---

## Status

| Phase | Status |
|-------|--------|
| Baseline | ✅ Complete |
| Pruning | ✅ Complete |
| Quantization | ✅ Complete |
| Distillation | 🔄 In Progress |
| Final Analysis & Report | ⏳ Planned |

---

## Repo Structure

```
practicum_project/
│
├── pruned/                        # Pruned model checkpoints
│   ├── pruned_10.pth
│   ├── pruned_30.pth
│   ├── pruned_50.pth
│   ├── pruned_70.pth
│   └── pruned_90.pth
│
├── quantized/                     # Quantized model checkpoints
│   ├── resnet18_dynamic_int8.pth
│   ├── resnet18_int8_10pruned.pth
│   ├── resnet18_int8_30pruned.pth
│   ├── resnet18_int8_50pruned.pth
│   ├── resnet18_int8_70pruned.pth
│   ├── resnet18_int8_90pruned.pth
│   └── resnet18_static_int8_base.pth
│
├── .gitignore
├── README.md
│
├── resnet18_cifar10_baseline.pth  # Baseline trained model
├── resnet_base_training.ipynb
├── resnet_dynamic_quantization.ipynb
├── resnet_pruning.ipynb
└── resnet_static_quantization.ipynb
```
