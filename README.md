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

## Phase 3 — Quantization (In Progress)

**Method:** Dynamic INT8 quantization via `torch.quantization`

| Metric | FP32 | INT8 |
|--------|------|------|
| Accuracy | 95.34% | 94.81% |
| Model Size | 44.77 MB | 44.76 MB |
| Latency (CPU) | 28.21 ms | 28.18 ms |

**Key findings:**
- 0.53% accuracy drop — acceptable, essentially negligible
- Size reduction: ~0.01 MB — practically zero
- Latency improvement: ~0.03 ms — noise, not a result
- Dynamic quantization quantizes weights statically but activations at runtime; for conv-heavy models like ResNet-18, this yields little benefit — the compute bottleneck is in the convolutions, not matrix multiplications where INT8 shines
- Static quantization with proper calibration is the right next step if actual speedup is needed

> Both techniques preserve accuracy well but deliver no real deployment gains. The bottleneck is the compression method, not the aggressiveness. Unstructured pruning can't remove weights; dynamic quant can't speed up convolutions.

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

## Repo Structure (Expected)

```
│   .gitignore
│   README.md
│   resnet18_cifar10_baseline.pth
│   resnet18_dynamic_int8.pth
│   resnet_base.ipynb
│   resnet_pruning.ipynb
│   resnet_quantization.ipynb
│
└───pruned
        pruned_10.pth
        pruned_30.pth
        pruned_50.pth
        pruned_70.pth
        pruned_90.pth
```
