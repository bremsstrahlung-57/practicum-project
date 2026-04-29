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

## Phase 2 — Unstructured Pruning (Complete)

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
- This is expected, not a failure — it's the canonical limitation of unstructured pruning and directly motivates structured pruning

> Unstructured sparsity ≠ hardware speedup. Clean empirical confirmation of a known theoretical limitation.

---

## Phase 3 — Structured Pruning + Quantization (Complete)

### Phase 3a — Structured Pruning

**Method:** L1-norm magnitude-based channel pruning via `torch-pruning` (MagnitudePruner)  
**Dependency handling:** Automatic skip-connection coupling via `torch-pruning` dependency graph  
**Fine-tuning:** 15 epochs, LR 1e-4, SGD + cosine annealing  
**Channel removal ratios tested:** 30%, 50%, 70%

| Model | Acc (FP32) | Latency FP32 (ms) | Size FP32 (MB) | Params (M) | MACs (M) | Speedup vs Baseline |
|-------|------------|-------------------|----------------|------------|----------|---------------------|
| Baseline | 95.24% | 21.48 ms | 42.70 MB | 11.17M | 557.2M | 1.00× |
| Structured-30% | 93.31% | 18.62 ms | 20.89 MB | 5.46M | 269.8M | 1.15× |
| Structured-50% | 91.41% | 7.19 ms | 10.73 MB | 2.80M | 140.2M | 2.99× |
| Structured-70% | 86.45% | 4.56 ms | 3.85 MB | 1.00M | 50.0M | 4.71× |

**Key findings:**
- Unlike unstructured pruning, structured pruning produces real FLOPs and size reduction — tensors physically shrink
- 50% channel removal is the sweet spot: 3× latency reduction, 4× size reduction, only 3.8% accuracy drop
- 70% pushes to 4.7× speedup but accuracy drops 8.8% — acceptable for latency-critical applications, borderline for general use
- `torch-pruning` handles ResNet skip-connection coupling automatically — pruned channels stay consistent across residual branches

### Phase 3b — Static Quantization (INT8)

**Method:** Post-training static quantization via `torch.ao.quantization` FX graph mode (`prepare_fx` / `convert_fx`)  
**Backend:** fbgemm (x86 CPU)  
**dtype:** `torch.qint8`  
**Calibration:** subset of CIFAR-10 train set  
**Note:** Eager mode static PTQ failed due to `aten::add.out` dispatch error on QuantizedCPU backend — residual additions in eager mode require explicit `FloatFunctional` wrappers. FX graph mode resolves this automatically by tracing the full compute graph and handling the residual add as a first-class node.

| Model | Acc (FP32) | Acc (INT8) | Latency FP32 (ms) | Latency INT8 (ms) | Size FP32 (MB) | Size INT8 (MB) | INT8 Speedup vs Baseline |
|-------|------------|------------|-------------------|-------------------|----------------|----------------|--------------------------|
| Structured-30% | 93.31% | 93.21% | 18.62 ms | 10.55 ms | 20.89 MB | 5.30 MB | 2.04× |
| Structured-50% | 91.41% | 91.27% | 7.19 ms | 4.95 ms | 10.73 MB | 2.75 MB | 4.34× |
| Structured-70% | 86.45% | 86.40% | 4.56 ms | 3.99 ms | 3.85 MB | 1.02 MB | 5.38× |

**Key findings:**
- Static PTQ adds near-zero accuracy cost (<0.15% across all ratios) — essentially free
- INT8 gains diminish at higher pruning ratios: 1.77× additional speedup at 30%, only 1.14× at 70% — at ~1M params the model is too small for memory bandwidth to be the bottleneck
- Size compression from INT8 is consistent (~4×) regardless of pruning ratio
- **Hero result: Structured-50% + INT8 — 4.34× faster, 15.5× smaller than baseline, only 3.97% accuracy drop**

### Combined Stack Summary

| Stage | Accuracy | Latency (ms) | Size (MB) | Speedup |
|-------|----------|--------------|-----------|---------|
| Baseline | 95.24% | 21.48 | 42.70 | 1.00× |
| + Structured 50% pruning | 91.41% | 7.19 | 10.73 | 2.99× |
| + Static INT8 | 91.27% | 4.95 | 2.75 | **4.34×** |

> Structured pruning + static quantization delivers a deployable edge model. 4.3× faster, 15.5× smaller, 4% accuracy cost.

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
| MACs | `tp.utils.count_ops_and_params` via torch-pruning |

---

## Theoretical Anchors

- **Lottery Ticket Hypothesis** — Frankle & Carlin (2019): motivates pruning; the 70% unstructured sparsity stability result is consistent with the existence of sparse trainable subnetworks
- **Deep Compression** — Han et al. (2016): this project reproduces the pruning + quantization stages in a controlled, reproducible setup

---

## Status

| Phase | Status |
|-------|--------|
| Baseline | ✅ Complete |
| Unstructured Pruning | ✅ Complete |
| Structured Pruning + Quantization | ✅ Complete |
| Knowledge Distillation | 🔄 Planned |
| Final Analysis & Report | ⏳ Planned |

---

## Repo Structure

```
practicum_project/
│
├── models/
│   ├── basline/                       # Baseline trained model
│   │   └── resnet18_cifar10_baseline.pth
│   ├── pruned/                        # Pruned model checkpoints
│   │   ├── pruned_10.pth
│   │   ├── pruned_30.pth
│   │   ├── pruned_50.pth
│   │   ├── pruned_70.pth
│   │   └── pruned_90.pth
│   ├── quantized/                     # Quantized model checkpoints
│   │   ├── resnet18_dynamic_int8.pth
│   │   ├── resnet18_int8_10pruned.pth
│   │   ├── resnet18_int8_30pruned.pth
│   │   ├── resnet18_int8_50pruned.pth
│   │   ├── resnet18_int8_70pruned.pth
│   │   ├── resnet18_int8_90pruned.pth
│   │   └── resnet18_static_int8_base.pth
│   └── structured_pruning/            # Structured pruning checkpoints
│       ├── pruned/                    # fine tuned too
│       │   ├── structured_pruned_30pct_fp32.pth
│       │   ├── structured_pruned_50pct_fp32.pth
│       │   └── structured_pruned_70pct_fp32.pth
│       └── pruned_and_quantized/
│           ├── structured_pruned_30pct_int8.pt
│           ├── structured_pruned_50pct_int8.pt
│           └── structured_pruned_70pct_int8.pt
│
├── .gitignore
├── README.md
│
├── resnet_base_traning.ipynb
├── resnet_dynamic_quantization.ipynb
├── resnet_pruning.ipynb
├── resnet_static_quantization.ipynb
└── resnet_structured_pruning.ipynb
```
