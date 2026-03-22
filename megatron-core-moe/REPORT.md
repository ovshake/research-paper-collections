# Scalable Training of Mixture-of-Experts Models with Megatron Core

**Authors:** NVIDIA (Zijie Yan, Hongxiao Bai, Xin Yao, Dennis Liu, Tong Liu, et al.)
**Date:** 2026-03-11
**Paper:** [PDF](https://arxiv.org/abs/2603.07685)
**Code:** https://github.com/NVIDIA/Megatron-LM

---

## TL;DR

A comprehensive systems paper describing Megatron-Core MoE, NVIDIA's open-source training stack for trillion-parameter MoE models. It identifies three fundamental efficiency walls (Memory, Communication, Compute) unique to MoE's parameter-compute mismatch, solves them through an integrated set of optimizations (Parallel Folding, DeepEP/HybridEP, CUDA Graphs, FP8/FP4, ECHO, Paged Stashing), and achieves **1,233 TFLOPS/GPU on GB300** and **1,048 TFLOPS/GPU on GB200** for DeepSeek-V3 (685B params, 256 experts).

---

## Key Novel Ideas

### 1. The Three Walls Framework — Memory, Communication, Compute

The paper's central organizing insight. MoE sparsity creates a **parameter-compute mismatch** (DeepSeek-V3: 685B total params but only 37B active = 18:1 ratio) that manifests as three tightly coupled barriers:

- **Memory Wall:** All E experts must reside in memory even though only K activate. Activations (131 GB for DeepSeek-V3) exceed weights + optimizer combined.
- **Communication Wall:** EP requires all-to-all collectives whose volume scales with tokens, not expert count. Can consume 20-60% of training time unoptimized.
- **Compute Efficiency Wall:** Fine-grained experts produce tiny GEMMs (~128 M-dimension) that underutilize Tensor Cores. Host overhead from many small kernel launches creates GPU bubbles.

**Critical insight:** These walls interact — fixing one often exposes or worsens another. FP8 reduces memory but introduces quantization kernel overhead (compute wall). Communication overlap requires extra memory buffers. CUDA Graphs need static shapes, conflicting with dropless routing. Effective optimization requires treating all three walls as a unified system.

### 2. MoE Parallel Folding — Breaking the Dense-Sparse Mismatch

Traditional frameworks force `EP ⊆ DP`, meaning attention and MoE layers share one parallelism configuration. This creates the **dense-sparse mismatch**: attention benefits from high TP while MoE benefits from high EP with low TP.

**Parallel Folding decouples them entirely:**
- Attention layers: TP × CP × DP × PP (optimized for dense computation)
- MoE layers: ETP × EP × EDP × PP (optimized for sparse expert routing)
- Only PP must be consistent across both

**Concrete impact:** With TP=4, CP=2, DP=8, PP=4 (256 GPUs), traditional approach limits EP ≤ 8. Parallel Folding enables EP=64 by folding across TP×CP groups, delivering 8× higher expert parallelism while keeping all communication within the NVLink domain.

### 3. Sync-Free Dropless MoE via Device-Initiated Kernels + ECHO + Paged Stashing

A three-part solution to enable full CUDA Graph coverage for dropless MoE (where token counts per expert are dynamic):

**Device-Initiated Kernels:** Redesigned Grouped GEMM and HybridEP kernels that read shape information directly from GPU memory instead of requiring host-device synchronization. The GPU kernel determines its own workload at runtime, eliminating the CPU from the critical path.

**ECHO (Elastic Cloning for Hot Experts):** Dynamically clones popular experts to underutilized GPU ranks. A planner computes per-expert *spillover* and per-rank *spare capacity*, then uses bin-packing to minimize clones while balancing load. Reduces both memory fragmentation (smaller worst-case buffers) and compute straggler effects.

**Paged Stashing:** Instead of O(layers × worst_case) memory for static CUDA Graph buffers, uses a single worst-case tmp buffer shared across layers plus a paged stashing buffer storing only actual tokens. Reduces memory from O(layers × worst_case) to O(worst_case + actual_total). Pages are managed via device-initiated stash/reload kernels overlapped with computation on dedicated CUDA streams.

### 4. Memory-Efficient Permutation — Zero-Overhead Activation Reduction

A mathematically elegant optimization. Standard MoE applies routing weights *after* expert computation, requiring storage of each expert's full output for the backward pass. Memory-Efficient Permutation absorbs the routing weight `p_i` *into* the activation function (before FC2), so:

Standard: `y = Σ p_i · W2(φ(W1·x))`
Memory-efficient: `y = Σ W2(p_i · φ(W1·x))`

Since W2 is linear, these are mathematically equivalent. But the backward pass for `∂L/∂p_i` now only needs `φ(z_i)` (already saved) rather than the full expert output. **Saves 26.3 GB per GPU for DeepSeek-V3 with zero computational cost.**

### 5. Communication-Computation Overlap via Merged FWD-BWD 1F1B

EP all-to-all remains on the critical path even with optimized dispatchers. The key insight: all-to-all latency can be hidden by overlapping it with computation from *adjacent micro-batches*.

**Merged FWD-BWD:** Interleaves the forward pass of micro-batch i+1 with the backward pass of micro-batch i. Combined with **W/D gradient split** (separating weight gradients from data gradients to break dependencies), this achieves **93% overlap ratio**, reducing EP communication from 30-40% to under 5% of iteration time on DeepSeek-V3 H100 training.

### 6. Comprehensive FP8/FP4 Training with Selective Precision

Four validated reduced-precision recipes, each optimized for specific hardware:

| Recipe | Granularity | Platform | Recommended For |
|---|---|---|---|
| Per-Tensor FP8 | 1 scale/tensor | Hopper/Blackwell | Experimentation |
| Blockwise FP8 | 128×128 blocks | Hopper | Production (proven at scale) |
| MXFP8 | 1×32 elements | Blackwell | Default on Blackwell |
| NVFP4 | 16 elements, 2-level scaling | Blackwell | Maximum throughput |

**Key principle: Selective Precision.** Router stays FP32, embeddings/output layers/optimizer states stay high precision, bulk expert GEMMs get quantized. NVFP4 additionally requires Random Hadamard Transforms (RHT), 2D weight scaling, and stochastic rounding for gradient stability.

**MoE-specific challenges solved:** Dynamic shape alignment (fused padding into permutation kernel), grouped quantization (fusing quantization of multiple expert tensors into one kernel), and NVFP4 quantization fusion (absorbing RHT + transpose + FP4 conversion into a single kernel to avoid BF16 materialization).

### 7. Dynamic Context Parallelism (Dynamic-CP)

For RL and SFT with variable-length sequences (100 to 1M tokens), static CP wastes resources on short sequences that don't need splitting. Dynamic-CP:
- Adaptively selects CP size per microbatch based on sequence length
- Pre-constructs multiple CP groups at initialization (powers of 2)
- Uses a lightweight 1D grid search solver that balances compute and memory
- Yields **35-60% end-to-end improvement** in variable-length training scenarios

---

## Architecture Details

**Megatron-Core MoE Layer — Four-Stage Forward Pass:**

| Stage | Operation | Communication |
|---|---|---|
| Route | TopK router selects experts per token | None |
| Dispatch | Permute + all-to-all to send tokens to expert GPUs | all-to-all |
| Compute | Grouped GEMM across local experts | None |
| Combine | all-to-all return + unpermute + weighted sum | all-to-all |

**Multi-Dimensional Parallelism Stack:**

| Dimension | Applies To | Purpose |
|---|---|---|
| TP (Tensor) | Attention | Shard large QKV/projection matrices |
| CP (Context) | Attention | Distribute long sequences |
| DP (Data) | Attention | Process different batches |
| PP (Pipeline) | Both | Split model by layers |
| EP (Expert) | MoE | Distribute experts across GPUs |
| ETP (Expert Tensor) | MoE | Shard within experts (rarely used) |
| EDP (Expert Data) | MoE | Replicate experts for throughput |

---

## Key Results

### Throughput Benchmarks (per-GPU)

| Model | System | #GPUs | Seqlen | Dtype | TFLOPS/GPU | Tokens/s/GPU |
|---|---|---|---|---|---|---|
| DeepSeek-V3 (685B) | GB300 | 256 | 4,096 | MXFP8 | **1,233** | 4,730 |
| DeepSeek-V3 | GB200 | 256 | 4,096 | MXFP8 | **1,048** | 4,020 |
| DeepSeek-V3 | GB200 | 256 | 4,096 | BF16 | 857 | 3,298 |
| DeepSeek-V3 | H100 | 1,024 | 4,096 | FP8-BLK | 368 | 1,412 |
| Qwen3-235B | GB300 | 256 | 4,096 | MXFP8 | **974** | 6,583 |
| Qwen3-235B | GB200 | 256 | 4,096 | MXFP8 | 919 | 6,212 |
| Qwen3-235B | GB200 | 256 | 4,096 | BF16 | 750 | 5,100 |
| Qwen3-235B | H100 | 256 | 4,096 | BF16 | 320 | 2,132 |
| Qwen3-235B | GB300 | 128 | 131,072 | MXFP8 | 1,150 | 1,556 |

**GB200/GB300 deliver ~3× token throughput vs H100** for both models.

### DeepSeek-V3 Optimized Configurations

| Category | GB200 (256 GPUs) | H100 (1,024 GPUs) |
|---|---|---|
| Parallelism (TP/PP/EP) | 1/4/64 | 2/8/64 |
| Precision | MXFP8 | FP8-Blockwise |
| Dispatcher | HybridEP | DeepEP |
| CUDA Graphs | Enabled | — |
| EP Overlap | — | Enabled |
| Performance | **1,048 TFLOPS** | **368 TFLOPS** |

### Communication Optimization Impact

| Optimization Stage | EP Communication Share |
|---|---|
| Baseline (no optimization) | 30-40% of iteration time |
| + DeepEP/HybridEP | ~20% |
| + FWD-BWD Overlap with W/D Split | **<5%** (93% overlap) |

### Memory Reduction for DeepSeek-V3

| Technique | Memory Saved per GPU |
|---|---|
| Memory-Efficient Permutation | 26.3 GB |
| FP8 Activations | 16 GB |
| Fine-grained Recomputation | 42.4 GB |
| Fine-grained Offloading | 10-18% reduction |
| Precision-Aware Optimizer | 10-12 GB |
| **Total: 199.5 GB → fits in 80 GB** | via all techniques combined |

### HybridEP vs Standard all-to-all (µs, EP=64)

| Operation | GB200 HybridEP | GB200 all-to-all | H100 HybridEP | H100 all-to-all |
|---|---|---|---|---|
| Dispatch | 675 | 930 | 4,626 | 9,164 |
| Combine | 744 | 827 | 4,398 | 8,727 |

HybridEP provides 1.4× speedup on GB200 and **2× speedup on H100** for EP=64.

---

## Key Takeaways

1. **MoE training is fundamentally communication-bound, not compute-bound.** Dense models maintain a "virtuous cycle" where more parameters = more compute = stable MFU. MoE breaks this: 18× more parameters but the same compute per token means communication dominates. This is a *qualitative* difference from dense training, not just a degree difference.

2. **The same model requires completely different optimization stacks on different hardware.** DeepSeek-V3 on GB200 is compute-efficiency-bound (CPU overhead is the bottleneck → CUDA Graphs + kernel fusions). On H100, it's communication-bound (cross-node EP → DeepEP + overlap). Hardware topology, not just model architecture, determines the optimization strategy.

3. **Memory-Efficient Permutation is the highest-value optimization: 26 GB saved with zero cost.** A pure algebraic rearrangement that eliminates redundant backward-pass storage. Every MoE implementation should adopt this.

4. **Parallel Folding is essential for MoE at scale.** The traditional EP ⊆ DP constraint forces suboptimal configurations on either attention or MoE layers. Decoupling them allows EP=64 with TP=1 for experts (full expert width, maximum GEMM efficiency) while attention uses TP=4 (efficient matrix sharding). This isn't optional for large MoE models — it's structurally necessary.

5. **CUDA Graphs for dropless MoE required three independent innovations.** Device-initiated kernels (eliminate host-device sync), ECHO (reduce load imbalance → smaller worst-case buffers), and Paged Stashing (fine-grained memory management for static buffers). None alone is sufficient; all three compose to achieve full coverage. This delivers 10% end-to-end speedup.

6. **FP8 training shifts bottlenecks rather than simply solving them.** FP8 halves activation memory and accelerates GEMMs, but the faster GPU execution exposes CPU overhead that was previously hidden. On GB200 with NVL72, enabling FP8 makes CPU launch latency the dominant bottleneck, necessitating CUDA Graphs and aggressive kernel fusion.

7. **The three-phase optimization workflow is the real practical contribution.** Phase 1: Establish memory-feasible parallelism. Phase 2: Select communication-optimal strategy. Phase 3: Profile and optimize the dominant wall. This iterative process, not any single technique, is how practitioners should approach MoE optimization.

8. **Fine-grained offloading can *increase* throughput by 15%.** Counterintuitively, offloading activations to CPU for Qwen3-235B freed enough GPU memory to reduce TP degree, which eliminated enough TP communication overhead to more than compensate for the PCIe transfer cost. Memory savings enable parallelism changes that improve throughput.

9. **RL post-training amplifies every MoE challenge.** Variable sequence lengths (100 to 1M tokens), interleaved inference/training phases with memory swapping, routing discrepancies between engines, and the need for packed sequences with dynamic CP all make RL the hardest workload for MoE training infrastructure.

10. **Reduced-precision training for MoE requires selective precision — not uniform quantization.** Router in FP32, embeddings/output/optimizer in high precision, bulk expert GEMMs in FP8/FP4. NVFP4 additionally needs RHT, 2D scaling, and stochastic rounding. Quantizing the router causes training instability and expert collapse.

---

## What's Open-Sourced

- **Framework:** Megatron-Core MoE — https://github.com/NVIDIA/Megatron-LM
- **TransformerEngine:** https://github.com/NVIDIA/TransformerEngine (Grouped GEMM, FP8/FP4 kernels)
- **HybridEP:** https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep
- **DeepEP:** https://github.com/deepseek-ai/DeepEP
- **Megatron-Bridge:** HuggingFace ↔ Megatron checkpoint interop for RL frameworks
- **Memory Estimator:** Interactive web GUI for parallelism tuning
- **All benchmark configurations** are documented in Appendix B for reproducibility
- Used in production by: NVIDIA (Nemotron-3), and adopted across academia and industry for models from billions to trillions of parameters
