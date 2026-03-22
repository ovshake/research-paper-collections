# Nemotron 3 Super: Open, Efficient MoE Hybrid Mamba-Transformer Model for Agentic Reasoning

**Authors:** NVIDIA
**Date:** 2026-03-11
**Paper:** [Technical Report (PDF)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

---

## TL;DR

Nemotron 3 Super is a 120B total / 12B active parameter hybrid Mamba-Attention MoE model that matches or beats GPT-OSS-120B and Qwen3.5-122B on benchmarks while delivering **2.2x and 7.5x higher inference throughput** respectively. It combines three architectural innovations — LatentMoE, Multi-Token Prediction (MTP), and hybrid Mamba-Attention interleaving — and is fully open-sourced (base, post-trained, and quantized checkpoints + datasets).

---

## Key Novel Ideas

### 1. LatentMoE — Hardware-Aware Expert Design

The most significant architectural contribution. Standard MoE layers route tokens in the full hidden dimension `d`, which is wasteful for both memory bandwidth (latency-bound) and all-to-all communication (throughput-bound).

**Core insight:** Project tokens down to a latent dimension `l < d` *before* routing and expert computation, then project back up afterward.

- Reduces per-expert weight loads and all-to-all traffic by factor `d/l`
- Uses the savings to increase both total experts (`N' = N * d/l`) and active experts per token (`K' = K * d/l`)
- Net effect: **more expert combinations at the same inference cost**, improving accuracy per byte
- Non-routed computations (gating network, shared experts) stay in full dimension `d` to preserve quality
- Nemotron 3 Super uses: `l = 1024`, `d = 4096`, giving a 4x compression, enabling 512 total experts with top-22 routing

**Why this matters:** Previous MoE scaling focused on accuracy per FLOP. LatentMoE optimizes for accuracy per *byte* (memory bandwidth) and accuracy per *parameter*, which are the actual bottlenecks in real deployment.

### 2. Shared-Weight Multi-Token Prediction (MTP)

Standard MTP uses N independent heads, each predicting a fixed future offset. This limits speculative decoding to N draft tokens, and causes train-inference mismatch when heads are reused autoregressively.

**Nemotron's approach:** Share parameters across all MTP heads so a single head is exposed to multiple offsets during training. Benefits:
- More robust to self-generated hidden states during autoregressive drafting
- Enables longer draft sequences with more stable acceptance rates
- Achieves highest average acceptance length (3.45) on SPEED-Bench, beating DeepSeek-R1 and Qwen3-Next
- No external draft model needed — native speculative decoding built into the architecture
- MTP with draft depth D=3 significantly shifts the throughput-latency Pareto frontier on Blackwell hardware

### 3. Hybrid Mamba-2 + Sparse Attention Architecture

88 layers following a periodic interleaving pattern:
- **Mamba-2 blocks** for efficient linear-time sequence modeling (constant-sized state, no quadratic KV cache growth)
- **Sparse global attention layers** inserted periodically as "anchors" for full-token interaction and long-range routing
- **LatentMoE layers** paired with both block types for sparse scaling

This avoids the quadratic memory cost of full attention while preserving global dependency modeling. Supports up to **1M context length**.

### 4. PivotRL — Efficient Agentic RL

A novel assistant-turn-level RL method for long-horizon agentic tasks:
- Reuses offline SFT expert trajectories during RL instead of requiring full online rollouts
- Focuses training on "pivots" — informative turns where the policy is uncertain
- Uses domain-appropriate reward matching similar (not exact) actions to expert actions
- Avoids both the OOD degradation of SFT and the massive cost of end-to-end RL
- Applied across agentic programming, search, terminal use, and conversational tool use

---

## Architecture Details

| Configuration | Value |
|---|---|
| Total Parameters | 120.6B |
| Active Parameters | 12.7B (12.1B excl. embeddings) |
| Total Layers | 88 |
| Model Dimension | 4096 |
| Q-Heads / KV-Heads | 32 / 2 (GQA) |
| Head Dimension | 128 |
| Mamba State Dim / Groups / Heads | 128 / 8 / 128 |
| Expert Hidden Dim | 2688 |
| Shared Expert Intermediate Size | 5376 |
| Total Experts per Layer | 512 |
| Top-K Active Experts | 22 |
| MoE Latent Size | 1024 |
| MTP Layers (shared weight) | 2 |
| Max Context Length | 1M tokens |

---

## Training Pipeline

### Pre-training (25T tokens, NVFP4 precision)
- **Phase 1 (80%, 20T tokens):** Broad coverage and diversity — web crawl, code, academic, multilingual
- **Phase 2 (20%, 5T tokens):** High-quality sources — Wikipedia, curated crawl, math, code-SFT
- **New synthetic pretraining data:** Code Concepts (15M problems), Unconditional Algorithmic, Economics, Formal Logic, Multiple Choice (~3.5M MMLU-style MCQs)
- **Learning rate:** WSD schedule, peak 4.5e-4, minus-sqrt decay to 4.5e-6 over final 5T tokens
- **Optimizer:** AdamW (weight decay 0.1, beta1=0.9, beta2=0.95)
- **Batch:** 3,072 sequences x 8,192 tokens = ~25.17M tokens/batch
- **First model to be fully pre-trained in NVFP4** (with selective BF16 for stability-critical layers)

### Long-Context Extension
- Continuous pretraining at 1M context length (34B tokens), followed by alternating 1M/4K training (17B tokens)

### Post-training
1. **SFT (7M samples, 80B tokens):** Two-stage loss — token-level average (Stage 1) then sample-level average (Stage 2) to prevent long outputs from dominating. Data blend: Agent 36%, Reasoning 31%, Chat 23%, Long Context 8%
2. **RLVR (3 rounds, 21 environments, 37 RL datasets):** Multi-environment RL from verifiable rewards across math, code, STEM, instruction following, safety, agentic tool use, long context, reasoning
3. **SWE-RL:** Separate stage for end-to-end software engineering — launches Apptainer containers, runs OpenHands agent loops, evaluates against ground-truth tests
4. **RLHF:** GenRM-based (principled, not vanilla) trained on Helpsteer 3 + lmarena-140k + human preferences
5. **MTP Healing (18B tokens):** Retrains MTP heads (weights frozen for backbone) on RLVR prompts with SFT-style loss — significantly recovers MTP accuracy after RL

### Quantization
- **FP8 (W8A8):** For Hopper GPUs
- **NVFP4 (W4A4):** For Blackwell GPUs — uses AutoQuantize (NAS-inspired) for optimal mixed-precision layer assignments
- NVFP4 achieves **99.8% median accuracy** relative to BF16 baseline
- **Mamba SSM cache quantization:** Stochastic rounding (Philox<5>) to FP16 solves verbosity blowup caused by RTNE bias accumulation in recurrent states

---

## Key Results

### vs. GPT-OSS-120B and Qwen3.5-122B

| Benchmark | N-3-Super | Qwen3.5-122B | GPT-OSS-120B |
|---|---|---|---|
| MMLU-Pro | 83.73 | 86.70 | 81.00 |
| AIME25 (no tools) | 90.21 | 90.36 | 92.50 |
| HMMT Feb25 (no tools) | 93.67 | 91.40 | 90.00 |
| LiveCodeBench v5 | 81.19 | 78.93 | 88.00 |
| SWE-Bench (OpenHands) | 60.47 | 66.40 | 41.90 |
| SWE-Bench (Codex) | 53.73 | - | 61.20 |
| Terminal Bench (hard) | 25.78 | 26.80 | 24.00 |
| TauBench V2 Average | 61.15 | 74.53 | 61.00 |
| IFBench (prompt) | 72.56 | 73.77 | 68.32 |
| RULER 1M | 91.75 | 91.33 | 22.30 |
| **Throughput (8k in / 64k out)** | **2.2x GPT-OSS** | **baseline** | **baseline** |

### Base Model vs. Peers (pre-training only)

Nemotron 3 Super Base significantly outperforms Ling-flash-Base-2.0 and GLM-4.5-Air-Base across the board:
- MMLU: 86.01 vs 81.00 vs 81.00
- MATH: 84.84 vs 63.80 vs 50.36
- AIME 2024 (pass@32): 53.33 vs 30.00 vs 20.00
- RULER 1M: 74.39 (only model reporting this)

---

## Key Takeaways

1. **Latent-space MoE is the right abstraction for efficient scaling.** By compressing tokens before routing, you can have dramatically more experts and active parameters at the same serving cost. This is a more principled approach than just increasing expert count in full dimension.

2. **Hybrid Mamba-Attention is production-ready for long contexts.** The constant-state Mamba blocks eliminate KV cache growth for most layers, with sparse attention anchors preserving global reasoning. RULER scores at 512K and 1M context are remarkably strong (95.67 and 91.75).

3. **NVFP4 training at scale is viable.** First model to complete full 25T-token pretraining in NVFP4. Requires careful mixed-precision (BF16 for final 15% of layers, latent projections, MTP, attention QKV, embeddings; MXFP8 for Mamba output projection), but produces models with quality matching BF16 training.

4. **Shared-weight MTP is strictly better than independent heads** for speculative decoding. The regularization from multi-offset exposure makes the head more robust to autoregressive self-conditioning, yielding longer useful draft sequences.

5. **Multi-environment RL prevents catastrophic forgetting.** Training across 21 environments simultaneously yields stable gains, whereas single-environment RL causes severe regressions on other benchmarks. The scale of RL (37 datasets, up to 4,000 environment instances per batch) is notable.

6. **Agentic capability requires dedicated training infrastructure.** The SWE-RL setup (Apptainer containers, OpenHands agent loops, multi-harness training across Claude Code / Codex / OpenCode formats, memory watchdog daemons, command blocklists) reveals how much systems engineering goes into training agentic models.

7. **Two-stage SFT loss matters.** Switching from token-level to sample-level averaging in Stage 2 prevents long reasoning outputs from dominating the loss, restoring short-output performance without losing reasoning capability.

8. **Checkpoint merging saves ~16% of pretraining compute.** Weighted averaging over sliding windows of checkpoints during the stable LR phase produces 2-4 point improvements on benchmarks, eliminating the need for expensive intermediate decay runs.

9. **Mamba state quantization has unique challenges.** Unlike attention KV caches, Mamba SSM caches are recurrent — quantization errors accumulate over time. RTNE introduces systematic bias; stochastic rounding eliminates this, solving a 40% verbosity blowup issue at no accuracy cost.

10. **Open-source MoE models can match proprietary dense models.** With only 12B active parameters, Nemotron 3 Super is competitive with 120B+ dense models, demonstrating that the MoE + hybrid architecture approach can deliver frontier-class capabilities at dramatically lower serving costs.

---

## What's Open-Sourced

- **Checkpoints:** Base (BF16), Post-trained (BF16, FP8, NVFP4)
- **Datasets:** Nemotron-Pretraining-Specialized-v1.1, Nemotron-Super-Post-Training-Data (RL environments + SFT)
- **GenRM:** Qwen3-Nemotron-235B-A22B-GenRM-2603
- **Training recipe:** github.com/NVIDIA-NeMo/Nemotron
- **Evaluation configs:** Nemo Evaluator SDK
