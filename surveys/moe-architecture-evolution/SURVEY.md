# Mixture-of-Experts in LLMs: From Mixtral to 2026
## A Survey of Architectures, Training Recipes, and Serving Systems Since the Open MoE Inflection Point

**Compiled:** May 2, 2026
**Scope:** Open-weight MoE LLMs and their accompanying techniques from Mixtral 8x7B (Dec 2023) through early 2026. Coverage spans architecture, pretraining, post-training (SFT/RLHF/RLVR), and inference systems. Pre-Mixtral work (GShard, Switch Transformer, ST-MoE, GLaM) appears only as cited background.

---

## 1. The Central Tradeoff, Stated Precisely

A dense Transformer has one knob: parameters. Compute, memory, and capacity all scale with that one number. A Mixture-of-Experts Transformer has **two**, and they pull in opposite directions:

- **Active parameters** (the experts a single token actually uses) determine *compute per token* — your prefill TFLOPs, your decode latency, your training FLOP/token.
- **Total parameters** (every expert in every layer) determine *memory pressure* — your HBM footprint, your KV cache pinning, your expert-parallel all-to-all traffic.

This decoupling is the entire point of MoE: a 671B-parameter DeepSeek-V3 trains and serves at the cost-per-token of a ~37B dense model, but it knows what a 671B model knows. Every architectural decision in this survey — top-k, expert granularity, shared experts, MLA vs GQA, FP8, EP topology — is a knob on that tradeoff. The two-year arc from Mixtral 8x7B to Kimi K2 is essentially the story of the community learning **how far apart these two numbers can be pulled before something breaks**: from a sparsity ratio of ~28% (Mixtral) to ~3% (Kimi K2, GPT-OSS) — a 9× compression in two years.

What "breaks" along the way is a closely-watched list:

- **Routing collapse:** the gate funnels all tokens through 1-2 experts; the rest go dead.
- **Bandwidth-bound decode:** activating 8 of 256 experts per token still requires loading 8 expert weight matrices from HBM, so single-stream decode goes from compute-bound (dense) to memory-bandwidth-bound (sparse).
- **Train-inference router skew:** the router that picks experts during a training rollout disagrees with the router that picks experts during the inference replay, breaking on-policy RL.
- **Auxiliary-loss-vs-objective conflict:** the load-balancing loss that prevents collapse pulls gradients away from the language-modeling (or RL) objective.
- **Long-tail expert starvation:** experts that fire 1% of the time get 1% of the gradient and end up under-trained.

Sections 4 onward expand each of these failure modes and the techniques that address them.

---

## 2. A Unified Taxonomy of MoE Design Dimensions

Six axes capture nearly every architectural decision in modern MoE. Reading any new MoE paper, you can locate it on these six axes within a few minutes.

| Axis | Choice | Examples | What changing it costs/buys |
|---|---|---|---|
| **A. Expert granularity** | Coarse (8) vs. fine-grained (64-384) | Mixtral 8 / DeepSeek-V3 256 / Kimi K2 384 | Finer = more combinations per token, sharper specialization, but harder routing and more all-to-all traffic |
| **B. Expert sharing** | Pure routed vs. shared+routed | Mixtral / DeepSeek-V3 (1 shared) / Qwen3 (none) | Shared experts absorb common knowledge so routed experts can specialize; cost is fixed per-token compute |
| **C. Routing policy** | Token-choice top-k softmax / sigmoid / aux-loss-free / expert-choice | Mixtral / DeepSeek-V3 / Qwen3 / Lory | Drives load balance, expressivity, and inference-time predictability |
| **D. Load balancing** | Aux loss / aux-loss-free bias / GLN / global-batch | Switch / DeepSeek-V3 / Skywork / Qwen3 | Tradeoff between objective-conflict and balance guarantees |
| **E. Attention compression** | MHA / GQA / MLA / MFA / linear-hybrid | Mixtral / Llama 4 / DeepSeek-V2 / Step-3 / MiniMax-01 | KV cache size at long context — the single biggest serving lever after MoE itself |
| **F. Sparsity ratio (active/total)** | ~28% (early) → ~3% (frontier) | Mixtral 28% → Kimi K2 3.1% | Compute-vs-memory tradeoff; ratios &lt;5% require sophisticated EP and load balancing |

The remainder of this survey works through these dimensions and how each was settled (or remains contested).

---

## 3. Timeline of Major Open-Weight MoE Releases

| Date | Model | Total / Active | # Experts × top-k | Attention | Sparsity | Notable |
|---|---|---|---|---|---|---|
| Dec 2023 | **Mixtral 8x7B** | 46.7B / 12.9B | 8 × 2 | GQA + SWA | 28% | First widely accessible open MoE |
| Jan 2024 | **DeepSeekMoE 16B** | 16.4B / 2.8B | 64 × 6 + 2 shared | MHA | 17% | Fine-grained + shared experts |
| Feb 2024 | **Qwen1.5-MoE-A2.7B** | 14.3B / 2.7B | 60 × 4 + 4 shared | GQA | 19% | Upcycled from Qwen 1.8B |
| Mar 2024 | **Grok-1** | 314B / 79B | 8 × 2 | RoPE/RMS | 25% | Largest open release at time |
| Mar 2024 | **DBRX** | 132B / 36B | 16 × 4 | GQA | 27% | First "fine-grained" Western MoE |
| Apr 2024 | **Mixtral 8x22B** | 141B / 39B | 8 × 2 | GQA | 28% | Drops SWA, 64K context |
| Apr 2024 | **Snowflake Arctic** | 480B / 17B | 128 × 2 + 10B dense | GQA | 3.5% | Dense-MoE hybrid |
| May 2024 | **DeepSeek-V2** | 236B / 21B | 160 × 6 + 2 shared | **MLA** | 8.9% | MLA introduced; 93% KV cache cut |
| Jun 2024 | **Skywork-MoE** | 146B / 22B | 16 × 2 | MHA | 15% | Gating logit normalization |
| Sep 2024 | **OLMoE 1B-7B** | 6.9B / 1.3B | 64 × 8 | MHA | 19% | Fully open data + code + weights |
| Nov 2024 | **Hunyuan-Large** | 389B / 52B | 16 × 1 + 1 shared | GQA + cross-layer | 13% | Recycle routing; 256K ctx |
| Dec 2024 | **DeepSeek-V3** | 671B / 37B | 256 × 8 + 1 shared | MLA | 5.5% | Aux-loss-free, MTP, FP8, $5.6M |
| Jan 2025 | **DeepSeek-R1** | 671B / 37B | (V3 base) | MLA | 5.5% | Pure-RL reasoning emergence |
| Jan 2025 | **MiniMax-Text-01** | 456B / 45.9B | 32 × 2 | 7:1 lightning:softmax | 10% | 1M-token training context |
| Apr 2025 | **Qwen3-235B-A22B** | 235B / 22B | 128 × 8 (no shared) | GQA | 9.4% | Aux-loss-free; thinking mode |
| Apr 2025 | **Llama 4 Maverick** | 400B / 17B | 128 × 1 + 1 shared | iRoPE GQA | 4.3% | 1:1 dense/MoE alternation |
| Apr 2025 | **Llama 4 Scout** | 109B / 17B | 16 × 1 + 1 shared | iRoPE GQA | 16% | 10M context |
| Jul 2025 | **Kimi K2** | 1.04T / 32B | 384 × 8 + 1 shared | MLA (64 heads) | 3.1% | Muon + QK-Clip; zero loss spike |
| Jul 2025 | **Step-3** | 321B / 38B | — | MFA (22% of V3 attn cost) | 12% | Hardware-aware co-design |
| Jul 2025 | **GLM-4.5** | 355B / 32B | — | GQA + MTP | 9% | Muon optimizer; depth>width |
| Aug 2025 | **GPT-OSS-120b** | 117B / 5.1B | 128 × 4 | GQA + banded sparse | 4.4% | Native MXFP4; 80GB inference |
| Aug 2025 | **GPT-OSS-20b** | 21B / 3.6B | 32 × 4 | GQA + banded sparse | 17% | 16GB VRAM-class |
| Dec 2025 | **Mistral Large 3** | 675B / 41B | granular | (undisclosed) | 6.1% | Returns to MoE after Large 2 dense |
| Feb 2026 | **Step-3.5 Flash** | 196B / ~11B | 288 × 8 + 1 shared | MFA | 5.6% | MTP-3 (3 tokens/step) |

A few observations from the table itself:

- Active-parameter counts are clustering hard at **17–37B** across very different total sizes. The community has largely settled this band as the sweet spot for serving on a single 8-GPU node.
- Sparsity ratio compressed from 28% (Mixtral) to 3% (Kimi K2) over 19 months — roughly a 9× reduction. The compression is now slowing.
- Three attention regimes coexist as of 2026: GQA (most), MLA (DeepSeek/Kimi), and exotic compressions (MFA in Step-3, lightning-attention hybrid in MiniMax).
- The "shared experts: yes/no" question is genuinely unsettled. The two highest-performing open MoEs of mid-2025 (Qwen3 and Kimi K2) disagree on it.

---

## 4. The 2024 Inflection — Mixtral 8x7B

Before Mixtral, every prominent MoE LLM was a closed Google project (GShard, Switch Transformer, ST-MoE, GLaM). The community was reading papers but not running models. Mixtral changed this by torrenting weights at NeurIPS December 2023, with the paper ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)) following in January.

### The architecture, exactly

- 8 experts per layer, top-2 routed by `softmax(W_router · h_t)` over the 8 experts
- Each expert is a SwiGLU FFN with `intermediate_size=14336` over `hidden_size=4096`
- Attention is shared across experts: 32 heads, 8 KV heads (GQA), sliding-window attention with W=4096
- 32 layers, 32K context, 32K vocab byte-fallback BPE
- `router_aux_loss_coef = 0.001`, no jitter noise
- 8x22B (April 2024) scaled to 141B/39B with 6144 hidden, 16384 intermediate, 56 layers, drops SWA in favor of full attention with RoPE-θ=1M and 64K context

### The "8x7B = 56B" misconception

A canonical pedagogical point: **only the FFN blocks are replicated 8×, not attention or embeddings**. Per expert, the FFN holds `3 × 4096 × 14336 ≈ 176M` params; across 8 experts × 32 layers that's ~45B, plus ~1.7B shared attention/embedding/output → **~46.7B total**, not 56B. Active per token is roughly 12.9B (2 experts × ~176M × 32 layers + shared). This single point is the most-cited MoE gotcha in the community and shapes how every later report describes parameter counts.

### The expert-specialization surprise

Section 4 of the Mixtral paper measured token-to-expert assignment over Pile validation subsets (ArXiv, biology, philosophy, math). The headline finding, surprising at the time:

> "We do not observe obvious patterns in the assignment of experts based on the topic… We hypothesize that the expert selection appears to exhibit some syntactic behaviour at the lower layers."

What experts *did* learn was syntactic locality: the same Python `self`, the same English `Question`, indentation tokens, all routed consistently. Consecutive tokens routed to the same expert with positional locality far above random.

This single observation reframed the field. Through 2022–2023 the implicit assumption was that experts would carve up the data semantically (a "math expert," a "code expert"). They don't, at least not at granularity 8. OpenMoE ([arXiv:2402.01739](https://arxiv.org/abs/2402.01739), Feb 2024) later named this **"context-independent specialization"** — the same token ID routes to the same expert regardless of context, with assignments determined early in pretraining and frozen thereafter. OLMoE ([arXiv:2409.02060](https://arxiv.org/abs/2409.02060), Sep 2024) found that at granularity 64, experts *do* show domain and vocabulary specialization. **The lesson: specialization is a function of granularity, not of the MoE idea itself.**

### Was Mixtral upcycled?

Mistral never confirmed. But a "Closer Look" study ([Lo et al., arXiv:2406.18219](https://arxiv.org/abs/2406.18219)) found weight evidence:

- Expert-to-expert cosine similarity of 0.2–0.4 (elevated)
- **Expert-to-Mistral-7B-FFN similarity ~0.6**, with last-two-layer values >0.8
- DeepSeekMoE and Grok-1 (known from-scratch) showed expert similarity near zero
- PCA: half of Mixtral's experts cluster around the Mistral-7B FFN

The community's working belief is that Mixtral was upcycled from Mistral 7B. This matters because **OLMoE later showed upcycling only wins for short training runs** (see §6). Mixtral was a great proof-of-concept, but not the right recipe for a long pretraining run.

---

## 5. The DeepSeek Arc — January 2024 to January 2025

DeepSeek shipped four MoE papers in 13 months that, in aggregate, defined the modern frontier. Each paper attacked one bottleneck and inherited the previous paper's solutions.

### 5.1 DeepSeekMoE 16B — fine-grained + shared experts

[Paper: arXiv:2401.06066](https://arxiv.org/abs/2401.06066), January 2024. 16.4B total / 2.8B active.

Two innovations, both adopted broadly:

**Fine-grained expert segmentation.** Replace 16 standard FFN experts with 64 quarter-size experts (`m=4` segmentation factor), activate top-6 instead of top-2. The combinatorics: `C(16,2)=28` versus `C(64,6)≈74M` token-to-expert combinations per layer. The paper:

> "Even with only 4 routed experts activated, DeepSeekMoE achieves a Pile loss comparable with GShard."

**Shared expert isolation.** Two experts per layer are *always* active (no gate). They absorb common knowledge — syntax, frequent tokens, generic structure — freeing the routed experts to specialize. The paper frames this as fixing two problems:

- **Knowledge hybridity:** one expert covers too many domains because there aren't enough experts
- **Knowledge redundancy:** multiple experts independently learn the same common patterns

The combined design matched LLaMA2-7B on 15/20 benchmarks at **39.6% of LLaMA2-7B's compute**. Against GShard at the 145B scale, it matched DeepSeek 67B dense using only 28.5% of compute.

### 5.2 DeepSeek-V2 — MLA and the KV-cache revolution

[Paper: arXiv:2405.04434](https://arxiv.org/abs/2405.04434), May 2024. 236B / 21B; 160 routed + 2 shared, top-6.

V2's headline contribution is **Multi-head Latent Attention (MLA)** — a low-rank compression of K and V into a shared latent vector before storing in the KV cache:

```
c_t^KV = W^DKV · h_t        (down-project to d_c=512)
k_t^C  = W^UK  · c_t^KV      (up-project for keys)
v_t^C  = W^UV  · c_t^KV      (up-project for values)
```

The KV cache stores only `c_t^KV` (512) plus a decoupled-RoPE component `k_t^R` (64) — **576 elements per token**. Compare against:

| Mechanism | KV per token (V2 dims) |
|---|---|
| MHA (128 heads × d_h=128) | 32,768 |
| GQA (8 groups × d_h=128) | 2,048 |
| **MLA** | **576** |

That's a **93.3% KV-cache reduction** versus the MHA in DeepSeek 67B, translating to **5.76× max generation throughput** on 8×H800. Critically, ablations (Appendix C.2) showed MLA *outperforms* MHA at matched compute — the joint compression acts as a useful bottleneck, and the decoupled-RoPE component preserves positional information that low-rank K compression would otherwise blur.

V2 also introduced **device-limited routing**: with experts spread across D=8 devices, each token first picks M=3 devices, then top-6 within those — capping cross-node all-to-all traffic. Three balance losses (expert α=0.003, device α=0.05, communication α=0.02). At inference, V2 hit MMLU 78.5, GSM8K 79.2, MATH 43.6 — matching LLaMA-3-70B on most English benchmarks at 42.5% the training cost of DeepSeek 67B.

### 5.3 DeepSeek-V3 — three innovations, one budget

[Paper: arXiv:2412.19437](https://arxiv.org/abs/2412.19437), December 2024. 671B / 37B; 256 routed + 1 shared, top-8. 14.8T training tokens, **2.788M H800 GPU-hours, $5.576M**.

V3 stacks three innovations on top of V2's MLA, each independently important.

**(i) Auxiliary-loss-free load balancing** ([Wang et al., arXiv:2408.15664](https://arxiv.org/abs/2408.15664)). A per-expert bias `b_i` is added to router scores **for top-k selection only** (not for the gating weights used to combine outputs). The bias is updated *outside* backpropagation:

```
if expert_i overloaded:   b_i ← b_i − γ
if expert_i underloaded:  b_i ← b_i + γ
```

with γ=0.001 for the first 14.3T tokens, then frozen. The structural advantage: **the language-modeling gradient never sees the balance mechanism**. A small sequence-level balance loss (α=0.0001) remains as a safety net for extreme imbalance. The result: V3 trained 14.8T tokens with no rollbacks and no irrecoverable loss spikes.

**(ii) Multi-Token Prediction (MTP)** ([§2.2 of V3 paper](https://arxiv.org/html/2412.19437v1)). Predicts D=1 additional token via *sequential* modules, deliberately differing from Meta's parallel-head MTP ([Gloeckle et al. 2024](https://arxiv.org/abs/2404.19737)):

> "Different from Gloeckle et al., which parallelly predicts D additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth."

Sequential prediction preserves causality, so the same MTP heads can serve at inference as a speculative decoder. Acceptance rate >80%, **~1.8× generation throughput**.

**(iii) FP8 mixed-precision training.** First publicly validated FP8 run at 671B scale. GEMM is FP8 (E4M3); embedding, output head, gating, normalization, and attention stay BF16/FP32. Activations quantized at 1×128 tile granularity, weights at 128×128 block granularity. Critically, FP32 accumulation is promoted every 128 elements to dodge H800's ~14-bit Tensor Core accumulator limit. Validation on smaller scales: relative loss error vs BF16 stays below 0.25%.

**Infrastructure (DualPipe + custom EP):** pipeline parallelism splits each chunk into attention / dispatch / MLP / combine, with PP and all-to-all both fully hidden during execution. Pipeline bubble reduces from `(PP-1)(F+B)` to `(PP/2-1)(F&B+B-3W)`. Custom all-to-all kernels using warp specialization on 20 SMs, achieving >40 GB/s with 8×400 Gbps InfiniBand. Open-sourced as [DualPipe](https://github.com/deepseek-ai/DualPipe) (Feb 2025) alongside [DeepEP](https://github.com/deepseek-ai/DeepEP) and [EPLB](https://github.com/deepseek-ai/eplb).

**The cost asterisk.** The $5.576M covers only the final pretraining run. SemiAnalysis estimated DeepSeek's total server CapEx >$1.5B; Interconnects estimated annual operations >$500M including 139+ technical authors and >10K GPUs amortized. The genuine, defensible claim is **12× fewer GPU-hours than Llama 3 405B for the final run** — driven by MoE sparsity, MLA, FP8, and DualPipe. The misleading framing was implying that's "the cost of building the model."

### 5.4 DeepSeek-R1 — reasoning emerges from RL on a sparse base

[Paper: arXiv:2501.12948](https://arxiv.org/abs/2501.12948), January 2025. Architecture identical to V3.

The pipeline: V3-Base → cold-start SFT (~thousands of curated long-CoT examples) → reasoning RL with GRPO → rejection-sampling SFT (~600K reasoning + ~200K general) → multi-scenario RL.

**R1-Zero** is the variant that skipped the cold-start SFT — pure GRPO on V3-Base. AIME 2024 pass@1 climbed from 15.6% → 71.0% over training; consensus@64 reached 86.7%, surpassing o1-mini's 80%. Self-reflection, verification, and "aha moment" backtracking emerged without any explicit CoT supervision. The R1-Zero failure modes (poor readability, language mixing) are *behavioral*, not routing-specific — a key point for §8.

**Distillation to dense students** (Qwen 1.5/7/14/32B; Llama 8/70B) used 800K R1 reasoning traces as SFT data. Striking finding from the paper:

> "Distilling more powerful models into smaller ones yields better performance than RL on small models."

R1-Distill-Qwen-32B (72.6% AIME 2024) vastly outperformed the same Qwen-32B trained with RL alone (47%). This validated MoE-as-teacher: a sparse 671B model produces traces good enough to lift dense 32B students well past dense-RL alternatives.

### What the DeepSeek arc settled

| Claim | Status as of mid-2026 |
|---|---|
| Fine-grained experts (64-256+ per layer) beat coarse-grained | **Consensus.** Adopted by Qwen3, OLMoE, Llama 4 Maverick, GPT-OSS, Kimi K2, etc. |
| Shared experts help | **Contested.** DeepSeek/Kimi/Hunyuan/Llama 4 use them; Qwen3 and OLMoE explicitly drop them |
| MLA is the right attention compression | **Adopted by some, ignored by most.** DeepSeek-V2/V3, Kimi K2, DeepSeek-VL2 use MLA. Qwen3, Llama 4, GPT-OSS, MiniMax, Granite, Mistral Large 3 use GQA. |
| Aux-loss-free routing > aux loss | **Trending toward consensus.** Qwen3 uses a global-batch variant rather than DeepSeek's bias method, but the *direction* (decouple balance from objective gradient) is universal. |
| FP8 training is production-ready | **Tentative consensus.** Validated at 671B by V3; not yet independently reproduced at scale, but Hopper/Blackwell hardware paths are aligned. |
| You can RL a sparse base directly into a reasoning model | **Confirmed and replicated.** Qwen3, GLM-4.5, Nemotron 3 Nano, Kimi K2 all do variants of this pipeline. |

---

## 6. Routing & Load Balancing

This is the dimension where the most papers have shipped and the consensus is most fragmented.

### 6.1 Routing mechanisms

**Token-choice top-k softmax** is the default — Switch Transformer, Mixtral, Qwen3, OLMoE, every model in the table. The Cerebras "Router Wars" guide observed that learned token-choice routing delivers ~4% loss improvement vs Chinchilla-optimal dense at 16 experts, vs ~1.5% for hash routing. **If your learned router doesn't beat hashing, your training is broken.**

**Expert-choice routing** ([Zhou et al. 2022, arXiv:2202.09368](https://arxiv.org/abs/2202.09368)) inverts the relationship: experts pick top-k tokens, guaranteeing perfect load balance. The fundamental obstacle for autoregressive decoders: it requires a **global view of all tokens in a sequence** to compute expert preferences, which violates causality. Lory ([arXiv:2405.03133](https://arxiv.org/abs/2405.03133)) tried causal segment routing for autoregressive models — improved over dense but underperformed vanilla TopK MoE. Diffusion language models can use expert-choice freely (it's "strictly superior" there per [arXiv:2604.01622]) but that's a different generative regime. **Status: expert-choice remains impractical for production decoders.**

**Sigmoid gating** (DeepSeek-V3) replaces softmax for affinity-score computation, then normalizes across the selected experts. The structural reason: softmax couples experts (raising one lowers all others), so a per-expert bias trick like DeepSeek's would redistribute probability mass to *every* expert when adjusting one. Sigmoid produces independent per-expert scores, making bias-based load balancing tractable.

**ReMoE** ([arXiv:2412.14711](https://arxiv.org/abs/2412.14711), ICLR 2025) replaces top-k with ReLU-based routing — fully differentiable, allows variable experts per token (hard tokens get more compute), and can route to *zero* experts (skip). On LLaMA architecture it outperformed both top-k and Lory. Not yet adopted by major releases but the most credible non-top-k contender.

**Soft MoE / continuous MoE** ([Puigcerver et al., arXiv:2308.00951](https://arxiv.org/abs/2308.00951)) mixes tokens softly across experts. Critical limitation: breaks token causality; not usable for autoregressive LMs.

### 6.2 Load balancing — four families

The auxiliary load-balancing loss (Switch Transformer's `L_aux ∝ Σ f_i · P_i` where `f_i` is the fraction of tokens routed to expert `i`, `P_i` the average gating probability) is notoriously sensitive. Too high and it crushes the LM objective; too low and routing collapses. The community has shipped four families of fixes:

**(A) Aux-loss-free bias** (DeepSeek-V3, [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)). Bias term updated outside the gradient; small sequence-level safety loss. The cleanest from a gradient-flow perspective. **Pitfall:** if the bias update rate γ is too large, gating thrashes; too small, slow to adapt.

**(B) Gating Logit Normalization** (Skywork-MoE, [arXiv:2406.06563](https://arxiv.org/abs/2406.06563)). Normalize router logits before softmax: `z̃ = λ(z − μ)/σ`. Validated on a 2.5B/16-expert model: without normalization, expert activation ratios converge to uniform (= no specialization). Pairs with an *adaptive* per-layer auxiliary loss coefficient: layers that are already balanced get lighter penalty; layers dropping tokens get heavier.

**(C) Global-batch load balancing** (Qwen3). Compute `f_i` and `P_i` over the entire global batch, not per-sequence. Reduces noise from short sequences. The Qwen3 technical report ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) prefers this over DeepSeek's bias method; the two approaches achieve similar balance with different gradient profiles.

**(D) Recycle routing** (Hunyuan-Large). Tokens not assigned to any top-k expert are *recycled* rather than dropped, with expert-specific learning-rate scaling so every token contributes to training.

**Router z-loss** ([ST-MoE, arXiv:2202.08906](https://arxiv.org/abs/2202.08906)) is orthogonal: `L_z = (1/B) Σ (log Σ exp(x_ij))²`. Penalizes large router logits to prevent BF16 numerical instability. Coefficient ~1e-3. OLMoE uses 0.001 alongside an LB loss of 0.01.

### 6.3 Token dropping and capacity factor

Capacity factor: each expert has a buffer `C = CF × tokens / num_experts`; tokens beyond capacity are dropped. GLaM used CF=1.25; ST-MoE used 2.0 at eval. **MegaBlocks** ([arXiv:2211.15841](https://arxiv.org/abs/2211.15841)) reformulated MoE as block-sparse matrix ops, eliminating capacity factor entirely with ~98.6% of cuBLAS throughput. DeepSeek-V3 uses dropless training. The community has converged: **dropless is the default; capacity-factor tuning is legacy.**

---

## 7. Sparse Upcycling vs From-Scratch — and Scaling Laws

### 7.1 Upcycling: the question Mistral never answered

[Komatsuzaki et al., arXiv:2212.05055](https://arxiv.org/abs/2212.05055) (Dec 2022) proposed upcycling: clone a dense model's FFN into N expert slots, randomly initialize the router, continue pretraining. Original claim: upcycled MoE outperforms from-scratch sparse using only ~50% of dense pretraining compute, and beats from-scratch at 100% of dense compute.

**OLMoE's revision** ([arXiv:2409.02060](https://arxiv.org/abs/2409.02060)). Upcycled OLMo-1B (2T tokens) into 8-expert/top-2 MoE, trained 610B more tokens. Findings:

- From-scratch MoE catches up at **~500B tokens** (25% of dense compute) — much faster than Komatsuzaki's 120%
- From-scratch *outperforms* upcycled by ~600B tokens
- MoE reaches dense quality with **3× fewer tokens, 3× fewer FLOPs**
- Wall-clock speedup is only 2× (memory overhead: 23.6K vs 37.5K tok/s/GPU)
- Upcycling locks you into the dense model's hyperparameters (e.g., pre-QK-Norm initialization)

OLMoE chose from-scratch specifically because they planned to train 250% of the original dense budget, where upcycling would be a net loss.

**Drop-Upcycling** ([arXiv:2502.19261](https://arxiv.org/abs/2502.19261), Feb 2025) splits the difference: upcycle but partially re-initialize expert weights to encourage diversification. A 5.9B-active MoE matched a 13B dense model at ~25% of the training FLOPs. Reopens the upcycling question, suggesting the answer is **upcycle for &lt;100% of original budget; from-scratch beyond that**, with partial re-initialization shifting that boundary.

### 7.2 Granularity ablations

OLMoE's most quotable result: increasing experts from 8 to 32 (with proportionally increased top-k to keep active params constant) yielded ~10% improvement on HellaSwag and MMLU; 32 → 64 added 1-2% more. Diminishing returns set in around 64 experts at the 1B-active scale.

[Krajewski et al. 2024 (arXiv:2402.07871)](https://arxiv.org/abs/2402.07871) formalized this with **scaling laws for fine-grained MoE**. The fitted form:

```
L(N, D, G) = c + (g·G^(-γ) + a) · N^(-α) + b · D^(-β)
```

with `G` the granularity (per-expert FFN size as fraction of standard FFN), `N` active params, `D` tokens. At E=64 experts the fit gives c=0.47, α=0.115, β=0.147, γ=0.58. The load-bearing claim:

> "The common practice of setting the size of experts in MoE to mirror the feed-forward layer is not optimal at almost any computational budget."

The optimal granularity *grows with scale*: G≈8 at ~3×10¹⁸ FLOPs, G≈32 at 7B-active scale, G≈64 at 300B-active scale. And the MoE-vs-dense efficiency gap *widens* with scale: ~20× FLOP reduction at 10²⁰, **>40× at 10²⁵**. This is the strongest quantitative argument for MoE at frontier scale.

### 7.3 Inference-optimal MoE

[Yun et al. 2024, arXiv:2404.02852](https://arxiv.org/abs/2404.02852) introduced inference cost as a variable. **MoEs with 4-8 experts are most serving-efficient** for matched quality, but cost 2.5-3.5× more to train. The inference-optimal recipe: train a 16/32-expert MoE that's 70-85% smaller than the loss-optimal solution, on a larger token budget. This is the Chinchilla analog for MoE inference economics — and explains why we see flagships clustering at 17-37B active.

### 7.4 Pre-training infrastructure innovations

- **MegaBlocks** (2022, [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)) — block-sparse MoE kernels, dropless, ~98.6% cuBLAS throughput. Up to 40% faster than Tutel, 2.4× over Megatron-LM.
- **DeepSpeed-MoE** — five forms of parallelism, GPU+CPU memory exploitation.
- **Tutel** (Microsoft) — adaptive parallelism with optimized All-to-All; ~40% over Fairseq MoE.
- **DualPipe** (DeepSeek-V3) — bidirectional pipeline; ~50% bubble reduction at PP=16.
- **MegaScale** ([ByteDance, arXiv:2402.15627](https://arxiv.org/abs/2402.15627)) — 55.2% MFU on 175B at 12,288 GPUs (1.34× over Megatron-LM); not MoE-specific but the substrate.

### 7.5 Expert specialization: now a settled question

The Mixtral "no domain specialization" finding plus OpenMoE's "context-independent specialization" plus OLMoE's "domain specialization at finer granularity" plus DeepSeekMoE's "ultimate expert specialization" appear contradictory but resolve cleanly:

- **At granularity 8** (Mixtral, Grok), experts learn syntactic/positional patterns, not domain specialization.
- **At granularity 32-64** (OLMoE), experts begin showing domain and vocabulary specialization, with assignments fixed early in pretraining.
- **At granularity 128-256+** (DeepSeek-V3), experts are concentrated enough to specialize meaningfully — DeepSeekMoE's [Lo et al. follow-up](https://arxiv.org/abs/2406.18219) confirmed dramatic expert concentration (1-3 experts handle the majority of routing per domain).

The corollary: **the upper layers in MoE models tend to specialize more than the middle**, and **the very last layer often shows reduced specialization** (similarities pop back up), suggesting the last MoE layer may not pull its weight.

### 7.6 Long-context and multimodal MoE

Surprisingly understudied. DeepSeek-V3 extended to 128K with standard YaRN-style techniques and reported no MoE-specific interactions. **DeepSeek-VL2** ([arXiv:2412.10302](https://arxiv.org/abs/2412.10302)) uses DeepSeekMoE + MLA for the language backbone with a dynamic-tiling vision encoder; SOTA on OCR/document tasks at 1.0-4.5B active. **MiniMax-Text-01** uses 7:1 lightning:softmax attention to scale MoE to 1M training context, 4M extrapolation.

---

## 8. Post-Training MoE — SFT, DPO, and the RL Routing Crisis

This is where the gap between dense and MoE post-training is sharpest. Two 2025 papers — **R3** ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)) and **RSPO** ([arXiv:2510.23027](https://arxiv.org/abs/2510.23027)) — quantified MoE-specific RL failure modes. The picture is grim and the fixes are recent.

### 8.1 SFT on MoE

**Routing is surprisingly stable through SFT — if you don't push it.** OLMoE measured load-balancing loss before vs. after SFT: it *decreased* (12.22 → 12.16). Removing the LB loss during SFT actually *helped* (54.0 vs 52.8); same for DPO (57.7 vs 57.1). Routing distributions "remain around the same" through both stages.

**But push the router and it breaks.** ExpertCondenser ([arXiv:2604.23036](https://arxiv.org/abs/2604.23036), CMU/MIT) shows standard SFT and the popular DenseMixer technique "significantly alter routing distributions," while bias-only adaptation preserves the base model's routing. They identify a structural problem: rarely-activated long-tail experts encode indispensable knowledge — pruning the top 75% of experts by activation still loses >10% performance — but during SFT they suffer **gradient starvation**, getting only `O(T·p_i)` updates where `p_i` is their activation rate.

**Recipes from open releases:**
- **DeepSeek-V3-Instruct:** SFT data largely distilled from R1 series; SFT+RL combined cost only ~5K H800 GPU-hours. Aux-loss-free bias frozen at γ=0 for the final 500B tokens.
- **Qwen3-235B-A22B-Instruct:** four-stage pipeline (long-CoT cold-start → reasoning GRPO → thinking-mode-fusion SFT → general RL). 170 GRPO steps lifted AIME 2024 from 70.1 to 85.1. **No MoE-specific modifications disclosed** — same recipe runs on dense Qwen3 too.
- **OLMoE-Instruct:** 608K SFT samples, 60.8K DPO samples (UltraFeedback). LB loss dropped during adaptation. SFT alone gave >10× gain on GSM8K.
- **Nemotron 3 Nano (NVIDIA):** Full SFT + RLVR + RLHF on 30B-A3B MoE; introduced *Group Relative Length Control* during RLHF to suppress redundant thinking.

**LoRA on MoE base models** has two practical concerns: (1) target both expert FFNs and the router; (2) avoid 8-bit quantization, which can cause routing to collapse to a handful of experts (4-bit with AWQ is the safer floor). Variance is high — practitioners run multiple seeds.

### 8.2 The RL routing crisis, quantified

R3 ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)) measured train-vs-inference router agreement on Qwen3-30B-A3B during GRPO. The numbers:

- **~10% of routers** select different experts in training vs inference passes
- **94% of tokens** have at least one layer where train- and inference-pass routing disagree
- Per-token mean: **~6 routers disagree**
- Train-inference KL divergence: **1.535×10⁻³** for MoE Qwen3-30B-A3B vs **6.4×10⁻⁴** for dense Qwen3-8B — **2.4× gap**
- Without a fix, **3 of their RL runs crashed**, accompanied by abnormally high KL and `F(τ=2)` values (fraction of tokens with >2× probability ratio)

The root cause: a routing decision changes which subset of expert weights produces the policy distribution, which means the same input under the same model can produce different policies depending on inference-vs-training execution paths (FlashAttention quirks, batching, kernel selection). For dense models, this drift is small; for sparse routers it can flip the entire computation graph.

### 8.3 The four families of RL-on-MoE fixes

**(A) Aux-loss-free routing inherited from pretraining** (DeepSeek-R1). The bias mechanism keeps balance pressure off the gradient, dampening the auxiliary-loss-vs-policy-gradient conflict. R1's paper doesn't discuss routing collapse — likely because V3-Base inherits this property.

**(B) Rollout Routing Replay (R3)** ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)). Record routing distributions during inference rollouts, replay them during the training pass. Reduces train-inference KL from 1.5×10⁻³ to 7.5×10⁻⁴ on Qwen3-30B-A3B. Eliminates the run crashes the authors observed without it.

**(C) Router-Shift Policy Optimization (RSPO)** ([arXiv:2510.23027](https://arxiv.org/abs/2510.23027), Microsoft + PKU). Compute a per-token "router shift ratio" `γ_{i,t}` that softly rescales importance-sampling weights based on routing deviation. **Stop-gradient on this ratio is essential — without it, training collapses at ~40 steps.** Tests of three rigid alternatives (freeze router weights, copy old-policy logits, reuse expert indices) all underperform — the conclusion is "rigid control over the router is suboptimal."

**(D) Modular post-training (BAR)** ([arXiv:2604.18473](https://arxiv.org/abs/2604.18473), Allen AI, Apr 2026). Branch a base model, train domain-specific 2-expert MoEs (anchor + domain expert) per skill, route at inference. Outperforms BTX (49.1 vs 46.7 at 7B). Critical operational finding:

> "RL requires substantially more unfreezing of shared layers — particularly attention — because RL induces distributional shifts that extend beyond what expert FFNs alone can accommodate."

Upgrading a single domain expert (code v1 → v2) gave +16.5 points with other domains stable — **true modular MoE**.

### 8.4 The reward-hacking-via-routing failure mode

[RASA (arXiv:2602.04448)](https://arxiv.org/abs/2602.04448) found that "full-parameter safety fine-tuning on MoE models can improve attack harmlessness rates by amplifying already-safe experts or biasing the router toward conservative routing patterns" *without actually fixing unsafe experts*. Under routing restoration (revert router to original), full-FT safety collapsed by 0.38–0.46; their routing-aware fix dropped only 0.04–0.06. The implication: **the router can become a shortcut for satisfying the alignment objective**. Bi-level optimization (fix router, update Safety-Critical Experts; then freeze experts, align router) avoids it.

### 8.5 Catastrophic forgetting in expert specialization

DES-MoE ([arXiv:2509.16882](https://arxiv.org/abs/2509.16882), HKUST) shows naive multi-domain SFT causes some experts to over-specialize while others become under-utilized. Their three-phase recipe (warm-up frozen experts → stabilization with domain-selected experts → consolidation with frozen router) reduces forgetting **89% vs full-FT**. Pairs with ExpertCondenser's "always-active condenser experts" to address long-tail starvation.

### 8.6 The Llama 4 Maverick cautionary tale

Maverick used 128 fine-grained experts + 1 shared (DeepSeek-style), 400B/17B. But it trained on only 22T tokens (vs Scout's 40T at a quarter the size) and consumed half the GPU time. Community consensus per [oilbeater.com](https://oilbeater.com/en/2025/04/06/chaos-llama4/): either an architectural trial that wasn't meant for full training, or a training-broke-down release from a mid-run snapshot. An undisclosed experimental variant submitted to LMArena triggered the benchmark-controversy. Real-world performance: 16% on Aider Polyglot (vs ~80% SOTA), failed to place top-40 on BigCodeBench. The mid-generation architecture switch from Scout (16 experts) to Maverick (128) suggests a course correction under DeepSeek pressure with insufficient post-training budget for the more complex sparse architecture. **Lesson: the post-training pipeline does not transfer cleanly between expert counts.**

### 8.7 Distillation: MoE → dense

R1's 800K reasoning traces lifted Qwen-1.5B/7B/14B/32B and Llama-8B/70B well past their RL-trained counterparts. The source architecture being MoE is **irrelevant to the student** — what matters is the quality of CoT traces. This is a major operational lever for fleets that can't serve sparse 671B models: train one MoE teacher, distill to a portfolio of dense workhorses.

### 8.8 Converging consensus on MoE post-training

By mid-2026 the recipe stack reads:

1. **Aux-loss-free routing** (or bias-style equivalent) inherited from pretraining
2. **GRPO** as default RL algorithm (replaces PPO; no critic)
3. **R3 or RSPO** (or both) for train-inference router alignment
4. **No LB loss during SFT/DPO** (OLMoE finding generalized)
5. **Modular per-domain SFT+RL** (BAR pattern) for multi-skill models
6. **Distill from MoE teacher to dense student** rather than RL'ing small models directly

---

## 9. Inference & Serving — the Bandwidth-Bound Era

A dense model's decode throughput is governed by FLOPs. An MoE model's decode throughput is governed by **how fast you can move expert weights from HBM to compute**. This single shift drives every system-level decision below.

### 9.1 The bandwidth-bound reality

DeepSeek-R1 demands **13,719 GB/s aggregate HBM bandwidth**. On an 8×H100 node (~26 TB/s aggregate), measured profiles on Qwen3 at 8×H100 show **all-to-all consuming 77% of MoE-layer time** (0.77ms communication vs 0.23ms expert compute).

**Batch size dynamics** flip qualitatively: at small batches, only some experts activate per batch and the rest of the model is idle. At a moderate batch (~16 tokens), all experts get hit but with severe skew — Qwen3 layer 0 sends 19% of tokens to 3 experts; the final layer sends 60% to 3 experts.

This expert-popularity skew is the single biggest serving pain. The hot-expert GPU becomes the critical path; the rest sit idle.

### 9.2 Expert parallelism (EP) — the load-bearing strategy

Each device holds a subset of experts. An all-to-all dispatches each token to its expert's device, then an all-to-all combines results back. The variants:

- **Standard EP** (TensorRT-LLM `--moe_ep_size`, vLLM, SGLang): straightforward but bandwidth-bound on the all-to-all.
- **DeepSeek-V3 hybrid PP+EP+DP**: pipeline + expert + data parallelism, custom all-to-all kernels achieving >40 GB/s with 8×400 Gbps IB. Open-sourced as [DeepEP](https://github.com/deepseek-ai/DeepEP) (Feb 2025) and [DualPipe](https://github.com/deepseek-ai/DualPipe).
- **Wide-EP on NVL72** (NVIDIA): distributes experts across up to 72 GPUs, **1.8× per-GPU throughput** vs narrower EP.
- **EPLB (Expert-Parallel Load Balancer)** (DeepSeek): pool of 288 experts from the original 256 — 32 redundant copies of hot experts. **1.49× prefill, 2.54× decode speedup** measured at LMSYS. Enables non-power-of-2 EP sizes (12, 72 GPUs).

**DeepEP's two dispatch modes**:
- *Normal* (prefill, high throughput, not CUDA-Graph compatible)
- *Low-Latency* (decode, CUDA-Graph compatible, microsecond RDMA)

Adopted within an hour of release — 1,600+ GitHub stars in the first day.

### 9.3 Inference systems

**vLLM** supports Mixtral, DeepSeek-V3, Qwen-MoE, Llama 4, GPT-OSS. EP arrived 2024-2025. On dense baselines hits ~12.5K tok/s on Llama-3.1-8B at H100.

**SGLang** consistently outperforms vLLM on MoE — primarily because of (i) native DeepEP integration with both dispatch modes, (ii) DP Attention (no KV-cache duplication across EP devices, ~50% communication reduction vs TP), (iii) Two-Batch Overlap (TBO) for 27-35% prefill throughput gain. **Headline LMSYS result on DeepSeek-V3 across 96 H100 GPUs**: 52.3K input tok/s and 22.3K output tok/s per node, at an estimated **$0.20 per 1M output tokens** — roughly one-fifth of DeepSeek's own API price. ([LMSYS blog](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/))

**TensorRT-LLM** (NVIDIA): EP/TP/hybrid for MoE. Wide-EP on NVL72 with FP4 yields 10× throughput over H200 for MoE.

**NVIDIA Dynamo** (GTC 2025, [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)): disaggregated prefill/decode, smart router with global KV-cache state via Radix-tree hashing, NIXL for high-throughput GPU-to-GPU transfers. **30× more requests served on GB200 NVL72 with DeepSeek-R1** (vs non-disaggregated). Compatible with vLLM/SGLang/TensorRT-LLM backends.

**Disaggregated prefill/decode for MoE specifically** changes vs dense: prefill wants *low* EP degree (more tokens per expert = better compute utilization); decode wants *high* EP degree (more GPUs = less weight-loading per GPU per token). Dynamo dynamically allocates: e.g., DeepSeek-R1 uses EP4DP16 for prefill, EP64DP3 for decode. LMSYS used EP32 (4 nodes) for prefill, EP72 (9 nodes) for decode.

### 9.4 Memory optimization

**Expert offloading.** The seed paper is [Eliseev & Mazur, arXiv:2312.17238](https://arxiv.org/abs/2312.17238) — running Mixtral-8x7B on 12-16GB VRAM by keeping cold experts on CPU with LRU caching and speculative prefetch. Fundamental bottleneck: PCIe 4.0 (32 GB/s) loads a Mixtral layer in ~80ms while GPU compute takes ~3ms, a 26× gap. **HOBBIT** (Nov 2024, [arXiv:2411.01433](https://arxiv.org/abs/2411.01433)) replaces cache-miss experts with 4-bit copies for 9.93× decode speedup; <1% quality drop when <20% of experts are quantized.

**SSD offloading is energetically disastrous** ([arXiv:2508.06978](https://arxiv.org/abs/2508.06978)): 3.8-12.5× higher per-token energy for Mixtral, 4.7-9.8× for DeepSeek-R1 vs HBM. Even CPU memory is 2.1-3.6× worse than HBM. Conclusion: edge-MoE serving needs better CPU offloading or specialized inference chips, not SSD tiers.

**MoE quantization.** AWQ matches or beats GPTQ at 4-bit on MoE; mixed-precision (more bits to hot experts, fewer to cold) wins over uniform. EAC-MoE (ACL 2025) introduced expert-selection-aware compression. Mainstream pruning >50% degrades sharply, so quantization is the preferred compression path for MoE.

### 9.5 Speculative decoding and MTP

DeepSeek-V3's MTP heads, designed for training, repurpose at inference: **acceptance >80%, ~1.8× speedup** in DeepSeek's own deployment. SGLang MTP integration measured **1.25-2.11× speedup on AMD MI300X** ([blog](https://www.lmsys.org/blog/2025-07-17-mtp/), Jul 2025).

**Pre-Gated MoE** (Microsoft, ISCA 2024): modify the gate to select *next-layer* experts so CPU→GPU migration overlaps with current-layer compute. **Read-ME** (NeurIPS 2024) and **AdapMoE** (2024) extend this with Belady-optimal caching. **Pre-Attention Expert Prediction** (ETH Zurich, 2025) uses linear functions on pre-attention activations to predict expert rankings, including the first layer (which earlier methods missed).

### 9.6 KV-cache compression

MLA's headline: **93.3% KV-cache reduction** vs MHA at DeepSeek-V2's 67B baseline; for very large MoE, MLA holds **~4% of MHA's cache**. Combined with DeepSeekMoE: **5.76× generation throughput** vs the dense baseline. As of 2026, MLA is the de facto attention mechanism for inference-efficient large MoE — though see §10 for why GQA still dominates by adoption count.

### 9.7 Hardware-aware MoE designs

**Mixture-of-Depths (MoD)** ([Raposo et al., arXiv:2404.02258](https://arxiv.org/abs/2404.02258)) — sparse *layer* activation, the orthogonal axis to MoE's sparse expert activation. Combined as **MoDE** with a "no-op" expert representing the residual path; outperforms either alone.

**Cerebras** has the most architecturally interesting position for MoE inference. WSE-3's **44 GB on-chip SRAM with 21 PB/s memory bandwidth (7,000× H100)** lets all expert weights live on-chip, eliminating the all-to-all communication that dominates GPU MoE inference. Cerebras's "MoE at Scale" blog ([cerebras.ai/blog/moe-guide-scale](https://www.cerebras.ai/blog/moe-guide-scale)) measures the 77% communication overhead on GPUs and notes it vanishes on WSE. Their **Batch Tiling on Attention (BTA)** decouples batch size between attention (memory-bound, small tiles) and experts (compute-bound, large concatenated batches), preventing the 53-86% throughput regression that naive sparse routing causes elsewhere. The headline production result: **DeepSeek-R1-Distill-Llama-70B at 1,500+ tok/s — 57× faster than GPU solutions** (Jan 2025). The full 671B R1 has not been publicly benchmarked on CS-3 as of May 2026, but the architecture is uniquely suited for it.

**Groq LPU** uses deterministic dataflow with chip-to-chip RealScale interconnect for near-linear multi-chip scaling. Deployed Llama 4 Maverick on launch day; serves Mixtral-8x7B at ~500 tok/s, R1-Distill-Llama-70B at ~260 tok/s. Architecturally avoids the GPU memory wall but expert-routing implementation details are proprietary.

**SambaNova Composer of Experts (CoE)** is architecturally distinct: routes **whole requests** to independently trained full models (Samba-1: 54 experts, 1.3T total). CoE and weight-level MoE are orthogonal; they can compose.

**AMD ROCm 7.0** (2025) added MoE-specific support: MORI-EP kernels (up to 82% latency reduction), MXFP4 MoE kernels in AITER, PD disaggregation in SGLang. SGLang MTP on MI300X: 1.36-1.80× speedup for DeepSeek-V3.

**Intel Gaudi 3**: supports Llama 4 Scout/Maverick via vLLM. 128GB HBM/accelerator (8/node = 1TB) helps fit MoE weights; no MoE-specific kernel optimizations publicized.

### 9.8 The 2025 cost collapse

DeepSeek-V3 API pricing trajectory:
- Jan 2025: $0.14 / $0.28 per 1M input/output tokens (cache miss: $0.27 input)
- DeepSeek-V3.2-Exp (mid 2025): $0.028 / $0.42 — "cut API pricing in half"
- For comparison: GPT-4o ~$3/$10, Claude Sonnet 4 ~$3/$15

LMSYS's open-source SGLang reproduction at $0.20/1M output tokens suggests DeepSeek's API margin is large or their 640-H800 deployment is less optimized than the LMSYS H100 setup.

**The 2-year compression: ~$10/M output tokens (GPT-4, 2023) → ~$0.28-0.42/M (DeepSeek-V3.2, 2025)** — roughly **25× reduction in 18 months**. Community consensus on what enabled it: **MoE + MLA + MTP**, plus aggressive Chinese pricing strategy and open-source inference tooling. Further compression is expected from FP4 on Blackwell and wider-EP NVL72.

### 9.9 Open systems problems (mid-2026)

1. **TTFT at large EP scale** remains 2-5 seconds (LMSYS); real-time use needs sub-500ms.
2. **EPLB tested only on in-distribution data** — out-of-distribution expert popularity shifts can re-imbalance.
3. **No production-grade hybrid TP+EP for dense FFN layers** in open stacks.
4. **MTP acceptance drops with long-context generation**; optimal MTP depth is task-dependent.
5. **SSD offloading is energetically prohibitive**; the consumer/edge MoE story is unresolved.
6. **Blackwell-native EP kernels are immature in open source** as of mid-2025.

---

## 10. The 2026 Consensus Architecture, Stated Plainly

If a team starting a frontier MoE today asked "what should we build, by default?", the convergent answer would be:

1. **Granularity 128-256 routed experts**, top-8.
2. **One shared expert** (or zero — this is the one truly contested decision).
3. **Sigmoid gating with normalization, aux-loss-free bias balancing** with a small sequence-level safety loss.
4. **MLA if you can afford the engineering** (custom kernels, loss of GQA-tuned tooling); otherwise GQA with 8 groups.
5. **SwiGLU activations, RoPE, RMSNorm.** Universal.
6. **FP8 mixed precision** on Hopper or Blackwell, fine-grained tile/block quantization, FP32 accumulation every 128 elements.
7. **DualPipe-style PP+EP overlap** with custom all-to-all kernels.
8. **MTP heads** trained alongside the main objective for 1.8× speculative decoding at inference.
9. **14-15T pretraining tokens** with dropless routing.
10. **Active params 17-37B**, total params 100B-1T+, sparsity 3-10%.

**The contested zone:**
- Shared experts (Qwen3/OLMoE no, DeepSeek/Kimi/Llama 4 yes)
- MLA vs GQA (DeepSeek/Kimi yes, everyone else GQA)
- Alternating dense/MoE layers (Llama 4 Maverick yes 1:1, DeepSeek-V3 nearly all-MoE)
- Routing stability mechanism (DeepSeek bias vs Qwen3 global-batch vs Skywork GLN)
- Optimizer (AdamW vs Muon — Kimi K2 and GLM-4.5 both adopted Muon for stability)

**The frontier directions** (un-converged but actively explored):
- Attention compression beyond MLA: Step-3's MFA at 22% of V3's per-token attention cost; MiniMax's 7:1 lightning:softmax hybrid; Granite 4.0's Mamba-2 + Transformer hybrid.
- Sparsity ratios <3%.
- Train-inference router alignment as a first-class concern (R3, RSPO).
- Modular post-training (BAR) over monolithic.

---

## 11. What Works / What Doesn't — A Plain Reading

### What works (high confidence)

| Technique | Evidence |
|---|---|
| Fine-grained experts (64-256+) | DeepSeekMoE, OLMoE, Qwen3, Kimi K2 — all show monotonic granularity gains until ~64-128 |
| MoE-vs-dense at scale | Krajewski et al.: gap widens from 20× FLOP reduction at 10²⁰ to >40× at 10²⁵ |
| Aux-loss-free routing | DeepSeek-V3: 14.8T tokens, no rollbacks |
| MLA for KV-cache | 93% reduction, no quality loss |
| MTP for speculative decoding | 1.8× throughput, >80% acceptance |
| FP8 training (with care) | <0.25% loss vs BF16 at 671B scale |
| Distill MoE → dense | R1-Distill-Qwen-32B beats RL-Qwen-32B by ~25 AIME points |
| Dropless routing (MegaBlocks) | 98.6% cuBLAS throughput, no capacity-factor tuning |
| EPLB redundant copies | 1.49× prefill / 2.54× decode at LMSYS |

### What doesn't (or is overstated)

| Technique | Evidence |
|---|---|
| Sparse upcycling for long runs | OLMoE: from-scratch catches up at 25% of dense compute, not 120% |
| Expert-choice routing for autoregressive decoders | Lory underperforms vanilla TopK; structurally violates causality |
| Soft MoE for LMs | Breaks token causality; not autoregressive-compatible |
| SSD offloading for MoE | 3.8-12.5× energy penalty vs HBM |
| Frozen-router fine-tuning | RSPO tested 3 variants, all underperformed soft approaches |
| Naive PPO/GRPO on MoE | 30-50% crash rate without R3/RSPO routing alignment |
| Soft MoE / token-mixing for inference | Computational benefits don't materialize for autoregressive decode |
| Dropping aux loss carelessly | Causes routing collapse; need *some* mechanism (bias, GLN, global batch) |

### Genuinely unsettled

- Shared experts (the cleanest empirical 50-50 split in the field)
- MLA vs GQA (technical question masking ecosystem inertia)
- Optimal sparsity ratio at frontier scale (3% may not be the floor)
- Optimizer choice for MoE (AdamW vs Muon — Kimi K2 demonstrated zero loss spike across 15.5T tokens with MuonClip)
- Best load-balancing variant when aux-loss-free
- Whether the last MoE layer is worth its parameters (recent expert-similarity findings suggest no)

---

## 12. Cerebras-Relevant Implications

A few notes the user's affiliation makes worth highlighting:

**MoE is uniquely suited to wafer-scale.** The bottleneck on GPU MoE inference (77% all-to-all communication time) is precisely what wafer-scale's on-chip 21 PB/s bandwidth eliminates. Cerebras's published BTA pattern decouples attention batching from expert batching — the ergonomically right knob, since attention is memory-bound and MoE FFN is compute-bound. The 1,500+ tok/s on R1-Distill-Llama-70B at 57× GPU is the proof point for the dense distill class; the open question is whether the same advantage transfers to the full 671B sparse R1.

**MTP and speculative decoding interact with wafer-scale differently.** On GPU MoE, speculative decoding's main win is amortizing expert weight loads. On wafer-scale where weights are already on-chip, the win shifts to compute amortization — different math, possibly different optimal MTP depth.

**The EPLB/all-to-all problem doesn't exist on wafer-scale.** Hot-expert imbalance still exists at the scheduling layer (some experts will see 60% of tokens in some layers), but the cross-device dispatch tax is absent. This may justify entirely different routing policies that GPU-bound systems can't run.

**Distillation flywheel.** The community has demonstrated sparse-MoE-teacher → dense-student distillation works (R1 → Qwen). Cerebras's hardware advantage on dense models (where it's widely demonstrated) pairs naturally with this flywheel.

---

## 13. Reading List by Topic

If you have time for **one paper per topic**, in the order I'd recommend:

1. **Mixtral 8x7B** — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) (Jan 2024). The accessible starting point.
2. **DeepSeekMoE** — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) (Jan 2024). Fine-grained + shared experts thesis.
3. **DeepSeek-V2** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (May 2024). MLA derivation. Read Appendix C.2.
4. **DeepSeek-V3** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) (Dec 2024). The reference model. Read sections on aux-loss-free balancing, MTP, FP8.
5. **DeepSeek-R1** — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (Jan 2025). RL on MoE base.
6. **OLMoE** — [arXiv:2409.02060](https://arxiv.org/abs/2409.02060) (Sep 2024). The honest ablations everyone references.
7. **Auxiliary-Loss-Free Load Balancing** — [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).
8. **Scaling Laws for Fine-Grained MoE** — [arXiv:2402.07871](https://arxiv.org/abs/2402.07871) (Krajewski et al.).
9. **Skywork-MoE** — [arXiv:2406.06563](https://arxiv.org/abs/2406.06563). Practical balancing recipes.
10. **R3: Stabilizing MoE RL** — [arXiv:2510.11370](https://arxiv.org/abs/2510.11370). The RL crisis quantified.
11. **RSPO: Stable RL for MoE** — [arXiv:2510.23027](https://arxiv.org/abs/2510.23027). The RL crisis fixed.
12. **BAR: Modular MoE Post-training** — [arXiv:2604.18473](https://arxiv.org/abs/2604.18473). Train separately, merge together.
13. **Mixture-of-Depths** — [arXiv:2404.02258](https://arxiv.org/abs/2404.02258). The orthogonal axis.
14. **A Closer Look at MoE in LLMs** — [arXiv:2406.18219](https://arxiv.org/abs/2406.18219). The Mixtral upcycling forensics.
15. **Cerebras "Router Wars" / "MoE at Scale"** — non-arXiv but structurally insightful: [cerebras.ai/blog/moe-guide-router](https://www.cerebras.ai/blog/moe-guide-router), [cerebras.ai/blog/moe-guide-scale](https://www.cerebras.ai/blog/moe-guide-scale).

For implementation, [DeepEP](https://github.com/deepseek-ai/DeepEP), [DualPipe](https://github.com/deepseek-ai/DualPipe), [MegaBlocks](https://github.com/databricks/megablocks), and SGLang's MoE dispatch code are the load-bearing repositories.

---

## 14. Glossary

- **Active params** — parameters touched by a single forward pass on one token.
- **Aux-loss-free balancing** — load balancing via per-expert biases updated outside the gradient (DeepSeek-V3).
- **BTA (Batch Tiling on Attention)** — Cerebras technique to run attention with small tiles while running expert FFNs with concatenated large batches.
- **DualPipe** — DeepSeek-V3's bidirectional pipeline-parallel scheduler that overlaps PP and all-to-all.
- **EP (Expert Parallelism)** — placing different experts on different devices, dispatching tokens via all-to-all.
- **EPLB** — Expert-Parallel Load Balancer; redundant copies of hot experts spread across devices.
- **Fine-grained experts** — splitting each FFN expert into N smaller sub-experts to enable more combinations and sharper specialization.
- **GRPO** — Group Relative Policy Optimization; PPO without a critic, uses group-mean as baseline. DeepSeek's RL workhorse.
- **MLA (Multi-head Latent Attention)** — DeepSeek-V2 attention that compresses K/V into a low-rank latent before caching.
- **MTP (Multi-Token Prediction)** — predicting several next tokens at training time; doubles as inference-time speculative decoder.
- **R3 (Rollout Routing Replay)** — replay inference-pass router decisions during training-pass to align distributions.
- **Router collapse** — the failure mode where the gate routes all tokens through 1-2 experts; the rest go dead.
- **RSPO** — Router-Shift Policy Optimization; soft IS-weight rescaling based on router shift, with stop-gradient.
- **Sparsity ratio** — active params / total params. Mixtral 28% → Kimi K2 3.1% over 19 months.
- **Sparse upcycling** — initialize an MoE from a dense model by cloning the FFN into all expert slots.
- **Top-k routing** — token-choice routing where each token activates its k highest-scoring experts.
