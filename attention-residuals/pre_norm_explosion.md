# PreNorm Dilution: The Hidden Cost of Stable Training

**A deep-dive companion to the Attention Residuals paper (Kimi Team, 2026)**

---

## 1. Introduction — The PreNorm Bargain

The original Transformer (Vaswani et al., 2017 — [52] in AttnRes) placed LayerNorm *after* the residual addition:

```
h_l = LayerNorm(h_{l-1} + f_{l-1}(h_{l-1}))     # PostNorm (original)
```

This **PostNorm** arrangement keeps hidden-state magnitudes bounded—each layer's output is re-normalized before the next layer sees it. But PostNorm introduced a brutal training problem: gradient vanishing compounds exponentially with depth. Repeated normalization on the residual path distorts the gradient signal, causing it to decay as O(1/2^{L/2}) through the network (Xiong et al., 2020 — [60]). In practice, PostNorm Transformers beyond ~12 layers required careful learning-rate warmup and could not scale to 100+ layers without heroic engineering.

GPT-2 (Radford et al., 2019) popularized the fix that would become the industry default: move LayerNorm *before* the sublayer:

```
h_l = h_{l-1} + f_{l-1}(PreNorm(h_{l-1}))        # PreNorm (GPT-2 onward)
```

PreNorm preserves a clean **identity path** through the residual stream. Gradients flow through the identity shortcut without any normalization distortion, enabling stable training at arbitrary depth. Nguyen & Salazar (2019 — [34]) demonstrated "Transformers without Tears"—no warmup needed, stable at 1000+ layers. This was the decisive breakthrough that unlocked modern LLM scaling.

**But PreNorm introduces a hidden cost.** By normalizing the *input* to each sublayer but never re-normalizing the residual stream itself, PreNorm allows the accumulated hidden state to grow without bound. Each layer contributes to an ever-expanding sum, progressively burying earlier layers' contributions under the magnitude of the accumulation.

This is the **PreNorm Bargain**: stable gradients in exchange for unbounded representations. Every modern LLM—GPT-4, Llama, DeepSeek-V3, Gemma—pays this price.

---

## 2. The Mathematics of Magnitude Growth

### The residual recurrence

Unrolling the standard residual connection reveals a simple accumulation:

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
    = h_1 + Σ_{i=1}^{l-1} f_i(h_i)
```

where `h_1` is the token embedding and `f_i` is the i-th sublayer (attention or MLP). The hidden state at layer `l` is just the embedding plus the sum of all preceding layer outputs, with every term weighted equally at 1.

### Why magnitudes grow

PreNorm normalizes the *input* to each `f_i` (via LayerNorm or RMSNorm), but the residual stream `h_l` itself is never re-normalized. Since `h_l` is a sum of `l` terms, its magnitude grows with depth.

**Random-walk model (lower bound).** If layer outputs `f_i(h_i)` are approximately uncorrelated in direction and have roughly unit norm, then the residual stream behaves like a random walk in d-dimensional space. Ziming Liu's analysis ("Depth 1 — Understanding Pre-LN and Post-LN", blog post, Jan 2026) demonstrates this with 100-layer random-initialization experiments:

```
||h_l|| ≈ √l               (norm grows as square root of depth)
1 - cos(h_l, h_{l+1}) ~ 1/l    (cosine distance between successive states shrinks)
```

Since the norm of each update is roughly constant while `||h_l||` keeps growing, the relative angular update satisfies `θ ~ 1/√l`, which implies the `1/l` cosine distance relationship. This is the *optimistic* case—deep layers barely perturb the representation's direction.

**Correlated case (upper bound).** If layer outputs are positively correlated (they often are—all layers process the same token), growth can be as fast as O(L) in norm. Sun et al. ("The Curse of Depth in Large Language Models", arXiv:2502.05795, NeurIPS 2025) formalize this precisely. Their **Lemma 1** proves that for standard Pre-LN (without any fix), output variance satisfies:

```
Θ(L) ≤ σ²(x_L) ≤ Θ(exp(L))
```

The lower bound is linear, the upper bound exponential—depending on the spectral properties of the layer Jacobians. They further demonstrate empirically that nearly half the layers in modern Pre-LN LLMs (Llama, Mistral, DeepSeek, Qwen) are less effective than expected, directly attributable to this variance explosion.

The Attention Residuals paper states the growth plainly:

> "PreNorm [34, 60]...its unweighted accumulation causes hidden-state magnitudes to grow as O(L) with depth, progressively diluting each layer's relative contribution [27]." — §1, Introduction

### Jacobian collapse

Sun et al. (2025) further show that in deep PreNorm networks, the Jacobian of the PreNorm operation approaches the identity. Their **Theorem 1** proves that when variance grows exponentially (`σ²(x_l) ~ exp(l)`), the gradient norm converges to a constant:

```
||∂y_l / ∂x_1||₂ ≤ M    (bounded above by constant M)
```

This implies that "for very large L, deeper layers behave nearly as an identity map"—the derivative of deep Transformer blocks approaches the identity matrix. Deeper layers barely contribute to training because the PreNorm normalization, operating on an increasingly large input, can barely distinguish meaningful variations from noise.

### The gradient product

For completeness, the gradient of the loss with respect to intermediate layer `l` factors as:

```
∂L/∂h_l = (∂L/∂h_L) · Π_{j=l}^{L-1} (I + ∂f_j/∂h_j)
```

The identity term `I` in each factor is what gives residual networks their training stability—there is always a direct gradient path from the loss to any layer. But this same identity path means that gradient magnitude is dominated by the identity component, with the `∂f_j/∂h_j` terms contributing less as their relative scale shrinks. This is why gradients flow freely (good for optimization) but concentrate in early layers (bad for expressiveness).

---

## 3. Three Consequences

### 3.1 Dilution: Deep layers contribute less

At layer `l`, the residual stream has accumulated contributions from `l` sources. When layer `l` adds its output `f_l(h_l)`, it is one term in a sum of `l+1` terms. Its *relative* contribution is approximately:

```
||f_l(h_l)|| / ||h_l|| ≈ 1/l
```

Even if `f_l` produces an output of the same magnitude as `f_1`, its influence on the residual stream's direction is `l` times smaller. Early layers define the representation; late layers merely nudge it.

This is **dilution**: the core problem that the Attention Residuals paper targets.

### 3.2 Compensatory output growth

Deeper layers don't accept their irrelevance quietly. During training, they learn to produce progressively larger outputs to maintain influence over the growing residual stream.

**Figure 5b from the Attention Residuals paper** shows this directly: in the baseline (standard PreNorm) model, each transformer block's output magnitude grows monotonically from ≈1 for the first block to ≈15 for the last block (index 27). This is a 15× amplification—deep layers are forced to "shout" to be heard over the accumulated noise.

This compensatory behavior is wasteful and potentially destabilizing. It means deeper layers devote capacity to magnitude rather than content, and it creates a feedback loop: larger outputs require the subsequent PreNorm to normalize from larger inputs, which further reduces the normalization's ability to differentiate between meaningful and trivial variations.

### 3.3 Gradient concentration in early layers

**Figure 5c from the Attention Residuals paper** reveals the gradient-side consequence: gradient magnitudes (×10⁻⁵) concentrate heavily in the earliest transformer blocks, with block 0 receiving roughly 3× the gradient of block 10+. The gradient magnitude drops steeply after the first few blocks, then flatlines near zero for the deeper half of the network.

With all residual weights fixed at 1, the baseline provides "no means of regulating gradient flow across depth, leading to disproportionately large gradients in the earliest layers" (§5.2, Training Dynamics). Early layers train fast while deep layers stagnate.

This gradient concentration creates an implicit **depth budget**: even if you stack 128 layers, only the first 30-50 are training effectively. The rest are along for the ride.

---

## 4. The Empirical Smoking Gun — Pruning Half Your Layers

The most striking empirical confirmation of PreNorm dilution comes from Gromov et al. ("The Unreasonable Ineffectiveness of the Deeper Layers", arXiv:2403.17887, 2024 — [11] in AttnRes).

Their key finding: **up to ~50% of layers can be pruned from trained LLMs with minimal degradation on QA benchmarks.** The exact threshold is model-dependent:

| Model | Prunable fraction before collapse |
|---|---|
| Llama-2 family (7B–70B) | **~45–55%** |
| Mistral-7B | ~35% |
| Phi-2 (2.7B) | ~25% |
| Qwen family | ~20% |

The Llama family is the most prunable—removing layers 16-32 (the deeper half) from Llama-2-70B causes negligible degradation on MMLU and BoolQ. Key observations:

- The quality loss from removing deep layers is dramatically smaller than removing the same number of early layers
- Performance stays in a "flat region" before a sharp transition to collapse (random-guessing accuracy)
- After identifying the optimal contiguous block to prune, "healing" with QLoRA finetuning on a single A100 recovers most remaining quality

This is precisely what PreNorm dilution predicts: deep layers contribute so little to the final representation that removing them barely matters. The model has learned to route most information through the first half of the network.

The Attention Residuals paper cites this directly:

> "Empirically, a significant fraction of layers can be pruned with minimal loss [11]." — §1, Introduction

If your architecture renders half its parameters nearly inert, something fundamental is broken. PreNorm dilution is that something.

---

## 5. PostNorm vs PreNorm — The Fundamental Tension

The normalization placement problem presents a genuine dilemma, not merely an engineering trade-off:

| Property | PostNorm | PreNorm |
|---|---|---|
| **Norm of h_l** | Bounded (re-normalized each layer) | Grows as O(L) |
| **Gradient flow** | Vanishes as O(1/2^{L/2}) | Clean identity path |
| **Deep layer utility** | Useful (bounded representations) | Diluted (buried in accumulation) |
| **Training stability** | Requires warmup, careful init | Stable out of the box |
| **Effective depth** | Early layers undertrained | Deep layers underutilized |

Ziming Liu's analysis (blog post, Jan 2026) captures this symmetry elegantly, a framing echoed in the SiameseNorm paper (Li et al., arXiv:2602.08064, 2026 — [27] in AttnRes):

> "Pre-LN suffers from the problem that later layers become 'useless'; Post-LN suffers from the opposite problem: early layers become 'useless'."

PostNorm's repeated normalization on the residual path causes gradient vanishing, which starves early layers of learning signal—so early layers become identity functions. PreNorm's unbounded accumulation causes dilution, which drowns out deep layers' contributions—so deep layers become identity functions.

Both schemes waste half the network's depth. The question that drove the 2022-2026 explosion of normalization research: **can we get bounded representations AND stable gradients?**

---

## 6. The Solution Landscape (2022–2026)

The past four years have seen an intense effort to break the PreNorm/PostNorm deadlock. Solutions can be organized by their approach to the underlying problem.

### 6.1 Scale the Residual

The simplest approach: adjust the weight of each layer's contribution.

- **DeepNet** (Wang et al., 2022, arXiv:2203.00555 — [54]): Scales the residual branch by a constant `α > 1` and the sublayer output by `β < 1`, using `h_l = α · h_{l-1} + β · f_{l-1}(PreNorm(h_{l-1}))`. This dampens magnitude growth while preserving gradient flow. Enabled training of 1000-layer PostNorm Transformers.

- **LayerScale** (Touvron et al., 2021 — [50]): Multiplies each layer's output by a learned diagonal matrix `diag(λ_l)`, initialized to small values (~0.1). Per-channel scaling lets the network learn to suppress or amplify individual features.

- **ReZero** (Bachlechner et al., 2020 — [2]): Multiplies each layer output by a learned scalar `α_l`, initialized to 0. Layers start as identity and gradually increase their contribution. Simple but effective for moderate depths.

- **GPT-2's 1/√(2L) scaling**: Scales the output of each residual sublayer by `1/√(2L)` at initialization, where `L` is the total number of layers. This ensures the variance contribution of each layer is O(1/L), keeping the total accumulation bounded at initialization. Many subsequent models adopt variants of this heuristic.

- **LayerNorm Scaling (LNS)** (Sun et al., arXiv:2502.05795, NeurIPS 2025): Scales the LayerNorm output inversely by the square root of the layer index: `h^(l) = LayerNorm(h^(l)) / √l`. No hyperparameters. Their **Lemma 2** shows this tightens the variance upper bound from O(exp(L)) to O(L^{2-ε}), where 0 < ε ≤ 1/4. Tested from 130M to 7B parameters, consistently outperforming prior normalization techniques.

### 6.2 Hybrid Normalization

Rather than choosing PreNorm or PostNorm, mix them.

- **SiameseNorm** (Li et al., 2026, arXiv:2602.08064 — [27]): Maintains **two parallel streams** with shared parameters—one PreNorm, one PostNorm. The PreNorm stream provides stable gradients; the PostNorm stream provides bounded representations. Outputs are combined, getting the best of both worlds. Elegant but doubles the residual-stream memory.

- **HybridNorm** (Zhuo, 2025, arXiv:2503.04598 — [73]): Places normalization both before and after sublayers in a combined scheme, aiming to capture the gradient benefits of PreNorm and the representation benefits of PostNorm.

- **Mix-LN** (Li et al., arXiv:2412.13795, ICLR 2025): Uses **PostNorm for the first α·L layers** and **PreNorm for the remaining layers**, where α ∈ [0,1] is a hyperparameter (optimal α ≈ 0.25 for 250M, ≈ 0.06 for 7B). Post-LN in early layers keeps the base representation bounded and well-scaled; Pre-LN in deep layers ensures gradient stability where the Jacobian chain is longest. Simple insight: the two failure modes hit different parts of the network.

- **Keel** (Chen & Wei, 2026, arXiv:2601.19895 — [4]): Titled "Post-LayerNorm Is Back: Stable, Expressive, and Deep." Combines Highway-network-style gating with PostNorm, achieving stable training with bounded representations. Demonstrates that PostNorm can work at scale if paired with appropriate gating.

- **Peri-LN** (Kim et al., arXiv:2502.02732, ICML 2025): Places normalization **both before AND after each sublayer**: `y_l = x_l + Norm(Module(Norm(x_l)))`. The output-LN constrains residual spikes while the input-LN preserves gradient flow. Also normalizes embeddings and final output. Notes that recent models like Gemma 2 and OLMo have quietly adopted similar designs. Tested up to 3.2B.

- **FuseNorm** (OpenReview:azXOzJFwuf): Adopts Pre-LN's clean residual path for stable signal propagation while employing Post-LN-style computation that normalizes the residual connection output. Combined with a principled scaling strategy that maintains bounded signal variance.

- **SpanNorm** (Wang et al., arXiv:2601.22580, 2026): Establishes a clean residual connection that **spans the entire transformer block** (not just individual sublayers), stabilizing signal propagation while using PostNorm-style computation to normalize aggregated output. Tested on both dense and MoE architectures.

### 6.3 Multi-Stream Architectures

Instead of a single residual stream, maintain multiple parallel state vectors.

- **Hyper-Connections / mHC** (Zhu et al., 2025 [72]; Xie et al., 2026 [59]): Maintain `m` parallel residual streams updated via learned transition matrices `A_l ∈ R^{m×m}`. The mixing matrices are constrained to be doubly stochastic, stabilizing the cumulative products across depth. With `m=4` streams, mHC achieves substantial improvements over single-stream baselines. The semiseparable rank of the depth mixing matrix increases from 1 (standard residual) to `m`.

- **DDL — Deep Delta Learning** (Zhang et al., 2026 — [67]): Maintains a matrix state `H_l` updated via a delta-rule erase-and-write mechanism. Relates to the delta rule in sequence modeling (Gated Delta Networks).

### 6.4 Cross-Layer Access

Give each layer direct access to earlier layers' individual outputs (not just their compressed sum).

- **DenseFormer** (Pagliardini et al., 2024 — [36]): Each layer receives a weighted sum of all previous layer outputs with learned scalar coefficients, fixed after training. Grants cross-layer access but with static weights—the ablation in AttnRes shows this achieves only 1.767 loss vs baseline's 1.766, essentially no gain.

- **MRLA** (Fang et al., 2023 — [10]): Cross-Layer Retrospective Retrieving via Layer Attention. Uses element-wise sigmoid gating over all previous layers. Closer to linear attention than softmax attention over depth.

- **MUDDFormer** (Xiao et al., 2025 — [56]): Multiway Dynamic Dense connections via a small MLP that generates position-dependent weights across four decoupled streams.

- **ANCRe** (Zhang, 2026 — [68]): Adaptive Neural Connection Reassignment for efficient depth scaling.

### 6.5 Training Schedule

- **Progressive Residual Warmup (ProRes)** (Chen et al., arXiv:2603.05369, March 2026): Implements an "early layer learns first" philosophy. Each layer's residual branch is multiplied by a scalar that gradually warms from 0 to 1 during training. Deeper layers have **longer warmup periods**—they stay suppressed longer, waiting for earlier layers to stabilize. This creates a cascading activation: layer 1 activates first, then layer 2, etc. Improves depth scaling across model sizes with faster convergence.

### 6.6 Selective Aggregation

The Attention Residuals approach: replace fixed-weight accumulation with learned, input-dependent selection over depth.

- **Attention Residuals** (Kimi Team, 2026 — this paper): Replaces the unit-weight sum with softmax attention over all prior layer outputs. Each layer uses a learned pseudo-query to attend over all source layers, producing a convex combination (weights sum to 1). Block AttnRes groups layers into ~8 blocks for practical O(Nd) scaling.

### Summary Table

| Approach | Representative | Key Mechanism | Complexity |
|---|---|---|---|
| Scale residual | DeepNet, ReZero, LNS | Adjust per-layer weight | O(1) per layer |
| Hybrid norm | SiameseNorm, Mix-LN, Peri-LN | Combine Pre/Post placement | O(1)–O(2d) per layer |
| Multi-stream | mHC, DDL | m parallel residual streams | O(m²d) per layer |
| Cross-layer | DenseFormer, MRLA | Access individual prior outputs | O(Ld) per layer |
| Training schedule | ProRes | Progressive per-layer warmup | O(1) per layer |
| Selective aggregation | AttnRes | Softmax attention over depth | O(Nd) per layer (Block) |

---

## 7. How Attention Residuals Solves It

The AttnRes solution is elegant in its core mechanism: replace the fixed-weight sum with a **convex combination**.

### 7.1 Bounded magnitude via softmax

Standard residual: `h_l = Σ_{i=0}^{l-1} v_i` (unit weights, sum grows with l)

AttnRes: `h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i` where `Σ α_{i→l} = 1` (softmax normalization)

Because the attention weights sum to 1, the result is a **convex combination** of source layer outputs. The magnitude of `h_l` is bounded by the maximum magnitude of any individual source `v_i`, regardless of how many layers have been computed. This eliminates O(L) growth by construction.

### 7.2 RMSNorm on keys prevents large-output dominance

Without normalization, layers that produce naturally larger outputs would dominate the softmax through the key magnitudes, biasing attention weights regardless of content relevance. The kernel function:

```
φ(q, k) = exp(q^T · RMSNorm(k))
```

normalizes each source's key before computing compatibility, ensuring that attention weights reflect directional similarity, not magnitude. The ablation confirms this matters: removing RMSNorm degrades loss from 1.737→1.743 (Full) and 1.746→1.750 (Block).

### 7.3 Block AttnRes resets accumulation

Block AttnRes introduces a periodic structure: within each block of S layers, standard residual accumulation occurs (magnitude grows). At block boundaries, the selective aggregation resets the accumulation by computing a fresh convex combination of block-level representations.

**Figure 5b** shows the result directly: while the baseline's output magnitude grows monotonically to ~15, Block AttnRes shows a **bounded periodic pattern**—magnitude grows within each block, then resets at the boundary. The envelope stays bounded.

### 7.4 Training dynamics evidence

The improvements are visible across all three panels of Figure 5:

- **(a) Validation loss:** AttnRes achieves consistently lower loss throughout training, with the gap widening during the decay phase
- **(b) Output magnitude:** Bounded periodic pattern (AttnRes) vs. monotonic growth to ~15× (baseline)
- **(c) Gradient magnitude:** Substantially more uniform distribution across depth (AttnRes) vs. heavy concentration in earliest blocks (baseline)

The gradient uniformity is particularly significant: it means all layers are receiving meaningful learning signal throughout training, rather than the deep layers stagnating. This explains the +7.5 point gain on GPQA-Diamond—multi-step reasoning benefits most from having all layers contribute meaningfully.

### 7.5 The structured-matrix view

Through the depth mixing matrix framework (§6.2 in the paper), AttnRes represents the progression:

```
Standard Residual:  M = all-ones lower triangular  (depth-wise linear attention, fixed weights)
Highway:            M = carry-product factored      (dynamic, rank 1)
mHC:                M = product of m×m transitions  (dynamic, rank m)
Full AttnRes:       M = dense lower triangular      (depth-wise softmax attention, rank L)
Block AttnRes:      M = block-structured            (dynamic, rank N+S)
```

Standard residuals are depth-wise linear attention with fixed unit weights. AttnRes upgrades this to depth-wise **softmax** attention with learned, input-dependent weights—the same linear→softmax transition that revolutionized sequence modeling.

---

## 8. Open Questions

### Will one approach dominate at frontier scale?

The solution landscape is crowded and no single method has been validated at true frontier scale (>400B parameters, >10T tokens). AttnRes shows a 1.25× compute advantage at 48B/1.4T tokens, but it's unclear whether this advantage holds, grows, or shrinks at GPT-4/Claude-scale training. The scaling law curves suggest the advantage persists (all variants have similar slopes), but extrapolation is uncertain.

SiameseNorm, Keel, and mHC are the strongest competing approaches, each with different engineering trade-offs:
- **SiameseNorm**: 2× residual memory but conceptually simple
- **Keel**: Revives PostNorm, avoiding the problem entirely
- **mHC**: Multi-stream with proven scaling but higher I/O cost (34d vs AttnRes's 5.5d)

### Is dilution worse for MoE architectures?

Modern frontier models (DeepSeek-V3, Kimi Linear, Mixtral) are all MoE. With expert routing adding another layer of sparsity and conditional computation, the interaction between PreNorm dilution and expert selection is unexplored. If different experts contribute outputs of very different magnitudes, dilution effects could be amplified non-uniformly across tokens.

The AttnRes paper validates on an MoE architecture (Kimi Linear, 48B/3B), but doesn't isolate the MoE interaction specifically.

### Can solutions compose?

Could you combine AttnRes with SiameseNorm (two streams, each using softmax aggregation over depth)? Or AttnRes with DeepNorm scaling? The paper doesn't explore compositions. The structured-matrix framework suggests these are orthogonal axes—AttnRes changes *how* layers are aggregated (weights), while SiameseNorm changes *how many* streams are maintained (state dimension). In principle, they could compose, but the interaction effects are unknown.

### What happens with much deeper models?

Current frontier models use 80-128 layers. If architectures push to 500+ layers (as some scaling hypotheses suggest for reasoning), dilution effects would be far more severe—layer 500's contribution would be 1/500th of the total. AttnRes's softmax attention would need to select from 500+ sources, which might require different attention mechanisms (e.g., local + global depth attention, analogous to sequence-length solutions).

### Is the dilution problem actually a feature?

A contrarian view: perhaps PreNorm dilution is a form of implicit regularization. By making deep layers nearly inert, it prevents the network from relying too heavily on very deep, highly nonlinear feature compositions. The Gromov et al. pruning result could be interpreted as evidence that the network has learned a reasonable allocation of capacity, concentrating representational power where it matters (early layers). If so, "fixing" dilution might require corresponding changes to regularization and capacity allocation.

The evidence weighs against this interpretation—AttnRes consistently improves downstream performance, suggesting that diluted layers *could* be useful if given the mechanism to contribute—but it's worth considering as architectures evolve.

---

## Sources

### Primary
- **Attention Residuals** (Kimi Team, 2026) — the companion paper for this document

### Key citations from the AttnRes paper
- [52] Vaswani et al. "Attention Is All You Need." NeurIPS, 2017. (Original Transformer / PostNorm)
- [60] Xiong et al. "On Layer Normalization in the Transformer Architecture." arXiv:2002.04745, 2020. (PreNorm analysis)
- [34] Nguyen & Salazar. "Transformers without Tears." IWSLT, 2019. (PreNorm stability)
- [27] Li et al. "SiameseNorm: Breaking the Barrier to Reconciling Pre/Post Norm." arXiv:2602.08064, 2026.
- [11] Gromov et al. "The Unreasonable Ineffectiveness of the Deeper Layers." arXiv:2403.17887, 2024/2025.
- [54] Wang et al. "DeepNet: Scaling Transformers to 1,000 Layers." arXiv:2203.00555, 2022.
- [4] Chen & Wei. "Post-LayerNorm Is Back: Stable, Expressive, and Deep." (Keel) arXiv:2601.19895, 2026.
- [45] Srivastava et al. "Highway Networks." arXiv:1505.00387, 2015.
- [72] Zhu et al. "Hyper-Connections." arXiv:2409.19606, 2025.
- [59] Xie et al. "mHC: Manifold-Constrained Hyper-Connections." arXiv:2512.24880, 2026.
- [73] Zhuo. "HybridNorm." arXiv:2503.04598, 2025.
- [36] Pagliardini et al. "DenseFormer." arXiv:2402.02622, 2024.
- [10] Fang et al. "Cross-Layer Retrospective Retrieving via Layer Attention." (MRLA) arXiv:2302.03985, 2023.

### Supplementary external references
- Sun et al. "The Curse of Depth in Large Language Models." arXiv:2502.05795, NeurIPS 2025.
- Li et al. "Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN." arXiv:2412.13795, ICLR 2025.
- Kim et al. "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture." arXiv:2502.02732, ICML 2025.
- Wang et al. "SpanNorm: Reconciling Training Stability and Performance in Deep Transformers." arXiv:2601.22580, 2026.
- Chen et al. "ProRes: Progressive Residual Warmup." arXiv:2603.05369, March 2026.
- FuseNorm. OpenReview: azXOzJFwuf.
- Ziming Liu. "Depth 1 — Understanding Pre-LN and Post-LN." Blog post, Jan 2026.
