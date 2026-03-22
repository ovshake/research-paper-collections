# Research Ideas: Delightful Policy Gradient

Based on [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608v1) (Osband, 2026). Hardware assumption: 4× A100 80GB.

---

## 1. DG for LLM RLHF/GRPO

**Priority: Highest | Compute: Medium (days) | Novelty: Very high**

The paper flags RLHF as future work but never tests it. The entire LLM post-training field uses GRPO/REINFORCE variants (DeepSeek, Nemotron-Cascade, Qwen) — if DG helps even modestly here, it's a significant result.

### Hypothesis

In GRPO-based LLM post-training, the same budget allocation pathology exists: easy prompts the model already handles dominate the gradient; hard prompts where the model needs to improve are starved. DG should rebalance gradient budget toward hard prompts, improving sample efficiency and final performance.

### Experimental Plan

1. **Base model:** Qwen2.5-1.5B-Instruct or SmolLM-1.7B
2. **Task:** Math RL with verifiable rewards (AIME-style, or a subset of NuminaMath with answer verification)
3. **Implementation:** Drop-in modification to GRPO — multiply each sample's policy gradient term by σ(χ_t/η) where:
   - χ_t = U_t · ℓ_t (advantage × surprisal)
   - U_t = group-normalized reward (standard GRPO advantage)
   - ℓ_t = −log π(response | prompt), clipped to [−10, 10] per Algorithm 2
4. **Baselines:** Standard GRPO, GRPO + entropy bonus
5. **Metrics:**
   - Overall pass@1 on held-out math problems
   - Pass@1 stratified by difficulty (easy/medium/hard buckets based on initial success rate)
   - Training efficiency: steps to reach X% accuracy
6. **Ablations:** η ∈ {0.5, 1.0, 2.0}, response-level vs. token-level surprisal

### Key Implementation Detail

For full-response surprisal: ℓ = −Σ_t log π(y_t | y_{<t}, x) is the sum of per-token negative log-probs. This can be very large for long responses. Options:
- Clip ℓ to [−10, 10] before computing delight (as in Algorithm 2)
- Whiten: χ̃ = (χ − μ_χ) / (σ_χ + ε) before gating
- Use per-token delight instead of per-response (each token gets its own gate)

### Expected Outcome

Token Reversal results (Figure 9) show DG's advantage grows with output-space size M^H. LLM output spaces are enormous — if the mechanism transfers, gains should be substantial on hard problems.

---

## 2. Diagnosing the Budget Allocation Pathology in GRPO

**Priority: High | Compute: Low-Medium | Novelty: High**

A focused diagnostic study: does GRPO exhibit the self-reinforcing budget allocation problem on real LLM prompts?

### Hypothesis

During GRPO training, the effective gradient contribution from each prompt is proportional to the model's current success rate on that prompt (analogous to c_n ∝ p_n in Section 4.2). Easy prompts that the model already solves attract increasingly more gradient budget over training, while hard prompts stagnate.

### Experimental Plan

1. **Setup:** Standard GRPO math RL run on Qwen2.5-1.5B (same as Idea 1)
2. **Logging (the key contribution):**
   - For each prompt in each batch, log: prompt difficulty (initial pass@16), gradient norm contribution, advantage values, success rate over time
   - Bucket prompts into quintiles by initial difficulty
   - Track how each quintile's share of total gradient norm evolves over training
3. **Visualizations:**
   - Gradient budget share per difficulty quintile over training steps (the LLM analog of Figure 4b)
   - Per-quintile accuracy curves — do hard prompts stagnate?
   - Compare GRPO vs. DG-GRPO on these metrics
4. **Controls:** Run the same analysis on a uniform-weighted baseline (all prompts get equal gradient weight, like CE oracle) to establish the ideal allocation

### Why This Matters Even Without DG Working

If the pathology exists in GRPO (which the theory predicts), documenting it empirically is a contribution regardless of whether DG is the right fix. The community needs to see this self-reinforcing dynamic in action.

### Presentation

Modeled on the MNIST diagnostic (Section 3): clean plots showing the problem first, then DG as one potential solution.

---

## 3. Temperature Schedule for η

**Priority: Medium | Compute: Very low (hours) | Novelty: Medium**

The paper uses η=1 everywhere. This is a quick, publishable ablation.

### Hypothesis

The optimal η may change during training. Early on, aggressive gating (low η) helps by suppressing noise when the policy is bad everywhere. Late in training, the directional effect (which requires η > 1/2 per Proposition 2) matters more.

### Experimental Plan

Run on MNIST bandit and Token Reversal (nearly free compute):

1. **Constant baselines:** η ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0}
2. **Warmup schedules:**
   - η: 0.1 → 1.0 (linear warmup over first 50% of training)
   - η: 0.1 → 1.0 (exponential warmup)
3. **Decay schedules:**
   - η: 2.0 → 0.5 (start gentle, end aggressive)
   - η: 2.0 → 1.0 (start gentle, end standard)
4. **Cosine schedule:** η oscillates between 0.5 and 2.0
5. **Measure both mechanisms separately:** alignment to g*_PG (Mechanism 1) and g*_CE (Mechanism 2) as in Figure 4, for each schedule

### Key Question

Is there a point in training where η < 1/2 is actually beneficial (variance reduction dominates) before switching to η > 1/2 (directional improvement dominates)?

### Extension

If a schedule works well on MNIST/Token Reversal, validate on the LLM GRPO setup from Idea 1.

---

## 4. DG for Rejection Sampling / Best-of-N Training

**Priority: Medium | Compute: Medium | Novelty: Medium-high**

### Hypothesis

In best-of-N / rejection sampling fine-tuning, rejected samples contain useful learning signal that is typically discarded. DG provides a principled way to weight learning from all N samples:
- High-reward, unlikely response (breakthrough) → high delight → learn strongly
- Low-reward, unlikely response (blunder) → low delight → suppress
- Mediocre, likely response → moderate delight → standard weight

### Experimental Plan

1. **Task:** Math or code generation with verifiable rewards
2. **Generate:** N=16 responses per prompt using a 1.5B model
3. **Training variants:**
   - **Standard rejection sampling:** Fine-tune on top-1 response only
   - **Weighted rejection sampling:** Fine-tune on all N, weighted by reward
   - **DG-weighted:** Fine-tune on all N, weighted by σ(reward × surprisal / η)
4. **Metric:** Pass@1 on held-out problems after K training steps, stratified by difficulty
5. **Ablation:** N ∈ {4, 8, 16, 32}, η ∈ {0.5, 1.0, 2.0}

### Why This Is Interesting

This bridges the DG mechanism with the offline/off-policy setting. The paper only considers on-policy DG. Showing it works for offline data reweighting extends the contribution.

---

## 5. DG × GAE Interaction

**Priority: Medium-low | Compute: Low | Novelty: Medium**

### Question

GAE (Generalized Advantage Estimation) is itself a variance reduction technique. Does DG's benefit stack with GAE, or are they redundant?

### Hypothesis

- Mechanism 1 (within-context variance reduction) may overlap with GAE → subadditive
- Mechanism 2 (cross-context directional improvement) is orthogonal to GAE → benefits should stack

### Experimental Plan

1. **Environment:** DeepMind Control Suite (paper's setup: actor-critic MLP, 10M steps)
2. **Grid:** GAE λ ∈ {0, 0.5, 0.9, 0.95, 1.0} × {PG, DG}
3. **Measure:** Final reward, aggregate regret, and separately the two mechanism diagnostics if feasible
4. **Prediction:** DG's advantage should persist at high λ (where GAE already reduces variance well), confirming the directional mechanism is the dominant effect in practice

### Feasibility

The paper's Control Suite runs on a single GPU. The full 5×2 grid runs in parallel on 4 GPUs overnight.

---

## 6. Convergence Properties (Theory-Focused)

**Priority: Low | Compute: Negligible | Novelty: Medium**

### Open Questions From the Paper

- Does DG converge to the same fixed point as standard PG?
- Is the convergence rate faster or slower?
- What happens at η < 1/2 where Proposition 2 breaks?
- Does DG introduce fixed points that PG doesn't have?

### Experimental Plan (Empirical Theory)

All tabular — runs on CPU, no GPU needed:

1. **Very long runs** (10^6 steps) on symmetric K-armed bandits with K ∈ {3, 10, 100, 1000}
2. Track ε = 1 − π(y*) for both PG and DG to convergence
3. Measure convergence rate: fit ε(t) ~ t^{-α} and compare α
4. Sweep η ∈ {0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 10.0} — especially the η < 1/2 regime
5. Multi-context version: N=100 contexts with varying difficulty, check whether DG and PG converge to the same policy

### Potential Theoretical Direction

The softplus potential Ψ_η(χ) = η · softplus(χ/η) from Appendix A might enable a Lyapunov-style convergence proof. The strict concavity of the entropy-regularized objective gives a unique optimal gate — this structure might yield a contraction argument.

---

## Suggested Execution Order

```
Week 1:  Idea 3 (η schedules — quick wins, builds intuition)
         Idea 6 (tabular convergence — runs on CPU alongside GPU work)

Week 2-3: Idea 2 (GRPO diagnostics — sets up infrastructure for Idea 1)

Week 3-5: Idea 1 (DG for LLM GRPO — the main experiment)

Week 5-6: Idea 4 (rejection sampling — if Idea 1 shows promise)
          Idea 5 (GAE interaction — if Control Suite results are interesting)
```

---

## References

- Delightful Policy Gradient (Osband, 2026) — [paper](https://arxiv.org/abs/2603.14608v1)
- GRPO (Shao et al., 2024) — DeepSeekMath
- Nemotron-Cascade 2 (Yang et al., 2026) — Cascade RL with GRPO
- Kelly Criterion (Kelly, 1956) — optimal long-run compounding
