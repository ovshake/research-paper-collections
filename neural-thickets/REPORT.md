# Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights

**Authors:** Yulu Gan, Phillip Isola (MIT CSAIL)
**Date:** 2026-03-12
**Paper:** [PDF](https://arxiv.org/abs/2603.12228)
**Project:** https://thickets.mit.edu | **Code:** https://github.com/sunrainyg/RandOpt

---

## TL;DR

Pretrained LLM weights are not a single point to be optimized from — they define a *distribution* over diverse task-specialist models. In large, well-pretrained models, the local neighborhood is so densely populated with task-improving perturbations ("thickets") that **random guessing + ensembling (RandOpt)** matches or beats PPO, GRPO, and ES for post-training, with O(1) wall-clock time and zero gradient computation.

---

## Key Novel Ideas

### 1. The "Thicket" Phenomenon — Dense, Diverse Specialists Surround Pretrained Weights

The central discovery. The paper reframes pretrained weights not as a starting point for optimization, but as the center of a *distribution* over models:

- **Solution Density** (Definition 2.1): The probability that a random Gaussian perturbation improves task performance. Measured as `delta(m) = P[s(theta + epsilon) >= s(theta) + m]` where `epsilon ~ N(0, sigma^2 I)`.
- For large models (32B Qwen2.5), **60-80% of random perturbations improve performance** on tasks like GSM8K and Countdown. For small models (0.5B), this drops to near zero.
- **Scaling law**: Solution density increases monotonically with model scale across all performance thresholds.

**Why this matters:** It overturns the assumption that post-training requires sophisticated optimization. For sufficiently large models, the loss landscape is so favorable that brute-force search works.

### 2. Spectral Discordance — Perturbations Are Specialists, Not Generalists

The paper tests two hypotheses about what random perturbations do:
- H1 (Generalists): There's a uniformly better model nearby
- H2 (Specialists): Perturbations trade off — improving one task while hurting others

**Result: H2 wins.** They introduce *Spectral Discordance* (Definition 2.2) — measuring how orthogonal the task-ranking vectors are across perturbations. Discordance increases monotonically with model scale (from ~0.45 at 0.5B to ~1.1 at 32B), meaning larger models have more specialized, diverse neighborhoods.

This is visualized beautifully: PCA of 7-dimensional performance vectors shows distinct clusters of specialists (math experts, code experts, chemistry experts), and RGB composites of task-accuracy landscapes show a richly multi-colored pattern rather than uniform gray.

### 3. RandOpt — Random Guessing as Post-Training

A deliberately simple algorithm that exploits thicket properties:

**Training phase:**
1. Sample N random Gaussian perturbations: `theta_i = theta + sigma_i * epsilon(s_i)`
2. Evaluate each on a small training/validation set
3. Select top-K performers

**Inference phase:**
- Generate predictions from all K models, aggregate via majority vote

Key properties:
- **O(1) wall-clock time** — all N evaluations are embarrassingly parallel, no sequential optimization steps
- **Zero gradients** — no backpropagation at all
- **Fully decentralized** — workers never communicate during training, only at inference via ensembling

### 4. Three Regimes of the Loss Landscape

Demonstrated via a minimal 1D signal prediction experiment:

| Regime | Condition | Behavior |
|---|---|---|
| **Needle in Haystack** | No/insufficient pretraining | Good solutions are vanishingly rare; requires structured search (gradient descent) |
| **Thicket** | Diverse pretraining (mixed signals) | Dense, diverse solutions nearby; random search works |
| **Plateau** | Narrow pretraining (one signal type) | Pretrained weights already minimize the test task; no room for improvement |

**Critical insight:** Thickets emerge specifically from *diverse* pretraining. Training on many signal types creates a "jack of all trades" whose neighborhood contains diverse specialists.

### 5. Distillation — Collapsing the Ensemble into One Model

RandOpt's main practical limitation is K forward passes at inference. The paper shows this can be resolved:
- Generate 25,000 responses from top-50 models on 500 hard training examples
- SFT the base model on selected (question, reasoning trace, answer) triples for 2 epochs
- **Cost: ~2% of training compute**
- Result: Distilled single model achieves accuracy comparable to the K=50 ensemble (e.g., Qwen2.5-3B on GSM8K: Base 79.8, Distill 84.3, RandOpt ensemble 87.1)

### 6. Types of Thickets — Not Just Reasoning

Decomposition of gains on GSM8K (Figure 9) reveals two distinct types of thickets:
- **Reasoning thickets** (12.3%): Perturbations that enable solving previously unsolvable problems
- **Format thickets** (19.0%): Perturbations that fix answer formatting so checkers can parse correct answers

Both contribute substantially. This suggests thickets are not one-dimensional — they include variations in reasoning approaches, answer formats, personalities, domain knowledge, etc.

---

## Architecture Details

Not applicable — this is a methods/analysis paper that works with existing model architectures (Qwen2.5, Llama 3.1, OLMo3) without modifying them.

**RandOpt Hyperparameters:**

| Parameter | Typical Value | Notes |
|---|---|---|
| Population size (N) | 2,000 - 5,000 | More is better; log-linear scaling |
| Ensemble size (K) | 50 | Top-K from N |
| Noise scales (Sigma) | {0.001, 0.002, ..., 0.01} | Set of scales; each seed gets one uniformly sampled |
| Perturbation distribution | N(0, I_d) | Standard Gaussian, scaled by sigma |
| Ensembling method | Majority vote | At inference over K model outputs |
| Models tested | 0.5B - 32B | Qwen2.5, Llama-3.1-8B, OLMo3-7B |

---

## Key Results

### RandOpt vs. Standard Post-Training (Qwen2.5-3B-Instruct, equal FLOPs)

| Benchmark | Base | PPO | GRPO | ES | RandOpt (K=50) |
|---|---|---|---|---|---|
| Countdown (math) | 37.6 | ~45 | ~54 | ~54 | **58.8** |
| GSM8K (math) | 79.8 | ~78 | ~83.5 | ~82 | **87.1** |
| MATH-500 (math) | 43.0 | ~48 | ~54 | ~54 | **59.7** |
| OlyBench (math) | ~35 | ~37 | ~37 | ~35 | **39.4** |
| MBPP (code) | ~60 | ~64 | ~70 | ~69 | **70.3** |
| ROCStories (writing) | ~53 | ~44 | ~49 | ~49 | **60.3** |
| USPTO (chemistry) | ~30 | ~31 | ~32 | ~31 | **34.5** |

RandOpt matches or exceeds all baselines across most model scales (0.5B-8B) and task categories, even when baselines are given equivalent ensembling (50-pass TT-MV).

### Solution Density Scaling (Qwen2.5, sigma=0.005)

| Model Scale | GSM8K (>=100% base) | Countdown (>=100% base) |
|---|---|---|
| 0.5B | ~0% | ~8% |
| 1.5B | ~18% | ~45% |
| 3B | ~37% | ~54% |
| 32B | **~64%** | **~60%** |

### Vision-Language Model (Qwen2.5-VL-3B-Instruct)

| Method | GQA Accuracy |
|---|---|
| Base | 56.6 |
| RandOpt | **69.0** (+12.4%) |

### Distillation Results (GSM8K)

| Model | Base | Distill | RandOpt (K=50) |
|---|---|---|---|
| Qwen2.5-1.5B-Inst | 58.8 | 74.9 | 76.4 |
| Qwen2.5-3B-Inst | 79.8 | 84.3 | 87.1 |

### Wall-Clock Efficiency

RandOpt with N=2000, K=50 on a 200-GPU GH200 cluster: **3.2 minutes** to post-train OLMo-3-7B-Instruct on Countdown (achieves 70% accuracy).

---

## Key Takeaways

1. **Pretrained weights are a distribution, not a point.** The paper's most profound conceptual contribution. Rather than thinking of the pretrained model as "the base model" to fine-tune, think of it as specifying a Gaussian distribution over models whose samples include diverse task specialists. This reframes post-training as *selection* rather than *optimization*.

2. **Solution density scales as a power law with model size.** At 32B parameters, 60-80% of random perturbations improve task performance. At 0.5B, it's near zero. This means the thicket phenomenon is an emergent property of scale — small models live in a "needle in haystack" regime where structured search (gradient descent) is essential.

3. **Random guessing is competitive with GRPO/PPO/ES — but only for large enough models.** RandOpt fails below ~1.5B parameters (Figure 8). The takeaway is not "throw away gradient descent" but rather "once you have a good enough representation, the choice of adaptation algorithm matters less than you'd think."

4. **The flat pretraining loss landscape hides rich per-task structure.** The aggregate pretraining loss is smooth and flat near the optimum (consistent with prior work on flat minima). But decomposing into individual task losses reveals a spiky, multi-colored landscape. The "flatness" is an artifact of averaging over many tasks.

5. **Ensembling is critical — K=1 underperforms significantly.** RandOpt with K=1 (best single perturbation) is much weaker than K=50. The diversity of the thicket is not just a curiosity; it's practically necessary for the approach to work. The specialists are complementary, and majority voting aggregates their strengths.

6. **Thicket density explains why spurious rewards can work.** Recent findings show post-training sometimes improves with random or spurious reward signals. The thicket perspective explains this: if most random perturbations improve *something*, even a noisy reward signal will select beneficial perturbations.

7. **Distillation can compress the ensemble at ~2% additional cost.** The K forward passes at inference can be collapsed into a single model via SFT distillation on the ensemble's outputs, recovering most of the accuracy gain without the inference overhead.

8. **Both format and reasoning thickets contribute substantially.** On GSM8K, 19% of gains come from format fixes and 12.3% from genuine reasoning improvements. This is important for interpreting benchmark results — gains from random perturbation are not purely superficial, nor purely deep.

9. **RandOpt is maximally parallelizable and decentralized.** Zero communication between workers during training, O(1) wall-clock time regardless of N. This makes it attractive for settings where compute is cheap but communication is expensive (federated learning, privacy-sensitive deployments).

10. **The connection to MAML and meta-learning is compelling.** The thicket phenomenon suggests that pretraining implicitly learns a MAML-like initialization — a point in weight space from which task-specific adaptations are easy to find. But unlike MAML, this isn't explicitly optimized for; it emerges naturally from diverse pretraining.

---

## What's Open-Sourced

- **Code:** https://github.com/sunrainyg/RandOpt (RandOpt implementation)
- **Project page:** https://thickets.mit.edu (visualizations and additional results)
- **Models used:** All experiments use publicly available models (Qwen2.5, Llama 3.1, OLMo3)
- **Datasets:** All benchmarks are public (Countdown, GSM8K, MATH-500, OlyBench, MBPP, ROCStories, USPTO, GQA)
