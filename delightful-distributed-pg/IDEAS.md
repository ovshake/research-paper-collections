# Delightful DPO: Token-Level Cross-Pollination in Preference Learning

## The Problem with Standard DPO

Standard DPO treats preferences as a hard binary label on entire sequences. Given a prompt $x$, a winning response $y_w$, and a losing response $y_l$, the loss is:

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\Bigl(\beta \sum_t \underbrace{\log \frac{\pi_\theta(y_{w,t} \mid y_{w,<t}, x)}{\pi_\text{ref}(y_{w,t} \mid y_{w,<t}, x)}}_{\Delta_t^w} - \beta \sum_t \underbrace{\log \frac{\pi_\theta(y_{l,t} \mid y_{l,<t}, x)}{\pi_\text{ref}(y_{l,t} \mid y_{l,<t}, x)}}_{\Delta_t^l}\Bigr)$$

The gradient pushes **every** token in $y_w$ up and **every** token in $y_l$ down, with the same sequence-level weight $(1 - \sigma(\beta \cdot \Delta))$.

This is wasteful and sometimes destructive:

**Example:** Suppose $y_l$ = "Let me think step by step. [excellent 10-step chain of thought] ... The answer is 43" (wrong final number) and $y_w$ = "The answer is 42" (correct but no reasoning). Standard DPO pushes down the entire excellent chain of thought in $y_l$ — including the 10 correct reasoning steps — just because the last token was wrong. Meanwhile it pushes up the terse $y_w$ uniformly.

This is exactly analogous to the pathology DG identifies in policy gradients: standard PG applies the same sequence-level advantage to every token in a trajectory, even though some tokens were brilliant moves and others were blunders. DG fixes this by gating per-token. We want the same fix for DPO.

## The Core Idea: Learn Good Parts from Both Responses

Instead of treating preferences as sequence-level binary labels, decompose the preference signal to the token (or span) level. Then:

1. **Strongly increase** probability of tokens in $y_w$ that are genuinely good (high-quality AND the model hasn't fully learned them yet)
2. **Weakly increase** probability of tokens in $y_w$ that are filler or already well-learned
3. **Preserve or increase** probability of tokens in $y_l$ that are genuinely good (good reasoning steps in the losing response)
4. **Decrease** probability of tokens in $y_l$ that are genuinely bad (the parts that actually caused the loss)

Points 1-2 are analogous to DG's amplification of breakthroughs and attenuation of common actions. Points 3-4 are the novel **cross-pollination**: learning good patterns from the losing response.

## Token-Level Delight for DPO

### What plays the role of advantage and surprisal?

In DG:
- **Advantage** $U_t$: how good was this action relative to baseline
- **Surprisal** $\ell_t = -\log \pi_\theta(A_t)$: how unexpected this action is under the current policy
- **Delight** $\chi_t = U_t \cdot \ell_t$

In DPO, we need analogous quantities per token:

**Token surprisal** (direct analog):

$$\ell_t = -\log \pi_\theta(y_t \mid y_{<t}, x)$$

This measures how much the model still has to learn about this token. Tokens the model already assigns high probability have low surprisal — learning from them is redundant.

**Token advantage** (requires more thought): We need a signal for "is this token good or bad for the response quality?" Several options exist, but the cleanest self-contained one uses the DPO implicit reward.

### Recommended Approach: Implicit Reward Decomposition

DPO defines an implicit reward $r_\theta(y, x) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$. Decompose this per-token:

$$r_t = \beta \log \frac{\pi_\theta(y_t \mid y_{<t}, x)}{\pi_\text{ref}(y_t \mid y_{<t}, x)}$$

The sign of $r_t$ tells us whether the model has moved this token's probability *up* (positive) or *down* (negative) relative to the reference.

**Token advantage for the winning response:**
$$u_t^w = -r_t^w$$

Negative implicit reward in $y_w$ means "the model hasn't learned to produce this token yet" — high value, worth emphasizing. Positive implicit reward means "already learned" — low marginal value.

**Token advantage for the losing response:**
$$u_t^l = r_t^l$$

Positive implicit reward in $y_l$ means "the model has been moving toward this token" — it's likely a shared good pattern. Negative means "the model already avoids this" — probably a genuinely bad pattern.

**Token delight:**

$$\chi_t^w = u_t^w \cdot \ell_t^w = \Bigl(-\beta \log \frac{\pi_\theta(y_{w,t})}{\pi_\text{ref}(y_{w,t})}\Bigr) \cdot \bigl(-\log \pi_\theta(y_{w,t})\bigr)$$

$$\chi_t^l = u_t^l \cdot \ell_t^l = \Bigl(\beta \log \frac{\pi_\theta(y_{l,t})}{\pi_\text{ref}(y_{l,t})}\Bigr) \cdot \bigl(-\log \pi_\theta(y_{l,t})\bigr)$$

High delight in $y_w$: the model hasn't learned this token yet AND it's surprising → a **breakthrough** worth emphasizing.

High delight in $y_l$: the model has been moving toward this token AND it's still surprising → a **good pattern worth preserving** (cross-pollination candidate).

## Formulation: D²PO (Delightful DPO)

### The Three-Term Gradient

Standard DPO gradient pushes all of $y_w$ up, all of $y_l$ down. D²PO adds per-token gates and a cross-pollination term:

**Term 1 — Gated winning-side preference:**
$$\sum_t \sigma(\chi_t^w / \eta) \cdot (1-\sigma(\beta\Delta)) \cdot \nabla_\theta \log \pi_\theta(y_{w,t} \mid \ldots)$$

Emphasize breakthroughs in $y_w$; de-emphasize filler.

**Term 2 — Gated losing-side preference:**
$$-\sum_t \sigma(-\chi_t^l / \eta) \cdot (1-\sigma(\beta\Delta)) \cdot \nabla_\theta \log \pi_\theta(y_{l,t} \mid \ldots)$$

Push down bad parts of $y_l$. But the gate $\sigma(-\chi_t^l / \eta)$ *closes* for tokens with positive delight — good parts of the losing response are protected.

**Term 3 — Cross-pollination:**
$$+\gamma \sum_t \sigma(\chi_t^l / \eta) \cdot \mathbb{1}[\chi_t^l > 0] \cdot \nabla_\theta \log \pi_\theta(y_{l,t} \mid \ldots)$$

Actively push UP high-delight tokens from $y_l$. These are the "good parts of the loser." Weight $\gamma$ controls cross-pollination strength.

### Algorithm

```
Input: prompt x, winning y_w, losing y_l, policy π_θ, reference π_ref
Hyperparameters: β (DPO temp), η (delight temp, default 1), γ (cross-pollination weight)

# Forward pass — compute per-token signals
for each token t in y_w:
    ℓ_t^w = -log π_θ(y_{w,t} | y_{w,<t}, x)           # surprisal
    r_t^w = β · log(π_θ(y_{w,t}) / π_ref(y_{w,t}))     # implicit reward
    u_t^w = -r_t^w                                       # advantage: value in unlearned tokens
    χ_t^w = u_t^w · ℓ_t^w                               # delight

for each token t in y_l:
    ℓ_t^l = -log π_θ(y_{l,t} | y_{l,<t}, x)
    r_t^l = β · log(π_θ(y_{l,t}) / π_ref(y_{l,t}))
    u_t^l = r_t^l                                        # positive = model already likes this
    χ_t^l = u_t^l · ℓ_t^l

# Sequence-level preference margin (standard DPO)
Δ = Σ_t r_t^w - Σ_t r_t^l
pref_weight = 1 - σ(Δ)

# Per-token gated gradient
For y_{w,t}:  weight = +σ(χ_t^w / η) · pref_weight       # gated push up
For y_{l,t}:  weight = -σ(-χ_t^l / η) · pref_weight      # gated push down
                       + γ · σ(χ_t^l / η) · 1[χ_t^l > 0]  # cross-pollination
```

## Why the Product (Not Sum) Matters Here Too

DG proved (Proposition 2) that additive combinations $\alpha U + (1-\alpha)\ell$ are sign-inconsistent: they can rank a failing rare action above a succeeding common one. The same problem transfers to DPO:

- **Surprisal alone** would protect ALL rare tokens in $y_l$, including rare bad patterns (e.g., a hallucinated entity name is rare and high-surprisal, but terrible)
- **Implicit reward alone** would protect tokens the model already likes, even if they're common and carry no information gain
- **The product** protects tokens that are both "model already likes them" AND "still surprising" — the intersection, not the union

This is the sign-consistency property: $\chi = u \cdot \ell$ is positive only when both factors agree on direction.

## Connections to Existing Work

| Method | Token-level? | Cross-pollination? | External model needed? |
|--------|:-----------:|:------------------:|:---------------------:|
| Standard DPO | No | No | No |
| Token-Level DPO (TDPO) | Yes | No | No |
| Step-DPO | Yes (per-step) | No | Yes (PRM) |
| Selective Reflection-Tuning | Span-level | Yes (via rewrite) | Yes (rewriter) |
| **D²PO (this idea)** | **Yes** | **Yes** | **No** |

D²PO is the first method that does both token-level gating AND cross-pollination using only signals available from the DPO forward pass itself.

## Potential Issues

### 1. Cross-Pollination Signal Noise
If token delight in $y_l$ is poorly calibrated, we might reinforce bad patterns. **Mitigation:** start with $\gamma$ small; require a minimum delight threshold; compute $\chi_t^l$ under a frozen snapshot of $\pi_\theta$ (stop gradient through delight).

### 2. Reward Hacking
If the model discovers that keeping $y_l$ tokens high-probability increases cross-pollination reward, it could game the system. **Mitigation:** compute $\chi_t^l$ using a detached/frozen copy of $\pi_\theta$, or EMA of the policy.

### 3. Sequence Coherence
Token-level signals can disagree with sequence-level preferences. A response can be "locally good everywhere but globally incoherent." **Mitigation:** the DPO preference weight $(1-\sigma(\beta\Delta))$ provides the global coherence signal; the gates modulate it, not replace it.

### 4. Gambling Pathology Transfers
DG's gambling pathology (high reward variance makes lucky draws look like breakthroughs) has a DPO analog: if preference labels are noisy (annotator disagreement), some tokens in $y_l$ might get high delight from label noise. **Mitigation:** use preference pairs with high annotator agreement; or weight cross-pollination by preference confidence.

## Suggested Experiments

1. **Math reasoning (GSM8K/MATH):** Many losing responses have correct intermediate steps. D²PO should learn from these while standard DPO pushes them all down. Metric: accuracy AND quality of intermediate reasoning.

2. **Code generation (HumanEval/MBPP):** Losing code often contains correct helper functions with a single bug. D²PO should preserve the good parts. Metric: pass@1 and edit distance to ground truth.

3. **Ablation: γ sweep.** Sweep cross-pollination weight from 0 (gated DPO, no cross-pollination) to 1.0. Expect an optimal range.

4. **Ablation: multiplicative vs additive token quality.** Compare $\chi = u \cdot \ell$ against $\alpha u + (1-\alpha)\ell$ to verify the product structure transfers.

5. **Visualization: cross-pollination targets.** Display which tokens in $y_l$ get high $\sigma(\chi_t^l / \eta)$ across training. Expect: correct reasoning steps get protected; actual errors get pushed down.

6. **Comparison: D²PO vs TDPO vs Step-DPO.** On the same preference data, compare (a) standard DPO, (b) TDPO (token-level but no cross-pollination), (c) Step-DPO if PRM available, (d) D²PO. The hypothesis: D²PO matches Step-DPO without needing a PRM and beats TDPO via cross-pollination.
