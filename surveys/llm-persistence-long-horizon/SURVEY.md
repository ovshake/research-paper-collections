# Encouraging LLMs to Persist in Long-Horizon Tasks
## A Literature Survey of Techniques That Keep Models Trying After Failure

**Compiled:** April 29, 2026
**Scope:** Methods (prompting, decoding, training, scaffolding) that prevent LLMs from prematurely abandoning multi-step tasks while avoiding unproductive thrashing.

---

## 1. The Problem, Stated Precisely

An LLM working on a long-horizon task — say, fixing a real GitHub issue, completing a 30-step web flow, or solving an Olympiad problem — must keep generating actions across many turns before any reward signal arrives. Two failure modes dominate empirically:

- **Premature termination:** the model emits an `<end>`, `terminate`, "I cannot help" or simply stops generating useful actions before the task is solved.
- **Unproductive thrashing:** the model keeps generating actions but loops repetitively, hallucinates progress, or wavers between contradictory answers — what Beyond pass@1 (arXiv:2603.29231) calls the "meltdown onset point."

The literature on persistence is essentially the literature on **walking the line between these two failures**. Every mechanism that pushes a model to keep trying risks pushing it into thrashing; every mechanism that prevents thrashing risks premature giving-up. The 2024-2026 work increasingly recognizes this duality and frames the problem as **adaptive persistence**: keep going on hard problems, stop on easy ones, and detect which is which from internal signals.

---

## 2. A Unified Taxonomy of Mechanisms

Six families of techniques have emerged, ordered roughly from prompt-level (cheap, model-agnostic) to training-level (expensive, durable):

| Family | What It Does | Where the Persistence Comes From |
|---|---|---|
| **A. Verbal self-correction** | Re-prompts the model with its own failure | Episodic memory of past attempts |
| **B. Test-time compute forcing** | Extends reasoning length at inference | Token-level intervention or learned budget |
| **C. Tree search & backtracking** | Explores multiple branches | Structural alternatives to single chain |
| **D. RL & reward shaping** | Trains the policy via dense/turn-aware rewards | Process reward models, curriculum, exploration bonuses |
| **E. Scaffold-level engineering** | Production retry budgets, ledgers, compaction | Outer loop above the model |
| **F. Calibration & anti-quitting** | Fixes confidence miscalibration that causes bail | Trained confidence signals, IDK regularization |

Sections 3-8 expand each family with mechanisms, key papers, and empirical evidence.

---

## 3. Family A — Verbal Self-Correction Loops

Re-prompt the model with feedback after failure, store the critique in memory, repeat.

### Methods using **external feedback** (works reliably)

- **Reflexion** (Shinn et al., NeurIPS 2023; [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)): Three-component loop — Actor generates, Evaluator scores from environment (test pass/fail, task success), Self-Reflection produces natural-language diagnoses stored in episodic memory and prepended to the next trial. **91% pass@1 on HumanEval** vs. 80% baseline. Pure prompt-based, no weight updates. Persistence trigger is the Evaluator's binary signal.
- **Self-Debug** (Chen et al., ICLR 2024; [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)): Code-execution feedback loop with rubber-duck-style explanations. +2-12% across SQL, MBPP, TransCoder. Matches baselines that sample 10x more candidates.
- **CRITIC** (Gou et al., ICLR 2024; [arXiv:2305.11738](https://arxiv.org/abs/2305.11738)): Verify-then-correct via tools (search, calculator, code interpreter). The paper's headline finding: **intrinsic self-feedback without tools has "limited and task-specific performance"** — external grounding is the load-bearing ingredient.
- **RCI** (Kim et al., 2023; [arXiv:2303.17491](https://arxiv.org/abs/2303.17491)): Recursively criticize-and-improve loop for computer task automation; SOTA on MiniWoB++.

### Methods using **pure self-feedback** (often fails)

- **Self-Refine** (Madaan et al., NeurIPS 2023; [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)): Same model serves as generator, critic, refiner. ~20% gains on subjective-quality tasks (dialogue, code optimization). Limited on formal reasoning.
- **"LLMs Cannot Self-Correct Reasoning Yet"** (Huang et al., ICLR 2024; [arXiv:2310.01798](https://arxiv.org/abs/2310.01798)): Foundational negative result. Intrinsic self-correction without oracle labels often **degrades** reasoning. Many reported successes secretly leak external signals (test cases, ground truth in prompts).
- **"Dark Side of Self-Correction"** (2024; [arXiv:2412.14959](https://arxiv.org/abs/2412.14959)): Documents **answer wavering** — GPT-3.5 changes 81.3% of answers more than 6 times across 10 self-correction rounds.

### Bridging the gap via training

- **SCoRe** (Kumar et al., 2024; [arXiv:2409.12917](https://arxiv.org/abs/2409.12917)): Multi-turn online RL that rewards improvement on the second attempt. The first work to show that genuine intrinsic self-correction requires **training**, not prompting. SFT-based self-correction fails due to (1) distribution mismatch — SFT correction traces don't match the errors the model actually makes, and (2) behavior collapse — narrow correction modes that don't generalize.

### Practical guidance

- **2-3 rounds is the sweet spot** ([arXiv:2604.10508](https://arxiv.org/abs/2604.10508)). Most gains concentrate in rounds 1-2; beyond that, diminishing returns and answer wavering risk dominate.
- The dichotomy is sharp: **with external feedback, refine; without it, don't** (or use SCoRe-style training).

---

## 4. Family B — Test-Time Compute Forcing

Make the model keep generating reasoning tokens at inference, either by force or by trained budget.

### Forced continuation

- **s1: Simple Test-Time Scaling** (Muennighoff et al., 2025; [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)): The canonical "force the model not to stop" paper. **Budget forcing** appends the literal string "Wait" whenever the model emits an end-of-thinking token, forcing it to re-examine its answer. SFT on 1,000 curated traces (s1K) + this inference-time hack lifts AIME24 from 50% → 57% on Qwen2.5-32B; matches o1-preview on MATH.
- **DeepSeek-R1** ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)): The "aha moment" — GRPO with rule-based outcome rewards alone causes **spontaneous emission of "Wait"** mid-reasoning. No supervised reasoning traces. Provides natural evidence that persistence is an emergent property of correct-answer rewards on hard problems.
- **Quiet-STaR** (Zelikman et al., 2024; [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)): Trained internal rationales at every token position via REINFORCE. Zero-shot GSM8K 5.9% → 10.9%, CommonsenseQA 36.3% → 47.2%.

### Parallel sampling + verification

- **Self-Consistency** (Wang et al., 2022; [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)): Sample N reasoning paths, majority vote. The model "persists" by producing parallel attempts rather than one longer chain.
- **Best-of-N + verifier** (Cobbe 2021 [arXiv:2110.14168](https://arxiv.org/abs/2110.14168), Lightman 2023 [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)): Sample N candidates, rank with a trained verifier (outcome RM or process RM).
- **Parallel-Distill-Refine (PDR)** ([arXiv:2510.01123](https://arxiv.org/abs/2510.01123), already in your repo as `rethinking-thinking-tokens`): Decouples sequential budget B_seq (latency) from total budget B_total (compute). +11% AIME 2024 vs long CoT at matched compute, with better latency.

### Trained adaptive budgets — knowing when to stop

- **DeepConf** (Meta, 2025; [arXiv:2508.15260](https://arxiv.org/abs/2508.15260)): Sliding-window confidence terminates traces when group confidence drops; 99.9% on AIME 2025 with **84.7% fewer tokens**. Training-free.
- **SelfBudgeter** ([arXiv:2505.11274](https://arxiv.org/abs/2505.11274)): Trains the model to **predict its required token budget** before reasoning, then enforces it via budget-guided GRPO. 61% length compression with negligible accuracy loss.
- **Don't Overthink It** ([arXiv:2505.17813](https://arxiv.org/abs/2505.17813)) and **Reasoning Completion Point** ([arXiv:2512.19585](https://arxiv.org/abs/2512.19585)): The negative-results companion. Shorter chains are **34.5% more likely correct** than the longest chain for the same question. Past the Reasoning Completion Point, additional tokens have **negative marginal returns** — redundant loops, erroneous self-corrections.

### Search-based inference exploration

- **Entropy-Gated Branching** ([arXiv:2503.21961](https://arxiv.org/abs/2503.21961)): Branch only at high-entropy reasoning steps (which empirically predict errors); +22.6% accuracy, 31-75% faster than beam search.
- **Latent Exploration Decoding** ([arXiv:2602.01698](https://arxiv.org/abs/2602.01698)): Aggregates intermediate-layer posteriors to combat **entropy collapse** from RL post-training (a key persistence-killing pathology).
- **"Oops, Wait" Token-Level Signals** ([arXiv:2601.17421](https://arxiv.org/abs/2601.17421)): "Wait" and "therefore" tokens are mutual-information peaks correlated with correctness — validating that s1's hack is grounded in real computational structure.

---

## 5. Family C — Tree Search & Backtracking

Explore multiple branches structurally so a wrong step doesn't kill the attempt.

### Inference-only search around a frozen LLM

- **Tree of Thoughts** (Yao et al., NeurIPS 2023; [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)): BFS/DFS over thought branches with self-evaluation. Canonical persistence demo: **74% on Game of 24 vs. 4% for CoT.**
- **Reasoning via Planning (RAP)** (Hao et al., EMNLP 2023; [arXiv:2305.14992](https://arxiv.org/abs/2305.14992)): MCTS with the LLM as both policy and world model. UCB exploration forces revisiting under-explored branches.
- **LATS** (Zhou et al., ICML 2024; [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)): MCTS using **environment interaction** (not internal world model) plus reflections propagated up the tree.
- **Verifier-guided / PRM-guided search**: Cobbe verifiers, Math-Shepherd ([arXiv:2312.08935](https://arxiv.org/abs/2312.08935)) PRM-guided beam search. Mistral-7B 28.6% → 43.5% on MATH.

### Training-time methods that internalize search

- **Stream of Search** (Gandhi et al., 2024; [arXiv:2404.03683](https://arxiv.org/abs/2404.03683)): Represents the **entire search process — including exploration, dead ends, backtracking — as a flat token sequence** and trains on it. +25% search accuracy on Countdown vs. optimal-only training.
- **Searchformer** (Meta, 2024; [arXiv:2402.14083](https://arxiv.org/abs/2402.14083)): Trained to predict A* search dynamics. Solves Sokoban 93.7% optimally with **26.8% fewer search steps than A* itself**.
- **Self-Backtracking** (Yang et al., 2025; [arXiv:2502.04404](https://arxiv.org/abs/2502.04404)): Special backtrack tokens; +40% accuracy gain over optimal-path SFT. The single sharpest demonstration that **teaching a model that dead ends are normal fundamentally changes persistence behavior**.
- **ASTRO** ([arXiv:2507.00417](https://arxiv.org/abs/2507.00417)): Models trained to explore, reflect, and backtrack within a single inference pass.

### Search distilled into the model (the 2025 frontier)

- **rStar-Math** (Microsoft, 2025; [arXiv:2501.04519](https://arxiv.org/abs/2501.04519)): MCTS + process preference model + 4 rounds of self-evolution. Qwen2.5-Math-7B 58.8% → **90.0% on MATH**, surpassing o1-preview. Top-20% of human IMO performance.
- **rStar2-Agent** ([arXiv:2508.20722](https://arxiv.org/abs/2508.20722)): Agentic extension; 14B model with GRPO-RoC achieves 80.6% on AIME24, **surpassing 671B DeepSeek-R1 with shorter responses**.
- **AlphaMath Almost Zero** ([arXiv:2405.03553](https://arxiv.org/abs/2405.03553)): MCTS produces both correct and incorrect paths with value estimates; trains policy and value models without human annotations.

### Caveat

PRM-guided tree search sometimes **shows no statistically significant gain** over best-of-N ([arXiv:2510.20272](https://arxiv.org/abs/2510.20272)) — PRM quality is the bottleneck.

---

## 6. Family D — RL & Reward Shaping (the bulk of the 2024-2026 literature)

Train the policy directly. The most active research area; four sub-axes.

### D1 — Richer reward signals (process supervision)

- **Lightman et al.** ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) and **Math-Shepherd** ([arXiv:2312.08935](https://arxiv.org/abs/2312.08935)): Step-level PRMs vs. outcome-only RMs. Step-level signal gives every intermediate action a gradient — without it, an agent that fails at step 8 of 15 receives no signal for the useful actions in steps 1-7.
- **AgentPRM** ([arXiv:2511.08325](https://arxiv.org/abs/2511.08325)): Step scores redefined as **promise** (proximity to goal) and **progress** (change in success likelihood); TD-based estimation with GAE — 8x more compute-efficient than rollout baselines.
- **ThinkPRM** ([arXiv:2504.16828](https://arxiv.org/abs/2504.16828)): Verbalized chain-of-thought verifier using only 1% of PRM800K labels.

### D2 — Better credit assignment over turns

The fundamental issue: standard GRPO/PPO operate at token-level MDPs, but agentic tasks have natural turn-level structure.

- **RAGEN / StarPO** (Wang et al., 2025; [arXiv:2504.20073](https://arxiv.org/abs/2504.20073)): Identifies the **"Echo Trap"** — reward variance cliffs cause agents to converge to repetitive templates. **StarPO-S** mitigates with trajectory-level reward normalization and entropy monitoring. *This is the most cited "premature mode collapse" paper.*
- **Turn-PPO** ([arXiv:2512.17008](https://arxiv.org/abs/2512.17008)): Turn-level MDP reformulation; PPO substantially more stable than GRPO on WebShop and Sokoban.
- **iStar** (ICLR 2026; [arXiv:2509.19199](https://arxiv.org/abs/2509.19199)): Implicit PRM via trajectory-DPO. SOTA on WebShop, VisualSokoban, and the unverifiable-reward SOTOPIA benchmark.
- **GTPO** ([arXiv:2511.14846](https://arxiv.org/abs/2511.14846)) and **Turn-Level Credit Assignment for Multi-Turn Reasoning** (ICML 2025; [arXiv:2505.11821](https://arxiv.org/abs/2505.11821)): Per-turn return-based advantages.

### D3 — Exploration bonuses & anti-collapse

- **SPEAR** (Tencent, ICLR 2026; [arXiv:2509.22601](https://arxiv.org/abs/2509.22601)): Self-imitation + **curriculum-staged entropy shaping** — early training maximizes entropy for tool diversity; later stages exploit. +16-21% on ALFWorld/WebShop over GRPO/GiGPO.
- **EMPG** ([arXiv:2509.09265](https://arxiv.org/abs/2509.09265)): Entropy-Modulated PG — avoids "punishing uncertain exploration harshly" while strongly correcting hallucinated confidence.
- **EPO** ([arXiv:2509.22576](https://arxiv.org/abs/2509.22576)): Phase-adaptive entropy smoothing breaks the "exploration-exploitation cascade failure" where sparse rewards cause premature convergence to flawed strategies. **+152% improvement on ScienceWorld.**
- **DAPO** ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)): Two specifically anti-collapse innovations — **Clip-Higher** (asymmetric clipping to maintain diversity) and **Soft Overlong Punishment** (length-aware penalty that doesn't punish sound-but-long reasoning).

### D4 — Length-aware shaping (the most direct "anti-give-up" mechanism)

- **Leash** ([arXiv:2512.21540](https://arxiv.org/abs/2512.21540)): **Adaptive length penalties scaling with estimated difficulty** — easy problems get strong brevity penalties, hard problems get slack to reason longer. Sharpest example of the "don't give up on hard problems" principle.
- **Progressive Reward Shaping** ([arXiv:2512.07478](https://arxiv.org/abs/2512.07478)): Stage-wise feedback — first reward parseable tool calls, then reward correctness. Bootstraps the agent past the cold-start abyss.
- **Shop-R1** (already in your repo): Caps `terminate` action reward at 0.8 to **prevent reward hacking via giving up early**. Concrete example of structurally disincentivizing premature termination — the 0.5B model overpredicted terminate (97.07%) absent this cap.

### D5 — Curriculum & difficulty-aware sampling

- **WebRL** (Qi et al., 2024; [arXiv:2411.02337](https://arxiv.org/abs/2411.02337)): **Self-evolving curriculum that generates new tasks from the agent's own failed attempts.** Llama-3.1-8B 4.8% → 42.4% on WebArena, beating GPT-4-Turbo (17.6%).
- **VCRL** ([arXiv:2509.19803](https://arxiv.org/abs/2509.19803)): Group-reward variance as difficulty proxy; upweights moderate-variance samples.
- **CDAS** ([arXiv:2505.17652](https://arxiv.org/abs/2505.17652)): Dynamic competence-difficulty alignment.

### D6 — Bootstrapping for long-horizon RL

The SFT-then-RL paper already in your repo ([arXiv:2604.23747](https://arxiv.org/abs/2604.23747)) is **directly relevant**: a model that hasn't been bootstrapped properly (or whose SFT was deflated by the documented DeepSpeed/OpenRLHF bugs) generates rollouts that produce no reward signal, making persistence un-trainable. SFT puts the model into a regime where it generates plausible-correct solutions, then RL refines toward higher-reward strategies. This is the structural argument for sequential training.

### Survey papers covering the entire field

- **The Landscape of Agentic RL for LLMs** (2025; [arXiv:2509.02547](https://arxiv.org/abs/2509.02547))
- **From Reasoning to Agentic: Credit Assignment Survey** (2026; [arXiv:2604.09459](https://arxiv.org/abs/2604.09459)) — catalogs 47+ papers on credit assignment alone

---

## 7. Family E — Scaffold-Level Engineering (Production Agents)

The often-overlooked layer above the model. Empirically, the AlphaEval paper in your repo found **scaffold matters as much as model** — same Opus 4.6 scores 53.45 via Codex vs. 64.41 via Claude Code, an 11-point spread.

### Coding agents

- **SWE-Agent** ([arXiv:2405.15793](https://arxiv.org/abs/2405.15793)): Agent-Computer Interface with **structured error feedback**: linter rolls back syntax-broken edits, shifts pre-existing errors so the agent isn't blamed, re-prompts. Configurable retry budgets (`cost_limit`, `max_attempts`, `min_budget_for_new_attempt`). Every termination path produces degraded-success autosubmit rather than crash.
- **Claude Code** ([arXiv:2604.14228](https://arxiv.org/abs/2604.14228) — recent paper): Five layered recovery mechanisms (max-output-token escalation up to 3 retries, reactive compaction, prompt-too-long handling, streaming fallback, fallback model substitution). **Five-stage compaction pipeline** (Budget Reduction → Snip → Microcompact → Context Collapse → Auto-Compact) ensures the agent never terminates from context overflow. Five stop conditions: no tool use, max turns, context overflow (handled by compaction), hook intervention, explicit abort.
- **OpenHands** ([arXiv:2511.03690](https://arxiv.org/abs/2511.03690)): Tenacity-based exponential backoff for transient errors; **event sourcing** (immutable event log) enables replay and incremental persistence so agent state survives crashes.

### Web/GUI agents

- **Magentic-One** ([arXiv:2411.04468](https://arxiv.org/abs/2411.04468)): **Dual-ledger architecture** — Task Ledger (facts, guesses, plan) and Progress Ledger (status, agent assignments). Orchestrator self-reflects on progress; if insufficient progress is detected for several steps, it revises the Task Ledger and replans. Explicit progress-detection prevents both premature termination and unproductive looping.
- **OS-Atlas** ([arXiv:2410.23218](https://arxiv.org/abs/2410.23218)) and **OSWorld** (NeurIPS 2024) for grounding/benchmarking.

### Embodied/game agents

- **Voyager** ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)): Three-component design — automatic curriculum, growing skill library, iterative prompting with self-verification. Discovers 160+ Minecraft items via continual exploration. Pure prompt-based; demonstrates that scaffold alone can produce indefinite persistence.
- **AutoGPT/BabyAGI** (the failure case, 2023): Recursive LLM-call architectures looped endlessly without progress tracking. **Persistence without progress tracking → thrashing.** BabyAGI archived September 2024.

### Cross-cutting failure-mode taxonomies

- **"Why Do Multi-Agent LLM Systems Fail?"** (Cemri et al., 2025; [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)): Across 1600+ annotated traces, **AppWorld suffers premature termination, OpenManus exhibits step repetition** — confirming the dual failure modes. Premature termination is a documented failure class FM-3.1 (6.2% of traced failures).
- **HORIZON / Long-Horizon Task Mirage** (2026; [arXiv:2604.11978](https://arxiv.org/abs/2604.11978)): 3,100+ trajectory benchmark across WebArena, AgentBench, MAC-SQL, Isaac Sim with LLM-as-Judge attribution.
- **Beyond pass@1** (2026; [arXiv:2603.29231](https://arxiv.org/abs/2603.29231)): Defines **Meltdown Onset Point (MOP)** via sliding-window entropy over tool-call sequences. Documents the paradox: **agents with the highest graceful-degradation scores also have the highest meltdown rates** — aggressive multi-step strategies either succeed spectacularly or spiral into incoherent looping.

---

## 8. Family F — Calibration & Anti-Quitting Mechanisms

The negative-space view: why models give up too easily, and how to fix it without inducing thrashing.

### Sycophancy → premature give-up under pushback

- **Sharma et al.** (ICLR 2024; [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)): Five SOTA assistants wrongly admit mistakes when challenged, abandon correct reasoning to match user beliefs.
- **How RLHF Amplifies Sycophancy** (2026; [arXiv:2602.01002](https://arxiv.org/abs/2602.01002)): Closed-form agreement penalty as a minimal correction.

### Overcautious refusal

- **XSTest** ([arXiv:2308.01263](https://arxiv.org/abs/2308.01263)) and **OR-Bench** (ICML 2025; [arXiv:2405.20947](https://arxiv.org/abs/2405.20947), 80,000 prompts across 32 LLMs): Document widespread false refusal — surface-level lexical overlap triggers safety classifiers on solvable tasks.

### Underthinking ↔ Overthinking duality

- **Underthinking / TIP** (Wang et al., Jan 2025; [arXiv:2501.18585](https://arxiv.org/abs/2501.18585)): "Thoughts Are All Over the Place" — frequent thought-switching prevents committed exploration. TIP penalizes premature transitions at decoding time.
- **Don't Think That Much for 2+3=?** (Chen et al., Dec 2024; [arXiv:2412.21187](https://arxiv.org/abs/2412.21187)): The dual problem.
- **OptimalThinkingBench** ([arXiv:2508.13141](https://arxiv.org/abs/2508.13141)): Explicitly benchmarks both directions.

### Confidence calibration

- **Rewarding Doubt** ([arXiv:2503.02623](https://arxiv.org/abs/2503.02623)): Logarithmic scoring rule penalizing both over- and under-confidence via RL.
- **The Format Tax** (in your repo, `the-format-tax`): Structured output degrades reasoning 3-10pp — relevant because format constraints can cause premature give-up patterns.
- **Too Sharp Too Sure** (in your repo): ECE tracks loss curvature throughout training; CalMO reduces ECE by 71%. *Direct relevance: undercalibrated models bail on solvable problems.*

### IDK pathology

- **Cohen et al.** (NeurIPS 2024; [arXiv:2412.06676](https://arxiv.org/abs/2412.06676)): Adds [IDK] token shifting probability mass for incorrect predictions, with **L_FP-reg anti-false-positive regularizer** to prevent collapse to abstention on solvable problems. Without the regularizer, the model learns abstention is always safe — premature give-up encoded into training.

### Truncation-induced give-up

- **Solving Context Window Overflow** ([arXiv:2511.22729](https://arxiv.org/abs/2511.22729)): Memory-pointer architecture — agents operate on references rather than raw content, ~7x token reduction. Addresses the case where the agent gives up because context is exhausted, not because the task is hard.

---

## 9. The Core Tension (and Why the Field Is Converging on Adaptive Persistence)

Several findings, taken together, make the duality unmistakable:

| Mechanism that increases persistence | Pathology it can cause | Counterexample |
|---|---|---|
| s1 budget forcing ("Wait") | Overthinking past Reasoning Completion Point | Don't Overthink, RCP |
| Reflexion / self-correction loops | Answer wavering (81% of answers change ≥6 times) | Dark Side of Self-Correction |
| Entropy regularization | Persistent exploration without convergence | EPO cascade-failure |
| Tree search branching | Compute waste vs. flat sampling | Limits of PRM-Guided Search |
| Removing refusals | Genuine safety regressions | OR-Bench trade-off curves |
| IDK token | Collapse to abstention on solvable problems | L_FP-reg fix |

The most compelling synthesis emerging in 2025-2026 is **adaptive persistence**: spend compute proportional to problem difficulty, terminated by internal signals (confidence, progress, entropy), with phase-aware schedules during training. The exemplars:

- **DeepConf**: confidence-based termination — 99.9% AIME 2025 with 84.7% fewer tokens
- **SelfBudgeter**: trained budget prediction — 61% length compression, no accuracy loss
- **Leash**: difficulty-aware length penalty — easy problems get strict brevity, hard problems get slack
- **EPO**: phase-adaptive entropy — explore early, exploit late
- **Magentic-One**: progress-ledger detection — replan when stagnation persists for several steps
- **Agent-R**: trained mid-trajectory recovery via MCTS-spliced trajectories

These methods don't ask "should we always keep trying" — they ask **"how do we know when more trying helps."** That's the question the next generation of work is built around.

---

## 10. Connection to Existing Repo Papers

| Repo paper | Relevance |
|---|---|
| `search-r1-rl-search-reasoning` | RL-emergent multi-turn tool persistence; outcome-only reward suffices |
| `webserv-rl-web-agents` | Infrastructure for scalable agent RL training (Incus + ZFS + DOM parser) |
| `shop-r1` | **Anti-give-up reward shaping**: terminate action capped at 0.8 |
| `alphaeval-agents-production` | Evidence that scaffold matters as much as model (11-pt spread) |
| `rethinking-thinking-tokens` | Decouples B_seq (latency) from B_total (compute) for refinement |
| `abstract-cot-latent-reasoning` | Compresses persistent reasoning into latent space (11.6x fewer tokens) |
| `sft-then-rl-outperforms-mixed-policy` | **Bootstrapping prerequisite for RL persistence** — bug-deflated SFT made the field misjudge mixed-policy methods |
| `too-sharp-too-sure-calibration` | ECE/calibration directly relates to underconfident-bail-out pathology |
| `the-format-tax` | Structured-output degradation; relevant to scaffold persistence |
| `delightful-policy-gradient` family | Variance reduction during long-horizon RL — keeps gradients stable across noisy trajectories |
| `nemotron-cascade-2` | Cascade RL with on-policy distillation — relevant to long-horizon stability |

---

## 11. Recommended Reading Priority (if time-limited)

If you want a 10-paper foundation:

1. **Reflexion** (arXiv:2303.11366) — the prompt-level baseline
2. **DeepSeek-R1** (arXiv:2501.12948) — emergent persistence from outcome RL
3. **s1: Simple Test-Time Scaling** (arXiv:2501.19393) — the most direct "force don't stop" technique
4. **Lightman et al. PRM800K** (arXiv:2305.20050) — process reward foundation
5. **WebRL** (arXiv:2411.02337) — self-evolving curriculum from failures
6. **RAGEN / StarPO** (arXiv:2504.20073) — Echo Trap mode-collapse documentation
7. **rStar-Math** (arXiv:2501.04519) — search distilled into model
8. **Stream of Search** (arXiv:2404.03683) — internalizing backtracking
9. **DeepConf** (arXiv:2508.15260) — adaptive termination by confidence
10. **"LLMs Cannot Self-Correct Reasoning Yet"** (arXiv:2310.01798) — the foundational negative result

For the next 5 to round out the picture:

11. **Don't Overthink It** (arXiv:2505.17813) — overthinking dual
12. **Leash** (arXiv:2512.21540) — adaptive length penalty
13. **Agent-R** (arXiv:2501.11425) — trained mid-trajectory recovery
14. **Beyond pass@1** (arXiv:2603.29231) — Meltdown Onset Point and the persistence paradox
15. **The Landscape of Agentic RL for LLMs** (arXiv:2509.02547) — comprehensive survey

---

## 12. Open Questions

The literature has cleared the easy ground. The hard questions remaining as of 2026:

- **Can adaptive persistence be learned from a single signal?** Or do we need ensembles (confidence + progress + entropy + budget) all together?
- **Is internalized search (rStar, SoS) inherently better than external search (ToT, RAP)?** The 14B rStar2-Agent beating 671B DeepSeek-R1 suggests yes, but at training cost.
- **How do we benchmark persistence directly, not just task success?** HORIZON and Beyond pass@1 are first attempts but the field lacks consensus metrics.
- **Does the Echo Trap (RAGEN) imply a fundamental limitation of GRPO for long-horizon agents?** Turn-PPO and iStar are early evidence of yes.
- **How does persistence interact with safety alignment?** Reducing refusals can regress safety; the OR-Bench Pareto frontier is unsolved.
- **Is the asymmetry between premature-give-up (bad) and overthinking (also bad) symmetric in cost?** Production deployment usually treats the former as worse — but no rigorous accounting exists.

---

*This survey complements existing repo coverage of RL post-training, agentic systems, and test-time scaling. It is intentionally cross-cutting: persistence is not a single mechanism but a property that emerges from coordinated work across reward design, scaffold engineering, decoding, and calibration.*
