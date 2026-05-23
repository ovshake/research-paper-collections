# Research Papers Index

## Human Behavior Simulation & Evaluation
*Can LLMs accurately mimic real individual humans?*

| Paper | Folder | One-Line Summary |
|---|---|---|
| Generative Agent Simulations of 1,000 People | [generative-agent-simulations-1000](generative-agent-simulations-1000/) | Digital twins of 1,052 real people from 2-hour interviews, replicating GSS responses 85% as accurately as the individuals themselves |
| LLM-Powered Digital Twins of Consumers | [llm-digital-twins-consumers](llm-digital-twins-consumers/) | LoRA fine-tuning + RAG to predict future purchases (86% accuracy) and generate reviews (0.94 cosine similarity) at $0.13/consumer |
| Can LLM Agents Simulate Multi-Turn Human Behavior? | [llm-simulate-human-behavior](llm-simulate-human-behavior/) | Benchmarks LLMs on 31,865 real shopping sessions; best prompt-based model achieves only 11.86% accuracy, fine-tuning reaches 17.26% |
| OPeRA: Observation, Persona, Rationale, and Action | [opera-shopping-behavior-simulation](opera-shopping-behavior-simulation/) | First public dataset with observation+persona+rationale+action from 51 real Amazon shoppers; best LLM gets 22% next-action accuracy |
| Shop-R1 | [shop-r1](shop-r1/) | First RL framework for training LLMs to simulate real shopping behavior, achieving 27.72% action accuracy (65% improvement over SFT) |
| SimBench | [simbench](simbench/) | Standardized benchmark across 20 datasets in moral/economic/psychological/social domains; best LLM scores 40.80/100 |

## LLM Agent Systems & Domain Simulations
*Building useful systems with LLM agents*

| Paper | Folder | One-Line Summary |
|---|---|---|
| Agent Hospital | [agent-hospital](agent-hospital/) | Virtual hospital where LLM doctor agents evolve by treating synthetic patients, achieving 92.2% on MedQA without labeled data |
| AgentSociety | [agent-society-large-scale-simulation-llm-driven-generative-agents](agent-society-large-scale-simulation-llm-driven-generative-agents/) | Distributed social simulator scaling to 10K+ agents, replicating opinion polarization and hurricane-induced mobility changes |
| CitySim | [citysim-llm-driven-urban-simulation](citysim-llm-driven-urban-simulation/) | Urban simulation with Maslow-grounded LLM agents generating daily schedules via recursive value-driven planning |
| TradingAgents | [trading-agents](trading-agents/) | Multi-agent LLM trading firm with adversarial bull-bear debates, achieving 23-27% cumulative returns over 3 months |
| UXAgent | [uxagent](uxagent/) | Simulated usability testing with dual-loop architecture; 14/16 UX researchers improved their study designs using its output |
| AlphaEval | [alphaeval-agents-production](alphaeval-agents-production/) | Production-grounded benchmark of 94 tasks from 7 companies across 6 O*NET domains; best agent config (Claude Code + Opus 4.6) scores only 64.41/100, scaffold matters as much as model |
| WebServ | [webserv-rl-web-agents](webserv-rl-web-agents/) | Full-stack browser-server environment for scalable RL training of web agents; auto DOM parser + Incus/ZFS containers cut launch 5x and storage 240x vs Docker, SOTA single-prompt on WebArena |

## A/B Testing with LLM Agents
*Can LLM agents replace real-user experiments?*

| Paper | Folder | One-Line Summary |
|---|---|---|
| Agent A/B | [agent-ab](agent-ab/) | 1,000 LLM agents with structured personas reproduce directional outcomes of a 2M-user human experiment at ~3% cost |
| SimAB | [simab](simab/) | Screenshot-based A/B simulation achieving 67% accuracy (75% high-confidence) across 47 historical tests |
| RL-Enhanced LLM A/B Testing | [rl-llm-abtest](rl-llm-abtest/) | Actor-Critic RL policy on top of LLM content generator with GRU-based memory for long-term preference drift |

## RL for LLM Post-Training
*Better training and optimization signals for LLMs*

| Paper | Folder | One-Line Summary |
|---|---|---|
| Delightful Policy Gradient | [delightful-policy-gradient](delightful-policy-gradient/) | Drop-in PG replacement gating each term with sigmoid of delight (advantage x surprisal), provably reducing variance |
| Delightful Distributed PG | [delightful-distributed-pg](delightful-distributed-pg/) | Extends delight gating to distributed RL, suppressing surprising failures to achieve ~10x lower error under stale data |
| Does This Gradient Spark Joy? | [gradient-spark-joy](gradient-spark-joy/) | Kondo gate uses delight to skip ~97% of backward passes while matching full DG accuracy |
| Self-Distilled RLVR | [self-distilled-rlvr](self-distilled-rlvr/) | Fixes GRPO credit assignment via environment reward for direction + self-distillation for magnitude, +2.32% on multimodal reasoning |
| SRPO: Sample Routing Policy Optimization | [srpo-sample-routing](srpo-sample-routing/) | Routes correct rollouts to GRPO, failed to SDPO, beating GRPO by 3.4% on Qwen3-8B while 17.2% faster |
| Nemotron-Cascade 2 | [nemotron-cascade-2](nemotron-cascade-2/) | Cascade RL (sequential domain-wise RL) + on-policy distillation; 30B MoE reaching gold-medal on IMO/IOI/ICPC 2025 |
| Neural Thickets | [neural-thickets](neural-thickets/) | 60-80% of random perturbations improve task performance; RandOpt matches PPO/GRPO/ES with zero gradient computation |
| Self-Improving Pretraining | [self-improving-pretraining](self-improving-pretraining/) | Uses a post-trained model as rewriter+judge to RL-train during pretraining; 86.3% quality win rate, +36.2% factuality, +18.5% safety over standard NTP |
| Target Policy Optimization | [target-policy-optimization](target-policy-optimization/) | Constructs closed-form target distribution over scored candidates and fits via cross-entropy; gradient vanishes at convergence, outperforms GRPO under sparse reward on 1.7B LLMs |
| SD-Zero | [sd-zero-self-distillation](sd-zero-self-distillation/) | Self-revision turns binary rewards into dense token-level supervision; +10.5% on Qwen3-4B over base, outperforms GRPO/RFT/SDFT with 1 response per question |
| Search-R1 | [search-r1-rl-search-reasoning](search-r1-rl-search-reasoning/) | RL trains LLMs to autonomously issue multi-turn search queries during reasoning; retrieved token loss masking + outcome reward yields +24% over RAG on 7 QA benchmarks |
| SFT-then-RL Outperforms Mixed-Policy | [sft-then-rl-outperforms-mixed-policy](sft-then-rl-outperforms-mixed-policy/) | Discovers DeepSpeed optimizer + OpenRLHF loss bugs that deflated SFT baselines in many papers; once fixed, plain SFT→RL beats all mixed-policy methods by +3.8-22.2 pts |
| Co-Evolving Policy Distillation | [co-evolving-policy-distillation](co-evolving-policy-distillation/) | Interleaves branch-specific RLVR with bidirectional mutual on-policy distillation; unified Qwen3-VL-4B beats domain-specific experts on text, image, and video reasoning |
| DORA: Async RL System for LLM Training | [dora-async-rl-training](dora-async-rl-training/) | Multi-version streaming training eliminates rollout bubbles from skewed generation; 2.12× end-to-end throughput (8.2× rollout speedup) via dynamic orchestration + zero-re-prefill KV-Cache reuse |
| dTRPO: Trajectory Reduction for Diffusion LLM Alignment | [dtrpo-diffusion-llm-alignment](dtrpo-diffusion-llm-alignment/) | Proves policy ratios over dLLM trajectories reduce to products over newly-unmasked tokens (schedule cancels); enables DPO-cost offline training with 4 forward passes, +9.6% STEM on 7B dLLMs |
| RPG: KL-Regularized Policy Gradient Design | [rpg-kl-regularized-policy-gradient](rpg-kl-regularized-policy-gradient/) | Unifies KL-regularized PG design space (fwd/rev, norm/unnorm, differentiable/REINFORCE); finds GRPO's KL missing importance weight; RPG-Style Clip achieves 52% AIME25 on 4B, beating Qwen3-4B-Instruct (47%) |
| PrefixRL: Reuse FLOPs via Off-Policy Prefixes | [prefixrl-reuse-flops-offpolicy](prefixrl-reuse-flops-offpolicy/) | Conditions on-policy RL on prefixes of off-policy correct traces instead of supervising on them; discovers back-generalization (training on prefixed problems improves no-prefix accuracy); 2× compute efficiency, +23pt AIME '25 pass@1 |

## Architecture, Efficiency & Scaling
*Making LLMs faster, smaller, or more capable*

| Paper | Folder | One-Line Summary |
|---|---|---|
| ELF: Embedded Language Flows | [elf-embedded-language-flows](elf-embedded-language-flows/) | Continuous Flow Matching DLM in embedding space with final-step-only discretization; CFG + SDE sampling + x-prediction beats discrete DLMs (MDLM, Duo) at 32 steps with 10× fewer training tokens |
| ViTok-v2: 5B Native Resolution AEs | [vitok-v2-scaling-native-resolution-ae](vitok-v2-scaling-native-resolution-ae/) | Largest ViT image autoencoder (4.5B decoder); DINOv3 perceptual loss replaces GAN/LPIPS for stable billion-scale training; NaFlex+RoPE+SWA enables 8K images; decoder scaling benefits generation beyond reconstruction |
| TST: Token Superposition Training | [tst-token-superposition-training](tst-token-superposition-training/) | Average s consecutive embeddings + multi-hot CE loss for s× data throughput per FLOP during superposition phase; recovery to standard NTP is fast; 2.5× training speedup at 10B A1B MoE scale with no arch changes |
| Pion: Spectrum-Preserving Optimizer | [pion-spectrum-preserving-optimizer](pion-spectrum-preserving-optimizer/) | Updates weights via left/right orthogonal transforms preserving all singular values; Lie algebra momentum + RMS scaling; flat stability indicators, normalization-free training, best RLVR performance vs AdamW/Muon |
| SANA-WM: Minute-Scale World Model | [sana-wm-world-model](sana-wm-world-model/) | 2.6B world model: 60s 720p with 6-DoF camera control on single GPU; hybrid frame-wise GDN + softmax attention; dual UCPE + Plücker conditioning; 36× faster than LingBot-World at comparable quality |
| φ-Balancing for MoE Training | [phi-balancing-moe](phi-balancing-moe/) | Principled population-level load balancing via convex potential φ + mirror descent on EMA of routing probs; negative entropy mirror map gives 7× lower Gini, promotes domain specialization; one-line change beats ST-MoE/loss-free across scales |
| ConvexTok | [convextok](convextok/) | Reformulates tokenizer construction as a linear program (NP-hard → LP relaxation on a token-edge DAG); provably within 1% of optimal compression at common vocab sizes, beats BPE on bits-per-byte; LP nearly integral at 256k (90.5%) |
| Attention Residuals | [attention-residuals](attention-residuals/) | Learned softmax attention over depth replaces fixed residual connections, +7.5 points on GPQA-Diamond for 48B model |
| Mamba-3 | [mamba-3](mamba-3/) | Exponential-trapezoidal discretization + complex-valued SSM with data-dependent RoPE, +1.8 points over Gated DeltaNet at 1.5B |
| TriAttention | [triattention](triattention/) | Trigonometric KV cache compression exploiting pre-RoPE Q/K clustering, 2.5x throughput or 10.7x KV memory reduction |
| TurboQuant | [turboquant](turboquant/) | Random rotation + optimal per-coordinate quantization for 4.5-6x KV cache compression with zero quality loss at 3.5 bits |
| DroPE | [drope-context-extension](drope-context-extension/) | Removes RoPE after pretraining and recalibrates, outperforming all RoPE-scaling methods on long-context benchmarks |
| RoPE Fails in Long Contexts, Provably | [rope-fails-long-context](rope-fails-long-context/) | Proves RoPE loses both locality bias (position inversion → 50% probability) and token-relevance consistency as context grows; four failure modes; RoPE base B trades position vs token identification; 6 LLMs fail by 4K tokens |
| Positional Encoding & Length Generalization | [positional-encoding-length-generalization](positional-encoding-length-generalization/) | Systematic PE comparison finding NoPE outperforms all explicit methods at length generalization (MRR 0.69 vs 0.55) |
| Nemotron 3 Super | [nemotron-3-super](nemotron-3-super/) | 120B/12B-active hybrid Mamba-Attention MoE with 2.2-7.5x higher throughput via LatentMoE and Multi-Token Prediction |
| Megatron Core MoE | [megatron-core-moe](megatron-core-moe/) | Scalable MoE training via Parallel Folding, DeepEP, CUDA Graphs, and FP8/FP4, achieving 1,233 TFLOPS/GPU on GB300 |
| Composer 2 | [composer-2](composer-2/) | Cursor's coding model: continued pretraining on Kimi K2.5 (32B MoE) + large-scale RL, scoring 61.3 on CursorBench |
| Rethinking Thinking Tokens | [rethinking-thinking-tokens](rethinking-thinking-tokens/) | Parallel-Distill-Refine beats long chain-of-thought by +11% on AIME 2024 at lower latency via parallel compute |
| HISA | [hisa-hierarchical-sparse-attention](hisa-hierarchical-sparse-attention/) | Two-stage hierarchical indexer for sparse attention: block coarse filter + token refinement, 2-4x speedup at 32K-128K with >99% IoU to original DSA |
| PAPI | [papi-pim-llm-decoding](papi-pim-llm-decoding/) | Heterogeneous GPU+PIM architecture with dynamic kernel scheduling for LLM decoding, achieving 1.8x speedup and 3.4x energy efficiency over state-of-the-art |
| LUT Tensor Core | [lut-tensor-core](lut-tensor-core/) | Software-hardware co-design replacing MAC with LUT for mixed-precision GEMM in low-bit LLMs, achieving 4-6x power/area reduction and up to 5.51x inference speedup |
| CENT (PIM Is All You Need) | [cent-cxl-pim-llm-inference](cent-cxl-pim-llm-inference/) | GPU-free LLM inference via CXL-connected PIM/PNM devices: 2.3x throughput, 2.9x energy efficiency, 5.2x tokens/$ vs 4x A100 GPUs |
| OliVe | [olive-outlier-victim-quantization](olive-outlier-victim-quantization/) | Outlier-victim pair quantization sacrifices normal values adjacent to outliers for memory-aligned 4-bit PTQ, achieving 4.5x speedup and 4.0x energy reduction over GOBO |
| SpecEE | [specee-speculative-early-exiting](specee-speculative-early-exiting/) | Speculative model reduces early-exit predictor search space 10,000x; 2-layer MLP + heuristic scheduling + hyper-token mapping yields 2.25x cloud / 2.43x PC speedup with <1% accuracy loss |
| Step-3 | [step3-cost-effective-decoding](step3-cost-effective-decoding/) | 321B VLM (38B active) with MFA attention (arithmetic intensity 128) + AFD serving; 4,039 TGS on 32 GPUs vs DSv3's 2,324 on 128 GPUs, ~40% lower decoding cost |
| DFlash | [dflash-block-diffusion-speculative-decoding](dflash-block-diffusion-speculative-decoding/) | Block diffusion drafter with KV-injected target features for speculative decoding; 6x lossless speedup, 2.5x faster than EAGLE-3 |
| Seedance 2.0 | [seedance-2-video-generation](seedance-2-video-generation/) | Native multi-modal audio-video generation model; #1 on Arena.AI T2V (Elo 1450) and I2V (Elo 1449), first on all 6 dimensions of SeedVideoBench 2.0 over Kling 3.0, Sora 2 Pro, Veo 3.1 |
| Transformers Are Inherently Succinct | [transformers-inherently-succinct](transformers-inherently-succinct/) | Proves UHATs are exponentially more succinct than LTL/RNNs and doubly exponentially more succinct than finite automata; non-emptiness is EXPSPACE-complete |
| FASTER | [faster-value-guided-sampling-rl](faster-value-guided-sampling-rl/) | Noise-level critic filters diffusion policy candidates before denoising, recovering best-of-N gains at 8x fewer FLOPs; 1.7x faster inference on 3.3B VLA |
| DeepSeek-V4 | [DeepSeekV4](DeepSeekV4/) | 1.6T/49B-active MoE with hybrid CSA+HCA attention for 1M-token context; 27% FLOPs and 10% KV cache vs V3.2; Codeforces 3206, PutnamBench 120/120 |
| Neural Garbage Collection | [neural-garbage-collection](neural-garbage-collection/) | LM learns to evict KV cache entries via RLVR alongside reasoning; 2-3x cache compression with 49.6% vs 21.2% best baseline on Countdown |
| The Format Tax | [the-format-tax](the-format-tax/) | Structured output (JSON/XML/MD/LaTeX) degrades reasoning 3-10 pp in open-weight models; 92% from the prompt, not GCD; 2-Turn decoupling recovers most loss |
| Too Sharp, Too Sure | [too-sharp-too-sure-calibration](too-sharp-too-sure-calibration/) | ECE tracks loss curvature throughout training via shared margin-tail functional; CalMO training objective reduces ECE by up to 71% (Muon: 0.065→0.019) |
| Stochastic KV Routing | [stochastic-kv-routing](stochastic-kv-routing/) | Random cross-layer attention during training enables flexible depth-wise KV cache sharing at inference; 50-75% cache reduction with preserved or improved QA F1, regularization bonus |
| Abstract Chain-of-Thought | [abstract-cot-latent-reasoning](abstract-cot-latent-reasoning/) | Replaces verbal CoT with short sequences from a learned abstract vocabulary; bottlenecked SFT warm-up + GRPO yields up to 11.6x fewer reasoning tokens matching SFT+RL performance |
| Beyond MuP Pt.4: Parameter Stability (Jianlin Su blog) | [su-beyond-mup-4-parameter-stability](su-beyond-mup-4-parameter-stability/) | Unified framework for maintaining parameter norms throughout training: Post Clip vs Pre Decay (generalizes weight decay to any norm); under spectral norm reduces to SVC and spectral weight decay |

## Financial Markets & Econophysics
*Physics-inspired and quantitative models of market dynamics*

| Paper | Folder | One-Line Summary |
|---|---|---|
| Modeling Stock Market Dynamics Based on Conservation Principles | [stock-market-conservation-principles](stock-market-conservation-principles/) | Deterministic model using asset conservation + two trader types (fundamental + momentum) yields a neutral differential equation that bifurcates from stability through complex oscillations to apparent chaos |

## Surveys
*Cross-paper synthesis on a topic, with taxonomy and reading lists*

| Survey | Folder | Scope |
|---|---|---|
| MoE Architecture Evolution: Mixtral to 2026 | [surveys/moe-architecture-evolution](surveys/moe-architecture-evolution/) | Open-weight MoE LLMs Dec 2023 → 2026: architectures (Mixtral, DeepSeek arc, Qwen3, Llama 4, Kimi K2, GPT-OSS), routing/balancing, FP8 training, MLA/MTP/EP serving, SFT/RL on MoE; consensus design + open questions |
| Beyond MuP Series Walkthrough (Jianlin Su) | [su-beyond-mup-series-walkthrough](su-beyond-mup-series-walkthrough/) | Plain-English walkthrough of all 4 posts: three stability conditions → Muon for linear → per-layer optimizers (Embedding/LM Head/RMSNorm γ) → Post Clip / Pre Decay; one principle, six algorithms |
| LLM Persistence in Long-Horizon Tasks | [surveys/llm-persistence-long-horizon](surveys/llm-persistence-long-horizon/) | Methods (prompting, decoding, training, scaffolding) that prevent premature termination and unproductive thrashing across multi-step tasks |
