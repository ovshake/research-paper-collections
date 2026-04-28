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

## Architecture, Efficiency & Scaling
*Making LLMs faster, smaller, or more capable*

| Paper | Folder | One-Line Summary |
|---|---|---|
| Attention Residuals | [attention-residuals](attention-residuals/) | Learned softmax attention over depth replaces fixed residual connections, +7.5 points on GPQA-Diamond for 48B model |
| Mamba-3 | [mamba-3](mamba-3/) | Exponential-trapezoidal discretization + complex-valued SSM with data-dependent RoPE, +1.8 points over Gated DeltaNet at 1.5B |
| TriAttention | [triattention](triattention/) | Trigonometric KV cache compression exploiting pre-RoPE Q/K clustering, 2.5x throughput or 10.7x KV memory reduction |
| TurboQuant | [turboquant](turboquant/) | Random rotation + optimal per-coordinate quantization for 4.5-6x KV cache compression with zero quality loss at 3.5 bits |
| DroPE | [drope-context-extension](drope-context-extension/) | Removes RoPE after pretraining and recalibrates, outperforming all RoPE-scaling methods on long-context benchmarks |
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
