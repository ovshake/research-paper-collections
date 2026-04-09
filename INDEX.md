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
