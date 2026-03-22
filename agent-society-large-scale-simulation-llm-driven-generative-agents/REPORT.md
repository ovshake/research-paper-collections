# AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society

**Authors:** Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, Chen Gao, Fengli Xu, Fang Zhang, Ke Rong, Jun Su, Yong Li
**Date:** February 12, 2025
**Paper:** [PDF](https://arxiv.org/abs/2502.08691)

---

## TL;DR

AgentSociety is a large-scale social simulator that integrates LLM-driven generative agents (with psychologically grounded minds including emotions, needs, and cognition), a realistic societal environment (urban, social, and economic spaces), and a distributed simulation engine powered by Ray and MQTT. It scales to 10,000+ agents with ~500 interactions per agent per day, and successfully replicates real-world social phenomena across four experiments: opinion polarization, inflammatory message spread, universal basic income effects, and hurricane-induced mobility changes.

---

## Key Novel Ideas

### 1. Theory-Grounded Agent Mind Architecture (Emotion-Needs-Cognition Framework)

Unlike prior LLM agents that rely on simple role-playing prompts or basic agentic modules, AgentSociety integrates established social science theories into a three-level mental process:

- **Emotions**: Modeled using Shvo et al.'s framework with six core emotions (sadness, joy, fear, disgust, anger, surprise) rated 0-10 in intensity. Emotions drive rapid responses to stimuli and influence decision-making.
- **Needs**: Based on Maslow's Hierarchy of Needs, organized hierarchically from physiological needs to self-fulfillment. Needs serve as the fundamental motivational drivers, continuously updated by active behaviors, external events, and psychological states. The Theory of Planned Behavior is then used to formulate action plans that address priority needs.
- **Cognition**: Encompasses higher-level reasoning, planning, and attitude formation (support/opposition on topics rated 0-10). Cognition updates after every action, linking emotions and attitudes through the agent's experiences.

**Why it works:** By explicitly coupling these three psychological layers, behaviors emerge not from arbitrary LLM generation but from theoretically motivated internal states. This creates a feedback loop: actions update emotions, which shift needs, which drive cognition, which informs the next behavior -- mirroring real human psychological dynamics.

### 2. Mind-Behavior Coupling via Dual-Stream Memory

The agent workflow is unified through a novel Stream Memory system with two parallel streams:

- **Event Flow**: Chronological record of all proactive and passive events (actions taken, environmental changes). Each MemoryNode stores time, location, and event description.
- **Perception Flow**: Records the agent's subjective thoughts and attitudes toward events in the Event Flow. Each perception node links to one or more event nodes.

The agent's decision-making loop follows: (1) Action Determination from current state, (2) Event Feedback from environment, (3) Memory Update in both streams, (4) Emotion and Cognition Analysis, (5) Passive Event Processing. This dual-stream approach separates objective events from subjective experience, enabling richer and more human-like behavioral adaptation.

### 3. Needs-Driven Hierarchical Mobility with Gravity Model

Mobility is modeled not as random movement but through a four-step hierarchical decision framework grounded in the agent's needs:

1. **Intention Extraction**: Derive core mobility intentions from the needs hierarchy (e.g., "social demand" triggers "move to social venue")
2. **Place Type Selection**: Match demands with POI types in geographic databases
3. **Radius Decision**: Dynamically determine feasible travel ranges based on internal states (age, stamina) and environmental parameters (weather, traffic)
4. **Place Selection**: Apply the Gravity Model for spatial optimization: P_ij = (S_j / D_ij^beta) / sum(S_k / D_ik^beta), where S_j is location attractiveness, D_ij is distance, and beta is the distance decay coefficient

**Why it works:** The Gravity Model reduces LLM computational overhead while ensuring selections align with established human spatial patterns (proximity principle, agglomeration effects). The needs-driven chain (Need -> Plan -> Behavioral Sequence) creates spatiotemporal coupling that makes mobility a critical execution step for demand realization rather than an arbitrary choice.

### 4. Realistic Three-Space Societal Environment

The environment integrates three interconnected spaces:

- **Urban Space**: Built on OpenStreetMap road networks and SafeGraph POI data, with multi-modal mobility (driving via IDM model, walking, bus, taxi). Lanes, roads, junctions, AOIs, and POIs form a two-layer structure (static infrastructure + dynamic behavior).
- **Social Space**: Social network with three relationship types (family, friends, colleagues), each with strength values 0-100. Includes a **supervisor** module that monitors messages, filters based on algorithmic rules, and can ban users/connections -- simulating real social media platform moderation.
- **Economic Space**: Full macroeconomic model with four entities (agents, firms, government, banks). Implements progressive taxation, interest rates via Taylor Rule, wage/price adjustment by supply-demand, and a National Bureau of Statistics tracking GDP, per capita consumption, and other indicators.

### 5. MQTT-Powered Agent Messaging System for Scale

The system adopts MQTT (Message Queuing Telemetry Transport), originally designed for IoT device communication, as the inter-agent messaging protocol. This is a novel choice for multi-agent systems.

**Why it works:** MQTT's publish/subscribe architecture with lightweight packet structure handles hundreds of thousands of concurrent agent connections efficiently. Each agent subscribes to topics prefixed with `exps/<exp_uuid>/agents/<agent_uuid>/`, enabling targeted message delivery (agent-chat, user-chat, user-survey). Compared to alternatives tested (Redis Pub/Sub at 81,216 msg/s, RabbitMQ at 23,667 msg/s), MQTT achieved 44,702 msg/s with the added benefit of built-in GUI tools for monitoring and debugging. Kafka failed to even initialize 100,000 agents within 5 minutes.

### 6. Group-Based Distributed Execution with Ray + Asyncio

To solve the TCP port exhaustion problem (each agent needing multiple connections, with a system limit of 65,535 ports), AgentSociety introduces an intermediate "agent group" structure. Multiple agents share a single process with connection reuse, while Ray provides multi-process distributed execution and Python's asyncio enables concurrent LLM API calls within each group.

**Why it works:** This hybrid approach avoids the sequential bottleneck of single-process execution while preventing port exhaustion. Agents within a group share LLM API, MQTT, environment, database, and metrics recorder clients. The asynchronous design is particularly effective since LLM-driven simulations are I/O-intensive (waiting for API responses), allowing CPU resources to be used for computational tasks (e.g., running gravity models) during I/O waits.

---

## Architecture Details

### Agent Architecture

| Component | Details |
|---|---|
| Profile | Basic demographics (name, age, gender, education), personality |
| Status | Mental states, economic status, social relationships (dynamic) |
| Emotions | 6 core emotions (sadness, joy, fear, disgust, anger, surprise), intensity 0-10 |
| Needs | Maslow's hierarchy: physiological, safety, love & belonging, esteem, self-actualization |
| Cognition | Attitudes (topic support 0-10), thoughts, reasoning |
| Memory | Dual-stream: Event Flow (objective) + Perception Flow (subjective) |
| Behaviors | Mobility, social interactions, employment & consumption, other (sleeping, eating, etc.) |

### System Architecture

| Component | Technology |
|---|---|
| LLM API | OpenAI-compatible (DeepSeek-V3 used in experiments) |
| Messaging | MQTT via EMQX server |
| Database | PostgreSQL (batch writes via COPY FROM) |
| Metrics | MLflow server |
| Distributed Computing | Ray framework |
| Async Execution | Python asyncio |
| File Storage | AVRO format (local) + PostgreSQL (online) |
| Urban Environment | OpenStreetMap + SafeGraph POI data |
| Driving Model | IDM (acceleration) + MOBIL (lane-changing) |
| GUI | Frontend + Backend for real-time monitoring |

### Simulation Scale Parameters

| Parameter | Value |
|---|---|
| Max agents tested | 10,000+ |
| Avg. interactions per agent per day | ~500 (491.68 measured) |
| Total interactions simulated | 5 million+ |
| LLM used | DeepSeek-V3 |
| Infrastructure | Huawei Cloud c7.16xlarge.4 (64-core) |

---

## Training Pipeline

AgentSociety is not a trained model but a simulation framework. The pipeline for running a simulation is:

1. **Agent Initialization**: Load agent profiles (demographics, personality) and initialize status (mental states, economic status, social relationships). Profiles can be based on real demographic data (e.g., Census Block Group data for hurricane experiment).

2. **Environment Setup**: Configure the three-space environment:
   - Urban space with road network, AOIs, and POIs from real geographic data
   - Social network with relationship types and strengths
   - Economic space with firms, government, banks, and initial macroeconomic parameters

3. **Simulation Execution**: For each time step:
   - Agents assess current state (needs, emotions, cognition)
   - Generate plans via Theory of Planned Behavior
   - Execute behaviors (mobility, social interaction, economic activity, other)
   - Receive environment feedback
   - Update dual-stream memory (Event Flow + Perception Flow)
   - Update emotions, needs, and cognition based on outcomes

4. **Intervention (Optional)**: Three types available:
   - Pre-simulation agent configuration
   - During-simulation state manipulation
   - During-simulation message notification

5. **Data Collection**: Toolbox for social science research:
   - Surveys via MQTT (structured JSON responses)
   - Interviews via MQTT (one-on-one Q&A)
   - Intervention experiments
   - Metric recording via MLflow

---

## Key Results

### Environment Scalability (Table 2)

| # of Agents | Mean Time per Step (s) |
|---|---|
| 1,000 | 8.578 x 10^-3 +/- 3.0 x 10^-5 |
| 10,000 | 9.129 x 10^-3 +/- 1.5 x 10^-5 |
| 100,000 | 1.800 x 10^-2 +/- 5.66 x 10^-4 |
| 1,000,000 | 0.1680 +/- 5.34 x 10^-4 |

Minimal performance degradation even at 1M agents, confirming environment scalability.

### Messaging System Comparison (Table 3)

| System | Best Parallel Processes | Throughput (msg/s) | Auxiliary Tools |
|---|---|---|---|
| MQTT (emqx v5.8.1) | 32 | 44,702 +/- 111.3 | Built-in GUI |
| Redis Pub/Sub (v6.2) | 16 | 81,216 +/- 333.6 | - |
| RabbitMQ (v4.0.5) | 16 | 23,667 +/- 1,777.7 | Built-in GUI |
| Kafka | - | Failed to initialize | - |

MQTT was selected despite lower raw throughput than Redis due to built-in GUI tools for monitoring and debugging.

### Simulator Performance (Table 4)

| #Agents | #Groups | LLM Calls | Avg Time All (s/round) | Avg LLM Time (s/call) | Avg Env Time (ms/call) |
|---|---|---|---|---|---|
| 1,000 | 8 | 4,803 | 82.45 | 4.51 | 12.26 |
| 1,000 | 16 | 3,121 | 41.17 | 2.92 | 14.31 |
| 1,000 | 32 | 4,790 | 43.30 | 2.94 | 9.55 |
| 10,000 | 8 | 54,135 | 5,681.18 | 52.54 | 33.55 |
| 10,000 | 16 | 54,002 | 1,422.48 | 3.53 | 33.55 |
| 10,000 | 32 | 54,075 | 458.82 | 8.05 | 30.53 |

Key finding: LLM API calls are the primary bottleneck. With 32 groups for 10K agents, total time per round drops from 5,681s to 459s (12.4x speedup). Environment time remains in milliseconds throughout.

### Social Experiment 1: Polarization (Gun Control)

Three experimental groups with 100 agents each:
- **Control group**: No external intervention. 39% became more polarized, 33% more moderate.
- **Homophilic interaction group** (echo chamber): Only exposed to like-minded persuasive messages. 52% became more polarized -- demonstrating echo chamber effects.
- **Heterogeneous interaction group**: Only exposed to opposing viewpoints. 89% adopted more moderate opinions, 11% were persuaded to opposing views -- demonstrating exposure to diverse views curbs polarization.

### Social Experiment 2: Inflammatory Message Spread

- Inflammatory messages showed substantially higher information reach than non-inflammatory (control) content, confirming stronger viral potential.
- **Node-level intervention** (suspending accounts) was more effective than **edge-level intervention** (removing connections) at containing both information spread and emotional intensity.
- Agent interviews revealed sharing is driven by emotional reactions (sympathy, worry) and social responsibility.

### Social Experiment 3: Universal Basic Income (Texas)

- UBI policy ($1,000/month unconditional payment) increased consumption levels and reduced depression levels (measured via CES-D scale).
- Results align with real-world outcomes observed in Texas' UBI social experiment (Bartik et al., 2024).
- Agent interviews referenced key terms matching real-world UBI discourse: interest rates, long-term benefits, savings, necessities of life.

### Social Experiment 4: Hurricane Dorian (Columbia, SC)

- 1,000 agents simulated with real-time weather updates.
- Pre-hurricane activity level: 70-90% across census block groups.
- During hurricane: activity level dropped sharply to ~30%.
- Post-hurricane: gradual recovery to normal levels.
- Simulated daily visit patterns closely matched real SafeGraph mobility data (9-day normalized time series), validating the model's ability to replicate human mobility responses to external shocks.

---

## Key Takeaways

1. **LLM API latency is the dominant bottleneck** in large-scale agent simulations, not environment computation or messaging. Even with full parallelization (32 groups), LLM calls constitute the majority of execution time. Private LLM inference deployment may be necessary for >10K agent simulations.

2. **Theory-grounded agent design matters.** Integrating established psychological theories (Maslow's Hierarchy of Needs, Theory of Planned Behavior, Cognitive Appraisal Theory) into agent minds produces behaviors that are not only plausible but measurably aligned with real-world patterns across multiple domains.

3. **Echo chambers demonstrably intensify polarization.** Homophilic interactions led to 52% of agents becoming more polarized (vs. 39% in control), while heterogeneous exposure led 89% to moderate -- providing computational evidence for the echo chamber hypothesis.

4. **Node-level content moderation outperforms edge-level moderation** for controlling inflammatory message spread. Suspending offending accounts is more effective than severing individual connections, both for information containment and emotional intensity reduction.

5. **Environment realism is essential for simulation fidelity.** The paper argues that relying solely on LLM knowledge without modeling real-world operational laws leads to "hallucinated" social dynamics. Offloading physical constraints (traffic, economics, geography) to rule-based environment systems lets agents focus on subjective behavioral logic.

6. **The dual-stream memory (Event Flow + Perception Flow) enables richer agent behavior** by separating what happened (objective) from how the agent felt about it (subjective), enabling more nuanced decision-making and emotional responses over time.

7. **MQTT is a practical choice for agent messaging** despite not having the highest throughput. Its built-in monitoring tools, lightweight protocol, and proven scalability from IoT make it well-suited for social simulation where debugging and observability matter.

8. **The Gravity Model for mobility reduces LLM calls while maintaining realism.** Rather than asking the LLM to select specific locations (computationally expensive and prone to hallucination), the system uses the Gravity Model for spatial optimization, with the LLM only determining high-level intentions and constraints.

9. **UBI simulation results match real-world experimental data.** The alignment between simulated UBI effects (increased consumption, reduced depression) and actual Texas UBI outcomes validates that the system can serve as a practical policy evaluation platform.

10. **The paper frames three levels of social simulator maturity:** (1) Social twin systems (one-to-one mapping), (2) Mirroring worlds (prediction + intervention), and (3) Coexistent hybrid worlds (real-virtual integration). AgentSociety claims to be a pioneering attempt at level 3, where simulated individuals can interact with and influence real society.

---

## What's Open-Sourced

The paper states that AgentSociety is proposed as a platform and testbed. The system architecture uses established open-source components:

- **Ray** for distributed computing
- **EMQX** for MQTT messaging
- **PostgreSQL** for data storage
- **MLflow** for metrics recording
- **OpenStreetMap** for geographic data
- **SafeGraph** for POI data

The paper does not explicitly mention an open-source release of the AgentSociety codebase itself within the text reviewed, though it is presented as a framework intended for use by social scientists and policymakers. The code appears to be available at the associated project (the paper includes a clickable link on the first page). The LLM backbone used in experiments is DeepSeek-V3, accessed via its public API.
