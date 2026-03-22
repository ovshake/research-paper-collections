# CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation

**Authors:** Nicolas Bougie, Narimasa Watanabe (Woven by Toyota)
**Date:** June 26, 2025
**Paper:** [PDF](https://arxiv.org/abs/2506.21805)

---

## TL;DR

CitySim is a scalable LLM-driven urban simulation framework where agents autonomously generate realistic daily schedules using recursive, value-driven planning that balances mandatory activities, personal habits, and situational context. Agents are equipped with beliefs, long-term goals, spatial memory, and needs modules grounded in Maslow's Hierarchy, producing human-like behavior at both individual and city scale. Evaluated on up to 1,000 agents in the Tokyo metropolitan area, CitySim outperforms all prior agent-based baselines on time-use alignment, human preference, travel patterns, POI popularity prediction, well-being estimation, and crowd density modeling.

---

## Key Novel Ideas

### 1. Recursive Value-Driven Daily Planning
Unlike prior LLM agent frameworks that plan activities in a fixed sequential manner, CitySim uses a two-step recursive decomposition of the day into time blocks. First, mandatory activities (sleep, work, school) are assigned. Then remaining `[EMPTY]` blocks are recursively filled with medium-priority tasks (meals, hygiene). Blocks that still remain empty are filled at execution time using **value-driven planning**: the agent generates N=3 candidate activities, imagines the resulting desire state after each, and selects the one that best fulfills intrinsic needs. This matters because it allows agents to adapt to evolving internal states (hunger, energy, social needs) rather than committing to a rigid schedule upfront, producing more natural and context-sensitive behavior.

### 2. Belief-Aware Spatial Memory with Kalman Filtering
Each agent maintains belief vectors b_i in R^K for every POI, covering dimensions like price, atmosphere, satisfaction, and convenience. When an agent visits a POI, the LLM generates a subjective observation, which is integrated with the prior belief via a **Kalman filter** (with initial uncertainty sigma_b=0.25, observation noise sigma_o=0.2). Unvisited POIs have beliefs imputed from the k=10 most similar visited locations via embedding similarity. Beliefs also undergo daily **decay** toward the neutral value (0.5) with rate lambda=0.03 to model forgetting. This is novel because it gives agents principled uncertainty-aware belief updating rather than simple overwriting, enabling more realistic exploration-exploitation tradeoffs in place selection.

### 3. Belief-Weighted Gravity Model for Place Selection
Place selection uses a two-stage process (macro area, then micro POI) culminating in a belief-weighted gravity model. The selection probability for POI i is: p_ij = (b_j + epsilon) / D_ij^{1+gamma(b_j-0.5)} / sum_k(...), where b_j is belief-based attractiveness, D_ij is distance, and gamma=2.0 controls distance decay modulation. Crucially, higher beliefs about a place reduce the effective distance penalty (agents travel farther for places they believe are good). This extends traditional gravity models used in urban simulation by making them adaptive to each agent's personal experience.

### 4. Needs Module Grounded in Maslow's Hierarchy
Agents track four primary needs (hunger, energy, safety, social connection) as continuous scores in [0,1] that decay over time at need-specific rates. Needs are prioritized (hungry > safe > tired > social) and can **interrupt** ongoing plans when they cross thresholds (T_hunger=0.3, T_energy=0.3, T_safety=0.2, T_social=0.2). The LLM evaluates outcomes after each activity to update need scores. This mechanism ensures agents don't just follow static schedules but react dynamically to internal states, e.g., a hungry agent will abandon a leisure plan to find food.

### 5. Long-Term Goal Formation and Revision
Goals are generated monthly or after major life events using Maslow's framework. The LLM is conditioned on persona, financial status, social contacts, recent activities, current goals, and computed metrics: **need fulfillment** (proportion of day when needs exceed thresholds), **financial stress** (income < 0.9x expenses), **social isolation** (fewer than 3 unique contacts in 7 days), and **interest** (proportion of recently visited POIs with satisfaction belief > 0.5). This produces coherent short-term (weeks) and long-term goals that inform subsequent planning, enabling life-course consistency across multi-week simulations.

### 6. Three-Part Memory Architecture
CitySim agents have temporal, reflective, and spatial memory. **Temporal memory** stores chronological events with time, location, observation, and key fields, retrieving the top k_1=5 entries from the past 24 hours via cosine embedding similarity. **Reflective memory** synthesizes daily experiences into higher-level insights about habits, preferences, and evolving beliefs through a structured prompting process that generates up to 5 insights per day, each supported by explicit evidence references. **Spatial memory** maintains POI belief vectors updated via Kalman filtering. This three-part design enables agents to recall specific events, develop self-knowledge over time, and maintain grounded spatial preferences.

### 7. Dynamic Social Network with Evolving Beliefs
The social module maintains a weighted social network where each edge carries a belief vector b_{u,v} with dimensions for affinity, trust, and familiarity. Face-to-face interactions occur when agents are co-located (partner selected proportionally to current belief scores, limited to one per 30-minute tick). Online interactions are triggered dynamically when social satisfaction drops below a threshold. After each interaction, the LLM evaluates sentiment (positive/neutral/negative) to incrementally update social beliefs. This enables emergent social patterns rather than pre-scripted interaction schedules.

---

## Architecture Details

| Component | Key Parameters |
|---|---|
| LLM backbone | GPT-4o-mini (ChatGPT) |
| Simulation timestep | 5 minutes |
| Time block granularity | 5 minutes minimum |
| Agent count | Up to 1,000 (experiments); scalability tested to 10^6 |
| Setting | Tokyo metropolitan area (AgentSociety framework) |
| Spatial memory dimensions | K=4 (price, atmosphere, satisfaction, convenience) |
| Belief initial uncertainty (sigma_b^0) | 0.25 |
| Observation noise (sigma_o) | 0.2 |
| Belief decay rate (lambda) | 0.03 per simulated day |
| Neutral belief value (b_0) | 0.5 |
| Gravity model distance decay (gamma) | 2.0 |
| Gravity model stability constant (epsilon) | 10^-3 |
| Neighbor POIs for belief imputation (k) | 10 |
| Temporal memory retrieval (k_1) | 5 entries from past 24h |
| Need thresholds | T_hunger=0.3, T_energy=0.3, T_safety=0.2, T_social=0.2 |
| Need priority order | hunger > safety > energy > social |
| Candidate activities per empty block (N) | 3 |
| Candidate areas considered | Top 10 nearby |
| Candidate POIs per activity | Up to 200 |
| Personality traits | Big Five, 3-point scale (1=low, 2=medium, 3=high) |
| Available vehicles | Walk, bicycle, car, bus, train |
| Goal revision frequency | Monthly or after major life events |
| Reflective insights per day | Up to 5 |
| Long-term goal triggers | Need fulfillment index, financial stress (income < 0.9x expenses), social isolation (<3 contacts in 7 days) |

---

## Training Pipeline

CitySim is not a trained model but an inference-time simulation framework. The pipeline operates as follows:

1. **Persona Initialization:** Agent attributes (demographics, Big Five traits, habits, preferences, spatial anchors) are initialized from a proprietary survey-based dataset matching Japanese census distributions. Home and workplace locations are assigned based on population density from OpenStreetMap.

2. **Daily Simulation Loop (Algorithm 1):**
   - For each day, each agent calls `plan_day()` to generate the day's schedule via recursive decomposition.
   - For each 5-minute timestep:
     - `perceive()`: Agent receives environmental observation; perception module selects the appropriate action module.
     - `decide_action()`: Dispatcher invokes planning, social interaction, or other modules based on inferred needs.
     - If action requires movement: `select_POI()` (two-stage gravity model) then `select_vehicle()` then `move()`.
     - Otherwise: `execute(action)`.
     - `reflect()`: Update beliefs, goals, needs, habits at end of step.
   - End of day: Synthesize reflective memory from temporal memory entries.

3. **Goal Revision:** Monthly or triggered by life events; LLM generates coherent short and long-term goals conditioned on comprehensive agent context.

4. **Social Interactions:** Face-to-face (co-located agents) and online (triggered by low social satisfaction) interactions update social belief vectors based on LLM-evaluated sentiment.

---

## Key Results

### Macro-Level Time Use (Figure 1)
CitySim's time-use distribution across activity categories (Work, Commute, Housework, Personal Care & Sleep, etc.) closely matches ground-truth data from the 2021 Japanese national time-use survey across all age groups.

### Pairwise Human Preferences (Figure 2)
Win rates judged by GPT-4o on naturalness, coherence, and plausibility (outputs normalized via Llama-3.1 70B):

| | GeAn | AGA | HumanoidAgent | AgentSociety | MobileCity | CitySim |
|---|---|---|---|---|---|---|
| **CitySim** | **0.85** | **0.75** | **0.70** | **0.60** | **0.58** | **0.50** |

CitySim achieves the highest average win rate against all baselines.

### Human-Likeness Scores (Table 3, GPT-4o evaluated, 1-5 Likert scale)

| Method | Activity | Dialogue | Mobility | Event Reaction |
|---|---|---|---|---|
| GeAn | 3.11 +/- 0.18 | 3.96 +/- 0.04 | 3.08 +/- 0.17 | 3.03 +/- 0.21 |
| AGA | 3.22 +/- 0.28 | 4.00 +/- 0.03 | 3.16 +/- 0.24 | 3.15 +/- 0.19 |
| HumanoidAgent | 3.30 +/- 0.31 | 3.99 +/- 0.05 | 3.29 +/- 0.22 | 3.21 +/- 0.17 |
| AgentSociety | 4.02 +/- 0.22 | 4.08 +/- 0.06 | 3.82 +/- 0.25 | 3.75 +/- 0.21 |
| MobileCity | 4.09 +/- 0.27 | 4.04 +/- 0.06 | 3.96 +/- 0.18 | 3.89 +/- 0.17 |
| **CitySim** | **4.37 +/- 0.18** | **4.23 +/- 0.04** | **4.14 +/- 0.15** | **4.09 +/- 0.16** |

CitySim leads across all four domains.

### Well-Being Prediction (Table 1, 5-class macro F1)

| Method | F1-macro |
|---|---|
| GeAn | 0.19 +/- 0.03 |
| AGA | 0.20 +/- 0.03 |
| HumanoidAgent | 0.22 +/- 0.03 |
| AgentSociety | 0.28 +/- 0.02 |
| MobileCity | 0.21 +/- 0.02 |
| **CitySim** | **0.36 +/- 0.02** |
| GBDT (XGBoost, upper bound) | 0.45 +/- 0.04 |

CitySim is the best agent-based method, approaching the supervised GBDT baseline.

### Scalability (Table 2, mean time per simulation step)

| # Agents | CitySim (s) | AgentSociety (s) |
|---|---|---|
| 10^3 | 9.0x10^-3 +/- 3.2x10^-5 | 8.6x10^-3 +/- 3.0x10^-5 |
| 10^4 | 9.7x10^-3 +/- 2.1x10^-5 | 9.1x10^-3 +/- 1.5x10^-5 |
| 10^5 | 2.1x10^-2 +/- 5.0x10^-4 | 1.8x10^-2 +/- 5.7x10^-4 |
| 10^6 | 0.183 +/- 5.6x10^-4 | 0.168 +/- 5.3x10^-4 |

CitySim maintains comparable per-step time to AgentSociety even at 10^6 agents, with only modest overhead from its additional cognitive modules.

### Ablation Study (Table 4, human-likeness on 1-5 Likert scale)

| Variant | Activity | Dialogue | Mobility | Event Reaction |
|---|---|---|---|---|
| CitySim (full) | **4.37** | **4.23** | **4.14** | **4.09** |
| w/o Belief | 3.85 | 3.92 | 3.75 | 3.60 |
| w/o Recursive Planning | 3.72 | 3.85 | 3.80 | 3.65 |
| w/o Long-Term Goals | 3.80 | 3.95 | 3.88 | 3.70 |
| w/o Needs | 3.55 | 3.82 | 3.73 | 3.50 |
| w/o Persona | 3.60 | 3.60 | 3.72 | 3.58 |

---

## Key Takeaways

1. **Needs module is the single most impactful component.** Removing it causes the largest drop in activity (4.37 to 3.55) and event reaction (4.09 to 3.50) scores. Without explicit needs prioritization, agents produce rigid, unrealistic schedules and fail to interrupt plans for basic requirements like eating.

2. **Persona module is essential for diversity.** Without it, agents converge to a bland, homogenized pattern, causing the sharpest decline in dialogue quality (4.23 to 3.60). Demographic and psychological diversity is not optional for realistic simulation.

3. **Belief module prevents myopic behavior.** Removing beliefs drops activity scores by 0.52 points. Agents without beliefs cannot accumulate or leverage prior expectations about places, resulting in less adaptive and less contextually grounded plans.

4. **Recursive planning enables adaptive schedules.** Its removal particularly hurts activity coherence (4.37 to 3.72) and event reaction (4.09 to 3.65), as agents become less capable of adapting routines to new internal or environmental feedback.

5. **Long-term goals have a localized but meaningful impact.** Their removal most noticeably affects dialogue and mobility, suggesting that high-level goal management primarily benefits life-course consistency and contextual coherence over multi-day simulations.

6. **CitySim closely reproduces real-world macro patterns.** Time-use distributions match the 2021 Japanese national survey, travel distributions match real hourly patterns for both weekdays and weekends, and crowd density heatmaps align with ground-truth smartphone location data in Shibuya, Tokyo.

7. **Belief-weighted gravity model introduces a bias toward popular/branded POIs.** CitySim agents show an inflated estimate of real-world popularity for well-known POIs and underestimate crowd density in smaller streets, reflecting LLM popularity bias.

8. **Larger LLMs produce better belief estimation.** GPT-4o achieves the lowest mean absolute error for belief estimation of unvisited POIs across all five semantic categories (Restaurants, Parks, Shops, Transport, Entertainment), followed by GPT-4o mini and Qwen-14B.

9. **Scalability is maintained at city scale.** Per-step simulation time grows only modestly from 9ms (10^3 agents) to 183ms (10^6 agents), demonstrating the framework's suitability for large-scale urban modeling.

10. **Need trajectories are persona-specific and realistic.** Different agent profiles (office worker, student, night-shift nurse, freelance designer, retired senior) produce distinct, context-dependent need evolution patterns consistent with real-world routines, including irregular sleep for night workers and higher social satisfaction for seniors living with family.

---

## What's Open-Sourced

The paper does not mention any open-source code release or publicly available artifacts. The simulation is built on top of the AgentSociety framework (Piao et al., 2025). The persona dataset and well-being survey data used for evaluation are proprietary. The crowd density ground-truth is derived from proprietary smartphone location data, and POI popularity ground-truth comes from Google Maps ratings.
