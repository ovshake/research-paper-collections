# Generative Agent Simulations of 1,000 People

**Authors:** Joon Sung Park, Carolyn Q. Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, Michael S. Bernstein
**Affiliations:** Stanford, Northwestern, U. of Washington, Google DeepMind
**Date:** 2024-11-15 (Published in Nature)
**Paper:** [PDF](https://arxiv.org/abs/2411.10109)
**Code:** https://github.com/joonspk-research/generative_agent

---

## TL;DR

By conducting 2-hour qualitative interviews with 1,052 real people and feeding those transcripts into GPT-4o as agent memory, the authors create generative agents that replicate individuals' survey responses on the General Social Survey **85% as accurately as the individuals replicate their own answers two weeks later**. Interview-based agents significantly outperform demographic-based and persona-based agents, and reduce accuracy biases across racial and ideological subgroups.

---

## Key Novel Ideas

### 1. Interview-Conditioned Generative Agents — Idiographic Rather Than Demographic

Previous work on LLM-based human simulation used demographic descriptors (age, race, gender, ideology) or brief persona paragraphs to condition the model. These approaches flatten individuals into group stereotypes.

**The key insight:** A 2-hour qualitative interview (~6,500 words) captures the idiosyncratic details — personal history, values, reasoning patterns, life experiences — that actually determine attitudes and behavior. The full interview transcript is injected into the LLM prompt as agent "memory."

**Why this works better:** Interviews capture information that demographic categories cannot: a person's specific childhood experiences, their relationship with religion, their nuanced views on race relations shaped by personal encounters, etc. This information is far more predictive of individual-level responses than knowing someone is "a 45-year-old white liberal."

**Quantitative advantage:**
- GSS: Interview agents achieve 0.85 normalized accuracy vs. 0.71 for demographic-based and 0.70 for persona-based (14-15 point margin)
- Big Five: 0.80 normalized correlation vs. 0.55 demographic vs. 0.75 persona
- Economic games: 0.66 normalized correlation vs. 0.58 demographic vs. 0.65 persona

### 2. AI Interviewer Agent — Scalable Qualitative Data Collection

Conducting 1,000+ two-hour qualitative interviews with human interviewers would be prohibitively expensive and inconsistent. The authors built an AI interviewer agent using GPT-4o that:

- Follows a semi-structured script from the American Voices Project (Stanford)
- Dynamically generates follow-up questions based on participant responses
- Uses a **reflection module** that maintains running summary notes about the participant (e.g., `"place of birth": "New Hampshire"`, `"outdoorsy vs. indoorsy": "outdoorsy"`)
- Operates via voice-to-voice interaction (OpenAI TTS for speaking, Whisper for transcription)
- Completes interviews within 2 hours with ~4 second response latency

**Critical design choice:** The interview script was developed by sociologists independently of the study's evaluation metrics. This prevents data leakage — the interview doesn't ask about the GSS or Big Five directly, yet the information captured still predicts those outcomes.

### 3. Normalized Accuracy — Measuring Against Human Self-Consistency

A core methodological contribution. Humans don't perfectly replicate their own survey responses over time. Measuring agent accuracy against a single snapshot of human responses conflates model error with natural human variability.

**Solution:** Have participants retake all surveys/experiments two weeks later. Define:
- `Normalized accuracy = agent accuracy / participant self-replication accuracy`
- A score of 1.0 means the agent is as accurate as the person is consistent with themselves

This normalization is crucial because:
- Some questions are inherently unstable (people genuinely change their minds)
- Some people are more consistent than others
- Without normalization, the theoretical ceiling for agent accuracy is the human test-retest reliability, not 100%

### 4. Experimental Replication — Agents Replicate Treatment Effects

Beyond predicting individual survey responses, the agents replicated 4 out of 5 published social science experiments (the same 4 that human participants replicated). The effect sizes estimated from agents correlated **r = 0.98** with those from human participants.

This is significant because it suggests agents could serve as a pre-screening tool for experimental designs — testing whether a hypothesized effect is likely to replicate before running expensive human studies.

### 5. Bias Reduction Through Interviews

Demographic-based agents exhibit systematic biases — they overgeneralize group-level patterns. Interview-based agents reduce these biases:

| Metric | Demographic Agents | Interview Agents |
|---|---|---|
| Political ideology DPD (GSS) | 12.35% | 7.85% |
| Race DPD (GSS) | 3.33% | 2.08% |
| Political ideology DPD (Big Five) | 0.165 | 0.063 |
| Race DPD (Big Five) | 0.17 | 0.11 |
| Political ideology DPD (Games) | 0.50 | 0.19 |

(DPD = Demographic Parity Difference between best and worst performing subgroups)

---

## Architecture Details

**AI Interviewer Agent:**
- Model: GPT-4o
- Input: Interview protocol (JSON), participant utterances (via Whisper transcription)
- Components: Memory (accumulated reflections), Reflection module (summarizes key facts), Action generation (follow-up question or advance script)
- Context management: Reflection notes + last 5,000 characters of transcript (avoids lost-in-the-middle problem)
- Voice: OpenAI TTS for output, Whisper for input
- Average interview: 2 hours, ~6,491 words (std = 2,541)

**Generative Agent:**
- Model: GPT-4o
- Memory: Full interview transcript injected into system prompt
- For multi-step experiments: additionally given memory of prior stimuli and responses within that experiment
- Responds to any textual stimulus (forced-choice, surveys, game scenarios)

---

## Key Results

### Individual-Level Prediction (Normalized Accuracy/Correlation)

| Construct | Participants | Interview Agents | Demographic Agents | Persona Agents | Random |
|---|---|---|---|---|---|
| GSS (accuracy) | — | **0.85** | 0.71 | 0.70 | 0.27 |
| Big Five (correlation) | — | **0.80** | 0.55 | 0.75 | ~0 |
| Economic Games (correlation) | — | **0.66** | 0.58 | 0.65 | ~0 |

### Experimental Replications (5 studies)

| Study | Humans Replicated? | Agents Replicated? | Agent Effect Size |
|---|---|---|---|
| Ames & Fiske 2015 | Yes (d=9.45) | Yes (d=12.59) |
| Cooney et al. 2016 | Yes (d=0.40) | Yes (d=1.48) |
| Halevy & Halali 2015 | Yes (d=0.90) | Yes (d=2.98) |
| Rai et al. 2017 | No (p=0.040) | No (p=0.094) |
| Schilke et al. 2015 | Yes (d=0.33) | Yes (d=2.97) |

Effect size correlation between humans and agents: **r = 0.98**

### Ablation: Interview Content Requirements

| Condition | GSS Normalized Accuracy |
|---|---|
| Full interview | 0.85 |
| 80% removed (random) | 0.79 |
| Bullet-point summary (no linguistic cues) | 0.83 |
| Composite agent (with GSS/BFI/game answers) | 0.76 |

Even removing 80% of the interview or stripping it to bullet points, interview agents still outperform composite agents that have *direct access to related survey answers*.

---

## Key Takeaways

1. **Qualitative interviews are a surprisingly efficient way to condition LLMs for behavioral simulation.** A 2-hour interview captures more predictive information about a person than their actual survey responses on related instruments. This suggests interviews activate a different, richer channel of self-expression than structured questionnaires.

2. **The 85% normalized accuracy figure is remarkable.** It means agents predict an individual's responses nearly as well as the individual predicts their own responses two weeks later. The remaining 15% gap likely reflects both model limitations and the inherent unpredictability of human attitudes.

3. **Demographic-based agents are systematically biased; interview-based agents mitigate this.** Using demographics alone causes the LLM to rely on group stereotypes. Interviews provide the individuating information needed to overcome stereotypical predictions, reducing accuracy disparities across political, racial, and gender subgroups.

4. **Agents replicate experimental treatment effects with high fidelity (r = 0.98 with human effect sizes).** This opens a powerful application: using agent simulations to pre-screen experimental designs, estimate likely effect sizes, and identify studies unlikely to replicate — before spending resources on human participants.

5. **Interview content matters more than interview style.** Bullet-point summaries of interviews (removing all linguistic cues) achieve 0.83 normalized accuracy on the GSS, close to the full interview's 0.85. The factual content of what people share — their experiences, values, opinions — matters more than how they express it.

6. **Even 20% of the interview is more informative than full survey responses.** An agent with only 20% of the interview transcript (equivalent to 24 minutes) outperforms a composite agent that has direct access to the person's GSS, Big Five, and economic game responses (with same-category answers excluded).

7. **The normalization methodology is essential for honest evaluation.** Without accounting for human test-retest reliability, agent accuracy appears lower than it actually is. This normalization framework should become standard for evaluating human behavioral simulations.

8. **Agents tend to overestimate treatment effect sizes.** While agents correctly replicate which experiments show significant effects, their effect sizes are consistently larger (often 2-10x) than humans'. This suggests agents may be more "compliant" or stereotypical in their responses to experimental manipulations.

9. **The agent bank model enables replicable social science.** By providing a standardized bank of 1,000+ agents representing real individuals, researchers can run preliminary experiments, test hypotheses, and calibrate study designs before human data collection. The two-tiered access model (open for aggregated/fixed tasks, restricted for individual/open tasks) balances scientific value with privacy.

10. **Voice-to-voice AI interviewing scales qualitative research.** The AI interviewer conducted 1,052 two-hour interviews with consistent quality and ~4-second latency. This could transform qualitative social science by making large-scale, high-quality interview data collection economically feasible.

---

## What's Open-Sourced

- **Code:** https://github.com/joonspk-research/generative_agent (generative agent architecture)
- **Agent Bank:** Two-tiered access system hosted at Stanford
  - Open access: Aggregated responses on fixed tasks (GSS, etc.)
  - Restricted access: Individual agent responses on open tasks (requires review process)
- **Study protocol:** American Voices Project interview script included in supplementary materials
- **Pre-registration:** https://osf.io/mexkf/
- **Models used:** GPT-4o (not open-source)
