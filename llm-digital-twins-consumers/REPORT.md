# Predicting Behaviors with Large Language Model (LLM)-Powered Digital Twins of Consumers

**Authors:** Bingqing Li, Qiuhong (Owen) Wei, Xin (Shane) Wang*
**Affiliations:** Virginia Tech (Pamplin College of Business), Hong Kong University Business School
**Date:** 2025 (SSRN working paper)
**Paper:** [PDF](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5256223)

---

## TL;DR

A theory-grounded framework for building individual-level consumer digital twins by combining fine-tuning (to internalize personal traits from review history) with RAG (to inject product context at inference). Validated on Amazon e-commerce data with 304 consumers, the digital twins predict future purchases with **86% accuracy** and generate reviews with **0.94 cosine similarity** to actual consumer-written reviews, at a cost of **$0.13 per consumer**.

---

## Key Novel Ideas

### 1. Lewin's B = f(P, E) as Architectural Blueprint

The paper's central intellectual contribution is mapping a foundational psychological theory onto LLM adaptation techniques:

- **B = f(P, E)** — Behavior is a function of Person and Environment (Lewin, 1936)
- **P-component → Fine-tuning:** LoRA fine-tuning on a consumer's review history internalizes their enduring traits, preferences, communication style, and decision patterns into the model weights
- **E-component → RAG:** Retrieval-augmented generation dynamically injects product information (price, ratings, descriptions, peer reviews) at inference time, simulating how consumers seek and process contextual information

**Why this mapping matters:** Prior work used either prompting, fine-tuning, or RAG in isolation. This paper argues they serve fundamentally different psychological functions — fine-tuning captures *who you are*, RAG captures *what you're looking at* — and their combination is necessary for behaviorally plausible simulation.

### 2. UGC-Based Fine-Tuning for Individual Heterogeneity

Rather than using demographic labels ("35-year-old female") to condition the model, the paper fine-tunes on each consumer's actual review history — their own words, opinions, ratings, and product choices.

**Why UGC beats demographics:** Reviews capture the consumer's actual language patterns, sentiment tendencies, evaluation criteria, brand attitudes, and product reasoning. A consumer who writes "great value for the price" across many reviews reveals a price-sensitive decision style that no demographic label can capture.

**Implementation:** Each consumer's reviews (excluding the 50 most recent for testing) are formatted as instruction-input-output pairs and used to fine-tune Qwen 2.5-7B via LoRA (rank 8). One model per consumer.

### 3. Practical Cost Analysis for Scalable Deployment

The paper provides concrete cost-performance tradeoffs rarely seen in academic papers:

| Epochs | Accuracy | Time/Consumer | Cost/Consumer |
|---|---|---|---|
| 3 | 0.683 | 7m 56s | $0.036 |
| 5 | 0.703 | 14m 33s | $0.066 |
| 10 | 0.859 | 28m 28s | $0.128 |

On an RTX 4090D at $0.27/hour. This makes individual-level digital twins economically viable for marketing at scale.

---

## Architecture Details

| Component | Details |
|---|---|
| Base LLM | Qwen 2.5-7B (open-source) |
| Fine-tuning method | LoRA (rank 8, dropout 0) |
| Training data per consumer | ~502 reviews (median, excl. 50 held-out) |
| Fine-tuning epochs | 10 (best performance) |
| Learning rate | 5e-5 |
| Inference temperature | 0.1 |
| RAG embeddings | BGE-M3 |
| RAG retrieval | Top-2 chunks per product query |
| Knowledge base content | 21 product features (price, rating, category, reviews, descriptions) |
| Dataset | Amazon Reviews Dataset (McAuley Lab, 571.5M reviews) |
| Sample | 304 consumers (≥400 reviews each, excl. books/Kindle-only) |
| Test set per consumer | 50 purchased + 50 never-purchased products |
| Gender inference | facebook/bart-large-mnli (zero-shot from usernames) |

---

## Training Pipeline

1. **Data Collection:** Extract consumer review histories from Amazon Reviews Dataset (571.5M reviews, 54.5M unique consumers)
2. **Filtering:** Retain consumers with ≥400 reviews, exclude books/Kindle-only → 304 consumers
3. **Formatting:** Convert each consumer's reviews into instruction-input-output pairs; enrich with inferred gender
4. **Fine-tuning:** LoRA fine-tune Qwen 2.5-7B separately for each of 304 consumers (10 epochs)
5. **Knowledge Base Construction:** Embed all product metadata (21 features) into BGE-M3 vector space
6. **Inference:** For each test product, RAG retrieves top-2 relevant product chunks → prepend to prompt → fine-tuned model generates purchase decision + review

---

## Key Results

### Purchase Prediction (304 consumers × 100 products = 30,600 predictions)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Base LLM (Qwen 2.5-7B) | 0.541 | 0.749 | 0.085 | 0.141 | 0.541 |
| Base LLM + RAG | 0.623 | 0.757 | 0.288 | 0.361 | 0.623 |
| **Digital Twin (FT + RAG)** | **0.859** | **0.960** | **0.745** | **0.807** | **0.859** |

The biggest gain is in **recall** (0.085 → 0.745): without fine-tuning, the base model is too conservative to predict purchases, defaulting to "No."

### Review Generation Quality (10,520 true positive cases)

| Metric | Score |
|---|---|
| Word2Vec cosine similarity | 0.943 (SD = 0.052) |
| Sentence-BERT / BGE-M3 cosine similarity | 0.650 (SD = 0.125) |

### Scaling with Data Volume

| Reviews Available | Purchase Accuracy |
|---|---|
| 50-100 | 0.547 |
| 101-200 | 0.683 |
| 201-300 | 0.780 |
| 301-400 | 0.834 |
| 400+ (main sample) | 0.859 |

Performance saturates around **300-400 reviews** — more data yields diminishing returns.

### Category Breadth Has No Effect

Correlation between number of product categories and accuracy: r = -0.069, p = 0.229. Digital twins work equally well for specialized and diverse shoppers.

---

## Key Takeaways

1. **Fine-tuning + RAG is strictly better than either alone for consumer simulation.** The base LLM with RAG only reaches 62.3% accuracy — it knows about the product but not the person. Fine-tuning alone would know the person but not the product. The combination (85.9%) demonstrates that both components are necessary.

2. **The recall gap is the most revealing metric.** Base LLM recall is 8.5% — it almost never predicts a purchase. This exposes a fundamental limitation of generic LLMs for consumer simulation: they default to conservative, non-committal responses. Fine-tuning teaches the model to say "yes" when the consumer would.

3. **~300 reviews is the practical minimum for high-quality digital twins.** Below that, accuracy drops sharply (55% at 50-100 reviews). Above 400, gains are marginal. This sets a clear data threshold for practitioners deciding which consumers to build twins for.

4. **$0.13 per consumer makes this commercially viable.** At 10 epochs on a single RTX 4090D, creating a digital twin costs less than a cup of coffee. For a company with 10,000 high-value customers with sufficient review history, the total cost would be ~$1,300 — trivial compared to the cost of traditional focus groups or surveys.

5. **Generated reviews capture both content and style.** The 0.943 Word2Vec similarity and 0.650 Sentence-BERT similarity show the twins don't just predict *what* consumers buy, but replicate *how they would describe their experience*. This has implications for synthetic UGC, product testing, and sentiment forecasting.

6. **The theoretical grounding (Lewin's equation) is more than decoration.** It provides a principled answer to the question "why combine fine-tuning and RAG?" — because behavior is jointly determined by stable personal traits (P) and dynamic environmental context (E), and these map cleanly onto the two LLM adaptation techniques.

7. **Privacy-compliant by design.** Digital twins built from first-party data (CRM, UGC) and public product information don't require cookies, third-party tracking, or invasive data collection. This positions the approach well for GDPR/CCPA compliance as cookie-based targeting phases out.

8. **Category generalization is robust.** The non-significant correlation between category breadth and accuracy means a consumer who shops exclusively in electronics gets an equally good twin as someone who shops across 9 categories. The model learns *how the person evaluates products*, not just *which products they like*.

9. **Key limitation: only text-based behavior is modeled.** Browsing patterns, cart abandonment, visual attention, and non-verbal cues are entirely absent. The twin knows what the consumer *says* but not what they *do silently* — which may be the majority of consumer behavior.

10. **No comparison against proprietary models.** The paper uses only Qwen 2.5-7B. Performance with GPT-4, Claude, or Gemini is unknown and could be significantly better, especially for the review generation task where language quality matters.

---

## What's Open-Sourced

- **Nothing released** — no code, fine-tuned models, or processed datasets
- **Data source is public:** Amazon Reviews Dataset (McAuley Lab) is publicly available
- **Model is open-source:** Qwen 2.5-7B is freely available
- **Pipeline is reproducible:** All hyperparameters and implementation details are documented
