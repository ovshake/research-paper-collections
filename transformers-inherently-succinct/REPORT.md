# Transformers Are Inherently Succinct

**Authors:** Pascal Bergsträßer (RPTU Kaiserslautern-Landau), Ryan Cotterell (ETH Zürich), Anthony W. Lin (RPTU Kaiserslautern-Landau & MPI-SWS)
**Date:** October 23, 2025 (arXiv v2)
**Paper:** [PDF](https://arxiv.org/abs/2510.19315)

---

## TL;DR

This paper proves that transformers can describe formal languages *far more compactly* than traditional models. Specifically, transformers (UHATs) are **exponentially** more succinct than Linear Temporal Logic (LTL) and Recurrent Neural Networks (RNNs), and **doubly exponentially** more succinct than finite automata. As a byproduct of this extreme compactness, verifying even simple properties of transformers (like "does this transformer accept any string at all?") is EXPSPACE-complete — meaning it requires double-exponential time in the worst case.

---

## Key Novel Ideas

### 1. Succinctness as a Measure of Transformer Power

Previous work on transformer expressivity focused on *what* languages transformers can recognize. This paper flips the question: given that transformers and other models (LTL, finite automata, RNNs) can all recognize the same class of languages (star-free regular languages), **how compactly** can each model describe a given language?

The succinctness of a language L with respect to a class C of language recognizers measures the smallest (denotational) size of any representation T ∈ C that recognizes L. Formally:

- C₁ is **exponentially more succinct** than C₂ if for every function f ∈ 2^(o(n)), there exists R₁ ∈ C₁ such that any R₂ ∈ C₂ recognizing the same language has size |R₂| > f(|R₁|). In other words, translating from C₁ to C₂ blows up the description size by at least an exponential.
- **Doubly exponentially more succinct** is the same idea but with f ∈ 2^(2^(o(n))) — the blowup is at least a tower of two exponentials.

This is important because succinctness directly determines how hard it is to analyze a model. A more succinct representation is harder to verify because it packs more information into fewer bits.

### 2. Transformers Can Count to 2^(2^n) — The Doubly Exponential Counter

The central technical construction shows how a transformer (UHAT) of polynomial size in n can recognize languages whose shortest accepted string has length at least 2^(2^n) — doubly exponential in the transformer's size.

Here's the intuition: the transformer encodes a **tiling problem** (a grid-coloring puzzle) as a string. The string represents a function τ that maps cells of a {1,...,2^n} × {1,...,m} grid to colored tiles. Each row of the grid is encoded as a sequence of tiles separated by binary counters (counting from 0 to 2^n − 1), and multiple rows are stacked on top of each other.

The transformer checks:
1. **Binary counter correctness:** Each counter increments properly (e.g., 0000, 0001, 0010, ...). It does this with attention — looking at the nearest # separator to the left and comparing bit patterns.
2. **Horizontal tile constraints:** Adjacent tiles in the same row have compatible colors (checked by attending to the rightmost matching position to the left).
3. **Vertical tile constraints:** Tiles in the same column of adjacent rows have compatible colors (checked by attending to positions with the same binary counter value in the previous row).

Because the grid can have 2^(2^n) cells (2^n columns × 2^(2^n) rows in the worst case), the shortest valid encoding has doubly exponential length — yet the transformer that checks it has only polynomial size.

### 3. UHAT-to-LTL Translation in Singly Exponential Time

Previous work (Yang et al., 2024) showed how to translate a UHAT into an LTL formula, but the translation incurred a **doubly exponential** blowup. This paper significantly improves this to a **singly exponential** translation.

The key observation (Proposition 11) is that all intermediate values in a UHAT computation can be represented with only a **polynomial number of bits**. This is because:
- ReLU layers don't increase bit size (just max with 0).
- Attention layers only combine two input vectors linearly with fixed rational coefficients.
- Applying linearly many layers can at most produce values that are exponential in the number of layers, but still representable with polynomially many bits.

Since the set F of possible intermediate values is at most exponentially large, and the LTL formula essentially enumerates over possible values at each layer, the resulting formula is exponential in the transformer size — not doubly exponential.

### 4. EXPSPACE-Completeness of the Non-Emptiness Problem

The paper proves a tight complexity bound: deciding whether a given UHAT (or B-RASP program) accepts **any** string is EXPSPACE-complete.

- **Lower bound (EXPSPACE-hard):** Reduces from the 2^n-tiling problem, which is known to be EXPSPACE-complete. The reduction encodes tiling instances as B-RASP programs of polynomial size (described in detail in idea #2 above).
- **Upper bound (in EXPSPACE):** Translates the UHAT to an LTL formula in exponential time (idea #3), then checks non-emptiness of the LTL formula in polynomial space (using a known result from Sistla & Clarke, 1985). The composition gives exponential space total.

This is much stronger than previously known — Sälzer et al. (2025) showed transformers are at least NEXP-hard, but this paper pins the complexity exactly at EXPSPACE-complete.

---

## Formal Framework

### Unique Hard-Attention Transformers (UHATs)

The paper works with **Unique Hard-Attention Transformers (UHATs)**, which are the simplest well-studied class of transformers in formal language theory. A UHAT is defined as:

1. A **token embedding** emb: Σ → ℚ^d mapping each symbol to a rational vector.
2. A sequence of **UHA (attention) layers** and **ReLU layers**, all of matching width.
3. An **acceptance vector** t ∈ ℚ^s for deciding whether to accept.

Each UHA layer has:
- Three **affine transformations** A, B (for query/key scoring) and C (for combining attended values).
- A **mask predicate** M (no masking, strict future masking, or strict past masking).
- A **tie-breaking function** τ (leftmost or rightmost when multiple positions have the same score).

The attention mechanism selects exactly **one** position (hard attention) based on the maximum score, with ties broken deterministically. This is weaker than softmax attention but is standard in the theoretical literature.

### Boolean RASP (B-RASP)

B-RASP is an intermediate programming language equivalent in expressivity to UHATs. A B-RASP program computes a sequence of Boolean vectors through position-wise operations (Boolean combinations) and attention operations (selecting a position based on a mask, score, and value predicate). B-RASP serves as a more convenient formalism for the proofs — constructions are easier to express as programs than as transformer weight matrices.

### Linear Temporal Logic (LTL)

LTL is a logic used in formal verification, with temporal operators **S** (since) and **U** (until). LTL formulas recognize exactly the **star-free regular languages** — the same class recognized by UHATs (shown by Yang et al., 2024). The key prior result is that LTL's non-emptiness problem is PSPACE-complete, which contrasts with the much harder EXPSPACE-completeness for UHATs.

---

## Key Results

| Result | Statement |
|--------|-----------|
| **Theorem 5** | The non-emptiness problem for UHATs and B-RASP programs is **EXPSPACE-complete** |
| **Theorem 14** | UHATs are **exponentially more succinct** than LTL |
| **Theorem 16** | UHATs are **doubly exponentially more succinct** than finite automata |
| **Corollary 17** | UHATs are **exponentially more succinct** than RNNs |
| **Proposition 12** | Any UHAT can be translated to an LTL formula of **exponential** size (improving the prior doubly-exponential bound) |
| **Proposition 15** | Any LTL formula can be translated to a UHAT in **polynomial** time |
| **Theorem 18** | The **equivalence problem** for UHATs is EXPSPACE-complete |
| **Corollary 10** | EXPSPACE-hardness holds even for restricted UHATs using only strict future masking with rightmost tie-breaking |
| **Corollary 13** | For restricted UHATs (strict future + leftmost only), non-emptiness drops to **NEXP** |

### Succinctness Hierarchy

The results establish a clean hierarchy:

```
Finite Automata ≤ RNNs ≤ LTL ≤ UHATs (transformers)
                    ↑          ↑         ↑
              doubly exp    exp      poly
              gap to UHAT  gap      (UHAT can simulate LTL in poly time)
```

Transformers sit at the top: they can represent languages in polynomial size that require exponential-size LTL formulas or doubly-exponential-size automata.

---

## Key Takeaways

1. **Transformers are extremely compact representers of patterns.** A transformer with n parameters can encode patterns that would require ~2^n parameters in an LTL formula or ~2^(2^n) states in a finite automaton. This suggests that transformers' practical power comes partly from representational efficiency, not just from recognizing a broader class of languages.

2. **Compactness comes at a cost: verification is very hard.** The EXPSPACE-completeness of non-emptiness means there's likely no efficient algorithm to check whether a transformer accepts any input at all. Under standard complexity assumptions, this requires double-exponential time — far worse than the polynomial time needed for the same question on finite automata.

3. **The doubly exponential counter trick is the core construction.** Transformers can use attention to implement binary counters that count to 2^n, and by stacking multiple rows, they can count to 2^(2^n). This is because attention can compare the current position with a distant position (via score functions) in a way that sequentially-processed models like RNNs cannot.

4. **The gap between transformers and RNNs is at least exponential.** Since RNNs with fixed precision k have 2^(kd) states (Proposition 3), they are equivalent to finite automata of exponential size. UHATs are doubly exponentially more succinct than finite automata and at least exponentially more succinct than RNNs. This includes state-space models (SSMs) like Mamba.

5. **The UHAT-to-LTL translation was tightened from doubly exponential to singly exponential.** This is a technically important improvement that enabled the tight EXPSPACE upper bound. The key insight is that intermediate transformer values need only polynomially many bits.

6. **LTL can be simulated by transformers in polynomial time.** Going the other direction is cheap: any LTL formula can be converted to a UHAT of polynomial size (Proposition 15). This means transformers are strictly more succinct than LTL — there's no reverse exponential blowup.

7. **Equivalence checking for transformers is also EXPSPACE-complete.** Determining whether two transformers recognize the same language has the same worst-case complexity as non-emptiness. This means automated comparison of transformer behaviors is computationally intractable in general.

8. **Restricting attention patterns reduces complexity.** If all layers use the same masking/tie-breaking pattern (e.g., strict future + leftmost), the non-emptiness problem drops from EXPSPACE to NEXP. This suggests that the full power of transformers comes from mixing different attention patterns.

9. **The results hold for the weakest meaningful transformer variant.** UHATs are the weakest well-studied class of transformers — they use hard attention (selecting one position) instead of softmax. Since the lower bounds already hold for this weak model, they automatically apply to more powerful variants. The upper bounds hold for fixed-precision (real-world) transformers.

10. **Connection to Zipf's law and natural language.** The paper draws an interesting parallel: just as the Hindu-Arabic numeral system is exponentially more succinct than Roman numerals (you can write "1000000" much more compactly than "MMMMMM..."), transformers achieve a similar compactness advantage over simpler models. This echoes Zipf's observation that frequently-used concepts tend to get succinct encodings.

---

## What's Open-Sourced

Nothing was released — this is a pure theoretical paper with formal proofs. No code, models, or datasets.
