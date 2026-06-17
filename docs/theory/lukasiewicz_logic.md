# Łukasiewicz Many-Valued Logic — Theoretical Foundations

**Author:** Carlos Leandro  
**Context:** Theoretical basis for Łukasiewicz Neural Networks (ŁNNs). Refer to the original paper: Leandro (2009), *Symbolic Knowledge Extraction using Łukasiewicz Logics*, ALT 2009, arXiv:1604.03099.

---

## 1. Motivation

Classical Boolean logic restricts every proposition to one of two truth values: 0 (false) or 1 (true). This binary constraint is adequate for modeling crisp symbolic rules but ill-suited for reasoning over real-valued sensor data, probability estimates, or partial membership. Łukasiewicz many-valued logic, introduced by Jan Łukasiewicz in 1920 as an extension of classical logic to a third truth value "possible" (later generalised to a continuum), provides a principled algebraic framework for graded truth that connects naturally to neural computation over [0, 1].

The key property that makes Łukasiewicz logic useful for neural-symbolic integration is that its connectives are **piecewise-linear functions of [0, 1]** — they can be implemented exactly by neurons with linear pre-activation and truncated-identity activation, without approximation.

---

## 2. The Continuous Łukasiewicz Algebra

The continuous Łukasiewicz algebra is the structure **[0, 1]_Ł** = ⟨[0, 1], ⊗, ⟹, ¬, 0, 1⟩ where the truth domain is the unit interval and the connectives are defined as follows.

### 2.1 Core Connectives

| Connective | Symbol | Definition | Classical limit |
|---|---|---|---|
| Conjunction (strong) | x ⊗ y | max(0, x + y − 1) | x AND y |
| Implication (residuum) | x ⟹ y | min(1, 1 − x + y) | NOT x OR y |
| Negation | ¬x | 1 − x | NOT x |
| Disjunction (strong) | x ⊕ y | min(1, x + y) | x OR y |
| Weak conjunction | x ∧ y | min(x, y) | x AND y |
| Weak disjunction | x ∨ y | max(x, y) | x OR y |

The **strong conjunction** ⊗ and **implication** ⟹ form a *residuated pair*: the defining property of a residuated lattice is that x ⊗ y ≤ z if and only if x ≤ y ⟹ z. This adjointness relation ensures the logic is *algebraically complete* — every tautology provable semantically is also provable by the Hilbert-style axiom system (Łukasiewicz completeness theorem).

### 2.2 Geometric Interpretation

For x, y ∈ [0, 1]:

```
x ⊗ y:  the signed distance from the diagonal x + y = 1, clipped at 0
          (zero when x and y do not "overlap" in probability mass)

x ⊕ y:  the signed distance from the diagonal x + y = 0, clipped at 1
          (saturation of the combined evidence)

x ⟹ y:  the "degree of inclusion" of x in y;
          equals 1 whenever y ≥ x (x is already implied), decreases linearly
          as y falls below x
```

These interpretations make Łukasiewicz logic a natural language for expressing graded inclusion, partial evidence combination, and soft constraints.

### 2.3 Algebraic Identities

Selected identities that hold in [0, 1]_Ł (used in formula simplification and neural-to-symbolic translation):

```
¬¬x          = x                                 (involution)
x ⊕ ¬x       = 1                                 (excluded middle in Ł)
x ⊗ ¬x       = 0                                 (non-contradiction)
x ⟹ y        = ¬x ⊕ y                           (material implication)
¬(x ⊗ y)     = ¬x ⊕ ¬y                          (de Morgan strong)
¬(x ⊕ y)     = ¬x ⊗ ¬y                          (de Morgan strong)
x ⊗ y        = ¬(x ⟹ ¬y)                        (conjunction from implication)
x ⊕ y        = ¬x ⟹ y                            (disjunction from implication)
```

The last two identities show that ⟹ and ¬ form a *functionally complete* basis for the strong fragment: ⊗ and ⊕ are derivable.

---

## 3. Finite-Valued Łukasiewicz Logics

### 3.1 The (n+1)-Valued Space

For any integer n ≥ 1, the **(n+1)-valued Łukasiewicz logic** restricts truth values to the finite set:

```
S_n = {0, 1/n, 2/n, …, (n-1)/n, 1}
```

The connectives are defined by the same formulas as in the continuous case, evaluated at rational points and clamped to S_n. For x, y ∈ S_n, the results x ⊗ y, x ⊕ y, ¬x, and x ⟹ y are always in S_n — the algebra is closed.

The classical Boolean case corresponds to n = 1 (S_1 = {0, 1}).  
The 3-valued logic (n = 2, S_2 = {0, 0.5, 1}) adds a "half-true" element that the original paper uses to generate training data for formula reverse-engineering.

### 3.2 Truth Sub-Tables

Given a formula f(x₁, …, x_m) and truth domain S_n, its **truth sub-table** is the enumeration of all (n+1)^m truth assignments and their outputs. This is the training target for the reverse-engineering task in the original paper.

For the 3-valued case (n = 2):

- m = 6 variables → (2+1)^6 = 729 rows
- The formula is treated as a function [0,1]^m → [0,1] evaluated at all rational points in S_2^m

The truth sub-table is faithful: two formulas with identical truth sub-tables over S_n are logically equivalent in the (n+1)-valued logic, even if they have different syntactic forms.

### 3.3 Choice of n = 2 (3-valued) in the paper

The paper uses S_2 = {0, 0.5, 1} for truth table generation because:

1. It is the smallest S_n that distinguishes conjunction from minimum: for x = y = 0.5, `x ⊗ y = 0` but `min(x,y) = 0.5`.
2. The truth table size grows as 3^m, which is feasible for m ≤ 6 features (729 rows) and small enough for LM's quadratic-cost Jacobian computation.
3. A network trained on 3-valued data generalises to the continuum: if it correctly outputs S_2 values on all 3^m inputs, the identity of the formula is recovered because S_2 separates all connectives.

---

## 4. Connection to Neural Computation

### 4.1 Linear Pre-Activation and Łukasiewicz Connectives

Consider a neuron computing `ψ_b(w₁x₁ + w₂x₂ + … + wₙxₙ)` where ψ_b is the **truncated identity** (clipped linear):

```
ψ_b(z) = clamp(z + b, 0, 1) = min(1, max(0, z + b))
```

For integer weights w_i ∈ {−1, 0, +1} and integer bias b, this neuron computes an exact Łukasiewicz connective (Proposition 3 of the paper; proved in §5.2.1 below).

The key insight is that **every** Łukasiewicz connective of two variables can be written as ψ_b(w₁x + w₂y) for appropriate integer (w₁, w₂, b):

| Connective | w₁ | w₂ | b | Formula |
|---|---|---|---|---|
| x ⊗ y | +1 | +1 | −1 | clamp(x + y − 1, 0, 1) |
| x ⊕ y | +1 | +1 | 0 | clamp(x + y, 0, 1) |
| ¬x | −1 | — | +1 | clamp(1 − x, 0, 1) |
| x ⟹ y | −1 | +1 | +1 | clamp(1 − x + y, 0, 1) |
| min(x,y) | +1 | +1 | −1 followed by additional | not representable as a single ψ_b neuron |

**Weak conjunction (min) and disjunction (max) are not directly representable** as a single truncated-identity neuron with weights ±1. They require either more complex architectures or the λ-approximation mechanism (Definition 4 of the paper).

### 4.2 Proposition 3 Restated

Let α = ψ_b(−x₁ − … − x_n + x_{n+1} + … + x_m) with n negative-weight inputs and p = m − n positive-weight inputs, all weights ±1.

- α computes x_{n+1} ⊗ … ⊗ x_m ⊗ ¬x₁ ⊗ … ⊗ ¬x_n **iff** b = −p + 1
- α computes x_{n+1} ⊕ … ⊕ x_m ⊕ ¬x₁ ⊕ … ⊕ ¬x_n **iff** b = n

This provides the complete classification of two-connective neurons and is the foundation for symbolic extraction: after crystallization (w_i ∈ {−1, 0, +1}, b ∈ ℤ), every neuron in the network is either a conjunction, a disjunction, or a λ-similar formula (intermediate case).

### 4.3 Formal Language over Ł

A first-order Łukasiewicz formula over variables x₁, …, x_m is defined recursively:

```
φ ::= x_i                      (atomic proposition)
     | ¬φ                       (negation)
     | φ ⊗ ψ                    (strong conjunction)
     | φ ⊕ ψ                    (strong disjunction)
     | φ ⟹ ψ                   (implication)
     | φ ∧ ψ                    (weak conjunction — not directly representable)
     | φ ∨ ψ                    (weak disjunction — not directly representable)
```

The neural network computes a formula in this language when every neuron satisfies Proposition 3. The depth of the formula tree equals the number of hidden layers; each edge of the formula DAG corresponds to a non-zero weight.

---

## 5. Comparison with Other Logics

| Property | Classical Boolean | Łukasiewicz | Fuzzy (Zadeh) |
|---|---|---|---|
| Truth domain | {0, 1} | [0, 1] or S_n | [0, 1] |
| Conjunction | AND (idempotent) | ⊗ = max(0, x+y−1) | min(x,y) |
| Disjunction | OR (idempotent) | ⊕ = min(1, x+y) | max(x,y) |
| Negation | ¬x = 1−x | ¬x = 1−x | ¬x = 1−x |
| Excluded middle | holds | holds | fails in general |
| Non-contradiction | holds | holds | fails in general |
| Completeness theorem | Gödel 1930 | Łukasiewicz 1930 | none (not axiomatisable) |
| Neural implementation | threshold gate | truncated-identity ψ_b | various |

Łukasiewicz logic occupies a privileged position: it is the unique extension of classical logic to [0, 1] that preserves both the excluded middle (x ⊕ ¬x = 1) and the residuated structure (x ⊗ y ≤ z ↔ x ≤ y ⟹ z). Zadeh's fuzzy logic uses min/max connectives that are idempotent and do not satisfy the residuated property.

---

## 6. References

- Łukasiewicz, J. (1920). *O logice trójwartościowej*. Ruch Filozoficzny, 5, 170–171.
- Hájek, P. (1998). *Metamathematics of Fuzzy Logic*. Kluwer Academic Publishers.
- Cignoli, R., D'Ottaviano, I. M. L., & Mundici, D. (2000). *Algebraic Foundations of Many-Valued Reasoning*. Kluwer.
- Leandro, C. (2009). *Symbolic Knowledge Extraction using Łukasiewicz Logics*. ALT 2009. arXiv:1604.03099.
