# Theoretical Validation: Residual Skip Connections in Łukasiewicz Neural Networks

**Author:** Carlos Leandro | **Date:** 2026-06-03

---

## 1. Background

A Castro Neural Network (CNN) has activation ψ(x) = clamp(x, 0, 1), integer weights in {−1, 0, 1}, and integer biases. A ŁNN is a binary CNN — each neuron has at most two inputs. A neuron is *representable* if and only if it can be classified as a conjunction or a disjunction (Proposition 3 of the paper).

**Classification (Prop. 3):** given α = ψ_b(−x₁,…,−x_n, x_{n+1},…,x_m) with n negative weights and p positive weights:
- Conjunction iff `b = −p + 1`
- Disjunction iff `b = n`

---

## 2. The Residual Fusion Neuron

A residual block computes, for each dimension j:

```
y_j = ψ_{b_j}(F(x)_j + x_j)  =  clamp(F(x)_j + x_j + b_j, 0, 1)
```

This is a neuron with exactly **two inputs** — F(x)_j and x_j — both with weight +1:

```
n_neg = 0,  n_pos = 2
```

Applying Proposition 3:

| Condition     | Resulting formula            | Representable?     |
|---------------|------------------------------|--------------------|
| b_j = 0       | y_j = F(x)_j ⊕ x_j          | ✓ Disjunction      |
| b_j = −1      | y_j = F(x)_j ⊗ x_j          | ✓ Conjunction      |
| b_j ∉ {0,−1}  | not directly classifiable    | λ-similar (Def. 4) |

**Immediate conclusion:** the fusion neuron is representable for exactly two bias values (0 and −1). For any other integer value, the λ-similar mechanism already present in the implementation applies (rule R, §2.3 of the paper).

---

## 3. Conditions for Guaranteed Representability

For a crystallized residual network to be 100% representable:

| Requirement                          | Guarantee mechanism                                                          |
|--------------------------------------|------------------------------------------------------------------------------|
| Skip-connection weight fixed at +1   | Non-trainable parameter; unaffected by soft crystallization drift            |
| Inner layers F representable         | Standard crisp crystallization of each `LukasiewiczLinear`                   |
| Fusion bias ∈ {0, −1}               | Υ_n pushes toward the nearest integer; initializing at 0 steers toward ⊕    |

---

## 4. Translation to a Łukasiewicz Formula

After full crystallization, a residual block B with inner function F translates as:

```
b_j = 0  →  B(x)_j = F(x)_j ⊕ x_j
b_j = −1 →  B(x)_j = F(x)_j ⊗ x_j
```

Substituting F recursively (which is itself a ŁNN) yields a well-formed Łukasiewicz formula in each dimension j. Proposition 1 continues to guarantee that the network encodes the truth table of the formula.

**Example** with 2 inner layers (F = L₂ ∘ L₁) and bias = 0:

```
B(x)_j  =  (L₂(L₁(x))_j)  ⊕  x_j
```

---

## 5. Compatibility with Table 1 of the Paper

| Fusion neuron configuration       | Interpretation  | Table 1 row           |
|-----------------------------------|-----------------|-----------------------|
| ψ_0(F(x)_j, x_j)                 | F(x)_j ⊕ x_j   | x ⊕ y, weights +1/+1 |
| ψ_{−1}(F(x)_j, x_j)             | F(x)_j ⊗ x_j   | x ⊗ y, weights +1/+1 |

Table 1 does not list (+1, +1) weights as an explicit column, but Proposition 4 guarantees that any disjunctive/conjunctive configuration with weights in {−1, +1} is representable and reduces to these forms.

---

## 6. Constraints

1. **Dimension matching:** skip connections are only valid inside blocks with the same width. The first layer (n_inputs → hidden_width) is always a standard projection without a skip connection.
2. **Skip weight ≠ ±1 after crystallization:** if the skip weight were learned and crystallized to 0, the skip connection would vanish. The implementation fixes the weight at +1 to guarantee residual behaviour.
3. **Fusion bias ≠ integer:** unlikely after Υ_n, but covered by the λ-similar mechanism.

---

## 7. Conclusion

Adding element-wise residual skip connections with a fixed weight of +1 and a learned bias is **fully compatible with Łukasiewicz representability** (Propositions 3 and 4). The fusion neuron is a disjunction (⊕) or a conjunction (⊗) after crystallization. No new non-representable neuron type is introduced. The existing λ-similar mechanism covers the edge cases.
