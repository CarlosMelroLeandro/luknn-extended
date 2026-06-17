# Castro Neural Networks and Łukasiewicz Neural Networks

**Author:** Carlos Leandro  
**Context:** Architectural foundations for ŁNN training and symbolic extraction. Prerequisite: `docs/theory/lukasiewicz_logic.md`.

---

## 1. Castro Neural Networks

A **Castro Neural Network (CNN)** is a feed-forward network where every neuron computes the **truncated identity activation** (also called the Łukasiewicz activation):

```
ψ_b(w₁x₁, …, wₙxₙ) = clamp(Σ wᵢxᵢ + b, 0, 1)
                      = min(1, max(0, Σ wᵢxᵢ + b))
```

subject to the constraints:

- **Weights:** w_i ∈ {−1, 0, +1}  (integers; zero means the connection is absent)
- **Biases:** b ∈ ℤ  (integers; any integer value)
- **Activations:** all layer outputs remain in [0, 1] by construction (the clamp guarantees this)
- **Domain:** inputs x_i ∈ [0, 1]

These constraints are what distinguish a CNN from a standard feed-forward network with truncated activations. The integer restriction on weights and biases is not merely a quantization convenience — it is the algebraic condition that makes each neuron implement an exact Łukasiewicz connective (see §2).

### 1.1 Why Truncated Identity?

The truncated identity ψ_b is the simplest activation that:

1. Maps any real pre-activation to [0, 1] without saturation of the gradient (unlike sigmoid), except at the boundaries.
2. Is **piecewise linear** with slope 1 in the active region — the gradient either flows fully or not at all, with no scaling distortion.
3. Computes the exact Łukasiewicz operators when weights are ±1 integers (Proposition 3, proved in §2).

Standard activations (ReLU, sigmoid, tanh) do not satisfy property 3. A ReLU neuron with weights ±1 computes `max(0, Σ wᵢxᵢ + b)`, which does not in general equal a Łukasiewicz connective because it has no upper clamp.

### 1.2 Representation Theorem (Proposition 3)

**Theorem (Proposition 3 of Leandro 2009):** Let α be a neuron with n negative-weight inputs x₁, …, x_n and p positive-weight inputs x_{n+1}, …, x_m (all weights ±1). Then:

- **α implements conjunction** iff b = −p + 1:
  ```
  α = x_{n+1} ⊗ … ⊗ x_m ⊗ ¬x₁ ⊗ … ⊗ ¬x_n
  ```

- **α implements disjunction** iff b = n:
  ```
  α = x_{n+1} ⊕ … ⊕ x_m ⊕ ¬x₁ ⊕ … ⊕ ¬x_n
  ```

**Proof sketch:** For the conjunction case, the pre-activation is:
```
z = (x_{n+1} + … + x_m) − (x₁ + … + x_n) + (−p + 1)
```

When all inputs are in [0,1] and weights are ±1, substituting x_i → 1 − x_i for negative-weight inputs and using the definition of ⊗ (strong conjunction applied n+p−1 times) yields:

```
clamp(z, 0, 1) = max(0, x_{n+1} + … + x_m − x₁ − … − x_n − p + 1)
               = (… ((x_{n+1} ⊗ x_{n+2}) ⊗ x_{n+3}) … ⊗ x_m) ⊗ ¬x₁ ⊗ … ⊗ ¬x_n
```

The induction is on the number of inputs; the base case (n+p = 2) is verified directly from the connective definitions. The disjunction case follows by the de Morgan duality ¬(x ⊗ y) = ¬x ⊕ ¬y.

### 1.3 The λ-Similar Approximation (Definition 4)

Neurons that satisfy neither the conjunction nor disjunction condition (b ∉ {−p+1, n}) do not represent an exact Łukasiewicz connective. For such neurons, the paper defines the **best λ-similar approximation**: the closest representable connective under the metric

```
λ(α, β) = Σ_{t ∈ S_n^m} |α(t) − β(t)| / |S_n^m|
```

evaluated over the truth sub-table. In practice: after crystallization, each neuron is classified by testing both conditions; if neither holds, the closer one (by mean absolute deviation on the truth sub-table) is chosen as the symbolic approximation, and the approximation quality λ ∈ [0,1] is reported.

---

## 2. Łukasiewicz Neural Networks

A **Łukasiewicz Neural Network (ŁNN)** is a CNN with the additional constraint that **every neuron has at most two active inputs** (at most two non-zero weights). This binary-fan-in constraint ensures direct, unambiguous translation to a first-order Łukasiewicz formula: each neuron corresponds to a single binary connective (conjunction, disjunction, or implication via a negative weight), and the network can be read as a formula tree.

### 2.1 Structural Consequence

With fan-in ≤ 2, a 2-hidden-layer ŁNN of width h computes:

```
output = ψ_{b₃}(ψ_{b₂}(ψ_{b₁}(x, x), ψ_{b₁'}(x, x)), ψ_{b₂'}(…))
```

Each ψ call corresponds to one binary connective. The formula depth equals the number of hidden layers. A 3-layer ŁNN can represent formulas with up to 3 levels of nesting.

In the PyTorch replication, the strict fan-in ≤ 2 constraint is relaxed during training (weights are continuous, fan-in is full) and enforced only at crystallization time through sparsification. The symbolic extraction step then approximates the resulting dense integer-weight network by the closest ŁNN.

### 2.2 Multi-Input Extension (Castro generalisation)

The paper also discusses multi-input generalizations where fan-in > 2 is allowed. For n + p inputs with bias b = −p + 1, the neuron computes the n+p−1-fold strong conjunction. This corresponds to a chain of binary conjunctions, which is associative in Łukasiewicz logic:

```
x ⊗ y ⊗ z = (x ⊗ y) ⊗ z = x ⊗ (y ⊗ z)
```

So multi-input conjunctive or disjunctive neurons can be unambiguously flattened to a formula tree without loss of information.

---

## 3. Network Architecture

The feed-forward architecture used in the paper and this replication consists of:

- **Input layer:** raw features normalized to [0, 1]
- **Hidden layers:** LukasiewiczLinear layers, each applying the truncated identity activation
- **Output layer:** single neuron with truncated identity, output in [0, 1]

```
x ∈ [0,1]^m → [L₁: ReLU→clamp] → [L₂: ReLU→clamp] → … → ŷ ∈ [0,1]
```

In code:

```python
from luknn.layers.lukasiewicz_linear import make_lukasiewicz_net

model = make_lukasiewicz_net(
    n_features=17,          # input dimension
    n_hidden_layers=2,      # depth (2 recommended by paper)
    hidden_width=17,        # neurons per layer (= n_features in our replication)
    mode="ste"              # weight mode: 'continuous', 'ste', or 'clamp'
)
```

The `mode` parameter controls how weights are constrained during training (see optimizer docs); it does not affect the forward computation formula — all modes use the same `clamp(z + b, 0, 1)` activation.

---

## 4. Crystallization

**Crystallization** is the process of mapping continuous trained weights to integers in {−1, 0, +1}, producing an exact CNN from a soft trained model. It is a necessary post-processing step to enable symbolic extraction.

### 4.1 Smooth Crystallization Function Υ_n

The paper's smooth crystallization function:

```
Υ_n(w) = sign(w) · (cos((1 − frac(|w|)) · π/2)^n + ⌊|w|⌋)
```

where `frac(w) = w − ⌊w⌋` is the fractional part. Properties:

- **Fixed points:** Υ_n(k) = k for all integers k (crystallized weights are unchanged)
- **Monotone:** |Υ_n(w) − k| < |w − k| for the nearest integer k (attraction toward integers)
- **Smooth:** continuously differentiable; gradient flows through training
- **Increasing n:** sharper attraction (Υ₂ is gentle; Υ₁₆ is nearly hard)

For n = 2 (paper default), the function applies a squared-cosine taper, gently pulling weights toward their nearest integer at each training step without eliminating gradient information.

**Mathematical justification:** The substitution `u = 1 − frac(|w|)` maps the interval [k, k+1) to [0, 1]. The function cos(u·π/2)^n maps [0, 1] → [0, 1], with cos(0)^n = 1 at u = 0 (w = k+0, already at integer) and cos(π/2)^n = 0 at u = 1 (midpoint w = k + 0.5, moved to k). Adding ⌊|w|⌋ restores the integer offset. The result is that each weight is non-linearly compressed toward the nearest integer.

### 4.2 Progressive Crystallization Schedule

The replication uses a progressive schedule Υ₂ → Υ₄ → Υ₈ → Υ₁₆ applied at thresholds {0%, 50%, 75%, 90%} of training. This provides:

1. **Phase 1 (0–50%):** Gentle Υ₂ — weights move freely, crystallization pressure is low
2. **Phase 2 (50–75%):** Υ₄ — moderate pressure, most weights begin to commit
3. **Phase 3 (75–90%):** Υ₈ — strong pressure, near-integer weights are forced
4. **Phase 4 (90–100%):** Υ₁₆ — near-binary forcing; almost all weights ≥ 0.8 round to 1

### 4.3 Crisp Crystallization

After training, the final crisp step:

```python
w_crisp = round(w).clamp(-1, 1)    # weights: integers in {-1, 0, +1}
b_crisp = round(b)                   # biases: any integer (no clamp)
```

Note: the paper's original formulation uses `floor(w)` for biases, which incorrectly maps w = 0.94 → 0 instead of 1. The replication uses `round()` for correctness.

### 4.4 Representation Error Δ(N)

The **representation error** Δ(N) measures how far a trained network is from being integer-valued:

```
Δ(N) = Σᵢ |wᵢ − round(wᵢ)|
```

summed over all weights. When Δ(N) = 0, the network is fully crystallized (all weights are already integers). The normalized metric Δ(N)/n_params ∈ [0, 0.5] is used in stopping criteria.

---

## 5. Symbolic Knowledge Extraction

After crystallization (all weights in {−1, 0, +1}, all biases in ℤ), the extraction pipeline:

1. **Neuron classification:** For each neuron, count n (negative-weight inputs) and p (positive-weight inputs). Apply Proposition 3 to classify as conjunction, disjunction, or λ-similar.

2. **Formula assembly:** Recursively substitute layer outputs. Each neuron's symbolic label is substituted back to its input formula. The result is a first-order Łukasiewicz formula in the input variables.

3. **OBS pruning:** Before extraction, apply Optimal Brain Surgeon (OBS) to remove weights whose deletion does not increase MSE beyond the tolerance. This yields sparser, more interpretable formulas. OBS removes the weight w_i that minimises the second-order approximation to the loss increase:

   ```
   ΔL ≈ w_i² / (2 · [H⁻¹]_{ii})
   ```

   where H is the Hessian of the training loss. Weights are removed in order of ascending ΔL until MSE exceeds the budget.

4. **λ-Approximation fallback:** For neurons not satisfying Proposition 3, compute the best λ-similar formula over the 3-valued truth sub-table. Report the approximation quality λ.

---

## 6. Scalability Limits

The truth-sub-table approach to classification (step 1–2 above) is exponential in the number of active inputs per neuron:

| Active inputs per neuron | 3-valued sub-table size | Feasible? |
|---|---|---|
| 6 | 3⁶ = 729 | ✓ fast |
| 10 | 3¹⁰ = 59 049 | ✓ feasible |
| 12 | 3¹² = 531 441 | ✓ slow |
| 15 | 3¹⁵ ≈ 14M | ✗ memory/time limit |
| 22 | 3²² ≈ 31B | ✗ infeasible (≈1.5 TB) |
| 111 | 3¹¹¹ | ✗ astronomically infeasible |

**Practical implication:** for real-world datasets with many features, the Proximal optimizer's regularization-induced sparsity (see `docs/optimizers/proximal_algorithm.md`) is a prerequisite for extraction — it reduces each neuron's fan-in to a tractable level. Alternatively, feature pre-selection (dimensionality reduction before training) must be applied.

---

## 7. References

- Castro, J. L., Mantas, C. J., & Benítez, J. M. (2002). *Interpretation of artificial neural networks by means of fuzzy rules*. IEEE Transactions on Neural Networks.
- Leandro, C. (2009). *Symbolic Knowledge Extraction using Łukasiewicz Logics*. ALT 2009. arXiv:1604.03099.
- LeCun, Y., Denker, J. S., & Solla, S. A. (1990). *Optimal Brain Damage*. NeurIPS 1990. (Precursor to OBS used here.)
- Hassibi, B., & Stork, D. G. (1993). *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon*. NeurIPS 1993.
