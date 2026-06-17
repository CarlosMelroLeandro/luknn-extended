# Differentiable Łukasiewicz Machine (DLM)

## Overview

The Differentiable Łukasiewicz Machine (DLM) is a neuro-symbolic architecture that combines:

1. **Leandro (2009)**: Castro neural networks — every neuron implements a Łukasiewicz connective with integer weights ±1 and integer bias (Proposition 3)
2. **Nguy & Wasilewski (2025)**: Differentiable Logic Networks (DLN) — softmax-weighted mixture of gate outputs for gradient-based learning

The key contribution of DLM over DLN: DLM restricts the gate set to **G_rep** (the 12 representable gates), guaranteeing that every neuron is a valid Castro neuron after crystallization. DLN uses G_full (16 gates including non-representable XOR, XNOR, Gödel min/max), so crystallized DLN neurons may not satisfy Proposition 3.

## Gate Set G_rep

Every gate in G_rep corresponds to a single neuron ψ_b(w₁a, w₂b) = clamp(w₁a + w₂b + b, 0, 1) with w₁, w₂ ∈ {−1, 0, +1}:

| Gate | w₁ | w₂ | b | Symbol | Truth table {00,01,10,11} |
|------|----|----|---|--------|---------------------------|
| CONJ | +1 | +1 | −1 | a ⊗ b | 0,0,0,1 |
| DISJ | +1 | +1 |  0 | a ⊕ b | 0,1,1,1 |
| IMP  | −1 | +1 | +1 | a ⟹ b | 1,1,0,1 |
| RIMP | +1 | −1 | +1 | b ⟹ a | 1,0,1,1 |
| NCONJ| −1 | −1 | +2 | ¬(a⊗b)| 1,1,1,0 |
| NDISJ| −1 | −1 | +1 | ¬(a⊕b)| 1,0,0,0 |
| ANEG | +1 | −1 |  0 | a⊗¬b  | 0,0,1,0 |
| BNEG | −1 | +1 |  0 | ¬a⊗b  | 0,1,0,0 |
| NEGA | −1 |  0 | +1 | ¬a    | 1,1,0,0 |
| NEGB |  0 | −1 | +1 | ¬b    | 1,0,1,0 |
| PRJA | +1 |  0 |  0 | a     | 0,0,1,1 |
| PRJB |  0 | +1 |  0 | b     | 0,1,0,1 |

**Excluded from G_rep** (not implementable as single ψ_b): XOR = |a−b|, XNOR = 1−|a−b|, Gödel min(a,b), Gödel max(a,b).

**Complementary pair structure**: G_rep contains 6 complementary pairs that sum to 1 for any (a,b):
CONJ+NCONJ=1, DISJ+NDISJ=1, IMP+ANEG=1, RIMP+BNEG=1, NEGA+PRJA=1, NEGB+PRJB=1.

This structure means that with uniform softmax weights, the output of any neuron is exactly 0.5 regardless of input — explaining the initialization problem.

## Architecture

```
Input (n_features)
    ↓
GateLayer(n_features → hidden_width)    [hidden layer 1]
    ↓
GateLayer(hidden_width → hidden_width)  [hidden layer 2, × (n_hidden_layers−1)]
    ↓
GateLayer(hidden_width → n_output_heads) [output layer]
    ↓
mean(n_output_heads)  →  sigmoid threshold → binary prediction
```

Each **GateLayer** contains N neurons. Neuron k:
1. Receives a fixed pair of inputs (a_k, b_k) selected from the previous layer
2. Maintains logits θ_k ∈ ℝ^{12} (one per gate in G_rep)
3. Computes a **soft mixture** during training:

   out_k = Σ_{g ∈ G_rep} softmax(θ_k / T)_g · gate_g(a_k, b_k)

4. After training, **crystallizes** to the argmax gate:

   g* = argmax(θ_k),  out_k^crys = ψ_{b_{g*}}(w_{1,g*}·a_k, w_{2,g*}·b_k)

## Input Pair Selection

Each neuron selects exactly 2 inputs from the previous layer. The pairing is **fixed at initialization** and not learned. Two strategies:

- **random** (default): each neuron independently draws 2 distinct indices uniformly
- **sequential**: neuron k uses inputs (2k mod fan_in, (2k+1) mod fan_in)

Random pairing relies on probabilistic coverage: with N neurons over fan_in features, the expected fraction of feature pairs covered is approximately 1 − e^{−N/fan_in}.

## Multiple Output Heads

A critical architectural choice for training stability. With a single output neuron (n_output_heads=1), the gradient path from the output to hidden layer neurons covers only 2/hidden_width neurons per step (≈3% for hidden_width=68). This causes gradient sparsity: most hidden neurons receive no signal per iteration.

With n_output_heads=K output neurons and mean aggregation:
- The gradient covers K×2 neurons of the last hidden layer per step
- For K=n_features (recommended): coverage ≈ 50–90% of hidden neurons per step
- After crystallization, majority vote of K binary outputs gives the final prediction

Recommended setting: **n_output_heads = n_features** (or hidden_width // 4).

## STE for Gradient Flow

Łukasiewicz gate functions involve clamp(·, 0, 1). For binary {0,1} inputs, the gate output is always exactly at the boundary (0 or 1), giving zero derivative through the clamp. This blocks gradient flow through multi-layer architectures.

**Fix**: Straight-Through Estimator (STE) through clamp(0,1). The forward pass is exact (Łukasiewicz semantics); the backward pass treats the clamp as identity:

```
STE_clamp(x): forward = clamp(x, 0, 1)
              backward: ∂STE_clamp/∂x = 1 (always)
```

STE is activated automatically in `GateLayer.forward()` when the module is in training mode (`self.training == True`). During crystallization (`model.eval()`), the exact clamp is used.

## Training Protocol

### Phase 1: Gate Exploration (p1_fraction of budget)
- Adam optimizer on BCE(mean_output, y)
- Temperature T = T_init (broad: softmax near-uniform)
- Best logit checkpoint saved at minimum BCE
- Stagnation-based early stopping

### Phase 2: Gate Sharpening (remaining budget)
- Adam optimizer on BCE + λ_H · H(softmax(θ)) (entropy regularization)
- Temperature anneals T_init → T_final (sharpening gate distributions)
- Entropy regularization warm-up: 0 → λ_entropy
- Best logit checkpoint updated
- Dual stopping: MSE < tol_mse AND gate_confidence > conf_threshold

### Post-training Crystallization
- Restore best-BCE logit checkpoint
- Argmax gate selection per neuron
- Verify 100% representability (guaranteed for gate_set='rep')

### Hyperparameter Recommendations

| Parameter | Small datasets (MONK) | Large datasets (Mushroom, Spambase) |
|-----------|----------------------|--------------------------------------|
| lr | 5e-3 | 5e-3 |
| T_init | 2.0 | 2.0 |
| T_final | 0.05 | 0.05 |
| lambda_entropy | 0.15 | 0.15 |
| p1_fraction | 0.5 | 0.5 |
| conf_threshold | 0.90 | 0.90 |
| max_iter | 3000 | 2000 |
| batch_size | None (full) | 512 |
| n_output_heads | n_features | n_features or 64 |
| hidden_width | 4 × n_features | 4 × n_features (capped at 256) |

## Crystallization and Symbolic Extraction

After training, `model.crystallize()` returns a `CrystallizedDLM`: a standard Castro neural network with integer weights. Every neuron satisfies Proposition 3 (guaranteed by G_rep construction).

The crystallized model is compatible with the existing extraction pipeline:
```python
from luknn.extraction.classifier import classify_neuron
from luknn.extraction.extractor   import extract_formula

crys = model.crystallize()
for layer in crys.linear_layers:
    for neuron_idx in range(layer.out_features):
        w = layer.weight.data[neuron_idx]
        b = layer.bias.data[neuron_idx]
        cfg = classify_neuron(w, b)  # returns CONJ/DISJ/etc.
```

## Comparison with DLN (Nguy & Wasilewski 2025)

| Property | DLN | DLM |
|----------|-----|-----|
| Gate set | G_full (16 gates) | G_rep (12 gates) |
| Representability | Not guaranteed | 100% guaranteed |
| Gradient for binary inputs | Non-zero (via XOR/Gödel) | Requires STE |
| Łukasiewicz semantics | Approximated | Exact |
| Formula extraction | Not always possible | Always possible (Proposition 3) |
| Crystallization | May produce non-representable neurons | Always produces valid Castro neurons |

## Theoretical Guarantee

**Proposition (DLM Representability)**: For a trained DLMNetwork with gate_set='rep', the crystallized model `crys = model.crystallize()` satisfies:
- Every neuron has weights w_i ∈ {−1, 0, +1} and bias b ∈ ℤ
- Every neuron is classified as CONJ, DISJ, IMP, RIMP, or a unary operation by Proposition 3
- `crys.representability_fraction() == 1.0` by construction

*Proof*: The argmax over G_rep always selects a gate in G_rep. Each gate in G_rep has a corresponding (w₁, w₂, b) tuple with integer values satisfying Proposition 3. The `to_weight_matrix()` method reads these integers directly. ∎
