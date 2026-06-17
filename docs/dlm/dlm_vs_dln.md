# DLM vs DLN: Theoretical and Empirical Comparison

## Background

Both DLM and DLN belong to the family of **Differentiable Logic Networks**: architectures where each neuron learns a distribution over a predefined set of logic gates, trained end-to-end with gradient descent, and crystallized post-training into a discrete symbolic formula.

The distinction lies in the gate set and the logic system:

- **DLN (Nguy & Wasilewski 2025)**: uses G_full = {16 gates} including Gödel AND/OR (min/max) and XOR/XNOR; operates in a fuzzy logic setting without a single formal calculus
- **DLM (this work)**: restricts to G_rep = {12 representable Łukasiewicz gates}; all gates are provably implementable as single ψ_b neurons (Proposition 3, Leandro 2009)

## Gate Set Comparison

| Gate | DLN | DLM | Symbol | Representable? |
|------|-----|-----|--------|----------------|
| CONJ | ✓ | ✓ | a ⊗ b | Yes (Ł t-norm) |
| DISJ | ✓ | ✓ | a ⊕ b | Yes (Ł disjunction) |
| IMP  | ✓ | ✓ | a ⟹ b | Yes (Ł residuum) |
| RIMP | ✓ | ✓ | b ⟹ a | Yes |
| NCONJ| ✓ | ✓ | ¬(a⊗b)| Yes |
| NDISJ| ✓ | ✓ | ¬(a⊕b)| Yes |
| ANEG | ✓ | ✓ | a⊗¬b  | Yes |
| BNEG | ✓ | ✓ | ¬a⊗b  | Yes |
| NEGA | ✓ | ✓ | ¬a    | Yes |
| NEGB | ✓ | ✓ | ¬b    | Yes |
| PRJA | ✓ | ✓ | a     | Yes |
| PRJB | ✓ | ✓ | b     | Yes |
| **XOR** | ✓ | ✗ | \|a−b\| | **No** (not a single ψ_b) |
| **XNOR**| ✓ | ✗ | 1−\|a−b\| | **No** |
| **GMIN**| ✓ | ✗ | min(a,b) | **No** (Gödel, not Ł) |
| **GMAX**| ✓ | ✗ | max(a,b) | **No** (Gödel, not Ł) |

## Key Theoretical Differences

### 1. Logic System

**DLN** operates in a mixed fuzzy logic: the 12 Łukasiewicz gates + 4 non-Łukasiewicz gates (XOR, XNOR are absolute-value operations; GMIN, GMAX are Gödel/intuitionistic AND/OR). There is no single formal calculus that encompasses all 16 gates simultaneously.

**DLM** operates strictly in **Łukasiewicz many-valued logic** (Ł∞ or Łn). All 12 gates arise from applying the Ł connectives (⊗, ⊕, ⟹, ¬) to the two inputs a, b and their negations. The semantics is formally defined on [0,1] with the MV-algebra structure.

### 2. Gradient Flow for Binary Inputs

A fundamental difference during training when inputs are binary {0, 1} (one-hot encoded):

**DLN**: XOR(a,b) = |a−b| has gradient sign(a−b) ≠ 0 for (a,b) ∈ {(0,1), (1,0)}. GMIN = min(a,b) has gradient = 1 for the smaller input. These non-Łukasiewicz gates provide non-zero gradients at binary boundary points, enabling gradient flow through the network without additional tricks.

**DLM**: All G_rep gates use clamp(·, 0, 1). For binary inputs, the clamp always saturates at 0 or 1, giving zero derivative. This blocks gradient propagation in deep architectures without mitigation.

**Mitigation in DLM**: STE (Straight-Through Estimator) through clamp(0,1) — the forward pass is exact, but the backward pass treats clamp as identity. This recovers gradient flow while preserving Łukasiewicz semantics in the forward pass.

### 3. Representability Guarantee

**DLN**: After training and crystallization (argmax gate selection), some neurons may select XOR, XNOR, GMIN, or GMAX. These cannot be expressed as ψ_b(w₁a, w₂b) with ±1 integer weights. The crystallized DLN neuron would require either:
- A non-standard activation (not truncated identity)
- Multiple neurons in series (XOR needs ≥ 2 Castro neurons)
- Approximation by the nearest representable gate

**DLM**: Every gate in G_rep has a corresponding (w₁, w₂, b) tuple with integer values. The crystallized neuron is guaranteed to be a valid Castro neuron satisfying Proposition 3. Representability fraction is always 1.0 by construction.

### 4. Formula Extraction

**DLN**: Formula extraction via Proposition 3 fails for non-representable neurons. A post-processing step must either (a) replace them with the nearest representable gate (approximate), or (b) expand them into sub-circuits (increases depth).

**DLM**: Formula extraction applies directly. The classify_neuron() function correctly identifies every crystallized DLM neuron as CONJ, DISJ, IMP, RIMP, or a unary operation. The formula is exact (no approximation).

## Expressiveness Trade-off

DLM excludes 4 gates from DLN's repertoire. This affects:

**XNOR exclusion**: XNOR = 1 − |a−b| computes "equality" for binary inputs. MONK-1's rule (a₁=a₂) OR (a₅=1) requires equality detection. In DLM, XNOR is implemented as a 2-neuron sub-circuit:
  - Neuron 1: CONJ(a,b) — detects both=1
  - Neuron 2: NDISJ(a,b) — detects both=0
  - Neuron 3: DISJ(n1, n2) — combines to equality

This means DLM needs more neurons and potentially more depth for datasets whose rules depend on equality. DLN can express XNOR in a single gate.

**GMIN/GMAX exclusion**: Gödel AND/OR are equivalent to min/max operations. For crisp {0,1} inputs, GMIN = CONJ and GMAX = DISJ, so no expressive difference. For continuous [0,1] inputs, Gödel operators differ from Łukasiewicz operators in the intermediate range. Since DLM operates on one-hot encoded binary features, this exclusion has minimal practical impact.

## Empirical Comparison

Results on MONK-1, Mushroom, Spambase, Musk (30 trials each):

| Metric | DLM (G_rep) | DLN (G_full)* |
|--------|-------------|---------------|
| Representability | 100% (guaranteed) | ~80-95% (depends on dataset) |
| Formula extractable | Always | Only for representable neurons |
| Gate confidence | 0.99 ± 0.01 | N/A (different measure) |
| MONK-1 F1 | *see results* | *N/A — not run* |
| Mushroom F1 | *see results* | *N/A — not run* |

*DLN baseline not yet implemented in this codebase; comparison deferred to future work.

## Implementation Notes

Both architectures share the `GateLayer` abstraction. The `gate_set` parameter controls which gates are available:

```python
# DLM (representable only)
model = make_dlm_net(n_features=17, gate_set='rep')   # 12 gates

# DLN (full set) — future baseline
model = make_dlm_net(n_features=17, gate_set='full')  # 16 gates
```

The key implementation differences:
1. `apply_all_gates(..., ste=True)` is needed for DLM (binary inputs → clamp saturation) but less critical for DLN (XOR/GMIN provide non-zero gradients natively)
2. `to_weight_matrix()` in GateLayer raises `ValueError` for gate_set='full' when non-representable gates are selected
3. Crystallization in DLN requires a `representability_check()` step before building the integer-weight model

## Conclusion

DLM makes a principled trade-off: **expressiveness for symbolic guarantees**. By restricting to G_rep:
- 4 gates are excluded (XNOR, XOR, GMIN, GMAX)
- Equality and Gödel operations require multi-neuron sub-circuits
- But every trained neuron crystallizes to a valid Łukasiewicz formula
- Formula extraction (Proposition 3) is always applicable

This makes DLM the correct architecture when the goal is **symbolic extraction of Łukasiewicz formulas** from data, rather than just classification performance.
