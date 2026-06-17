# Proximal Gradient Optimizer — Algorithm and Variants

**Author:** Carlos Leandro  
**Context:** One of three optimizer families for ŁNN training. Strongest sparsity induction; key failure modes well-characterised. Prerequisites: `docs/theory/castro_networks.md`.

---

## 1. Theoretical Background

### 1.1 Proximal Gradient Descent

For a composite objective L(w) = f(w) + g(w) where f is smooth (differentiable) and g is convex but potentially non-smooth (e.g., L1 regularization), the **proximal gradient** update is:

```
w_{k+1} = prox_{α·g}(w_k − α · ∇f(w_k))
```

where the **proximal operator** of g is:

```
prox_{α·g}(v) = argmin_u { g(u) + (1/2α) ‖u − v‖² }
```

For L1 regularization g(w) = λ‖w‖₁, the proximal operator is the **soft-threshold**:

```
prox_{α·λ‖·‖₁}(v)_i = sign(v_i) · max(0, |v_i| − α·λ)
```

This shrinks weights toward zero by exactly α·λ; weights below the threshold are set to zero (sparsification). The proximal operator can be interpreted as a projection plus contraction: it finds the point closest to v that simultaneously minimizes g.

### 1.2 Why Standard Proximal Gradient Fails for ŁNNs

The L1 proximal operator assumes the target is 0 (weights are compressed toward zero). For ŁNNs, the target is the **ternary lattice** {−1, 0, +1} — a multimodal attractor. Simple L1 sparsification:

1. Successfully drives weights to 0 (the zero attractor)
2. Has no mechanism to drive weights to ±1 (the non-zero attractors)

The result is a fully zeroed network that predicts the majority class — precisely the collapse observed in the baseline Proximal experiments.

### 1.3 Ternary Regularization as Proximal Approximation

The ternary penalty:

```
P(w) = w²(1 − w²)
```

acts as a non-convex regularizer with minima at {−1, 0, +1}. The full regularization loss is:

```
L_reg = MSE + λ_sparse · ‖w‖₁ + λ_attract · Σᵢ wᵢ²(1 − wᵢ²)
```

This is a non-convex composite: L1 drives weights toward 0, ternary attraction drives weights toward {−1, 0, +1}, and MSE provides task-relevant gradient. The interplay between these three terms determines which attractor each weight converges to.

**Critical issue:** when L1 dominates (high λ_sparse or early in training when weights are small), all weights collapse to 0. The two-phase training protocol addresses this by separating the MSE phase from the regularization phase.

### 1.4 Phase Structure

All Proximal variants use a three-phase structure:

```
Phase 1 (phase1_fraction × max_iter):
    Minimize MSE only (no regularization)
    Run until stagnation (not until tol_mse — let the continuous solution fully converge)

Phase 2 ((1−phase1_fraction) × max_iter):
    Minimize MSE + λ_sparse·‖w‖₁ + λ_attract·P(w)
    λ grows linearly 0 → target (warm-up)
    Soft-threshold applied after each step
    Stop when: MSE < tol_mse AND stuck_fraction < tol_dn (dual criterion)

Phase 3 (200 fixed steps, only if best_mse < 0.15):
    Hardening: same as Phase 2 but λ_sparse×10 and λ_attract×10
    Forces near-integer weights over the crystallization threshold
```

The dual stopping criterion in Phase 2 prevents exit while weights are still in the "stuck zone" (0.1 < |w| < 0.9), ensuring that crystallization is non-destructive.

### 1.5 Stuck Fraction Metric

```python
def stuck_fraction(model):
    parts = []
    for name, p in model.named_parameters():
        if "weight" in name:
            stuck = ((p.data.abs() > 0.1) & (p.data.abs() < 0.9)).float()
            parts.append(stuck)
    return torch.cat([s.flatten() for s in parts]).mean().item()
```

A weight is "stuck" if it is neither pruned (near 0, |w| ≤ 0.1) nor committed (near ±1, |w| ≥ 0.9). Phase 2 should run until stuck_fraction < 0.05 (at most 5% of weights undecided).

---

## 2. Benchmark Results (30 trials per dataset/variant)

| Dataset | Variant | F1 [95% CI] | Notes |
|---|---|---|---|
| MONK-1 (17f) | Proximal | 0.000 [0.000, 0.000] | collapse; 500 iters insufficient |
| MONK-1 (17f) | ProximalTopK | 0.034 [−0.017, 0.085] | marginal; mostly collapse |
| MONK-1 (17f) | ProximalGroupLasso | 0.000 [0.000, 0.000] | collapse |
| MONK-1 (17f) | ProximalL0 | 0.000 [0.000, 0.000] | collapse |
| Mushroom (111f) | Proximal | 0.051 [−0.007, 0.109] | near-collapse |
| Mushroom (111f) | **ProximalTopK** | **0.338 [0.247, 0.429]** | only variant that learns |
| Mushroom (111f) | ProximalGroupLasso | 0.000 [0.000, 0.000] | collapse |
| Mushroom (111f) | ProximalL0 | 0.000 [0.000, 0.000] | collapse |
| Spambase (57f) | Proximal | 0.000 [0.000, 0.000] | collapse |
| Spambase (57f) | **ProximalTopK** | **0.419 [0.331, 0.507]** | only variant that learns |
| Spambase (57f) | ProximalGroupLasso | 0.000 [0.000, 0.000] | collapse |
| Spambase (57f) | ProximalL0 | 0.000 [0.000, 0.000] | collapse |
| Musk (166f) | all | 0.000 [0.000, 0.000] | all collapse |

---

## 3. Variant: `Proximal` (Corrected Baseline)

### Description

Two-phase optimizer with dual stopping. This is the corrected version of the original `ProximalOptimizerOLD`:

- **Fix 1:** Phase 1 runs until stagnation (not early-exit at tol_mse), ensuring the continuous solution fully converges before regularization starts.
- **Fix 2:** Phase 2 uses dual stopping (MSE + stuck_fraction), preventing premature exit while weights are undecided.

### Hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| lr | 1e-2 | Adam learning rate |
| lambda_sparse | 1e-3 | L1 coefficient (Phase 2 and 3) |
| lambda_attract | 0.1 | Ternary-attraction coefficient |
| prox_threshold | 5e-4 | Soft-threshold magnitude per step |
| phase1_fraction | 0.6 | Phase 1 budget fraction |

### Mechanism

Phase 1: pure Adam on MSE with weight projection to [−1, 1] after each step. The projection is a hard clamp — any weight driven outside [−1, 1] by Adam (e.g., large gradient steps) is immediately clipped.

Phase 2: Adam on MSE + ternary regularization. The soft-threshold `sign(w) · max(0, |w| − threshold·scale)` is applied after each step — this shrinks weights toward 0 proportionally to their magnitude, providing additional sparsification beyond the L1 gradient alone.

### Failure Mode on High-Dimensional Datasets

On Mushroom (111 features): Phase 1 converges to a continuous solution where each neuron has small weights spread across many of the 111 inputs. When Phase 2 begins, the L1 term shrinks all small weights toward 0, collapsing the distributed representation. The ternary attraction P(w) = w²(1−w²) has near-zero gradient at w ≈ 0 (P'(0) = 0), so it cannot rescue weights that have already been pushed to near-zero by L1.

**Root cause:** the distributed-small-weight equilibrium in high-dimensional input spaces is a fixed point of the L1 penalty. There is no mechanism in the base Proximal to break this symmetry and concentrate weights on a sparse subset of inputs.

### Justification for Inclusion

Despite poor empirical performance on Mushroom and Spambase, the base Proximal remains in the benchmark as the theoretical reference: it is the simplest implementation of the phase-separated training protocol, and its failure modes illuminate why the specialized variants (TopK, GroupLasso, L0) were designed.

---

## 4. Variant: `ProximalTopK` (Top-K Fan-In Pruning)

### Description

ProximalTopK inserts a **hard sparsification step** between Phase 1 and Phase 2: each output neuron retains only its k largest-magnitude input weights; the remaining weights are zeroed and **permanently masked** for the rest of training.

```python
def _apply_topk_mask(model, k):
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() == 2 and p.shape[1] > k:
            threshold = p.data.abs().topk(k, dim=1).values[:, -1:]
            mask = (p.data.abs() >= threshold).float()
            p.data *= mask
```

The mask is enforced at every Phase 2 step: `p.data *= mask` ensures gradient updates cannot reactivate pruned connections.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| k_per_neuron | 10 | Max active inputs per output neuron after pruning |

### Why k=10?

After Phase 1, each neuron has learned which inputs are most relevant. The top-10 selection is a heuristic: large enough to retain complex interactions (k ≥ 4 is needed for conjunction of 4 variables), small enough to prevent weight dilution (k = 10 out of 111 features = 9% active). The specific value of k should be tuned per dataset; k=10 was chosen as a conservative default.

### Theoretical Justification

The distributed-small-weight trap arises because Phase 2's L1 penalty cannot distinguish between weights that are "small and important" and "small and irrelevant." Top-K selection, applied after Phase 1 has sorted weights by importance (magnitude), removes the ambiguity: the surviving weights are the k most important per neuron. Phase 2 then has a sparse, high-magnitude weight matrix to work with — the L1 penalty shrinks irrelevant connections (already zeroed), while ternary attraction commits the surviving weights to ±1.

This is a form of **magnitude-based pruning** followed by **ternary crystallization** — a two-stage pipeline analogous to weight pruning in model compression (Han et al., 2016).

### Phase 2 Behaviour after Top-K

After masking, the network typically has MSE higher than Phase 1's best (the pruning removes information). Phase 2 re-optimizes from this degraded starting point. The mask prevents reactivation of pruned weights; Phase 2 can only recover by adjusting the surviving weights more aggressively.

### Empirical Outcome

**ProximalTopK is the only Proximal variant that consistently learns on Mushroom and Spambase:**
- Mushroom: F1 = 0.338 [0.247, 0.429] — significantly better than all other Proximal variants (p_holm < 0.0001 for all pairwise comparisons)
- Spambase: F1 = 0.419 [0.331, 0.507] — again significantly better (p_holm < 0.0001)

On MONK-1, all Proximal variants collapse (F1 ≈ 0). This is likely an iteration budget issue (500 iters) rather than a structural failure: Phase 1 on MONK (124 training samples, 17 features) converges to near-zero MSE quickly, but the subsequent Phase 2 regularization disrupts the solution and 500 remaining iterations are insufficient to recover.

**ProximalTopK is the preferred variant for datasets with n_features > 40.**

---

## 5. Variant: `ProximalGroupLasso` (Structured Group Sparsity)

### Description

ProximalGroupLasso modifies Phase 1 to include a **Group Lasso** penalty:

```
L_GL = MSE + λ_group · Σᵢ ‖w_i‖₂
```

where w_i is the vector of all input weights to output neuron i (one row of the weight matrix). Unlike the element-wise L1 penalty, the Group Lasso applies **joint** pressure: the gradient is proportional to `w_i / ‖w_i‖₂`, which uniformly shrinks the entire row of weights.

The Group Lasso produces **structured sparsity**: instead of shrinking individual weights, it drives entire rows toward zero (eliminating whole input connections), leaving a small number of rows with high-magnitude weights.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| lambda_group | 0.01 | Group Lasso coefficient (Phase 1 only) |

### Theoretical Justification

The Group Lasso penalty:

```
R(W) = Σᵢ ‖W_i·‖₂ = Σᵢ √(Σⱼ wᵢⱼ²)
```

has the property that when `‖w_i‖₂` is small, the gradient `∂R/∂wᵢⱼ = wᵢⱼ / ‖w_i‖₂` has large magnitude for each element — the group is collectively driven to zero. This is the "all-or-nothing" property that distinguishes Group Lasso from element-wise L1: either the whole group survives (if any weight is large) or the whole group is eliminated.

For ŁNNs, where each row represents all inputs to one neuron, this corresponds to **neuron-level pruning**: a neuron either receives inputs from many features (large group norm) or none (group norm = 0). This should promote sparser, more interpretable formulas.

### Empirical Outcome

**ProximalGroupLasso collapses on all datasets (F1 = 0.000)**, performing no better than the base Proximal. The failure mode is the same: on high-dimensional inputs, Phase 1 with Group Lasso drives all rows to small norms — the group penalty converges to the zero solution, and Phase 2 cannot recover.

**Post-hoc analysis:** Group Lasso is theoretically sound for structured sparsity but requires careful tuning of `lambda_group`. With lambda_group = 0.01 (our default), the penalty is strong enough to drive all groups near zero but not selective enough to retain any. A smaller lambda_group (e.g., 0.001) might preserve some rows while zeroing others. However, given that ProximalTopK achieves this selectivity more reliably through hard pruning, Group Lasso is not the recommended approach.

---

## 6. Variant: `ProximalL0` (Hard Concrete L0 Regularization)

### Description

ProximalL0 extends the base Proximal with a learned binary connectivity mask based on the **Hard Concrete** distribution (Louizos et al., ICLR 2018). Each weight wᵢⱼ is multiplied by a stochastic gate zᵢⱼ during Phase 1:

```
effective_weight = w_ij · z_ij
```

where z_ij is sampled from the Hard Concrete distribution — a continuous relaxation of Bernoulli that can reach exactly 0 and 1 (unlike the standard Concrete distribution which is bounded away from the extremes):

```
u ~ Uniform(0, 1)
s = sigmoid((log u − log(1−u) + log_alpha_ij) / beta)
z = clamp(s · (zeta − gamma) + gamma, 0, 1)
```

Parameters: beta = 2/3 (temperature), gamma = −0.1 (left stretch), zeta = 1.1 (right stretch).

The L0 penalty is the **expected number of open gates** (differentiable w.r.t. log_alpha_ij):

```
L0_penalty = Σᵢⱼ sigmoid(log_alpha_ij + C)
```

where C = −beta · log(−gamma/zeta) is a correction term.

### Motivation

Unlike Top-K (which selects by magnitude) and Group Lasso (which selects by group norm), L0 selects by **gradient information**: the gate log_alpha_ij learns whether the connection between input j and neuron i is useful for the MSE objective. This is theoretically superior because magnitude is a noisy proxy for importance (a small weight might be important if the corresponding input has low signal-to-noise ratio).

### Training Protocol

Phase 1: MSE + λ_l0 · L0_penalty, stochastic gates (train mode). Gates are sampled fresh at each forward pass — this provides exploration over the connectivity structure.

Transition: gates are binarized (log_alpha > 0 → z = 1, else z = 0). Gated-out weights are zeroed permanently.

Phase 2: standard ternary regularization on surviving weights (hard gates fixed, no more sampling).

Pre-crystallization: parametrizations are removed (`parametrize.remove_parametrizations`), restoring the normal weight tensor.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| lambda_l0 | 1e-4 | L0 penalty coefficient |

### Empirical Outcome

**ProximalL0 collapses on all datasets (F1 = 0.000).** The failure is similar to GroupLasso but with a different mechanism: with lambda_l0 = 1e-4, the L0 penalty drives log_alpha toward large negative values (gates tend to close) for all connections. After Phase 1, most connections are gated out, leaving a sparse network with insufficient capacity. The surviving weights from Phase 2 ternary regularization cannot recover the required function.

The Hard Concrete L0 is sensitive to the lambda_l0 scale: too small and all gates open (no sparsification); too large and all gates close (collapse). The default lambda_l0 = 1e-4 was set without per-dataset tuning, which likely contributes to the failures.

**Theoretical note:** the L0 approach is fundamentally sound (it is theoretically the correct formulation of discrete weight selection as a differentiable optimization problem) but requires careful calibration and is empirically fragile compared to ProximalTopK's hard magnitude-based pruning.

---

## 7. Analysis: Why ProximalTopK Succeeds Where Others Fail

The four variants attempt different solutions to the same underlying problem: distributing weights across many inputs prevents any single weight from being large enough to survive ternary crystallization.

| Variant | Solution | Phase 1 exit | Phase 2 input |
|---|---|---|---|
| Proximal | Phase 2 regularization alone | distributed small weights | kills them all |
| GroupLasso | Group-level shrinkage in Phase 1 | groups tend to small norm | kills them all |
| ProximalL0 | Gate learning in Phase 1 | most gates close | surviving weights still small |
| **ProximalTopK** | **Hard hard-cut by magnitude after Phase 1** | **distributed small weights** | **only k survive, become large** |

The key difference for TopK: after the hard cut, the surviving k weights per neuron are the k most informative connections. Phase 2's ternary regularization then operates on a sparse, high-magnitude weight matrix where the L1 term has less "mass" to shrink and the ternary attraction P(w) = w²(1−w²) has larger gradient (because |w| is larger). The Phase 2 success for TopK is a consequence of the Phase 1→Phase 2 transition leaving a favourable initialisation.

**Soft vs hard pruning:** the fundamental distinction is that TopK's hard masking is **irreversible** — pruned connections cannot be recovered. This is what prevents the distributed-weight trap: the network cannot "spread" weights back to pruned connections during Phase 2. GroupLasso and L0 allow the full weight matrix to be revisited in Phase 2 (GroupLasso zeroes rows but they can un-zero; L0 gates are fixed but surviving weights can redistribute).

---

## 8. Collapse Analysis: MONK-1

All Proximal variants collapse on MONK-1 despite it being the simplest dataset. The likely cause is iteration budget (max_iter = 500) rather than a structural limitation:

- Phase 1 (300 iters on MONK): MSE converges very quickly (124 training samples, 17 features) — by iter ~50, MSE < 0.001. The remaining ~250 Phase 1 iters are spent in stagnation.
- Phase 2 (200 iters): begins from a near-zero MSE solution. The ternary regularization's warm-up reaches full strength at iter 200, but 200 iters is too short for Adam to navigate from the continuous solution to a ternary crystallized solution while maintaining low MSE.

**Evidence for this hypothesis:** STE variants do NOT collapse on MONK-1 (F1 = 0.639–0.716), and STE runs for max_iter = 2000 (4× more iterations). The Proximal variants on Mushroom use max_iter = 300 and do not universally collapse (TopK reaches F1 = 0.338), suggesting that the MONK collapse is budget-specific.

**Recommendation:** rerun MONK with max_iter = 2000 to test the budget hypothesis.

---

## 9. Summary and Recommendations

| Variant | MONK | Mushroom | Spambase | Recommended? |
|---|---|---|---|---|
| Proximal | collapse | near-collapse | collapse | ✗ (except as reference) |
| **ProximalTopK** | collapse (budget) | F1=0.338 ✓ | F1=0.419 ✓ | **✓ Default choice** |
| ProximalGroupLasso | collapse | collapse | collapse | ✗ |
| ProximalL0 | collapse | collapse | collapse | ✗ (requires tuning) |

**General rules:**
1. Use ProximalTopK for any dataset with n_features > 20
2. Use max_iter ≥ 2000 for MONK-class datasets
3. ProximalTopK with k_per_neuron tuning (k = round(n_features / 10)) as a starting point
4. The ProximalL0 implementation is correct but requires careful lambda_l0 calibration; it is not recommended for production use without hyperparameter search

---

## 10. References

- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso*. Journal of the Royal Statistical Society: Series B, 58(1), 267–288.
- Yuan, M., & Lin, Y. (2006). *Model selection and estimation in regression with grouped variables*. Journal of the Royal Statistical Society: Series B, 68(1), 49–67.
- Louizos, C., Welling, M., & Kingma, D. P. (2018). *Learning Sparse Neural Networks through L₀ Regularization*. ICLR 2018.
- Han, S., Pool, J., Tran, J., & Dally, W. J. (2016). *Learning both Weights and Connections for Efficient Neural Networks*. NeurIPS 2015.
- Parikh, N., & Boyd, S. (2014). *Proximal Algorithms*. Foundations and Trends in Optimization, 1(3), 127–239.
