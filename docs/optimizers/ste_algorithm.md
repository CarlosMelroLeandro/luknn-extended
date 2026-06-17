# Straight-Through Estimator Optimizer — Algorithm and Variants

**Author:** Carlos Leandro  
**Context:** One of three optimizer families for ŁNN training. Best empirical performance across datasets. Prerequisites: `docs/theory/castro_networks.md`.

---

## 1. Theoretical Background

### 1.1 The Quantization Gradient Problem

Training a neural network with quantized (discrete) weights requires computing gradients through the quantization function. For ternary quantization to {−1, 0, +1}:

```
hard_snap(w) = round(w).clamp(-1, 1)
```

The derivative of `hard_snap` is zero almost everywhere (the function is piecewise constant) and undefined at the snap thresholds. Back-propagation through zero gradients produces no weight updates, making standard SGD inapplicable.

### 1.2 The Straight-Through Estimator (STE)

The Straight-Through Estimator (Hinton 2012; Bengio et al. 2013) addresses this by using a **biased gradient estimator** that ignores the quantization in the backward pass:

```
Forward:   ŷ = f(hard_snap(w))          ← uses discrete weights
Backward:  ∂L/∂w ≈ ∂L/∂hard_snap(w)    ← gradient passes through as if identity
```

The mathematical identity exploited is:

```
hard_snap(w) = (hard_snap(w) − w).detach() + w
```

The `.detach()` call stops gradient flow through the first term. During forward, the first term is non-zero (the quantization gap). During backward, only the second term (`w`) carries gradient — which has derivative 1. The result is:

```
∂ hard_snap(w) / ∂w ≈ 1    (STE approximation)
```

This is the "straight-through" property: the gradient passes through the quantization as if it were the identity. The estimator is **biased** (the true gradient is 0) but **useful** because it provides information about the loss landscape of the continuous proxy w.

### 1.3 STE in the ŁNN Context

For ŁNNs, the STE operates on **latent continuous weights** w_cont ∈ ℝ maintained by the optimizer. The ternary snap-threshold is:

```
hard_snap(w) = {  +1  if w >  1/3
              {   0   if |w| ≤ 1/3
              {  -1   if w < -1/3
```

(The threshold 1/3 is the midpoint of the [0, 1] and [−1, 0] intervals in the ternary lattice.)

During each forward pass, `w_ternary = hard_snap(w_cont)` is computed; the STE identity is applied so gradients flow back to w_cont. Adam updates w_cont continuously; the ternary values seen by the network at each step are the rounded projection of w_cont.

**Crystallization is free:** since the STE always runs the forward pass with ternary weights, the final network is already ternary — no post-hoc crystallization step is needed for weights. Biases are handled separately via rounding.

### 1.4 Adam Optimizer with Cosine LR Annealing

All STE variants use **Adam** as the continuous optimizer:

```
m_k = β₁ m_{k-1} + (1−β₁) g_k              ← first moment
v_k = β₂ v_{k-1} + (1−β₂) g_k²            ← second moment
ŵ = m_k / (1 − β₁^k)                        ← bias correction
v̂ = v_k / (1 − β₂^k)
w_{k+1} = w_k − lr_k · ŵ / (√v̂ + ε)
```

with **cosine learning rate schedule**:

```
lr_k = lr_min + 0.5 (lr_max − lr_min)(1 + cos(π · k / max_iter))
```

(Here lr_min = 0, so lr_k = 0.5 · lr · (1 + cos(π · k / T)).)

The cosine schedule provides warm behaviour early in training (high LR → exploration) and fine convergence late (low LR → exploitation). For the STE, this also means ternary boundary decisions are revisited frequently early and locked in progressively as LR decays.

---

## 2. Ternary Regularization

### 2.1 The Ternary Penalty P(w)

The ternary regularization term is:

```
P(w) = w²(1 − w²)
```

Properties:
- **Zeros at {−1, 0, +1}:** P(−1) = P(0) = P(+1) = 0 (penalty vanishes at ternary values)
- **Maximum at |w| = 1/√2 ≈ 0.71:** P(1/√2) = 0.25 (maximum ambiguity)
- **Gradient:** dP/dw = 2w − 4w³ = 2w(1 − 2w²)
  - Zero at {0, ±1/√2, ±1}
  - Positive for 0 < w < 1/√2 (pushes w toward either 0 or +1)
  - Negative for w > 1/√2 (pushes w toward +1)

The gradient has the desired "trichotomic" behaviour: for w near 0, the gradient pulls w toward 0; for w near ±1, the gradient pulls w toward ±1; for w in the "ambiguous zone" (|w| ≈ 0.71), the gradient is maximal.

### 2.2 Regularization Term in the Loss

```
L_reg(w) = MSE(ŷ, y) + λ_attract · Σᵢ P(wᵢ)
          = MSE(ŷ, y) + λ_attract · Σᵢ wᵢ²(1 − wᵢ²)
```

The coefficient λ_attract is applied with a **linear warm-up** from 0 to λ_attract over training:

```
λ_k = λ_attract · (k / max_iter)
```

This prevents the cold-start problem: at initialization, all weights are near 0, and P(0) = 0 — but the gradient dP/dw|_{w=0} = 0 as well, so the regularizer provides no force. As training progresses and weights move away from 0, the ternary penalty gradient grows. The warm-up ensures the penalty does not compete with the MSE signal before the network has learned a reasonable solution.

### 2.3 Boundary Fraction Diagnostic (bf_pre)

The **boundary fraction** (bf_pre) is defined as the fraction of latent weights within ±0.15 of the snap threshold (|w| ∈ [0.18, 0.48] or equivalently, within 0.15 of 1/3):

```
bf_pre = fraction of {|w| : |w − 1/3| < 0.15}
```

A high bf_pre (> 0.35) indicates many weights are "on the fence" between 0 and ±1 — the STE will snap them to either value with low confidence, making crystallization unstable. The dual stopping criterion uses this as a trigger: training should continue until bf_pre < tol_boundary, ensuring ternary decisions are made with high confidence.

A **low bf_pre with poor F1** (as observed for STE_reg/STE_dual on Mushroom and Spambase) indicates **premature boundary forcing**: the regularizer has pushed weights to the boundary too early, before MSE has converged, resulting in a suboptimal frozen solution.

---

## 3. Mode='ste' and Weight Clamping

All STE variants require the model to be created with `mode='ste'`:

```python
model = make_lukasiewicz_net(n_features, n_hidden_layers=2,
                             hidden_width=n_features, mode='ste')
```

In `mode='ste'`, the `LukasiewiczLinear` layer performs the STE snap in its forward pass. The optimizer maintains continuous latent weights; the layer applies `hard_snap` before the truncated-identity activation.

- `STE_base` clamps latent weights to [−1.5, 1.5] — the wider range prevents gradient starvation at exactly ±1
- `STE_reg`, `STE_dual`, `STE_hybrid` clamp to [−1, 1] — necessary with regularization to keep w²(1−w²) well-defined

---

## 4. Benchmark Results (30 trials per dataset/variant)

| Dataset | Variant | F1 [95% CI] | bf_pre | Notes |
|---|---|---|---|---|
| MONK-1 (17f) | STE_base | 0.639 [0.527, 0.752] | 0.41 | — |
| MONK-1 (17f) | STE_reg | **0.716 [0.654, 0.778]** | 0.38 | best; reg helps in low-dim |
| MONK-1 (17f) | STE_dual | 0.708 [0.647, 0.769] | 0.36 | similar to reg, faster |
| MONK-1 (17f) | STE_hybrid | 0.673 [0.579, 0.767] | 0.39 | MSE-first then reg |
| Mushroom (111f) | STE_base | **0.324 [0.171, 0.478]** | 0.44 | best; reg hurts in high-dim |
| Mushroom (111f) | STE_reg | 0.055 [0.012, 0.097] | 0.328 | premature forcing (low bf_pre) |
| Mushroom (111f) | STE_dual | 0.055 [0.012, 0.097] | 0.328 | same collapse as STE_reg |
| Mushroom (111f) | STE_hybrid | 0.218 [0.121, 0.314] | 0.36 | partial mitigation |
| Spambase (57f) | STE_base | **0.685 [0.626, 0.744]** | 0.45 | only CI_lower > 0.5 |
| Spambase (57f) | STE_reg | 0.318 [0.231, 0.404] | 0.264 | strongest collapse (bf_pre=0.264) |
| Spambase (57f) | STE_dual | 0.318 [0.231, 0.404] | 0.264 | identical to STE_reg |
| Spambase (57f) | STE_hybrid | 0.449 [0.341, 0.557] | 0.33 | partial recovery |
| Musk (166f) | all | ≈ 0.000 | — | architecture insufficient |

---

## 5. Variant: `STE_base` (Baseline)

### Description

The original STE: Adam with cosine LR, MSE-only loss, latent weights clamped to [−1.5, 1.5]. Best-state checkpoint is saved and restored before crystallization (to avoid using a final-step model that may have regressed from the best training MSE).

### Hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| lr | 5e-3 | Adam learning rate |
| clip_grad | 1.0 | Gradient norm clip (0 = disabled) |

### Justification

The wider clamp [−1.5, 1.5] allows latent weights to "overshoot" slightly beyond ±1 before snapping, which can accelerate convergence compared to a hard [−1, 1] constraint. The absence of regularization means the only pressure toward ternary values comes from the STE mechanism itself — the network learns ternary weights by the discrete nature of the forward pass, not by explicit penalty.

### When to use

- Default choice for all datasets
- Only option that remains effective for n_features > 40 (where regularization causes premature boundary forcing)
- CI_lower > 0.5 on Spambase (57f) and MONK-1 — the only operationally reliable STE configuration on those datasets

---

## 6. Variant: `STE_reg` (Ternary Regularization)

### Description

STE_base plus the ternary regularization term `λ_attract · w²(1−w²)` with linear λ warm-up (0 → λ_attract over training). Clamp reduced to [−1, 1].

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| lambda_attract | 0.05 | Peak ternary regularization coefficient |

### Justification

The ternary penalty provides explicit gradient information about the distance from the integer lattice, which the pure STE does not (its "crystallization" is purely mechanical through the snap function). In low-dimensional settings (≤ 17 features), this helps resolve ambiguous weights and reduces variance across trials.

### Empirical Outcome

**Helpful on MONK-1** (F1 = 0.716 vs 0.639 for base, improvement of 0.077). **Detrimental on Mushroom and Spambase** — the regularizer forces weights to the boundary (bf_pre drops to 0.328/0.264) before MSE converges. The cause is the **distributed-small-weight trap**: with 111/57 inputs, the MSE gradient is spread across many weights; each individual weight moves slowly, while the regularizer's warm-up reaches full strength before any weight is large enough to be informative. The warm-up fixes the *rate* but not the *magnitude* problem.

---

## 7. Variant: `STE_dual` (Dual Stopping Criterion)

### Description

STE_reg augmented with a dual stopping criterion: training stops when **both**:
1. MSE < mse_gate (a relaxed threshold, default 0.05 — much looser than tol_mse=2e-3 because STE's ternary MSE is typically higher than a continuous network's MSE)
2. boundary_frac < tol_boundary (fewer than 35% of weights on the snap fence)

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| mse_gate | 0.05 | MSE threshold for dual criterion (relaxed) |
| tol_boundary | 0.35 | Max allowed boundary fraction at stopping |

### The mse_gate Rationale

The standard tol_mse = 2e-3 is calibrated for continuous-weight networks where near-zero MSE is achievable. STE networks with ternary weights rarely reach MSE < 2e-3 on real datasets; the ternary snap introduces irreducible quantization noise. The mse_gate = 0.05 is a realistic target for STE-ternary networks on moderately difficult datasets.

### Empirical Outcome

On MONK-1: F1 = 0.708, very similar to STE_reg (0.716). The dual criterion successfully reduces the number of iterations (faster convergence per trial) but does not improve accuracy. On Mushroom/Spambase, the collapse observed in STE_reg propagates identically: the boundary forcing happens before mse_gate is reached, so the dual criterion provides no protection.

---

## 8. Variant: `STE_hybrid` (Two-Phase)

### Description

Two-phase approach to avoid cold-start collapse in the regularized variants:

**Phase 1** (~40% of budget, p1_fraction=0.4): pure MSE + cosine LR. Finds a good ternary solution before any regularization is applied.

**Phase 2** (remaining 60%): ternary regularization warm-up (0 → λ_attract) + dual stopping criterion. Uses lr/2 to avoid undoing Phase 1.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| p1_fraction | 0.4 | Fraction of max_iter allocated to Phase 1 |
| lambda_attract | 0.05 | Peak ternary reg coefficient in Phase 2 |
| mse_gate | 0.05 | MSE gate for dual criterion (Phase 2) |
| tol_boundary | 0.35 | Boundary fraction threshold (Phase 2) |

### Motivation

The Phase 1 — Phase 2 separation mirrors the LM_hybrid design. The key insight: cold-start regularization (STE_reg) fails because the regularizer has full strength before any MSE gradient has shaped the latent weights. By running Phase 1 first, the latent weights are already at a good continuous ternary solution; Phase 2's regularization then fine-tunes rather than disrupts.

### Empirical Outcome

On Mushroom: F1 = 0.218 [0.121, 0.314] — a partial recovery compared to STE_reg (0.055), but still significantly below STE_base (0.324). The two-phase separation helps but does not fully resolve the premature forcing: in 111-dimensional space, Phase 1 with 40% of the budget (800 iters at max_iter=2000) is too short to fully converge.

On MONK-1: F1 = 0.673 [0.579, 0.767] — slightly below STE_reg and STE_dual. The Phase 1 MSE-only training appears to over-commit to a solution that Phase 2 cannot significantly improve.

---

## 9. The Regularization Reversal Phenomenon

The empirical results reveal a striking interaction between feature dimensionality and regularization effectiveness:

| Dataset | n_features | STE_base F1 | STE_reg F1 | Direction |
|---|---|---|---|---|
| MONK-1 | 17 | 0.639 | 0.716 | Reg **helps** (+0.077) |
| Spambase | 57 | 0.685 | 0.318 | Reg **hurts** (−0.367) |
| Mushroom | 111 | 0.324 | 0.055 | Reg **hurts** (−0.269) |

**Mechanistic explanation:**

In low dimensions (MONK-1, 17 features), each weight carries relatively high individual gradient signal from the MSE. The ternary penalty's force is small relative to the MSE gradient, so the warm-up allows both to coexist. The penalty helps stabilize weights near ternary values without destabilizing MSE convergence.

In high dimensions (57+ features), the MSE gradient is distributed across many weights. Each individual weight receives a small MSE signal. The ternary penalty's warm-up reaches full strength while individual MSE gradients are still small, so the penalty dominates and forces weights to the boundary prematurely. Once a weight snaps, the gradient information for that weight is lost in the STE mechanism — ternary decisions made under poor MSE information are frozen.

**Informal theorem:** *For a fixed λ_attract and warm-up schedule, there exists a critical dimensionality n* above which ternary regularization is harmful. n* depends on the ratio of per-weight MSE gradient magnitude to the regularization force at the inflection point of P(w).*

The bf_pre diagnostic (`boundary_frac_pre`, the boundary fraction before crystallization) is a reliable indicator: bf_pre below ~0.30 reliably predicts poor F1 in high-dimensional settings (premature forcing), while bf_pre above ~0.35 is compatible with good F1.

---

## 10. Summary and Recommendations

| Variant | Best use case | Avoid when |
|---|---|---|
| STE_base | n_features > 40; default choice | never — always competitive |
| STE_reg | n_features ≤ 20; rule-based datasets | n_features > 40 |
| STE_dual | n_features ≤ 20; efficiency matters | n_features > 40 |
| STE_hybrid | 20 < n_features ≤ 40; experimental | n_features > 40 (insufficient budget for Phase 1) |

**The STE_base is the single most robust configuration:** it achieves CI_lower > 0.5 on both MONK-1 and Spambase, is statistically indistinguishable from STE_reg on MONK-1 (after Holm correction), and does not degrade on high-dimensional inputs.

---

## 11. References

- Hinton, G. (2012). *Neural Networks for Machine Learning*. Lecture notes, University of Toronto (Lecture 9c: Using noise as a regularizer).
- Bengio, Y., Léonard, N., & Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*. arXiv:1308.3432.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. ICLR 2015.
- Loshchilov, I., & Hutter, F. (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983.
