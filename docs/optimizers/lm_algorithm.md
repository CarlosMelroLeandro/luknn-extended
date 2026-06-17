# Levenberg-Marquardt Optimizer — Algorithm and Variants

**Author:** Carlos Leandro  
**Context:** One of three optimizer families for ŁNN training. Closest to the original ALT 2009 paper. Prerequisites: `docs/theory/castro_networks.md`.

---

## 1. Theoretical Background

### 1.1 The Gauss-Newton Method

For a nonlinear least-squares problem min_w ‖F(w)‖², Gauss-Newton iterates:

```
w_{k+1} = w_k − (JᵀJ)⁻¹ Jᵀ r
```

where **J** is the Jacobian matrix ∂F/∂w (shape: n_outputs × n_params) and **r = F(w_k)** is the residual vector (network output minus target). The Gauss-Newton Hessian approximation **JᵀJ** ignores second-derivative terms, which is valid when residuals are small near the optimum. This gives *quadratic* local convergence near a zero of F.

For ŁNNs, F(w) is the vector of per-sample errors: F(w)_i = ŷ_i(w) − y_i, so n_outputs = n_samples. The MSE loss is ‖F(w)‖² / n_samples.

### 1.2 Levenberg-Marquardt Augmentation

When **JᵀJ** is singular or near-singular (common early in training), the Gauss-Newton step is ill-conditioned. The Levenberg-Marquardt (LM) algorithm adds a regularisation term:

```
Δw = −(JᵀJ + μI)⁻¹ Jᵀ r
```

The damping factor μ > 0 interpolates between:
- **μ → 0:** Gauss-Newton (fast, quadratic convergence)
- **μ → ∞:** Gradient descent with step size 1/μ (slow, but always downhill)

The **trust-region interpretation:** the LM step is the solution to minimising the quadratic model of the loss within a trust region ‖Δw‖ ≤ δ(μ). As μ increases, the step shrinks; as μ decreases, the step approaches Gauss-Newton.

**Damping adaptation (Marquardt rule):**
- If the step decreases loss: μ ← μ / factor_down (grow trust region)
- If the step increases loss: μ ← μ × factor_up (shrink trust region, reject step)

### 1.3 Adaptation to ŁNNs — Key Design Decisions

#### Jacobian via forward-mode AD

The standard approach computes J via finite differences (n_params forward passes) or backpropagation (n_outputs backward passes). For ŁNNs:

- n_params ≈ n_hidden_layers × hidden_width² (small for the architectures we use)
- n_outputs = n_samples (large for real datasets)

Therefore **forward-mode AD** (`torch.func.jacfwd`) is efficient: it computes J using n_params forward JVP passes, each producing one column of Jᵀ. Cost: O(n_params × n_samples) per iteration.

For large datasets (Mushroom, Spambase), mini-batch Jacobian is used: a batch of size B replaces the full dataset, giving approximate JᵀJ with O(n_params × B) cost.

#### Levenberg vs Marquardt damping

The paper's text describes "Marquardt scaling" μ·diag(JᵀJ), but this produces near-zero damping when weights are small (early in training), causing numerically explosive steps (‖Δw‖ ≈ 47 observed). The replication uses **standard Levenberg damping μI**, which is better conditioned and equivalent to Marquardt scaling when ‖w‖ ≈ 1 (after partial crystallization).

#### Smooth crystallization within the LM step

The paper's modified LM applies Υ_n after each **accepted** step:

```
w_{k+1} = Υ_n(w_k + Δw)   (if loss decreases)
w_{k+1} = w_k              (if loss increases — step rejected)
```

This is different from applying Υ_n as a regularizer: the crystallization function is applied post-hoc, not as part of the gradient computation. It gently attracts weights toward integers without changing the loss landscape.

### 1.4 Mini-Batch Jacobian

For datasets where n_samples × n_params is too large to store in memory as a dense matrix, the mini-batch Jacobian approximates:

```
JᵀJ ≈ (1/B) Σᵢ jᵢ jᵢᵀ
```

where jᵢ is the Jacobian of a single sample. With batch_size B, this is unbiased if the batch is randomly sampled. The approximation introduces gradient noise but allows LM to scale to tens of thousands of samples.

---

## 2. Training Protocol

```
Initialize: μ = μ_init; best_mse = ∞; stagnation_counter = 0

For k = 1, …, max_iter:
    1. Draw batch B (or use full dataset if batch_size = 0)
    2. Compute J = jacfwd(model, B) using forward-mode AD
    3. Compute Δw = −(JᵀJ + μI)⁻¹ Jᵀ r  via Cholesky solve
    4. Tentatively set w' = w + Δw
    5. Compute mse' = MSE(model(w'), x_full, y_full)
    6. If mse' < mse:
         μ ← μ / factor_down
         w ← Υ_n(w')          # accepted step + smooth crystallization
         mse ← mse'
    7. Else:
         μ ← μ × factor_up   # rejected step
    8. If mse < tol_mse: stop (converged)
    9. Update stagnation counter; if counter ≥ patience: stop (stagnation)

Post-training:
    if converged:
        progressive_crystallize(w)   # Υ₂ → Υ₄ → Υ₈ → Υ₁₆
        crisp_crystallize(w)          # round to {-1, 0, +1}
        if Δ(N) < 0.01 and MSE_crisp < tol_mse:
            obs_prune(model)          # remove redundant weights
```

---

## 3. Benchmark Results (30 trials per dataset, 5×2 CV for Spambase/Musk)

| Dataset | Variant | F1 [95% CI] | Conv% | Notes |
|---|---|---|---|---|
| MONK-1 | LM_base | 0.464 [0.343, 0.585] | 7% | — |
| MONK-1 | LM_delayed | 0.156 [0.048, 0.263] | 33% | worst; 33% degenerate trials |
| MONK-1 | LM_progressive | 0.442 [0.317, 0.566] | 13% | similar to base |
| MONK-1 | LM_dual | 0.464 [0.343, 0.585] | 0% | same as base; dual stop prevents premature exit |
| MONK-1 | LM_hybrid | 0.482 [0.382, 0.581] | 3% | best; LM → Adam handoff helps |
| Mushroom | LM_base | 0.277 [0.149, 0.405] | 0% | hw=8 caps capacity |
| Mushroom | LM_delayed | 0.140 [0.029, 0.251] | 27% | delayed crys. causes 27% degenerate |
| Mushroom | LM_progressive | 0.258 [0.135, 0.381] | 0% | — |
| Mushroom | LM_dual | 0.277 [0.149, 0.405] | 0% | — |
| Mushroom | LM_hybrid | 0.235 [0.110, 0.360] | 0% | worse here (capacity limited) |
| Spambase | LM_hybrid | 0.455 [0.239, 0.671] | 0% | best; n=10 (wide CI) |
| Musk | all | ≈ 0.000 | 0% | hw=8 insufficient for 166f |

**Interpretation:** LM is competitive only on MONK (hw=17 = n_features, full capacity). On Mushroom and Spambase, the hidden width cap (hw=8/12) imposed by Jacobian cost eliminates representational capacity. The method is structurally unsuitable for high-dimensional datasets without feature pre-selection.

---

## 4. Variant: `LM_base` (Baseline)

### Description

Direct implementation of the paper's modified Levenberg-Marquardt with:
- Standard Levenberg damping μI (not Marquardt diag(JᵀJ))
- Smooth crystallization Υ₂ at each accepted step
- Crisp crystallization + OBS if converged

### Hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| mu_init | 1e-2 | Initial damping factor |
| factor_up | 10.0 | μ multiplier on rejected step |
| factor_down | 10.0 | μ divisor on accepted step |
| crystallize_n | 2 | n in Υ_n (paper default) |
| patience | 50 | Stagnation patience (iters with no improvement) |
| prune | True | Apply OBS after crystallization |
| batch_size | 0 | 0 = full dataset; >0 = mini-batch |

### Justification

This is the closest variant to the paper's algorithm. The Υ₂ function provides gentle, continuous attraction toward integer weights throughout training. The factor_up/down = 10 follows the Marquardt recommendation for good convergence-stagnation balance.

**Known limitation:** Υ₂ is inert near w = 0.5 (the fractional midpoint), where `cos(0.5 · π/2)² = cos(π/4)² = 0.5` — the crystallization function maps 0.5 to 0.5, providing no attraction. Weights stuck near ±0.5 never crystallize under this scheme alone.

---

## 5. Variant: `LM_delayed` (Delayed Crystallization)

### Description

Identical to `LM_base` but Υ_n is not applied until `crystallize_start_fraction × max_iter` iterations have passed. During the delay phase, the LM step is applied without crystallization, allowing weights to find a continuous solution first.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| crystallize_start_fraction | 0.3 | Fraction of budget before Υ_n activates |

### Motivation

The hypothesis is that early crystallization may push weights into poor local minima before the quadratic convergence regime is reached. By delaying crystallization, the weights are free to explore the continuous loss landscape for the first 30% of training.

### Empirical Outcome

**LM_delayed is the worst LM variant on MONK-1** (F1 = 0.156 [0.048, 0.263]) and on Mushroom (F1 = 0.140 [0.029, 0.251]). The convergence flag is paradoxically *higher* (33% on MONK, 27% on Mushroom) because delayed crystallization allows MSE to drop below tol_mse in the continuous phase — but then the subsequent crisp crystallization step destroys the solution. The pattern is: converge in continuous space, crystallize destructively, report "converged" despite wrong output.

**Conclusion:** delaying crystallization is detrimental. The paper's in-step crystallization (Υ_n at every accepted step) is preferable because it progressively shapes the loss landscape around integer attractors during training, not after.

---

## 6. Variant: `LM_progressive` (Progressive Υ Schedule)

### Description

LM with a multi-stage Υ_n schedule: the crystallization strength n increases at pre-defined iteration thresholds, providing a warm-up analogous to the ternary regularization λ warm-up used in the Proximal/STE variants.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| n_schedule | (2, 4, 8, 16) | Sequence of n values for Υ_n |
| schedule_fractions | (0.0, 0.5, 0.75, 0.9) | Iteration thresholds for schedule transitions |

### Schedule Behaviour

```
Iterations 0%–50%:  Υ₂  — gentle, high weight-mobility
Iterations 50%–75%: Υ₄  — moderate pressure
Iterations 75%–90%: Υ₈  — strong pressure
Iterations 90%–100%: Υ₁₆ — near-binary forcing
```

### Motivation

`LM_base` uses a fixed Υ₂ throughout, which is too gentle in the final stages to resolve the ±0.5 stagnation problem. Progressive strengthening mirrors the annealing heuristic: start with a smooth landscape (Υ₂) to allow good initial convergence, then sharpen (Υ₁₆) to force remaining ambiguous weights to commit.

### Empirical Outcome

On MONK-1, LM_progressive (F1 = 0.442) performs similarly to LM_base (F1 = 0.464) — no significant difference. The progressive schedule does not solve the fundamental capacity limitation. On Mushroom, also comparable to base (F1 = 0.258 vs 0.277). The progressive schedule adds complexity without measurable benefit in these experiments.

---

## 7. Variant: `LM_dual` (Dual Stopping Criterion)

### Description

LM_base augmented with a dual stopping criterion: training stops only when **both** conditions hold:
- MSE < tol_mse (accuracy goal)
- Δ(N)/n_params < tol_dn (crystallization proximity goal)

This prevents early exit when the MSE target is reached but weights are still far from {−1, 0, +1}, which would leave crisp crystallization to make a large, potentially destructive rounding step.

### Hyperparameters (additional)

| Parameter | Default | Meaning |
|---|---|---|
| tol_dn | 0.05 | Max allowed normalised Δ(N) at stopping |
| dn_patience | 50 | Max iters waiting for Δ(N) improvement after MSE is met |

### The `stuck_fraction` Metric

The `_delta_n_norm` function computes the fraction of weights in the "stuck" zone 0.1 < |w| < 0.9 — weights that are neither pruned (near zero) nor active (near ±1). A weight at |w| = 0.5 contributes to this fraction. The dual criterion requires this fraction to drop below `tol_dn = 0.05` (at most 5% of weights stuck).

### Empirical Outcome

On MONK-1, LM_dual (F1 = 0.464) is identical to LM_base. The dual criterion prevents 0 premature exits (convergence drops from 7% to 0%), but the improvement in F1 is negligible. The stopping criterion is sound but the fundamental problem — insufficient crystallization depth from Υ₂ — is not addressed by stopping later.

---

## 8. Variant: `LM_hybrid` (Three-Phase: LM → Ternary Reg → Hardening)

### Description

The most sophisticated LM variant. Three sequential phases:

**Phase 1 — LM with Υ_n (p1_fraction of budget):**  
Standard LM with smooth crystallization. Stops at MSE convergence, stagnation, or budget exhaustion.

**Phase 2 — Adam + ternary regularization (remaining budget):**  
Switches to Adam optimizer with:
- Loss = MSE + λ_sparse·‖w‖₁ + λ_attract·w²(1−w²) (linear λ warm-up 0→λ)
- Weights projected to [−1, 1] after each step
- Dual stopping: MSE < tol_mse AND stuck_fraction < tol_dn

**Phase 3 — Hardening (p3_steps fixed budget):**  
Short burst with λ_sparse × 10 and λ_attract × 10. Skipped if Phase 2 dual-stopped or MSE ≥ 0.15.

### Hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| p1_fraction | 0.4 | Budget fraction for LM phase |
| p1_patience | 30 | LM stagnation patience |
| lr_p2 | 1e-2 | Adam LR for Phases 2 and 3 |
| lambda_sparse | 1e-3 | L1 coefficient (×10 in Phase 3) |
| lambda_attract | 0.1 | Ternary attraction coefficient (×10 in Phase 3) |
| prox_threshold | 5e-4 | Soft-threshold after each Phase-2 step |
| tol_dn | 0.05 | Stuck-fraction threshold for dual stopping |
| p3_steps | 200 | Fixed hardening steps |

### Motivation

LM alone fails to crystallize because Υ₂ provides no force at w = 0.5. The ternary penalty P(w) = w²(1−w²) has maximum gradient exactly at |w| = 1/√2 ≈ 0.71, pulling weights decisively toward 0 or ±1. Starting Phase 2 from LM's continuous solution (rather than random initialization) avoids the cold-start collapse that afflicts always-on ternary regularization.

Phase 3 closes the residual gap: Phase 2 often stagnates with a small fraction of weights at |w| ≈ 0.5. The 10× hardening burst provides a short, intense push to clear this residual.

### Empirical Outcome

**LM_hybrid is the best LM variant across all datasets:**

- MONK-1: F1 = 0.482 [0.382, 0.581] — best among all LM variants, tightest CI
- Mushroom: F1 = 0.235 [0.110, 0.360] — slightly worse than LM_base (capacity limited at hw=8; Phase 2 provides little benefit when the network cannot represent the function)
- Spambase: F1 = 0.455 [0.239, 0.671] (n=10, 5×2 CV) — best, but wide CI

The improvement on MONK is modest (0.482 vs 0.464 for LM_base) and not statistically significant at α = 0.05 after Holm correction. The hybrid is the recommended LM variant for new datasets, especially when hidden_width = n_features is feasible.

---

## 9. Summary Comparison

| Variant | MONK F1 | Mushroom F1 | Key mechanism | Recommended? |
|---|---|---|---|---|
| LM_base | 0.464 | 0.277 | Υ₂ at accepted steps | ✓ Reference |
| LM_delayed | 0.156 | 0.140 | Delayed Υ_n | ✗ Never |
| LM_progressive | 0.442 | 0.258 | Υ₂→Υ₁₆ schedule | neutral |
| LM_dual | 0.464 | 0.277 | Dual MSE+Δ(N) stop | neutral |
| **LM_hybrid** | **0.482** | 0.235 | LM + Adam/ternary reg | ✓ Best |

**When to use LM:**
- Only when n_features ≤ ~20 AND n_samples is small enough for full Jacobian or feasible mini-batch
- For symbolic extraction on clean synthetic data (truth tables): LM can achieve exact MSE=0 and full formula recovery
- Never on datasets with n_features > 50 without prior feature selection: Jacobian cost forces hidden width caps that eliminate representational capacity

---

## 10. References

- Levenberg, K. (1944). *A method for the solution of certain non-linear problems in least squares*. Quarterly of Applied Mathematics, 2(2), 164–168.
- Marquardt, D. W. (1963). *An algorithm for least-squares estimation of nonlinear parameters*. SIAM Journal on Applied Mathematics, 11(2), 431–441.
- Leandro, C. (2009). *Symbolic Knowledge Extraction using Łukasiewicz Logics*. ALT 2009. arXiv:1604.03099.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.
