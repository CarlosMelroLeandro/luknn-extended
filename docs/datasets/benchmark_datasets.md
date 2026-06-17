# Benchmark Datasets

**Author:** Carlos Leandro  
**Context:** Description of the four datasets used in the ŁNN optimizer benchmark. All datasets are available from the UCI Machine Learning Repository.

---

## Overview

| Dataset | Source | Samples | Features (raw) | Features (after preprocessing) | Task |
|---|---|---|---|---|---|
| MONK-1 | UCI (artificial) | 432 | 6 categorical | 17 binary | Binary classification |
| UCI Mushroom | UCI (biological) | 8 124 | 22 categorical | 111 binary | Binary classification |
| Spambase | UCI (email corpus) | 4 601 | 57 continuous | 57 normalized | Binary classification |
| Musk v2 | UCI (molecular) | 6 598 | 166 continuous | 166 normalized | Binary classification |

All features are normalized or binarized to [0, 1] to match the ŁNN input domain.

---

## 1. MONK-1

### Origin and Motivation

MONK-1 is one of three artificially generated datasets from the MONK's Problems benchmark (Thrun et al., 1991). It was designed to test the ability of learning algorithms to acquire rule-based classification from nominal attributes. The labeling rule is:

```
MONK-1 label = (a₁ = a₂) OR (a₅ = 1)
```

where a₁, a₂, a₅ are three of the six categorical attributes. This rule is exactly expressible as a first-order Boolean formula — and hence as a Łukasiewicz formula over the binarized features.

**Justification for inclusion:** MONK-1 is the canonical test for neuro-symbolic learning. Its rule has a clean logical structure that a correctly-trained and crystallized ŁNN should recover exactly. Failure to achieve F1 ≈ 1 on MONK-1 indicates either insufficient capacity, poor optimization, or crystallization failure — a useful diagnostic.

### Preprocessing

The six categorical attributes have values {1, 2, 3} (a₁, a₂, a₃, a₄, a₅) or {1, 2} (a₆). One-hot encoding produces 3+3+3+3+3+2 = 17 binary features.

```python
from luknn.benchmark.datasets import load_monk
ds = load_monk(problem=1, seed=42)
# ds.n_features = 17
# len(ds.X_train) = 124  (about half the 432 samples)
# len(ds.X_test)  = 308
```

### Benchmark Setup

| Parameter | Value |
|---|---|
| Architecture | 2 hidden layers, width = 17 |
| max_iter | STE: 2000; LM: 300; Proximal: 500 |
| tol_mse | 2e-3 |
| n_trials | 30 per variant |

### Results Synopsis

STE_reg achieves F1 = 0.716 [0.654, 0.778] — the best result across all methods and datasets in absolute terms. LM_hybrid achieves F1 = 0.482. All Proximal variants collapse at 500 iterations. No method achieves F1 ≈ 1 within the current iteration budgets, suggesting crystallization is incomplete — the network learns the right function but cannot commit all weights to integer values within the budget.

---

## 2. UCI Mushroom

### Origin and Motivation

The UCI Mushroom dataset (Schlimmer, 1987) contains descriptions of hypothetical mushroom samples from the Agaricus and Lepiota families. The task is binary classification: edible (class e) vs poisonous (class p). The dataset has 8 124 samples, no missing values in the version used, and 22 categorical attributes.

**Justification for inclusion:** Mushroom is the dataset used in the original ALT 2009 paper. It represents a high-dimensional categorical input space (111 binary features after one-hot encoding) and a largely deterministic labeling function — the classes are separable by a small number of features (principally `odor`, `spore-print-color`, and `ring-type`). This provides a realistic test of whether ŁNNs can identify the sparse, interpretable rule from a high-dimensional binary input.

### Preprocessing

One-hot encoding of all 22 categorical attributes yields:
- `cap-shape` (6), `cap-surface` (4), `cap-color` (10), `bruises` (2), `odor` (9), `gill-attachment` (2), `gill-spacing` (2), `gill-size` (2), `gill-color` (12), `stalk-shape` (2), `stalk-root` (5), `stalk-surface-above-ring` (4), `stalk-surface-below-ring` (4), `stalk-color-above-ring` (9), `stalk-color-below-ring` (9), `veil-type` (1, dropped as constant), `veil-color` (4), `ring-number` (3), `ring-type` (5), `spore-print-color` (9), `population` (6), `habitat` (7)

Total: 111 binary features after dropping the constant veil-type column.

**Note on the original paper's preprocessing:** the paper generates synthetic "half-valued" negative cases (§5) by adding a second training batch with y = 0.5 labels, to create a 3-valued training set for the Łukasiewicz interpretation. The current replication uses standard binary labels (y ∈ {0, 1}), which is equivalent for the MSE objective.

```python
from luknn.benchmark.datasets import load_mushroom
ds = load_mushroom(seed=42)
# ds.n_features = 111
# len(ds.X_train) = 6499
# len(ds.X_test)  = 1625
```

### Benchmark Setup

| Parameter | Value |
|---|---|
| Architecture | 2 hidden layers, width = n_features = 111 (STE, Proximal); width = 8 (LM, capacity-capped) |
| max_iter | STE: 1000; LM: 150; Proximal: 300 |
| tol_mse | 2e-3 |
| n_trials | 30 per variant |

The LM hidden width is capped at 8 (vs 111 for STE/Proximal) because the Jacobian cost scales as O(n_params × batch_size), and n_params ≈ 111 × 8 × 2 = 1776 parameters makes each LM step ~43 ms even with a mini-batch of 128.

### Results Synopsis

STE_base achieves F1 = 0.324 [0.171, 0.478]. ProximalTopK achieves F1 = 0.338 [0.247, 0.429]. All regularized STE variants and all Proximal variants except TopK collapse. LM variants reach F1 = 0.14–0.28 with hw=8. The dataset is partially solvable: a trained STE_base network identifies some of the relevant mushroom features, but the full formula is not recovered.

---

## 3. Spambase

### Origin and Motivation

The Spambase dataset (Hopkins et al., 1999) consists of 4 601 email messages (2 788 non-spam, 1 813 spam) from a corpus of personal emails. Each message is described by 57 continuous features:

- 48 word-frequency features: percentage of words matching spam-associated terms (`make`, `address`, `all`, `3d`, `our`, `over`, `remove`, etc.)
- 6 character-frequency features: percentage of characters matching `(`, `[`, `!`, `$`, `#`, `)`
- 3 run-length statistics: average, longest, and total run length of consecutive capital letters

**Justification for inclusion:** Spambase provides a challenging test for ŁNN optimizers because:
1. Features are continuous (not binary), requiring careful normalization
2. Decision boundary is non-trivial (57 features with mixed relevance)
3. The dataset is large enough (4 601 samples) that Jacobian-based methods face computational challenges
4. Feature distributions are highly skewed (most word-frequency features are 0 for most emails)

Spambase was not used in the original ALT 2009 paper; it is added here to assess the generalization of the benchmark beyond binary-coded categorical inputs.

### Preprocessing

All 57 features are normalized to [0, 1] by min-max scaling:

```python
from luknn.benchmark.datasets import load_spambase
ds = load_spambase(seed=42)
# ds.n_features = 57
# len(ds.X_train) = 3680  (80% split)
# len(ds.X_test)  = 921
```

No feature engineering is applied; the raw (normalized) word and character frequencies are used directly as ŁNN inputs.

### Benchmark Setup

| Parameter | Value |
|---|---|
| Architecture | 2 hidden layers, width = n_features = 57 (STE, Proximal); width = 12 (LM) |
| max_iter | STE: 1000; LM: 80 (5×2 CV); Proximal: 300 |
| tol_mse | 2e-3 |
| n_trials | STE, Proximal: 30; LM: 5×2 stratified CV (n=10) |

LM uses 5×2 stratified CV for Spambase because 80 iterations per trial is too short for meaningful statistics from 30 independent trials — 5×2 CV produces 10 observations with paired structure, which is more statistically efficient.

### Results Synopsis

STE_base achieves F1 = 0.685 [0.626, 0.744] — the only configuration across all methods with CI_lower > 0.5, making it the only operationally reliable result on Spambase. ProximalTopK achieves F1 = 0.419 [0.331, 0.507] (CI_lower = 0.331, below the 0.5 operational threshold). LM_hybrid achieves F1 = 0.455 [0.239, 0.671] (n=10, wide CI). All regularized STE variants and all Proximal variants except TopK collapse — the regularization reversal phenomenon is most pronounced here (STE_reg bf_pre = 0.264, lowest observed).

---

## 4. Musk v2

### Origin and Motivation

The Musk v2 dataset (Dietterich et al., 1997) contains descriptions of molecules (conformations) that have or have not been identified as musks by expert smellers. The task is multi-instance binary classification: a molecule is labeled as musk if at least one of its conformations is a musk.

This is a **multi-instance learning** problem: each molecule can have multiple conformations (instances), each described by 166 features derived from the molecular shape. The standard single-instance formulation (used here) treats each conformation independently and predicts the molecule-level label for each conformation.

| Statistic | Value |
|---|---|
| Total samples (conformations) | 6 598 |
| Positive class (musk conformations) | 1 017 (15.4%) |
| Negative class | 5 581 (84.6%) |
| Features per conformation | 166 |
| Features type | Continuous (molecular geometry descriptors) |

**Justification for inclusion:** Musk v2 is the largest and most challenging dataset in the benchmark. It tests the scalability limit of all three optimizer families and provides a negative result: flat ŁNN architectures (without multi-instance pooling) cannot classify Musk v2.

**Multi-instance structure:** the correct approach to Musk (as formalized in Maron & Lozano-Pérez, 1998) requires identifying at least one conformation per molecule that satisfies the musk concept. A flat ŁNN that predicts independently for each conformation, trained on molecule-level labels propagated to all conformations, faces label noise: non-musk conformations of a musk molecule are labeled 1. This structural mismatch is the primary reason all methods fail.

### Preprocessing

```python
from luknn.benchmark.datasets import load_musk
ds = load_musk(seed=42)
# ds.n_features = 166
# len(ds.X_train) = 5278  (80% split)
# len(ds.X_test)  = 1320
```

All 166 features normalized to [0, 1] by min-max scaling. No multi-instance aggregation.

### Benchmark Setup

| Parameter | Value |
|---|---|
| Architecture | 2 hidden layers, width = n_features = 166 (STE, Proximal); width = 8 (LM) |
| max_iter | STE: 1000; LM: 50 (5×2 CV); Proximal: 300 |
| tol_mse | 2e-3 |
| n_trials | STE, Proximal: 30; LM: 5×2 CV (n=10) |

### Results Synopsis

**All methods collapse (F1 ≈ 0 across all 30 trials for all variants).** The dataset is beyond the reach of the current ŁNN architectures for two independent reasons:

1. **Multi-instance structure:** the label propagation noise prevents any flat classifier from finding a consistent rule
2. **Dimensional overload:** 166 features × 2 hidden layers requires 166² × 2 ≈ 55 000 parameters. Even with ProximalTopK's k=10 pruning, each neuron still receives 10/166 ≈ 6% of inputs — the network cannot learn the targeted sparse musk concept

**Conclusion:** Musk v2 requires either (a) explicit multi-instance architecture (max-pooling over conformations per molecule), or (b) aggressive feature pre-selection reducing n_features to ≤ ~20 before applying ŁNN training. The current benchmark results confirm this negative: the flat ŁNN approach is insufficient, and this is a well-defined architectural limitation rather than an optimization failure.

---

## 5. Statistical Protocol

All benchmark experiments use the following statistical protocol:

### 5.1 Trial Design

- **30 independent trials** per (variant, dataset) for STE and Proximal methods
- **5×2 stratified cross-validation** (n=10 observations) for LM on Spambase and Musk (due to high per-trial cost from Jacobian computation)
- Seeds: trial k uses seed = 42 + k × 17 (spacing prevents seed correlation)

### 5.2 Confidence Intervals

95% CIs use the t-distribution with n−1 degrees of freedom:

```python
mean, se = np.mean(scores), np.std(scores, ddof=1) / np.sqrt(n)
ci95 = (mean - t.ppf(0.975, df=n-1) * se,
        mean + t.ppf(0.975, df=n-1) * se)
```

With n=30, the t-critical value is 2.045 (vs 1.960 for z-distribution). With n=10 (5×2 CV), it is 2.262 — CIs are approximately 10% wider.

### 5.3 Statistical Tests

Pairwise Wilcoxon signed-rank tests (two-sided) are used to compare variants within each dataset. Multiple comparisons are corrected using the **Holm-Bonferroni step-down procedure**, which controls the family-wise error rate (FWER) at α = 0.05.

The Wilcoxon test is non-parametric (no normality assumption) and paired (each trial uses the same seed across variants). For n=30 trials, the test has 80% power to detect a mean F1 difference of ~0.08 at α = 0.05.

### 5.4 Primary Metric

F1 score (macro-averaged for binary classification with threshold 0.5) is the primary metric. It is more informative than accuracy for class-imbalanced datasets (Musk: 15.4% positive). For LM (which does not always crystallize), F1 is computed on the post-crystallization network (`f1_crisp`).

---

## 6. References

- Thrun, S., et al. (1991). *The MONK's Problems: A Performance Comparison of Different Learning Algorithms*. CMU-CS-91-197.
- Schlimmer, J. (1987). *Concept Acquisition Through Representational Adjustment*. PhD thesis, UC Irvine.
- Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase*. UCI ML Repository.
- Dietterich, T. G., Lathrop, R. H., & Lozano-Pérez, T. (1997). *Solving the Multiple Instance Problem with Axis-Parallel Rectangles*. Artificial Intelligence, 89(1-2), 31–71.
- Maron, O., & Lozano-Pérez, T. (1998). *A framework for multiple-instance learning*. NeurIPS 1998.
