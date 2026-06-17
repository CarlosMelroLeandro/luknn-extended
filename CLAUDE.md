# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Replication package for **"Łukasiewicz Neural Networks Extended: Residual Architectures and Crystallization Strategies for Interpretable Rule Extraction"** by Carlos Leandro (ISEL). The `luknn` package implements three crystallization strategies (Levenberg-Marquardt, STE, Proximal) for neuro-symbolic ŁNNs with residual architectures, evaluated across six classification benchmarks.

## Setup

```bash
bash INSTALL.sh          # creates .venv/, installs CPU-only PyTorch + package in editable mode
source .venv/bin/activate
```

## Common commands

```bash
# Tests
pytest tests/ -v
pytest tests/ --cov=src/luknn --cov-report=term-missing

# Run a single test file
pytest tests/test_residual.py -v

# Full benchmark (all optimizers, 10 trials)
python benchmark/retrain_all_optimizers.py --n_trials 10

# LM_Residual vs baseline only
python benchmark/retrain_best.py --datasets mushroom heart monk_1 monk_2 monk_3 breast_cancer --n_trials 10

# Formula extraction from crystallized models
python benchmark/extract_formulas.py --datasets monk_1 monk_2 monk_3 --n_trials 10

# Hyperparameter search (per-optimizer shell scripts)
./tune_all.sh             # LM_Residual, background jobs per dataset
./tune_all_Proximal.sh    # Proximal optimizer
./tune_all_STE.sh         # STE optimizer

# Notebooks
jupyter notebook notebooks/
```

## Architecture

### Core library: `src/luknn/`

**Layers (`layers/`)**
- `LukasiewiczLinear` — the fundamental building block. Implements `linear + clamp(·,0,1)` with three training modes:
  - `'continuous'` — real weights; LM optimizer reads them via `jacfwd`
  - `'ste'` — ternary weights `{-1,0,1}` in the forward pass, identity in the backward (STE trick)
  - `'clamp'` — weights clamped to `[-1,1]` for Proximal/Adam
- `LukResidualBlock` — residual block `y = clamp(F(x) + x + b, 0, 1)`. After crystallization, `b=0` → disjunction (⊕), `b=-1` → conjunction (⊗). See `RESIDUAL_THEORY.md` for the derivation.

**Networks (`network/`)**
- `LukasiewiczNet` (in `layers/lukasiewicz_linear.py`) — flat feed-forward ŁNN
- `LukResidualNet` — residual ŁNN: optional projection layer → N residual blocks → output layer. Both networks expose `flat_weights()` / `load_flat_weights()` (needed by LM) and `crystallize()` / `is_crystallized()`
- `crystallization.py` — smooth Υ_n schedule and crisp post-training crystallization functions

**Optimizers (`optimizers/`)**
All three share `BaseOptimizer` and return a `TrainingResult`. `BenchmarkRunner` calls them through a unified interface.
- `LMOptimizer` — Modified Levenberg-Marquardt: Jacobian via `torch.func.jacfwd` (forward-mode AD), smooth crystallization applied after each accepted step, optional OBS pruning
- `STEOptimizer` — Adam with ternary quantization via STE; model must be built with `mode='ste'`
- `ProximalOptimizer` — Adam with ternary-attraction proximal term; model must be built with `mode='clamp'`

**Training (`training/`)**
- `lm.py` — low-level `lm_train` and variants (`lm_train_delayed`, `lm_train_progressive`, `lm_train_dual`, `lm_train_hybrid`)
- `obs_pruning.py` — Optimal Brain Surgeon weight pruning applied post-crystallization
- `ste.py` — STE training loop

**Extraction (`extraction/`)**
- `extractor.py` — translates a crystallized `LukNN` to a symbolic Łukasiewicz formula, layer by layer, using `classify_neuron` → proposition + λ-approximation fallback
- `residual_extractor.py` — same for `LukResidualNet`

**Benchmark (`benchmark/`)**
- `ExperimentConfig` — dataclass loaded from YAML; controls architecture, optimizer, dataset, training budget
- `BenchmarkRunner` — orchestrates N trials: load dataset → build network (mode selected from optimizer method) → instantiate optimizer → train → compute metrics → save JSON
- Supported optimizer method strings: `"LM"`, `"LM_Residual"`, `"STE"`, `"STE_Residual"`, `"Proximal"`, `"Proximal_Residual"`

**DLM (`dlm/`)**
- `DLMNetwork` — Differentiable Łukasiewicz Machine (an independent architecture using `GateLayer` with softmax gate selection and temperature annealing; not part of the main LukResidualNet experiments)

### Configs (`configs/`)

YAML files consumed by `ExperimentConfig.from_yaml()`. Schema sections: `experiment`, `architecture` (includes `hidden_width`, `n_blocks`, `n_inner` for residual), `optimizer` (method + params dict), `dataset`, `training`, `logging`.

### Results layout

- `results/tuning/` — JSON from grid-search runs
- `results/final/` — LM_Residual vs baseline (10 trials)
- `results/final3/` / `results/final5_clean/` — three/five-optimizer comparisons
- `results/formulas/` — extracted symbolic formulas

### Key invariants

- Weights are restricted to `{-1, 0, 1}` post-crystallization; biases are unrestricted integers.
- The LM optimizer requires `model.flat_weights()` / `model.load_flat_weights()` to exist.
- `crystallize()` must be called before calling the extractor; `is_crystallized()` checks representation error < tol.
- GPU is not used anywhere; all computation is CPU-only (`torch` without CUDA).
