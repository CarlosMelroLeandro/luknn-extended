# Łukasiewicz Neural Networks Extended

> Replication package for **"Łukasiewicz Neural Networks Extended: Residual Architectures and Crystallization Strategies for Interpretable Rule Extraction"** — Carlos Leandro (ISEL, Instituto Politécnico de Lisboa).

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen)

This repository contains the full implementation, experiments, and associated papers for an extension of the Łukasiewicz neural network (ŁNN) framework to residual architectures. The work introduces **LukResidualNet** — a family of residual ŁNNs whose skip connections have a provable logical interpretation (disjunction or conjunction) after weight crystallization — and systematically evaluates three crystallization strategies (Levenberg–Marquardt, STE, and Proximal) across six classification benchmarks.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Repository structure](#repository-structure)
4. [Running the experiments](#running-the-experiments)
5. [Reproducing published results](#reproducing-published-results)
6. [Notebooks](#notebooks)
7. [Tests](#tests)
8. [Associated papers](#associated-papers)
9. [License](#license)
10. [Citation](#citation)

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.2 (CPU wheel included in `INSTALL.sh`) |
| NumPy | 1.26 |
| SciPy | 1.12 |
| scikit-learn | 1.4 |
| OS | Linux / macOS / WSL 2 |

A GPU is **not required**. All benchmarks in the paper were run on CPU. Training times range from a few seconds (MONK) to ~15 minutes per optimizer (Mushroom, Breast Cancer) on a modern laptop.

---

## Installation

```bash
git clone https://github.com/CarlosMelroLeandro/luknn-extended.git
cd luknn-extended
bash INSTALL.sh
```

`INSTALL.sh` does three things:
1. Creates a virtual environment at `.venv/`.
2. Installs a CPU-only PyTorch wheel (no CUDA dependency).
3. Installs the `luknn` package in editable mode together with all dev dependencies (`pytest`, `pytest-cov`).

Activate the environment before any subsequent command:

```bash
source .venv/bin/activate
```

---

## Repository structure

```
luknn-extended/
│
├── src/luknn/              # Core library
│   ├── layers/             #   LukasiewiczLinear, LukResidualBlock
│   ├── network/            #   LukasiewiczNet, LukResidualNet
│   ├── optimizers/         #   LM, STE, Proximal optimizers
│   ├── extraction/         #   Symbolic formula extractor
│   ├── benchmark/          #   Dataset loaders, runner, metrics
│   ├── logic/              #   Łukasiewicz connective helpers
│   └── training/           #   Shared training utilities
│
├── experiments/            # Per-dataset standalone run scripts
│   ├── monk/run.py
│   ├── mushroom/run.py
│   ├── breast_cancer/run.py
│   ├── truth_table/        #   Formula reverse-engineering
│   └── noise_robustness/   #   Sensitivity analysis
│
├── tuning/                 # Hyperparameter grid-search drivers
│   ├── tune.py             #   Generic grid engine
│   ├── tune_mushroom.py
│   ├── tune_heart.py
│   ├── tune_monk.py
│   ├── tune_breast_cancer.py
│   ├── tune_ste.py         #   STE-specific grid
│   └── tune_proximal.py    #   Proximal-specific grid
│
├── benchmark/              # Post-tuning evaluation scripts
│   ├── retrain_best.py     #   LM_Residual vs baseline (10 trials)
│   ├── retrain_all_optimizers.py  # Three-way comparison
│   ├── compare_residual.py #   Quick config-file-based comparison
│   ├── extract_formulas.py #   Symbolic extraction pipeline
│   └── run_benchmark.py    #   Full benchmark runner
│
├── configs/                # YAML experiment configurations
├── notebooks/              # Jupyter analysis notebooks (01–05)
├── papers/                 # Submitted paper sources
│   ├── iberamia/
│   ├── IEEE/
│   ├── NAI/
│   └── SociadadePortuguesaMatematica/
│
├── scripts/                # Data download and stats utilities
├── data/                   # Cached datasets (git-ignored)
├── results/                # Output JSON / CSV (git-ignored)
├── logs/                   # Training logs (git-ignored)
├── tests/                  # pytest test suite
│
├── paper.tex / paper.pdf   # Manuscript (main submission)
├── RESIDUAL_THEORY.md      # Mathematical derivation of the residual block
├── REPORT.md               # Experimental narrative
├── references.bib          # BibTeX bibliography
├── INSTALL.sh              # One-shot setup
└── pyproject.toml
```

---

## Running the experiments

### Full LM_Residual grid search (all 6 datasets, background jobs)

```bash
./tune_all.sh
```

Launches one background process per dataset. Grid sizes:

| Dataset | Combinations | Trials | Total runs |
|---|---|---|---|
| Mushroom | 12 | 5 | 60 |
| Heart Disease | 36 | 5 | 180 |
| MONK-1/2/3 | 18 × 3 | 5 | 270 |
| Breast Cancer | 36 | 5 | 180 |

Results land in `results/tuning/`.

To monitor progress:

```bash
tail -f logs/tune_mushroom.log
```

To check whether jobs are still running:

```bash
jobs -l
```

---

### Proximal optimizer grid search

```bash
./tune_all_Proximal.sh
```

Same grid structure as above, using the Proximal (ternary-attraction Adam) optimizer. Proximal typically needs more iterations; budgets are adjusted accordingly.

---

### STE optimizer grid search

```bash
./tune_all_STE.sh
```

Uses the Straight-Through Estimator (STE) optimizer with Adam + ternary quantization. Datasets run sequentially within a single process.

---

### Individual dataset tuning

```bash
source .venv/bin/activate

# LM_Residual — single dataset
python tuning/tune_mushroom.py --n_trials 5 --results_dir results/tuning

# STE — single dataset
python tuning/tune_ste.py --dataset heart --n_trials 5

# Proximal — single dataset
python tuning/tune_proximal.py --dataset monk_1 --n_trials 5
```

---

## Reproducing published results

### Step 1 — Run grid search

Execute one or more of the tuning scripts above. Tuning JSON files are written to `results/tuning/`.

### Step 2 — Retrain best configuration (LM_Residual vs baseline)

```bash
python benchmark/retrain_best.py --datasets mushroom heart monk_1 monk_2 monk_3 breast_cancer --n_trials 10
```

Produces a comparison table (accuracy, F1, MSE, crystallization rate, Wilcoxon p-values) and saves JSON to `results/final/`.

### Step 3 — Three-optimizer comparison

```bash
python benchmark/retrain_all_optimizers.py --n_trials 10
```

Compares LM_Residual, STE, and Proximal side by side. Output goes to `results/final3/`.

### Step 4 — Formula extraction

```bash
python benchmark/extract_formulas.py --datasets monk_1 monk_2 monk_3 --n_trials 10
```

Trains crystallized models and prints the extracted Łukasiewicz formula layer by layer. Saves to `results/formulas/`.

---

## Notebooks

Five Jupyter notebooks cover the full experimental narrative:

| Notebook | Content |
|---|---|
| `01_truth_tables.ipynb` | Formula reverse-engineering from truth tables |
| `02_mushroom.ipynb` | UCI Mushroom benchmark — all three optimizers |
| `03_heart_disease.ipynb` | Cleveland Heart Disease — clinical interpretation |
| `04_monk.ipynb` | MONK-1/2/3 — ground-truth rule recovery |
| `05_breast_cancer.ipynb` | Breast Cancer Ljubljana — feature sparsity analysis |

Launch with:

```bash
jupyter notebook notebooks/
```

---

## Tests

```bash
pytest tests/ -v
```

The suite covers `LukResidualBlock`, `LukResidualNet`, `extract_formula_residual`, all three optimizers, and the layer primitives. Expected output: **61 tests passed**.

For coverage:

```bash
pytest tests/ --cov=src/luknn --cov-report=term-missing
```

---

## Associated papers

The `papers/` directory contains the LaTeX sources and compiled PDFs for the versions submitted to each venue:

| Folder | Venue |
|---|---|
| `papers/iberamia/` | IBERAMIA (Ibero-American Conference on AI) |
| `papers/IEEE/` | IEEE Transactions |
| `papers/NAI/` | NAI — Revista Portuguesa de Inteligência Artificial |
| `papers/SociadadePortuguesaMatematica/` | Boletim da Sociedade Portuguesa de Matemática |

The manuscript at the root (`paper.tex` / `paper.pdf`) is the primary submission copy.

Theoretical background for the residual block is documented in [`RESIDUAL_THEORY.md`](RESIDUAL_THEORY.md). The experimental narrative is in [`REPORT.md`](REPORT.md).

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{leandro2026luknn,
  title   = {Łukasiewicz Neural Networks Extended: Residual Architectures
             and Crystallization Strategies for Interpretable Rule Extraction},
  author  = {Leandro, Carlos},
  year    = {2026},
  note    = {Submitted}
}
```

The foundational ŁNN framework is described in:

```bibtex
@inproceedings{leandro2009symbolic,
  title     = {Symbolic Knowledge Extraction using Łukasiewicz Logics},
  author    = {Leandro, Carlos},
  booktitle = {Algorithmic Learning Theory (ALT 2009)},
  year      = {2009},
  note      = {\url{https://arxiv.org/abs/1604.03099}}
}
```
