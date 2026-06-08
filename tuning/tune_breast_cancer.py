"""
Hyperparameter grid search — Breast Cancer Ljubljana (UCI).

Dataset: 286 samples, ~20 binary/ordinal features, binary target
         (recurrence-events vs no-recurrence-events).

Grid (36 combos × 5 trials = 180 runs):
  hidden_width : [4, 6, 8]
  n_blocks     : [1, 2]
  mu_init      : [0.001, 0.01, 0.1]
  prune        : [True, False]

Fixed: n_inner=1, patience=100, tol_mse=0.15, max_iter=800, batch_size=0

Usage:
  python tuning/tune_breast_cancer.py [--n_trials N] [--results_dir DIR]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from luknn.benchmark.config import ExperimentConfig
from tune import run_grid


BASE_CONFIG = ExperimentConfig(
    name="LM_Residual — Breast Cancer [tuning]",
    seed=42,
    n_inputs=20,            # approx; overridden by actual n_features at runtime
    hidden_layers=[4, 4],   # unused by LM_Residual
    optimizer_method="LM_Residual",
    optimizer_params={
        "mu_init":       0.01,
        "patience":      100,
        "crystallize_n": 2,
        "prune":         True,
        "batch_size":    0,   # full batch (286 rows)
    },
    dataset_type="breast_cancer",
    hidden_width=6,
    n_blocks=1,
    n_inner=1,
    tol_mse=0.15,
    max_iter=800,
    results_dir="results/tuning",
    verbose=False,
)

GRID = {
    "hidden_width": [4, 6, 8],
    "n_blocks":     [1, 2],
    "mu_init":      [0.001, 0.01, 0.1],
    "prune":        [True, False],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",    type=int, default=5)
    p.add_argument("--results_dir", default="results/tuning")
    args = p.parse_args()

    run_grid(
        base_config=BASE_CONFIG,
        grid=GRID,
        n_trials=args.n_trials,
        results_dir=args.results_dir,
        label="breast_cancer",
    )


if __name__ == "__main__":
    main()
