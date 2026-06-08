"""
Hyperparameter grid search — Mushroom dataset (UCI, 111 features, 8124 rows).

Grid (12 combinations × 5 trials = 60 runs):
  hidden_width : [4, 6, 8]
  n_blocks     : [1, 2]
  mu_init      : [0.001, 0.01]

Fixed (based on config lm_residual_mushroom.yaml):
  n_inner=1, patience=80, tol_mse=0.15, max_iter=600, batch_size=512

Usage:
  python tuning/tune_mushroom.py [--n_trials N] [--results_dir DIR]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from luknn.benchmark.config import ExperimentConfig
from tune import run_grid


BASE_CONFIG = ExperimentConfig(
    name="LM_Residual — Mushroom [tuning]",
    seed=42,
    n_inputs=111,           # overridden by actual dataset features at runtime
    hidden_layers=[6, 4],   # unused by LM_Residual; kept for ExperimentConfig compat
    optimizer_method="LM_Residual",
    optimizer_params={
        "mu_init":       0.01,
        "patience":      80,
        "crystallize_n": 2,
        "prune":         False,
        "batch_size":    512,
    },
    dataset_type="mushroom",
    hidden_width=6,
    n_blocks=1,
    n_inner=1,
    tol_mse=0.15,
    max_iter=600,
    results_dir="results/tuning",
    verbose=False,
)

GRID = {
    "hidden_width": [4, 6, 8],
    "n_blocks":     [1, 2],
    "mu_init":      [0.001, 0.01],
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
        label="mushroom",
    )


if __name__ == "__main__":
    main()
