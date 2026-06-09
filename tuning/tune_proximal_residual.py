"""
Hyperparameter grid search — ProximalOptimizer on LukResidualNet.

Trains LukResidualNet(mode='clamp') with Adam in two phases:
  Phase 1: pure MSE minimization.
  Phase 2: MSE + L1 regularization + ternary attraction.

Grid per dataset (8 combos × 5 trials = 40 runs):
  lr          : [5e-3, 1e-2]
  lambda_sparse: [1e-4, 1e-3]
  hidden_width: [6, 8]

Fixed parameters per dataset (aligned with flat Proximal budgets):
  Mushroom      : tol_mse=0.15, max_iter=6000, n_blocks=1, n_inner=1
  Heart         : tol_mse=0.15, max_iter=8000, n_blocks=1, n_inner=1
  MONK-1/2/3    : tol_mse=0.10, max_iter=5000, n_blocks=1, n_inner=1
  Breast Cancer : tol_mse=0.15, max_iter=8000, n_blocks=1, n_inner=1

Usage:
  python tuning/tune_proximal_residual.py --dataset mushroom
  python tuning/tune_proximal_residual.py --dataset all --n_trials 5
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from luknn.benchmark.config import ExperimentConfig
from tune import run_grid

_BASES: dict[str, ExperimentConfig] = {

    "mushroom": ExperimentConfig(
        name="Proximal_Residual — Mushroom [tuning]",
        seed=42,
        n_inputs=111,
        hidden_layers=[8, 4],
        hidden_width=8,
        n_blocks=1,
        n_inner=1,
        optimizer_method="Proximal_Residual",
        optimizer_params={
            "lr":              5e-3,
            "lambda_sparse":   1e-3,
            "lambda_attract":  0.05,
            "prox_threshold":  2e-4,
            "phase1_fraction": 0.65,
        },
        dataset_type="mushroom",
        tol_mse=0.15,
        max_iter=6000,
        verbose=False,
    ),

    "heart": ExperimentConfig(
        name="Proximal_Residual — Heart Disease [tuning]",
        seed=42,
        n_inputs=13,
        hidden_layers=[6, 4],
        hidden_width=6,
        n_blocks=1,
        n_inner=1,
        optimizer_method="Proximal_Residual",
        optimizer_params={
            "lr":              8e-3,
            "lambda_sparse":   2e-3,
            "lambda_attract":  0.08,
            "prox_threshold":  3e-4,
            "phase1_fraction": 0.65,
        },
        dataset_type="heart_disease",
        heart_subset="cleveland",
        tol_mse=0.15,
        max_iter=8000,
        verbose=False,
    ),

    "monk_1": ExperimentConfig(
        name="Proximal_Residual — MONK-1 [tuning]",
        seed=42,
        n_inputs=17,
        hidden_layers=[6, 4],
        hidden_width=6,
        n_blocks=1,
        n_inner=1,
        optimizer_method="Proximal_Residual",
        optimizer_params={
            "lr":              5e-3,
            "lambda_sparse":   1e-3,
            "lambda_attract":  0.05,
            "prox_threshold":  2e-4,
            "phase1_fraction": 0.65,
        },
        dataset_type="monk",
        monk_problem=1,
        tol_mse=0.10,
        max_iter=5000,
        verbose=False,
    ),

    "monk_2": ExperimentConfig(
        name="Proximal_Residual — MONK-2 [tuning]",
        seed=42,
        n_inputs=17,
        hidden_layers=[6, 4],
        hidden_width=6,
        n_blocks=1,
        n_inner=1,
        optimizer_method="Proximal_Residual",
        optimizer_params={
            "lr":              5e-3,
            "lambda_sparse":   1e-3,
            "lambda_attract":  0.05,
            "prox_threshold":  2e-4,
            "phase1_fraction": 0.65,
        },
        dataset_type="monk",
        monk_problem=2,
        tol_mse=0.10,
        max_iter=5000,
        verbose=False,
    ),

    "monk_3": ExperimentConfig(
        name="Proximal_Residual — MONK-3 [tuning]",
        seed=42,
        n_inputs=17,
        hidden_layers=[6, 4],
        hidden_width=6,
        n_blocks=1,
        n_inner=1,
        optimizer_method="Proximal_Residual",
        optimizer_params={
            "lr":              5e-3,
            "lambda_sparse":   1e-3,
            "lambda_attract":  0.05,
            "prox_threshold":  2e-4,
            "phase1_fraction": 0.65,
        },
        dataset_type="monk",
        monk_problem=3,
        tol_mse=0.10,
        max_iter=5000,
        verbose=False,
    ),

    "breast_cancer": ExperimentConfig(
        name="Proximal_Residual — Breast Cancer [tuning]",
        seed=42,
        n_inputs=20,
        hidden_layers=[6, 4],
        hidden_width=6,
        n_blocks=1,
        n_inner=1,
        optimizer_method="Proximal_Residual",
        optimizer_params={
            "lr":              5e-3,
            "lambda_sparse":   1e-3,
            "lambda_attract":  0.05,
            "prox_threshold":  2e-4,
            "phase1_fraction": 0.65,
        },
        dataset_type="breast_cancer",
        tol_mse=0.15,
        max_iter=8000,
        verbose=False,
    ),
}

GRID = {
    "lr":            [5e-3, 1e-2],
    "lambda_sparse": [1e-4, 1e-3],
    "hidden_width":  [6, 8],
}

ALL_DATASETS = list(_BASES.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=ALL_DATASETS + ["all"])
    p.add_argument("--n_trials",    type=int, default=5)
    p.add_argument("--results_dir", default="results/tuning")
    args = p.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_grid(
            base_config=replace(_BASES[ds], results_dir=args.results_dir),
            grid=GRID,
            n_trials=args.n_trials,
            results_dir=args.results_dir,
            label=f"proximal_residual_{ds}",
        )


if __name__ == "__main__":
    main()
