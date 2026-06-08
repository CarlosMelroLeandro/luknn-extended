"""
Hyperparameter grid search — MONK problems 1, 2, 3 (UCI).

Dataset: 17 binary features (one-hot from 6 nominal attrs).
  MONK-1: 124 train / 432 test — rule: (a1==a2) OR (a5==1)
  MONK-2: 169 train / 432 test — rule: exactly 2 of {a1..a6} are 1
  MONK-3: 122 train / 432 test — noisy rule (5% label noise on train)

Grid per problem (18 combos × 5 trials = 90 runs; 270 total for 3 problems):
  hidden_width : [4, 6, 8]
  n_blocks     : [1, 2]
  mu_init      : [0.001, 0.01, 0.1]

Fixed: n_inner=1, patience=80, tol_mse=0.10, max_iter=500, batch_size=0

Usage:
  python tuning/tune_monk.py [--problems 1 2 3] [--n_trials N] [--results_dir DIR]
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from luknn.benchmark.config import ExperimentConfig
from tune import run_grid

# Base config for MONK (problem overridden per iteration below)
_BASE = ExperimentConfig(
    name="LM_Residual — MONK [tuning]",
    seed=42,
    n_inputs=17,            # 6 nominal attrs one-hot → 17 binary features
    hidden_layers=[4, 4],   # unused by LM_Residual
    optimizer_method="LM_Residual",
    optimizer_params={
        "mu_init":       0.01,
        "patience":      80,
        "crystallize_n": 2,
        "prune":         False,
        "batch_size":    0,   # full batch (train ≤ 169 rows)
    },
    dataset_type="monk",
    monk_problem=1,
    hidden_width=4,
    n_blocks=1,
    n_inner=1,
    tol_mse=0.10,       # MONK rules are clean → tighter stop criterion
    max_iter=500,
    results_dir="results/tuning",
    verbose=False,
)

GRID = {
    "hidden_width": [4, 6, 8],
    "n_blocks":     [1, 2],
    "mu_init":      [0.001, 0.01, 0.1],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems",    type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3], metavar="P",
                   help="Which MONK problems to tune (default: 1 2 3)")
    p.add_argument("--n_trials",    type=int, default=5)
    p.add_argument("--results_dir", default="results/tuning")
    args = p.parse_args()

    for problem in args.problems:
        base = replace(_BASE, monk_problem=problem,
                       name=f"LM_Residual — MONK-{problem} [tuning]")
        run_grid(
            base_config=base,
            grid=GRID,
            n_trials=args.n_trials,
            results_dir=args.results_dir,
            label=f"monk_{problem}",
        )


if __name__ == "__main__":
    main()
