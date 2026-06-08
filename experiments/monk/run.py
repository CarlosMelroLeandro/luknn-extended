"""
MONK problems experiment (§5 extension — ALT 2009 replication).

Runs problems 1, 2, 3 in sequence.  For each problem:
  - 5 independent LM training runs (best of 5 selected)
  - Crystallisation + formula extraction
  - Comparison with ground-truth rule

Ground-truth rules:
  MONK-1: (a1 == a2) OR (a5 == 1)
  MONK-2: exactly 2 of {a1=1,...,a6=1} are true
  MONK-3: (a5==3 AND a4==1) OR (a5!=4 AND a2!=3)  [5% label noise]

Usage:
    python -m experiments.monk.run
    python -m experiments.monk.run --problem 1
"""

import argparse
import torch
import numpy as np

from luknn.benchmark.datasets import load_monk, MONK_RULES
from luknn.network.luknn import make_network
from luknn.network.crystallization import crisp_crystallize
from luknn.training.lm import lm_train
from luknn.training.obs_pruning import obs_prune
from luknn.extraction.extractor import extract_formula

import torch.nn as nn

N_TRIALS = 5
HIDDEN_LAYERS = [8, 4]
TOL_MSE = 0.01


def _run_problem(problem: int) -> None:
    print(f"\n{'='*60}")
    print(f"MONK-{problem}  rule: {MONK_RULES[problem]}")
    print(f"{'='*60}")

    ds = load_monk(problem=problem)
    x_t, y_t = ds.X_train, ds.y_train
    x_e, y_e = ds.X_test,  ds.y_test

    print(f"Train: {x_t.shape}  Test: {x_e.shape}  "
          f"features: {ds.n_features}")

    best_mse   = float("inf")
    best_model = None

    for trial in range(N_TRIALS):
        torch.manual_seed(trial * 17 + problem * 100)
        model = make_network(ds.n_features,
                             n_hidden_layers=len(HIDDEN_LAYERS),
                             hidden_width=HIDDEN_LAYERS[0])
        result = lm_train(model, x_t, y_t,
                          max_iter=500, tol_mse=TOL_MSE, verbose=False)
        mse = (model(x_t) - y_t).pow(2).mean().item()
        with torch.no_grad():
            acc = ((model(x_e) >= 0.5).float() == y_e).float().mean().item()
        print(f"  Trial {trial+1}: converged={result['converged']}  "
              f"mse={mse:.5f}  acc={acc:.4f}")
        if mse < best_mse:
            best_mse   = mse
            best_model = model

    if best_model is None:
        print("  No model found.")
        return

    # Crystallise
    for m in best_model.net:
        if isinstance(m, nn.Linear):
            m.weight.data = crisp_crystallize(m.weight.data)
            m.bias.data   = m.bias.data.round()

    obs_prune(best_model, x_t, y_t, mse_budget=TOL_MSE)

    with torch.no_grad():
        test_acc = ((best_model(x_e) >= 0.5).float() == y_e).float().mean().item()
    print(f"\n  Best model test accuracy after crystallisation: {test_acc:.4f}")

    result = extract_formula(best_model, ds.feature_names)
    print(f"  Extracted formula: {result.formula}")
    print(f"  Fully representable: {result.representable}")


def run(problems=None):
    if problems is None:
        problems = [1, 2, 3]
    for p in problems:
        _run_problem(p)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, choices=[1, 2, 3],
                        help="Run a single MONK problem (default: all three)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    run(problems=[args.problem] if args.problem else None)
