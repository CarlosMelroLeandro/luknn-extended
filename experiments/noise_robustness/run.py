"""
Noise robustness experiment.

Adds Gaussian noise to a truth sub-table and measures whether the
reverse-engineering algorithm still recovers a valid formula.

The paper mentions the process is "stable for the introduction of Gaussian
noise", motivating its application to real data (§1).
"""

import torch
from luknn.logic.connectives import evaluate_formula, truth_subtable
from luknn.network.luknn import make_network
from luknn.network.crystallization import crisp_crystallize
from luknn.training.lm import lm_train
from luknn.extraction.extractor import extract_formula


def formula_target(x1, x2, x3):
    from luknn.logic.connectives import tnorm, residuum
    return tnorm(residuum(x1, x2), x3)


def run(sigma_values=(0.0, 0.02, 0.05, 0.1), n_values=5, n_trials=5):
    table = truth_subtable(3, n_values)
    clean = evaluate_formula(formula_target, 3, n_values)

    for sigma in sigma_values:
        successes = 0
        for _ in range(n_trials):
            noisy = (clean + sigma * torch.randn_like(clean)).clamp(0.0, 1.0)
            model = make_network(3, n_hidden_layers=3, hidden_width=3)
            result = lm_train(model, table, noisy, tol_mse=2e-3)
            if result["converged"]:
                successes += 1
        print(f"sigma={sigma:.2f}  recovery={successes}/{n_trials}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
