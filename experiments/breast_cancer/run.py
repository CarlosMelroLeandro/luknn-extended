"""
Breast Cancer (Ljubljana) experiment (§5 extension — ALT 2009 replication).

286 samples, 9 attributes (ordinal + nominal), binary target:
  0 = no-recurrence-events, 1 = recurrence-events

Runs 3 independent LM training trials.  Best model is crystallised and the
Łukasiewicz formula is extracted and printed.

Reference (Nguy & Wasilewski 2025 DLN_L baseline): acc=70.00%

Usage:
    python -m experiments.breast_cancer.run
"""

import torch
import numpy as np
import torch.nn as nn

from luknn.benchmark.datasets import load_breast_cancer
from luknn.network.luknn import make_network
from luknn.network.crystallization import crisp_crystallize
from luknn.training.lm import lm_train
from luknn.training.obs_pruning import obs_prune
from luknn.extraction.extractor import extract_formula

N_TRIALS     = 3
HIDDEN_WIDTH = 8
N_HIDDEN     = 2
TOL_MSE      = 0.05
REFERENCE_ACC = 0.7000


def run():
    print("Preparing Breast Cancer dataset…")
    ds = load_breast_cancer()
    x_t, y_t = ds.X_train, ds.y_train
    x_e, y_e = ds.X_test,  ds.y_test
    print(f"Train: {x_t.shape}  Test: {x_e.shape}  features: {ds.n_features}")
    print(f"Target balance (train): {y_t.mean():.3f} recurrence")

    best_mse   = float("inf")
    best_model = None

    for trial in range(N_TRIALS):
        torch.manual_seed(trial * 31 + 7)
        model = make_network(ds.n_features,
                             n_hidden_layers=N_HIDDEN,
                             hidden_width=HIDDEN_WIDTH)
        result = lm_train(model, x_t, y_t,
                          max_iter=500, tol_mse=TOL_MSE, verbose=True)
        mse = (model(x_t) - y_t).pow(2).mean().item()
        with torch.no_grad():
            acc = ((model(x_e) >= 0.5).float() == y_e).float().mean().item()
        print(f"Trial {trial+1}: converged={result['converged']}  "
              f"mse={mse:.5f}  acc={acc:.4f}")
        if mse < best_mse:
            best_mse   = mse
            best_model = model

    if best_model is None:
        print("No model found.")
        return

    # Crystallise
    for m in best_model.net:
        if isinstance(m, nn.Linear):
            m.weight.data = crisp_crystallize(m.weight.data)
            m.bias.data   = m.bias.data.round()

    obs_prune(best_model, x_t, y_t, mse_budget=TOL_MSE)

    with torch.no_grad():
        test_acc = ((best_model(x_e) >= 0.5).float() == y_e).float().mean().item()

    print(f"\nTest accuracy after crystallisation: {test_acc:.4f}")
    print(f"Reference DLN_L (Nguy & Wasilewski 2025): {REFERENCE_ACC:.4f}")
    print(f"Δ vs reference: {test_acc - REFERENCE_ACC:+.4f}")

    result = extract_formula(best_model, ds.feature_names)
    print(f"\nExtracted formula: {result.formula}")
    print(f"Fully representable: {result.representable}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run()
