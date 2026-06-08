"""
Mushroom dataset experiment (§5 of Leandro ALT 2009).

Target: reproduce the 100%-accurate model using 7 selected attributes
(A1–A7 in the paper) and extract the symbolic formula.

  Final formula (paper result):
    (A2 ⊗ ¬A5 ⊗ A7) ⊕ (A2 ⊗ A4 ⊗ ¬A7)

Usage:
    python -m experiments.mushroom.run
"""

import torch
import numpy as np

from luknn.network.luknn import make_network
from luknn.network.crystallization import crisp_crystallize
from luknn.training.lm import lm_train
from luknn.training.obs_pruning import obs_prune
from luknn.extraction.extractor import extract_formula
from .preprocess import prepare

# Attributes selected in paper (after initial feature selection pass)
# Indices correspond to binarized columns for:
# A1: bruises?=t  A2: odor∈{a,l,n}  A3: odor=c  A4: ring.type=e
# A5: spore.print.color=r  A6: population=c  A7: habitat=w
SELECTED_ATTR_NAMES = ["A1:bruises=t", "A2:odor∈{a,l,n}", "A3:odor=c",
                        "A4:ring.type=e", "A5:spore_color=r",
                        "A6:population=c", "A7:habitat=w"]


def run():
    print("Preparing mushroom dataset…")
    X, y = prepare()

    # Use only the 7 selected binary attributes
    # (you may need to adjust indices after binarization — see preprocess.py)
    n_selected = 7
    X_sel = X[:, :n_selected].copy()
    print(f"Using {n_selected} attributes, {len(y)} samples")

    x_t = torch.tensor(X_sel, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    best_mse = float("inf")
    best_model = None

    for trial in range(3):
        model = make_network(n_selected, n_hidden_layers=2, hidden_width=4)
        result = lm_train(model, x_t, y_t, max_iter=2000,
                          tol_mse=3e-3, verbose=True)
        mse_val = (model(x_t) - y_t).pow(2).mean().item()
        print(f"Trial {trial+1}: converged={result['converged']}  mse={mse_val:.5f}")
        if mse_val < best_mse:
            best_mse = mse_val
            best_model = model

    if best_model is None:
        print("No model found.")
        return

    import torch.nn as nn
    for m in best_model.net:
        if isinstance(m, nn.Linear):
            m.weight.data = crisp_crystallize(m.weight.data)
            m.bias.data = m.bias.data.round()

    obs_prune(best_model, x_t, y_t, mse_budget=3e-3)

    result = extract_formula(best_model, SELECTED_ATTR_NAMES)
    print("\nExtracted formula:")
    print(" ", result.formula)
    print(f"Fully representable: {result.representable}")


if __name__ == "__main__":
    torch.manual_seed(42)
    run()
