"""
Łukasiewicz logic connectives as differentiable PyTorch operations.

All operators are defined for truth values in [0, 1] (infinite-valued Ł-logic).
For finite (n+1)-valued logic, inputs/outputs are restricted to S_n = {0, 1/n, …, 1}.

Reference: §2.1 of Leandro (ALT 2009).
"""

import torch
from torch import Tensor


def tnorm(x: Tensor, y: Tensor) -> Tensor:
    """Łukasiewicz t-norm (conjunction/fusion ⊗): max(0, x + y − 1)."""
    return torch.clamp(x + y - 1.0, min=0.0)


def residuum(x: Tensor, y: Tensor) -> Tensor:
    """Łukasiewicz residuum (implication ⟹): min(1, 1 − x + y)."""
    return torch.clamp(1.0 - x + y, max=1.0)


def negation(x: Tensor) -> Tensor:
    """Strong negation ¬x = x ⟹ 0 = 1 − x."""
    return 1.0 - x


def disjunction(x: Tensor, y: Tensor) -> Tensor:
    """Disjunction ⊕ = ¬x ⟹ y = min(1, x + y)."""
    return torch.clamp(x + y, max=1.0)


def biconditional(x: Tensor, y: Tensor) -> Tensor:
    """Biconditional x ↔ y = (x ⟹ y) ⊗ (y ⟹ x)."""
    return tnorm(residuum(x, y), residuum(y, x))


# ---------------------------------------------------------------------------
# Truth sub-table generation
# ---------------------------------------------------------------------------

def truth_subtable(n_vars: int, n_values: int) -> Tensor:
    """
    Generate all rows of an (n_values)-valued truth sub-table.

    Returns a tensor of shape (n_values^n_vars, n_vars) with values in
    S_{n_values-1} = {0, 1/(n_values-1), …, 1}.
    """
    step = n_values - 1
    grid_1d = torch.arange(n_values, dtype=torch.float32) / step
    grids = torch.meshgrid(*([grid_1d] * n_vars), indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=1)


def evaluate_formula(formula_fn, n_vars: int, n_values: int) -> Tensor:
    """
    Evaluate a formula function over the full (n_values)-valued truth sub-table.

    formula_fn: callable that accepts n_vars positional Tensor arguments and
                returns a Tensor of shape (N,).
    """
    table = truth_subtable(n_vars, n_values)
    cols = [table[:, i] for i in range(n_vars)]
    return formula_fn(*cols)
