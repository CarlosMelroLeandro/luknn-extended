"""
λ-similarity and representable approximation for un-representable neurons.

Given an un-representable neuron α, rule R decomposes it into a finite set
S(α) of binary ŁNNs.  The best representable approximation is the one with
highest λ-similarity over the (n+1)-valued truth sub-table.

λ-similarity (§2.2 of Leandro ALT 2009):
    λ = exp( −mean_absolute_error(α, β)  over truth sub-table T )
"""

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..logic.connectives import truth_subtable


@dataclass
class Approximation:
    formula: str
    lam: float          # λ similarity in [0, 1]
    weights: list[int]
    biases: list[int]


def _psi(w: list[int], b: int, x: Tensor) -> Tensor:
    """Evaluate a multi-input neuron ψ_b(w·x) over a batch of rows."""
    wt = torch.tensor(w, dtype=torch.float32)
    return torch.clamp((x * wt).sum(dim=-1) + b, 0.0, 1.0)


def lambda_similarity(alpha_vals: Tensor, beta_vals: Tensor) -> float:
    """λ = exp(−MAE) over the shared truth sub-table."""
    mae = (alpha_vals - beta_vals).abs().mean().item()
    return math.exp(-mae)


def rule_R_decompositions(
    weights: list[int], bias: int
) -> list[tuple[list[int], int, list[int], list[int]]]:
    """
    Generate all binary-tree decompositions of an n-input neuron via rule R.

    Each decomposition is a tuple:
        (outer_weights, outer_bias, inner_weights, inner_bias)
    representing  ψ_{b0}(x_1,…,x_{n-1}, ψ_{b1}(x_{n-1}, x_n))
    with b = b0 + b1 and b1 ≤ b0.

    Only returns decompositions where neither neuron has constant output.
    """
    results = []
    n = len(weights)
    if n <= 2:
        return []

    # Try all splits: inner neuron gets last k inputs (k ≥ 2)
    for k in range(2, n):
        outer_w = weights[: n - k + 1]
        inner_w = weights[n - k :]
        b_sum = bias
        for b1 in range(-abs(b_sum), abs(b_sum) + 1):
            b0 = b_sum - b1
            if b1 > b0:
                continue
            # Check non-constant: outer in (-|outer|+1 .. |outer|), inner same
            outer_len = len(outer_w)
            inner_len = len(inner_w)
            if not (-outer_len < b0 < outer_len):
                continue
            if not (-inner_len < b1 < inner_len):
                continue
            results.append((outer_w, b0, inner_w, b1))
    return results


_MAX_TABLE_ROWS = 1 << 20   # 2^20 = ~1M rows; above this skip approximation


def best_approximation(
    weights: list[int], bias: int, n_values: int = 2
) -> Approximation | None:
    """
    Find the representable approximation with highest λ to the un-representable
    neuron ψ_b(weights) over the (n_values)-valued truth sub-table.

    Returns None (caller falls back to ψ_b notation) when the truth table
    would exceed _MAX_TABLE_ROWS rows (avoids OOM for large first layers).
    """
    n = len(weights)
    if n_values ** n > _MAX_TABLE_ROWS:
        return None
    table = truth_subtable(n, n_values)   # (N, n)
    alpha_vals = _psi(weights, bias, table)

    best_lam = -1.0
    best: Approximation | None = None

    for outer_w, b0, inner_w, b1 in rule_R_decompositions(weights, bias):
        # Evaluate the binary decomposition
        k = len(inner_w)
        inner_cols = table[:, n - k :]
        inner_out = _psi(inner_w, b1, inner_cols)

        outer_cols = torch.cat([table[:, : n - k], inner_out.unsqueeze(1)], dim=1)
        beta_vals = _psi(outer_w, b0, outer_cols)

        lam = lambda_similarity(alpha_vals, beta_vals)
        if lam > best_lam:
            best_lam = lam
            best = Approximation(
                formula=f"ψ_{b0}({outer_w}, ψ_{b1}({inner_w}))",
                lam=lam,
                weights=outer_w + inner_w,
                biases=[b0, b1],
            )

    return best
