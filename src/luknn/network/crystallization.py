"""
Smooth and crisp crystallization of ŁNN weights.

Crystallization forces weights toward integer values ({-1, 0, 1}) and
biases toward integers so the network converges to a Castro Neural Network (CNN).

Smooth crystallization (applied at each LM iteration):
    Υ_n(w) = sign(w) * ( cos((1 − frac(|w|)) * π/2)^n  +  floor(|w|) )

where frac(a) = a − floor(a).  n=2 was selected in the paper.

Crisp crystallization (post-training):
    round each weight to its floor integer, then snap to {-1, 0, 1}.

Reference: §2.3 of Leandro (ALT 2009).
"""

import math
import torch
from torch import Tensor


def smooth_crystallize(w: Tensor, n: int = 2) -> Tensor:
    """
    Apply Υ_n element-wise.  Differentiable; used inside the LM loop.
    n=2 balances convergence speed and learning plasticity (paper default).
    """
    s = torch.sign(w)
    a = torch.abs(w)
    floor_a = torch.floor(a)
    frac_a = a - floor_a                               # in [0, 1)
    curved = torch.cos((1.0 - frac_a) * (math.pi / 2)) ** n
    return s * (curved + floor_a)


def crisp_crystallize_weights(w: Tensor) -> Tensor:
    """
    Round to nearest integer then clamp to {-1, 0, 1}.
    For weight matrices only (NOT biases — biases can be any integer).
    """
    return torch.clamp(w.round(), -1.0, 1.0)


def crisp_crystallize_bias(b: Tensor) -> Tensor:
    """Round biases to nearest integer (no clamp — biases can be any integer)."""
    return b.round()


def progressive_crystallize(w: Tensor, schedule: tuple[int, ...] = (2, 4, 8, 16)) -> Tensor:
    """
    Apply smooth crystallization with increasing n until Δ(w) ≈ 0.
    More reliable than a single crisp floor when weights are near but not at integers.
    """
    for n in schedule:
        w = smooth_crystallize(w, n=n)
    return w


def representation_error(w: Tensor) -> Tensor:
    """
    Δ(N) = Σ_i (w_i − floor(w_i))   (eq. from §2.3).
    Zero when all weights are integers (i.e. the network is a CNN).
    """
    return (w - torch.floor(w)).sum()
