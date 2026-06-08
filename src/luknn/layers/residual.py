"""
LukResidualBlock — residual block for Łukasiewicz networks.

Architecture: y = ψ(F(x) + x + b)

The fusion neuron has fixed weights [+1, +1] and a learned bias.
After crystallization:
  b = 0  → y_j = F(x)_j ⊕ x_j   (disjunction, Prop. 3)
  b = -1 → y_j = F(x)_j ⊗ x_j   (conjunction, Prop. 3)

See RESIDUAL_THEORY.md for the full derivation.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .lukasiewicz_linear import LukasiewiczLinear


class LukResidualBlock(nn.Module):
    """
    A residual block: y = ψ(F(x) + x + b), element-wise.

    Parameters
    ----------
    width   : int   Input and output dimension (must be equal — direct skip connection).
    n_inner : int   Number of inner layers in F (default 1).
    mode    : str   Training mode, passed to the inner layers.
    """

    def __init__(self, width: int, n_inner: int = 1, mode: str = "continuous"):
        super().__init__()
        self.width = width
        self.inner_layers = nn.ModuleList([
            LukasiewiczLinear(width, width, mode=mode)
            for _ in range(n_inner)
        ])
        # Fusion neuron bias: learned, initialized to 0 → steers toward ⊕.
        self.merge_bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.inner_layers:
            h = layer(h)
        return torch.clamp(h + x + self.merge_bias, 0.0, 1.0)

    def crystallize(self) -> None:
        for layer in self.inner_layers:
            layer.crystallize()
        self.merge_bias.data = self.merge_bias.data.round()

    def representation_error(self) -> float:
        err = sum(layer.representation_error() for layer in self.inner_layers)
        err += (self.merge_bias.data - self.merge_bias.data.round()).abs().sum().item()
        return err

    def is_crystallized(self, tol: float = 1e-3) -> bool:
        return self.representation_error() < tol
