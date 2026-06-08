"""
LukResidualNet — ŁNN with residual blocks.

Architecture:
  n_inputs → proj (if n_inputs ≠ hidden_width) → [ResBlock × n_blocks] → output (→ 1)

The projection layer (proj) is a standard LukasiewiczLinear without a skip
connection, because the dimensions differ. Skip connections only exist inside
the ResBlocks, where input_width == output_width.

Identical interface to LukasiewiczNet: supports flat_weights / load_flat_weights
(for the LM optimizer) and crystallize / is_crystallized.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.lukasiewicz_linear import LukasiewiczLinear
from ..layers.residual import LukResidualBlock


class LukResidualNet(nn.Module):
    """
    Parameters
    ----------
    n_inputs      : int   Number of input variables.
    hidden_width  : int   Width of the hidden layers (and residual blocks).
    n_blocks      : int   Number of residual blocks (default 1).
    n_inner       : int   Inner layers per block (default 1).
    mode          : str   'continuous' | 'ste' | 'clamp'
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_width: int,
        n_blocks: int = 1,
        n_inner: int = 1,
        mode: str = "continuous",
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.hidden_width = hidden_width
        self.n_blocks = n_blocks

        # Projection: reduce/expand n_inputs → hidden_width (no skip connection).
        if n_inputs != hidden_width:
            self.proj: nn.Module | None = LukasiewiczLinear(n_inputs, hidden_width, mode=mode)
        else:
            self.proj = None

        self.blocks = nn.ModuleList([
            LukResidualBlock(hidden_width, n_inner=n_inner, mode=mode)
            for _ in range(n_blocks)
        ])

        self.output_layer = LukasiewiczLinear(hidden_width, 1, mode=mode)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)

    # ── Crystallization ──────────────────────────────────────────────────────

    def crystallize(self) -> None:
        if self.proj is not None:
            self.proj.crystallize()
        for block in self.blocks:
            block.crystallize()
        self.output_layer.crystallize()

    def is_crystallized(self, tol: float = 1e-3) -> bool:
        err = 0.0
        if self.proj is not None:
            err += self.proj.representation_error()
        for block in self.blocks:
            err += block.representation_error()
        err += self.output_layer.representation_error()
        return err < tol

    # ── Flat-weight interface (for the LM optimizer) ─────────────────────────

    def flat_weights(self) -> Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_flat_weights(self, w: Tensor) -> None:
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(w[offset: offset + n].view(p.shape))
            offset += n

    def weight_matrix_repr(self) -> list[tuple[Tensor, Tensor]]:
        """Return [(W, b), …] for all linear layers (excluding merge bias)."""
        result = []
        if self.proj is not None and isinstance(self.proj, LukasiewiczLinear):
            result.append((self.proj.weight.data.clone(), self.proj.bias.data.clone()))
        for block in self.blocks:
            for layer in block.inner_layers:
                result.append((layer.weight.data.clone(), layer.bias.data.clone()))
        result.append((self.output_layer.weight.data.clone(),
                        self.output_layer.bias.data.clone()))
        return result
