"""
LukasiewiczLinear — custom layer for Łukasiewicz NNs.

Supports three training modes:

  'continuous'  Continuous real weights (LM optimizer reads them via jacfwd).
  'ste'         Straight-Through Estimator: forward uses ternary weights
                {-1,0,1}; backward uses identity (STE trick).
  'clamp'       Continuous weights clamped to [-1,1]; gradient flows
                through clamp (for Proximal/Adam optimizer).

After training, call `.crystallize()` to snap weights to integer CNN form.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activation import TruncatedIdentityFn
from ..network.crystallization import (
    progressive_crystallize,
    crisp_crystallize_weights,
    crisp_crystallize_bias,
    representation_error,
)


def _hard_snap(w: Tensor, pos_thresh: float = 0.33, neg_thresh: float = -0.33) -> Tensor:
    """Map continuous weights to {-1, 0, 1} via symmetric thresholds."""
    return torch.where(
        w > pos_thresh,
        torch.ones_like(w),
        torch.where(w < neg_thresh, -torch.ones_like(w), torch.zeros_like(w)),
    )


class LukasiewiczLinear(nn.Module):
    """
    Linear + TruncatedIdentity layer with mode-dependent weight handling.

    Parameters
    ----------
    in_features, out_features : int
    mode : str   'continuous' | 'ste' | 'clamp'
    """

    def __init__(self, in_features: int, out_features: int, mode: str = "continuous"):
        super().__init__()
        assert mode in ("continuous", "ste", "clamp"), f"Unknown mode {mode!r}"
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._init_weights()

    def _init_weights(self) -> None:
        if self.mode == "ste":
            # Spread across [-0.7, 0.7] so ~50% of weights snap to ±1 at init.
            # With threshold=0.33, P(|U(-0.7,0.7)| > 0.33) ≈ 53%.
            nn.init.uniform_(self.weight, -0.7, 0.7)
            nn.init.uniform_(self.bias, 0.2, 0.8)
        else:
            # Small init keeps activations in the linear region of clamp(·,0,1).
            bound = 1.0 / (2 * self.in_features)
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.constant_(self.bias, 0.5)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "ste":
            w_snap = _hard_snap(self.weight)
            # STE: value of w_snap, gradient of self.weight
            w = (w_snap - self.weight).detach() + self.weight
        elif self.mode == "clamp":
            w = self.weight.clamp(-1.0, 1.0)
        else:
            w = self.weight

        # Use torch.clamp for jacfwd/vmap compatibility.
        # TruncatedIdentityFn is available for custom backward experiments.
        return torch.clamp(F.linear(x, w, self.bias), 0.0, 1.0)

    def crystallize(self) -> None:
        """In-place crystallization.  Strategy depends on training mode.

        STE: use hard_snap (same threshold as during forward) — progressive
             crystallization would move 0.4 → 0 even though it was snapping
             to +1 during training.
        continuous/clamp: progressive Υ-schedule pushes near-integer weights
             all the way to integers before rounding.
        """
        if self.mode == "ste":
            self.weight.data = _hard_snap(self.weight.data)
            self.bias.data = self.bias.data.round()
        else:
            self.weight.data = crisp_crystallize_weights(
                progressive_crystallize(self.weight.data)
            )
            self.bias.data = crisp_crystallize_bias(
                progressive_crystallize(self.bias.data)
            )

    def representation_error(self) -> float:
        return representation_error(
            torch.cat([self.weight.data.view(-1), self.bias.data.view(-1)])
        ).item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode={self.mode!r}"


class LukasiewiczNet(nn.Module):
    """
    Feed-forward ŁNN built from LukasiewiczLinear layers.

    Equivalent to LukNN but using the extended layer with mode support.
    Each layer applies TruncatedIdentityFn internally, so no separate
    activation is needed between layers.

    Parameters
    ----------
    n_inputs : int
    hidden_sizes : list[int]
    mode : str   Passed to every LukasiewiczLinear layer.
    """

    def __init__(self, n_inputs: int, hidden_sizes: list[int], mode: str = "continuous"):
        super().__init__()
        sizes = [n_inputs] + hidden_sizes + [1]
        self.layers = nn.ModuleList(
            LukasiewiczLinear(in_sz, out_sz, mode=mode)
            for in_sz, out_sz in zip(sizes[:-1], sizes[1:])
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)

    def crystallize(self) -> None:
        for layer in self.layers:
            layer.crystallize()

    def is_crystallized(self, tol: float = 1e-3) -> bool:
        for layer in self.layers:
            if layer.representation_error() > tol:
                return False
        return True

    def flat_weights(self) -> Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_flat_weights(self, w: Tensor) -> None:
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(w[offset : offset + n].view(p.shape))
            offset += n

    def weight_matrix_repr(self) -> list[tuple[Tensor, Tensor]]:
        return [(layer.weight.data.clone(), layer.bias.data.clone())
                for layer in self.layers]


def make_lukasiewicz_net(
    n_inputs: int,
    n_hidden_layers: int = 2,
    hidden_width: int | None = None,
    mode: str = "continuous",
) -> LukasiewiczNet:
    w = hidden_width if hidden_width is not None else n_inputs
    return LukasiewiczNet(n_inputs, [w] * n_hidden_layers, mode=mode)
