"""
TruncatedIdentity activation with correct autograd backward.

Standard `torch.clamp` has gradient 0 at the boundary endpoints;
`TruncatedIdentityFn` makes that explicit and gives clean gradient
masking (0 outside (0,1), 1 inside) needed for STE/Proximal backprop.
"""

import torch
import torch.nn as nn
from torch import Tensor


class TruncatedIdentityFn(torch.autograd.Function):
    """
    Forward : ψ(x) = clamp(x, 0, 1)
    Backward: dψ/dx = 1 if 0 < x < 1, else 0

    Uses new-style API (setup_context) + generate_vmap_rule so that
    functorch transforms (jacfwd / vmap used by LM Jacobian) work correctly.
    generate_vmap_rule=True is valid because the function is element-wise.
    """

    generate_vmap_rule = True  # lets jacfwd/vmap derive the batched rule automatically

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        mask = (x > 0.0) & (x < 1.0)
        return grad_output * mask.float()

    @staticmethod
    def jvp(ctx, *grad_inputs: Tensor) -> Tensor:
        """Forward-mode AD (required by jacfwd / torch.func.jvp)."""
        (x,) = ctx.saved_tensors
        mask = (x > 0.0) & (x < 1.0)
        return grad_inputs[0] * mask.float()


class TruncatedIdentity(nn.Module):
    """Module wrapper — drop-in replacement for the network package version."""

    def forward(self, x: Tensor) -> Tensor:
        return TruncatedIdentityFn.apply(x)
