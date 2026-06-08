"""
Truncated identity activation function.

ψ(x) = min(1, max(0, x)) = clamp(x, 0, 1)

This is the activation that makes each neuron represent a Łukasiewicz
connective exactly (Castro & Trillas, 1998).
"""

import torch
import torch.nn as nn
from torch import Tensor


class TruncatedIdentity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, 0.0, 1.0)


psi = TruncatedIdentity()
