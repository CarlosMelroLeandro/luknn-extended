"""
Łukasiewicz Neural Network (ŁNN) model.

Architecture: fully-connected feed-forward network with truncated-identity
activation (ψ).  The paper uses 3 hidden layers; depth is configurable.

Each neuron computes:
    ψ_b(w1*x1, w2*x2, …) = clamp(Σ_i w_i * x_i + b,  0,  1)

After training + crystallization, weights ∈ {-1, 0, 1} and biases ∈ ℤ,
making every neuron interpretable as a Łukasiewicz connective.
"""

import torch
import torch.nn as nn
from torch import Tensor
from .activation import TruncatedIdentity


class LukNN(nn.Module):
    """
    Feed-forward ŁNN.

    Parameters
    ----------
    n_inputs : int
        Number of propositional variables.
    hidden_sizes : list[int]
        Width of each hidden layer.  Paper default: 3 layers, width chosen
        by the reverse-engineering search.
    """

    def __init__(self, n_inputs: int, hidden_sizes: list[int]):
        super().__init__()
        act = TruncatedIdentity()
        sizes = [n_inputs] + hidden_sizes + [1]
        layers: list[nn.Module] = []
        for in_sz, out_sz in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_sz, out_sz, bias=True))
            layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

    def flat_weights(self) -> Tensor:
        """Return all weights and biases as a single 1-D vector (for LM)."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_flat_weights(self, w: Tensor) -> None:
        """Write a flat weight vector back into the model parameters."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(w[offset : offset + n].view(p.shape))
            offset += n

    def weight_matrix_repr(self) -> list[tuple[Tensor, Tensor]]:
        """
        Return [(W, b), …] per linear layer — matches the matrix notation
        used in the paper (§4, §5).
        """
        result = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                result.append((module.weight.data.clone(),
                                module.bias.data.clone()))
        return result


def _init_for_clamp(model: "LukNN") -> None:
    """
    Initialize weights so neurons start in the linear regime of clamp(x,0,1).

    PyTorch's default He init is designed for ReLU and produces large net inputs
    that saturate clamp immediately (gradient = 0 everywhere).

    Strategy: small uniform weights ± 1/(2*fan_in) + bias = 0.5 so that
    net input ≈ 0.5 ± small, keeping neurons squarely in (0,1) at the start.
    """
    import torch.nn as nn
    for module in model.net:
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            bound = 1.0 / (2 * fan_in)
            nn.init.uniform_(module.weight, -bound, bound)
            nn.init.constant_(module.bias, 0.5)


def make_network(n_inputs: int, n_hidden_layers: int = 3,
                 hidden_width: int | None = None) -> "LukNN":
    """
    Convenience constructor.  If hidden_width is None, uses n_inputs.
    Applies clamp-aware initialization automatically.
    """
    w = hidden_width if hidden_width is not None else n_inputs
    model = LukNN(n_inputs, [w] * n_hidden_layers)
    _init_for_clamp(model)
    return model
