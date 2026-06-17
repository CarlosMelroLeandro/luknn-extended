"""
DLMNetwork — Differentiable Łukasiewicz Machine.

Architecture:
    input (n_features)
    → GateLayer(n_features, width)
    → GateLayer(width, width)  ×  (n_hidden_layers - 1)
    → GateLayer(width, 1)       [output neuron — binary classification]

After training, call crystallize() to obtain a CrystallizedDLM with integer
weights that is verifiably 100% representable (for gate_set='rep').
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gate_layer import GateLayer
from ..layers.activation import TruncatedIdentityFn


class DLMNetwork(nn.Module):
    """
    Multi-layer Differentiable Łukasiewicz Machine.

    Parameters
    ----------
    n_features       : int    input dimensionality
    n_hidden_layers  : int    number of hidden GateLayers (default 2)
    hidden_width     : int    neurons per hidden layer (default n_features)
    temperature      : float  initial softmax temperature (anneals during training)
    gate_set         : str    'rep' (G_rep, 12 gates) or 'full' (16 gates)
    pair_mode        : str    'random' or 'sequential'
    seed             : int    random seed for reproducible pairings
    """

    def __init__(
        self,
        n_features: int,
        n_hidden_layers: int = 2,
        hidden_width: int | None = None,
        temperature: float = 1.0,
        gate_set: str = "rep",
        pair_mode: str = "random",
        seed: int | None = None,
        n_output_heads: int = 1,
    ):
        super().__init__()
        assert n_features >= 2, f"n_features must be >= 2, got {n_features}"
        assert n_hidden_layers >= 1, "n_hidden_layers must be >= 1"
        assert n_output_heads >= 1, "n_output_heads must be >= 1"

        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width or n_features
        self.gate_set = gate_set
        self._temperature = temperature
        self.n_output_heads = n_output_heads

        # Build hidden GateLayers
        layers = []
        fan_in = n_features
        for i in range(n_hidden_layers):
            s = seed + i if seed is not None else None
            layers.append(GateLayer(
                fan_in=fan_in,
                n_neurons=self.hidden_width,
                temperature=temperature,
                gate_set=gate_set,
                pair_mode=pair_mode,
                seed=s,
            ))
            fan_in = self.hidden_width

        # Output GateLayer — n_output_heads neurons (mean-aggregated during training).
        # More heads → denser gradient coverage of the last hidden layer.
        s_out = seed + n_hidden_layers if seed is not None else None
        self.output_layer = GateLayer(
            fan_in=fan_in,
            n_neurons=n_output_heads,
            temperature=temperature,
            gate_set=gate_set,
            pair_mode=pair_mode,
            seed=s_out,
        )

        self.hidden_layers = nn.ModuleList(layers)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value
        for layer in self.hidden_layers:
            layer.temperature = value
        self.output_layer.temperature = value

    @property
    def all_layers(self) -> list[GateLayer]:
        return list(self.hidden_layers) + [self.output_layer]

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch, n_features]  — values in [0, 1]

        Returns
        -------
        Tensor [batch, 1]  — soft output in [0, 1]

        When n_output_heads > 1, each head independently computes a soft gate
        output.  Training loss is computed on the mean; this provides denser
        gradient coverage of the last hidden layer (each head covers 2 neurons,
        so n_output_heads heads cover up to 2 × n_output_heads hidden neurons
        per gradient step, vs. 2 for a single head).
        """
        h = x
        for layer in self.hidden_layers:
            h = layer(h)                    # [batch, hidden_width]
        out = self.output_layer(h)          # [batch, n_output_heads]
        if self.n_output_heads > 1:
            return out.mean(dim=1, keepdim=True)  # [batch, 1]
        return out                           # [batch, 1]

    def entropy_loss(self) -> Tensor:
        """Mean gate-distribution entropy over all neurons (penalise diffuse gates)."""
        entropies = [layer.entropy() for layer in self.all_layers]
        return torch.stack(entropies).mean()

    def gate_confidence(self) -> float:
        """Mean max-gate-probability across all neurons."""
        confs = [layer.gate_confidence() for layer in self.all_layers]
        return float(torch.stack(confs).mean().item())

    def n_neurons(self) -> int:
        return self.n_hidden_layers * self.hidden_width + self.n_output_heads

    # ── Crystallisation ───────────────────────────────────────────────────────

    def crystallize(self) -> "CrystallizedDLM":
        """
        Select the argmax gate for every neuron and return an integer-weight
        CrystallizedDLM.

        For gate_set='rep': representability is guaranteed — every selected gate
        satisfies Proposition 3 by construction.
        """
        return CrystallizedDLM.from_dlm(self)

    def representability_report(self) -> dict:
        """Aggregate representability across all layers."""
        total = 0
        rep = 0
        gate_counts: dict[str, int] = {}
        for layer in self.all_layers:
            r = layer.representability_report()
            total += r["total"]
            rep += r["representable"]
            for g, c in r["gate_counts"].items():
                gate_counts[g] = gate_counts.get(g, 0) + c
        return {
            "total": total,
            "representable": rep,
            "fraction": rep / max(total, 1),
            "gate_counts": gate_counts,
        }

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, "
            f"n_hidden={self.n_hidden_layers}, "
            f"width={self.hidden_width}, "
            f"n_output_heads={self.n_output_heads}, "
            f"gate_set={self.gate_set!r}, "
            f"T={self._temperature:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────

class CrystallizedDLM(nn.Module):
    """
    Fully crystallised DLM: integer weights, truncated-identity activation.

    Every neuron has w_i ∈ {-1,0,+1} and b ∈ ℤ satisfying Proposition 3
    (for gate_set='rep').

    When n_output_heads > 1, forward() returns the majority vote of all output
    heads (rounded mean), giving a single scalar ∈ {0, 1}.

    Compatible with the existing extraction pipeline:
      from luknn.extraction.classifier import classify_neuron
      from luknn.extraction.extractor   import extract_formula
    """

    def __init__(self, layers: list[tuple[Tensor, Tensor]], n_output_heads: int = 1):
        """
        Parameters
        ----------
        layers         : list of (weight_matrix, bias_vector) per layer
                         weight_matrix : [n_out, n_in]  — integer values
                         bias_vector   : [n_out]         — integer values
        n_output_heads : number of output neurons in the last layer (for majority vote)
        """
        super().__init__()
        self.n_output_heads = n_output_heads
        self.linear_layers = nn.ModuleList()
        for W, b in layers:
            lin = nn.Linear(W.shape[1], W.shape[0], bias=True)
            lin.weight.data.copy_(W)
            lin.bias.data.copy_(b)
            lin.weight.requires_grad_(False)
            lin.bias.requires_grad_(False)
            self.linear_layers.append(lin)

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns [batch, 1].  For multi-head models, the output is the mean of
        all head outputs (equivalent to majority vote when heads are binary).
        """
        h = x
        for lin in self.linear_layers:
            h = TruncatedIdentityFn.apply(lin(h))
        # h: [batch, n_output_heads]
        if self.n_output_heads > 1:
            return h.mean(dim=1, keepdim=True)
        return h

    @classmethod
    def from_dlm(cls, dlm: DLMNetwork) -> "CrystallizedDLM":
        """Build a CrystallizedDLM from a trained DLMNetwork."""
        if dlm.gate_set != "rep":
            raise ValueError(
                "Crystallization with integer weights is only valid for gate_set='rep'. "
                "For gate_set='full', use argmax_gate() to check which neurons are "
                "non-representable before calling from_dlm()."
            )
        layers = []
        for layer in dlm.all_layers:
            W, b = layer.to_weight_matrix()
            layers.append((W, b))
        return cls(layers, n_output_heads=dlm.n_output_heads)

    def n_parameters(self) -> int:
        return sum(
            int((lin.weight.data.abs() > 0).sum().item())
            for lin in self.linear_layers
        )

    def n_neurons(self) -> int:
        return sum(lin.out_features for lin in self.linear_layers)

    def layers_info(self) -> list[dict]:
        """Return per-layer weight/bias summaries for inspection."""
        info = []
        for i, lin in enumerate(self.linear_layers):
            W = lin.weight.data
            info.append({
                "layer": i,
                "shape": tuple(W.shape),
                "non_zero_weights": int((W.abs() > 0).sum().item()),
                "unique_biases": sorted(lin.bias.data.int().unique().tolist()),
            })
        return info

    def classify_neurons(self) -> list[list]:
        """Apply Proposition 3 classification to every neuron."""
        from ..extraction.classifier import classify_neuron
        result = []
        for lin in self.linear_layers:
            layer_result = []
            for neuron_idx in range(lin.out_features):
                w = lin.weight.data[neuron_idx]
                b = lin.bias.data[neuron_idx]
                cfg = classify_neuron(w, b)
                layer_result.append(cfg)
            result.append(layer_result)
        return result

    def representability_fraction(self) -> float:
        """Fraction of neurons satisfying Proposition 3 (should be 1.0 for G_rep)."""
        from ..extraction.classifier import NeuronKind
        total = rep = 0
        for layer_cfgs in self.classify_neurons():
            for cfg in layer_cfgs:
                total += 1
                if cfg.kind != NeuronKind.UNREPRESENTABLE:
                    rep += 1
        return rep / max(total, 1)

    def flat_weights(self) -> Tensor:
        """Concatenate all weight tensors (for Δ(N) computation)."""
        parts = [lin.weight.data.flatten() for lin in self.linear_layers]
        return torch.cat(parts)


# ─────────────────────────────────────────────────────────────────────────────

def make_dlm_net(
    n_features: int,
    n_hidden_layers: int = 2,
    hidden_width: int | None = None,
    temperature: float = 1.0,
    gate_set: str = "rep",
    seed: int | None = None,
    n_output_heads: int = 1,
) -> DLMNetwork:
    """
    Factory function — mirrors make_lukasiewicz_net() API.

    Parameters
    ----------
    n_features      : input dimensionality
    n_hidden_layers : depth (default 2, as in paper)
    hidden_width    : neurons per hidden layer (default = n_features)
    temperature     : initial softmax temperature
    gate_set        : 'rep' for DLM (representable only) or 'full' for DLN comparison
    seed            : reproducible random pairing seed
    n_output_heads  : output gate neurons; >1 improves gradient coverage
                      (each head sees a different pair of last-hidden-layer neurons).
                      Typical: 4–8 for small datasets, 16–32 for large.

    Returns
    -------
    DLMNetwork ready for training with DLMOptimizer
    """
    return DLMNetwork(
        n_features=n_features,
        n_hidden_layers=n_hidden_layers,
        hidden_width=hidden_width,
        temperature=temperature,
        gate_set=gate_set,
        seed=seed,
        n_output_heads=n_output_heads,
    )
