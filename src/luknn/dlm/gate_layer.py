"""
GateLayer — differentiable softmax selection over G_rep.

Each GateLayer contains N neurons.  Each neuron:
  - Is assigned a fixed random pair of input indices (i, j) at initialisation
  - Maintains logits θ ∈ ℝ^{|G_rep|} (learnable)
  - Computes: out = Σ_k softmax(θ/T)_k · gate_k(x_i, x_j)

During training the output is a smooth blend of all gates.
After crystallisation each neuron snaps to its argmax gate.

Input pairing strategies:
  'random'     : each neuron independently draws 2 distinct indices uniformly
  'sequential' : neuron k uses inputs (2k mod fan_in, (2k+1) mod fan_in)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .gates import apply_all_gates, GATES, GATE_NAMES, N_GATES, N_GATES_FULL


class GateLayer(nn.Module):
    """
    A layer of N 2-input gate neurons with learnable gate distribution.

    Parameters
    ----------
    fan_in        : number of inputs to this layer
    n_neurons     : number of gate neurons (output dimension)
    temperature   : softmax temperature (lower → sharper → closer to argmax)
    gate_set      : 'rep' (12 representable gates) or 'full' (16 including XOR/XNOR)
    pair_mode     : 'random' or 'sequential'
    seed          : RNG seed for reproducible random pairing
    """

    def __init__(
        self,
        fan_in: int,
        n_neurons: int,
        temperature: float = 1.0,
        gate_set: str = "rep",
        pair_mode: str = "random",
        seed: int | None = None,
    ):
        super().__init__()
        assert fan_in >= 2, f"fan_in must be >= 2, got {fan_in}"
        assert gate_set in ("rep", "full"), f"gate_set must be 'rep' or 'full'"
        assert pair_mode in ("random", "sequential"), f"Unknown pair_mode {pair_mode!r}"

        self.fan_in = fan_in
        self.n_neurons = n_neurons
        self.gate_set = gate_set
        self.n_gates = N_GATES if gate_set == "rep" else N_GATES_FULL

        # Logits θ ∈ ℝ^{n_neurons × n_gates} — learnable
        self.logits = nn.Parameter(torch.zeros(n_neurons, self.n_gates))

        # Temperature: not a parameter, controlled externally
        self.temperature = temperature

        # Fixed input pairing — not learnable, registered as buffer
        pairs = self._make_pairs(fan_in, n_neurons, pair_mode, seed)
        self.register_buffer("pair_indices", pairs)  # shape [n_neurons, 2]

        self._init_logits()

    def _make_pairs(
        self,
        fan_in: int,
        n_neurons: int,
        mode: str,
        seed: int | None,
    ) -> Tensor:
        """Generate fixed input pair indices for each neuron."""
        if mode == "sequential":
            idx = torch.arange(n_neurons * 2) % fan_in
            pairs = idx.view(n_neurons, 2)
            # Ensure no self-pairing: if both indices equal, shift second
            mask = pairs[:, 0] == pairs[:, 1]
            pairs[mask, 1] = (pairs[mask, 1] + 1) % fan_in
            return pairs

        # Random mode
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        pairs = torch.zeros(n_neurons, 2, dtype=torch.long)
        for i in range(n_neurons):
            # Draw 2 distinct indices
            p = torch.randperm(fan_in, generator=rng)[:2]
            pairs[i] = p
        return pairs

    def _init_logits(self) -> None:
        # Small random init: slight preference diversity across neurons
        nn.init.normal_(self.logits, mean=0.0, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch, fan_in]

        Returns
        -------
        Tensor [batch, n_neurons]
        """
        # Gather the two inputs for each neuron
        # pair_indices: [n_neurons, 2]
        a = x[:, self.pair_indices[:, 0]]   # [batch, n_neurons]
        b = x[:, self.pair_indices[:, 1]]   # [batch, n_neurons]

        # Apply all gates with STE so gradients flow through clamp boundaries.
        # self.training controls: STE during forward/backward, exact during eval.
        gate_out = apply_all_gates(a, b, gate_set=self.gate_set, ste=self.training)

        # Softmax over gate dimension: [n_neurons, n_gates]
        probs = torch.softmax(self.logits / self.temperature, dim=-1)

        # Weighted sum: broadcast probs over batch
        # probs:    [n_neurons, n_gates] → [n_gates, n_neurons] → [n_gates, 1, n_neurons]
        probs_bc = probs.T.unsqueeze(1)                         # [n_gates, 1, n_neurons]
        output = (gate_out * probs_bc).sum(dim=0)               # [batch, n_neurons]

        return output

    def entropy(self) -> Tensor:
        """Mean gate-distribution entropy over all neurons (for regularisation)."""
        probs = torch.softmax(self.logits / self.temperature, dim=-1)
        # H(p) = -Σ p log p  (nats); clamp for numerical stability
        log_p = torch.log(probs.clamp(min=1e-12))
        h = -(probs * log_p).sum(dim=-1)        # [n_neurons]
        return h.mean()

    def gate_confidence(self) -> Tensor:
        """Mean max-probability over all neurons.  High → converging to pure gates."""
        probs = torch.softmax(self.logits / self.temperature, dim=-1)
        return probs.max(dim=-1).values.mean()

    # ── Crystallisation ──────────────────────────────────────────────────────

    def selected_gates(self) -> list[str]:
        """Return the argmax gate name for each neuron (post-training)."""
        gate_names = GATE_NAMES if self.gate_set == "rep" else (
            GATE_NAMES + [g.name for g in __import__(
                "luknn.dlm.gates", fromlist=["GATES_EXTRA"]
            ).GATES_EXTRA]
        )
        indices = self.logits.argmax(dim=-1).tolist()
        return [gate_names[i] for i in indices]

    def to_weight_matrix(self, input_names: list[str] | None = None) -> tuple[Tensor, Tensor]:
        """
        Convert crystallised layer to (weight_matrix, bias_vector) with integer values.

        weight_matrix : [n_neurons, fan_in]  — mostly zeros, at most 2 non-zero per row
        bias_vector   : [n_neurons]

        Only valid for gate_set='rep'.  Raises ValueError for non-representable gates.
        """
        from .gates import gate_weights, GATES_EXTRA

        if self.gate_set != "rep":
            raise ValueError(
                "to_weight_matrix() requires gate_set='rep'. "
                "Non-representable gates have no (w1,w2,b) integer representation."
            )

        gate_names_full = GATE_NAMES
        indices = self.logits.argmax(dim=-1)  # [n_neurons]

        W = torch.zeros(self.n_neurons, self.fan_in)
        bias = torch.zeros(self.n_neurons)

        for neuron_idx in range(self.n_neurons):
            gate_name = gate_names_full[indices[neuron_idx].item()]
            w1, w2, b = gate_weights(gate_name)
            i0 = int(self.pair_indices[neuron_idx, 0].item())
            i1 = int(self.pair_indices[neuron_idx, 1].item())
            W[neuron_idx, i0] = float(w1)
            W[neuron_idx, i1] = float(w2)
            bias[neuron_idx] = float(b)

        return W, bias

    def representability_report(self) -> dict:
        """
        After crystallisation, check representability of each neuron.

        For gate_set='rep': always 100% representable (guaranteed by design).
        For gate_set='full': counts non-representable gates.

        Returns dict with keys: total, representable, fraction, gate_counts.
        """
        from .gates import GATES_EXTRA

        extra_names = {g.name for g in GATES_EXTRA}
        gate_names_all = GATE_NAMES if self.gate_set == "rep" else (
            GATE_NAMES + list(extra_names)
        )
        selected = [gate_names_all[i] for i in self.logits.argmax(dim=-1).tolist()]
        n_rep = sum(1 for g in selected if g not in extra_names)
        gate_counts: dict[str, int] = {}
        for g in selected:
            gate_counts[g] = gate_counts.get(g, 0) + 1

        return {
            "total": self.n_neurons,
            "representable": n_rep,
            "fraction": n_rep / max(self.n_neurons, 1),
            "gate_counts": gate_counts,
        }

    def extra_repr(self) -> str:
        return (
            f"fan_in={self.fan_in}, n_neurons={self.n_neurons}, "
            f"n_gates={self.n_gates}, gate_set={self.gate_set!r}, "
            f"T={self.temperature:.3f}"
        )
