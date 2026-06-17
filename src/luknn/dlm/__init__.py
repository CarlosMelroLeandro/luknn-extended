"""
Differentiable Łukasiewicz Machine (DLM).

Architecture: multi-layer network where each neuron is a 2-input gate drawn
from the representable Łukasiewicz gate set G_rep (12 gates). During training,
each neuron holds a softmax distribution over G_rep; after crystallisation the
argmax gate is selected, giving a fully integer-weight Castro neural network
whose representability is guaranteed by construction.

Key difference from DLN (Nguy & Wasilewski 2025): DLM uses *exact* Łukasiewicz
operators (no fuzzy approximation) and restricts to gates provably implementable
as ψ_b(w₁a, w₂b) with w_i ∈ {-1,0,+1}, b ∈ ℤ (Proposition 3, Leandro 2009).
"""

from .gates import GATES, GATE_NAMES, N_GATES, gate_fn, gate_weights
from .gate_layer import GateLayer
from .network import DLMNetwork, make_dlm_net
from .optimizer import DLMOptimizer

__all__ = [
    "GATES", "GATE_NAMES", "N_GATES", "gate_fn", "gate_weights",
    "GateLayer",
    "DLMNetwork", "make_dlm_net",
    "DLMOptimizer",
]
