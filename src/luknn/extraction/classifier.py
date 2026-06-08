"""
Neuron configuration classification.

After crystallization every neuron has integer weights and bias.
Proposition 3 of Leandro (ALT 2009) gives the classification rule:

Given  α = ψ_b(−x_1,…,−x_n, x_{n+1},…,x_m)
  with n negative weights and p positive weights (m = n + p):

  • Conjunction  iff  b = −p + 1  (i.e. b = −(m−1) + n)
  • Disjunction  iff  b = n

A neuron is *representable* if it is a conjunction or disjunction (or constant).
Otherwise it is *un-representable* and must be approximated.
"""

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class NeuronKind(str, Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    CONSTANT_ZERO = "constant_zero"
    CONSTANT_ONE = "constant_one"
    UNREPRESENTABLE = "unrepresentable"


@dataclass
class NeuronConfig:
    weights: list[float]   # integer values (-1 or 1 after crystallization)
    bias: float
    kind: NeuronKind
    formula: str | None = None


def classify_neuron(weights: Tensor, bias: Tensor) -> NeuronConfig:
    """
    Classify a single crystallized neuron.

    weights : 1-D tensor of integer values (typically -1, 0, 1)
    bias    : scalar tensor
    """
    w = weights.round().int().tolist()
    b = int(bias.round().item())

    n_neg = sum(1 for wi in w if wi < 0)   # negative weights
    n_pos = sum(1 for wi in w if wi > 0)   # positive weights
    m = n_neg + n_pos

    if m == 0:
        # Bias-only neuron — constant
        kind = NeuronKind.CONSTANT_ONE if b >= 1 else NeuronKind.CONSTANT_ZERO
        return NeuronConfig(w, b, kind, formula="1" if b >= 1 else "0")

    b_conj = -n_pos + 1     # conjunction condition: b = -(m-1)+n = -p+1
    b_disj = n_neg          # disjunction condition: b = n

    if b == b_conj:
        kind = NeuronKind.CONJUNCTION
        formula = _build_formula(w, "otimes")
    elif b == b_disj:
        kind = NeuronKind.DISJUNCTION
        formula = _build_formula(w, "oplus")
    else:
        kind = NeuronKind.UNREPRESENTABLE
        formula = None

    return NeuronConfig(w, b, kind, formula)


def _build_formula(weights: list[int], op: str) -> str:
    """Build a symbolic formula string from integer weights."""
    terms = []
    for i, wi in enumerate(weights):
        var = f"x{i+1}"
        terms.append(f"¬{var}" if wi < 0 else var)
    sym = " ⊗ " if op == "otimes" else " ⊕ "
    return sym.join(terms)
