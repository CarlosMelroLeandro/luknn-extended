"""
Representable Łukasiewicz gate set G_rep.

Every gate in G_rep corresponds to a single neuron ψ_b(w₁a, w₂b) with
w_i ∈ {-1, 0, +1} and b ∈ ℤ that implements a valid Łukasiewicz connective.
Source: Leandro (2009) Table 1 + Proposition 3.

The set contains 12 binary (2-input) gates.  XOR and XNOR are *excluded*
because they are not implementable by any single ψ_b neuron with ±1 weights —
they require at least 2 neurons in Łukasiewicz logic.

Each entry in GATES stores:
  name    : short identifier
  w1, w2  : integer weights (-1, 0, or +1)
  b       : integer bias
  symbol  : Łukasiewicz notation
  verify  : truth-table check values for (a,b) ∈ {(0,0),(0,1),(1,0),(1,1)}

Non-representable gate set for DLN comparison (G_full = G_rep ∪ G_extra):
  XOR  : |a - b|              (not a single ψ_b neuron)
  XNOR : 1 - |a - b|         (not representable)
  GMIN : min(a, b)            (Gödel AND — requires different logic)
  GMAX : max(a, b)            (Gödel OR  — requires different logic)
"""

from __future__ import annotations
from typing import NamedTuple

import torch
from torch import Tensor


class GateSpec(NamedTuple):
    name: str
    w1: int           # weight on first input  ∈ {-1, 0, +1}
    w2: int           # weight on second input ∈ {-1, 0, +1}
    b: int            # integer bias
    symbol: str       # symbolic name for formula display


# ── Representable gate set G_rep ─────────────────────────────────────────────
#
# Proof of representability (Proposition 3 applied to each gate):
#   n_neg = #{i : w_i < 0},  n_pos = #{i : w_i > 0}
#   Conjunction iff b = -n_pos + 1
#   Disjunction iff b = n_neg
#
# CONJ:  n_pos=2, n_neg=0 → b_conj=-1 ✓  (b=-1)
# DISJ:  n_pos=2, n_neg=0 → b_disj=0  ✓  (b=0)
# IMP:   n_pos=1, n_neg=1 → b_disj=1  ✓  (b=+1, disjunction of b and ¬a)
# RIMP:  n_pos=1, n_neg=1 → b_disj=1  ✓  (b=+1, disjunction of a and ¬b)
# NCONJ: n_pos=0, n_neg=2 → b_disj=2  ✓  (b=+2, disjunction of ¬a and ¬b)
# NDISJ: n_pos=0, n_neg=2 → b_conj=1  ✓  (b=+1, conjunction of ¬a and ¬b)
# ANEG:  n_pos=1, n_neg=1 → b_conj=0  ✓  (b=0,  conjunction of a and ¬b)
# BNEG:  n_pos=1, n_neg=1 → b_conj=0  ✓  (b=0,  conjunction of ¬a and b)
# NEGA:  n_pos=0, n_neg=1 → b_disj=1  ✓  (b=+1, unary: ¬a)
# NEGB:  n_pos=0, n_neg=1 → b_disj=1  ✓  (b=+1, unary: ¬b)
# PRJA:  n_pos=1, n_neg=0 → b_disj=0  ✓  (b=0,  unary: a)
# PRJB:  n_pos=1, n_neg=0 → b_disj=0  ✓  (b=0,  unary: b)

GATES: list[GateSpec] = [
    #         name      w1   w2   b    symbol
    GateSpec("CONJ",   +1,  +1,  -1,  "a ⊗ b"),       # max(0, a+b-1)
    GateSpec("DISJ",   +1,  +1,   0,  "a ⊕ b"),       # min(1, a+b)
    GateSpec("IMP",    -1,  +1,  +1,  "a ⟹ b"),      # min(1, 1-a+b)
    GateSpec("RIMP",   +1,  -1,  +1,  "b ⟹ a"),      # min(1, 1+a-b)
    GateSpec("NCONJ",  -1,  -1,  +2,  "¬(a ⊗ b)"),   # min(1, 2-a-b)
    GateSpec("NDISJ",  -1,  -1,  +1,  "¬(a ⊕ b)"),   # max(0, 1-a-b)
    GateSpec("ANEG",   +1,  -1,   0,  "a ⊗ ¬b"),     # max(0, a-b)
    GateSpec("BNEG",   -1,  +1,   0,  "¬a ⊗ b"),     # max(0, b-a)
    GateSpec("NEGA",   -1,   0,  +1,  "¬a"),          # 1-a
    GateSpec("NEGB",    0,  -1,  +1,  "¬b"),          # 1-b
    GateSpec("PRJA",   +1,   0,   0,  "a"),            # a
    GateSpec("PRJB",    0,  +1,   0,  "b"),            # b
]

GATE_NAMES: list[str] = [g.name for g in GATES]
N_GATES: int = len(GATES)  # 12

# ── Non-representable gates for G_full (DLN comparison) ──────────────────────

GATES_EXTRA: list[GateSpec] = [
    GateSpec("XOR",   0, 0,  0,  "|a-b|"),             # non-Ł
    GateSpec("XNOR",  0, 0,  0,  "1-|a-b|"),           # non-Ł
    GateSpec("GMIN",  0, 0,  0,  "min(a,b)"),           # Gödel AND
    GateSpec("GMAX",  0, 0,  0,  "max(a,b)"),           # Gödel OR
]

GATES_FULL: list[GateSpec] = GATES + GATES_EXTRA
N_GATES_FULL: int = len(GATES_FULL)  # 16


# ── Gate function implementations ─────────────────────────────────────────────

def _conj(a: Tensor, b: Tensor) -> Tensor:   return (a + b - 1.0).clamp(0.0, 1.0)
def _disj(a: Tensor, b: Tensor) -> Tensor:   return (a + b).clamp(0.0, 1.0)
def _imp(a: Tensor, b: Tensor) -> Tensor:    return (1.0 - a + b).clamp(0.0, 1.0)
def _rimp(a: Tensor, b: Tensor) -> Tensor:   return (1.0 + a - b).clamp(0.0, 1.0)
def _nconj(a: Tensor, b: Tensor) -> Tensor:  return (2.0 - a - b).clamp(0.0, 1.0)
def _ndisj(a: Tensor, b: Tensor) -> Tensor:  return (1.0 - a - b).clamp(0.0, 1.0)
def _aneg(a: Tensor, b: Tensor) -> Tensor:   return (a - b).clamp(0.0, 1.0)
def _bneg(a: Tensor, b: Tensor) -> Tensor:   return (b - a).clamp(0.0, 1.0)
def _nega(a: Tensor, b: Tensor) -> Tensor:   return 1.0 - a
def _negb(a: Tensor, b: Tensor) -> Tensor:   return 1.0 - b
def _prja(a: Tensor, b: Tensor) -> Tensor:   return a
def _prjb(a: Tensor, b: Tensor) -> Tensor:   return b

# Non-representable (for G_full / DLN comparison)
def _xor(a: Tensor, b: Tensor) -> Tensor:    return (a - b).abs()
def _xnor(a: Tensor, b: Tensor) -> Tensor:   return 1.0 - (a - b).abs()
def _gmin(a: Tensor, b: Tensor) -> Tensor:   return torch.min(a, b)
def _gmax(a: Tensor, b: Tensor) -> Tensor:   return torch.max(a, b)


_REP_FNS = [_conj, _disj, _imp, _rimp, _nconj, _ndisj,
             _aneg, _bneg, _nega, _negb, _prja, _prjb]

_EXTRA_FNS = [_xor, _xnor, _gmin, _gmax]

_FULL_FNS = _REP_FNS + _EXTRA_FNS


# ── STE variants for training ─────────────────────────────────────────────────
#
# For binary {0,1} inputs, every Łukasiewicz gate output is always AT the
# clamp boundary (0 or 1), giving ∂gate/∂input = 0.  With standard clamp,
# gradient cannot propagate through hidden layers.
#
# Fix: STE (Straight-Through Estimator) through clamp(0,1).  Forward is exact
# (Łukasiewicz semantics preserved); backward passes gradient straight through
# the clamp — equivalent to treating clamp as identity for differentiation.

class _STE_Clamp01(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        return grad  # pass straight through


def _ste(x: Tensor) -> Tensor:
    return _STE_Clamp01.apply(x)


def _conj_ste(a: Tensor, b: Tensor) -> Tensor:  return _ste(a + b - 1.0)
def _disj_ste(a: Tensor, b: Tensor) -> Tensor:  return _ste(a + b)
def _imp_ste(a: Tensor, b: Tensor) -> Tensor:   return _ste(1.0 - a + b)
def _rimp_ste(a: Tensor, b: Tensor) -> Tensor:  return _ste(1.0 + a - b)
def _nconj_ste(a: Tensor, b: Tensor) -> Tensor: return _ste(2.0 - a - b)
def _ndisj_ste(a: Tensor, b: Tensor) -> Tensor: return _ste(1.0 - a - b)
def _aneg_ste(a: Tensor, b: Tensor) -> Tensor:  return _ste(a - b)
def _bneg_ste(a: Tensor, b: Tensor) -> Tensor:  return _ste(b - a)
# Linear in inputs — clamp never saturates when inputs ∈ [0,1]; no STE needed
def _nega_ste(a: Tensor, b: Tensor) -> Tensor:  return 1.0 - a
def _negb_ste(a: Tensor, b: Tensor) -> Tensor:  return 1.0 - b
def _prja_ste(a: Tensor, b: Tensor) -> Tensor:  return a
def _prjb_ste(a: Tensor, b: Tensor) -> Tensor:  return b

_REP_STE_FNS = [
    _conj_ste, _disj_ste, _imp_ste,  _rimp_ste,
    _nconj_ste, _ndisj_ste, _aneg_ste, _bneg_ste,
    _nega_ste, _negb_ste, _prja_ste, _prjb_ste,
]

# Non-representable extras use abs()/min()/max() — already differentiable.
_FULL_STE_FNS = _REP_STE_FNS + _EXTRA_FNS


def gate_fn(name: str) -> "callable":
    """Return the gate function for the given gate name."""
    try:
        idx = GATE_NAMES.index(name)
        return _REP_FNS[idx]
    except ValueError:
        extra_names = [g.name for g in GATES_EXTRA]
        idx = extra_names.index(name)
        return _EXTRA_FNS[idx]


def gate_weights(name: str) -> tuple[int, int, int]:
    """Return (w1, w2, b) for a representable gate by name.

    Raises ValueError for non-representable gates (no integer weights exist).
    """
    for g in GATES:
        if g.name == name:
            return g.w1, g.w2, g.b
    raise ValueError(f"Gate {name!r} is not in G_rep or has no (w1,w2,b) representation")


def apply_all_gates(
    a: Tensor,
    b: Tensor,
    gate_set: str = "rep",
    ste: bool = False,
) -> Tensor:
    """
    Apply all gates to inputs a, b and return stacked outputs.

    Parameters
    ----------
    a, b      : Tensor of shape [batch, n_neurons]
    gate_set  : 'rep' (12 gates) or 'full' (16 gates including non-Ł)
    ste       : if True, use STE-through-clamp variants for gradient flow

    Returns
    -------
    Tensor of shape [n_gates, batch, n_neurons]
    """
    if ste:
        fns = _REP_STE_FNS if gate_set == "rep" else _FULL_STE_FNS
    else:
        fns = _REP_FNS if gate_set == "rep" else _FULL_FNS
    return torch.stack([f(a, b) for f in fns], dim=0)


def verify_gate_table() -> dict[str, bool]:
    """
    Verify each gate against its expected 3-valued Łukasiewicz truth table.
    Returns {gate_name: passed} for all gates in G_rep.
    S_2 = {0, 0.5, 1}; checks (a,b) ∈ {0,1}² exactly.
    """
    test_cases = [
        (torch.tensor(0.0), torch.tensor(0.0)),
        (torch.tensor(0.0), torch.tensor(1.0)),
        (torch.tensor(1.0), torch.tensor(0.0)),
        (torch.tensor(1.0), torch.tensor(1.0)),
    ]
    # Expected: ψ_b(w1*a + w2*b) computed directly
    results = {}
    for i, (gate, fn) in enumerate(zip(GATES, _REP_FNS)):
        ok = True
        for a, b in test_cases:
            expected = (gate.w1 * a + gate.w2 * b + gate.b).clamp(0.0, 1.0)
            got = fn(a, b)
            if abs(float(expected) - float(got)) > 1e-6:
                ok = False
                break
        results[gate.name] = ok
    return results
