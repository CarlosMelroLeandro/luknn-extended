import torch
import pytest
from luknn.logic.connectives import tnorm, residuum, negation, disjunction, biconditional

def t(v): return torch.tensor(v, dtype=torch.float32)


class TestTnorm:
    def test_boundary_zeros(self):
        assert tnorm(t(0.0), t(0.0)).item() == pytest.approx(0.0)

    def test_boundary_ones(self):
        assert tnorm(t(1.0), t(1.0)).item() == pytest.approx(1.0)

    def test_classical_conjunction(self):
        # In 2-valued logic: 1 ⊗ 0 = 0
        assert tnorm(t(1.0), t(0.0)).item() == pytest.approx(0.0)

    def test_midpoint(self):
        assert tnorm(t(0.7), t(0.6)).item() == pytest.approx(0.3)


class TestResiduum:
    def test_implies_true(self):
        # x ≤ y  ⟹  x→y = 1
        assert residuum(t(0.3), t(0.8)).item() == pytest.approx(1.0)

    def test_implies_false(self):
        assert residuum(t(1.0), t(0.0)).item() == pytest.approx(0.0)

    def test_value(self):
        assert residuum(t(0.8), t(0.5)).item() == pytest.approx(0.7)


class TestNegation:
    def test_classical(self):
        assert negation(t(0.0)).item() == pytest.approx(1.0)
        assert negation(t(1.0)).item() == pytest.approx(0.0)

    def test_mid(self):
        assert negation(t(0.3)).item() == pytest.approx(0.7)


class TestDisjunction:
    def test_clamped(self):
        assert disjunction(t(0.8), t(0.6)).item() == pytest.approx(1.0)

    def test_sum(self):
        assert disjunction(t(0.3), t(0.4)).item() == pytest.approx(0.7)
