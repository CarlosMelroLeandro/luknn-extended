"""
Tests for LukResidualBlock, LukResidualNet and extract_formula_residual.

Covers:
  - Forward pass within [0, 1]
  - Crystallization of inner layers + merge bias
  - flat_weights / load_flat_weights (round-trip)
  - Formula extraction after crystallization: ⊕ and ⊗ correct
  - is_crystallized before and after crystallize()
  - 100% representability when bias ∈ {0, -1}
"""

import torch
import pytest
from luknn.layers.residual import LukResidualBlock
from luknn.network.residual_luknn import LukResidualNet
from luknn.extraction.residual_extractor import extract_formula_residual


# ── Helpers ───────────────────────────────────────────────────────────────────

def _force_merge_bias(block: LukResidualBlock, value: float) -> None:
    """Fix merge_bias to a given value for extraction testing."""
    with torch.no_grad():
        block.merge_bias.fill_(value)


def _force_inner_disjunction(block: LukResidualBlock) -> None:
    """Configure the first inner layer as disjunction x1 ⊕ x2 (for width=2)."""
    layer = block.inner_layers[0]
    with torch.no_grad():
        layer.weight.fill_(0.0)
        layer.weight[0, 0] = 1.0  # neuron 0: receives x_0 with weight +1
        layer.weight[1, 1] = 1.0  # neuron 1: receives x_1 with weight +1
        layer.bias.fill_(0.0)     # bias = n_neg = 0 → disjunction (identity here)


# ── Tests: LukResidualBlock ──────────────────────────────────────────────────

class TestLukResidualBlock:
    def test_output_in_unit_interval(self):
        block = LukResidualBlock(width=4, n_inner=1)
        x = torch.rand(10, 4)
        y = block(x)
        assert y.shape == (10, 4)
        assert (y >= 0.0).all() and (y <= 1.0).all()

    def test_crystallize_rounds_merge_bias(self):
        block = LukResidualBlock(width=3)
        with torch.no_grad():
            block.merge_bias.fill_(-0.4)   # should round to 0
        block.crystallize()
        assert (block.merge_bias.data == 0.0).all()

    def test_crystallize_rounds_inner_weights(self):
        block = LukResidualBlock(width=2)
        with torch.no_grad():
            block.inner_layers[0].weight.fill_(0.7)   # should round to 1
        block.crystallize()
        w = block.inner_layers[0].weight.data
        assert (w.abs() <= 1.0).all()
        assert ((w == 0.0) | (w == 1.0) | (w == -1.0)).all()

    def test_representation_error_zero_after_crystallize(self):
        block = LukResidualBlock(width=3)
        block.crystallize()
        assert block.representation_error() < 1e-6

    def test_is_crystallized(self):
        block = LukResidualBlock(width=2)
        assert not block.is_crystallized()   # continuous weights initially
        block.crystallize()
        assert block.is_crystallized()


# ── Tests: LukResidualNet ────────────────────────────────────────────────────

class TestLukResidualNet:
    def test_forward_shape(self):
        model = LukResidualNet(n_inputs=4, hidden_width=4, n_blocks=1)
        x = torch.rand(8, 4)
        y = model(x)
        assert y.shape == (8,)
        assert (y >= 0.0).all() and (y <= 1.0).all()

    def test_forward_with_proj(self):
        """n_inputs != hidden_width → automatic projection."""
        model = LukResidualNet(n_inputs=6, hidden_width=4, n_blocks=1)
        assert model.proj is not None
        x = torch.rand(5, 6)
        y = model(x)
        assert y.shape == (5,)

    def test_no_proj_when_same_width(self):
        model = LukResidualNet(n_inputs=4, hidden_width=4)
        assert model.proj is None

    def test_multiple_blocks(self):
        model = LukResidualNet(n_inputs=4, hidden_width=4, n_blocks=3)
        x = torch.rand(6, 4)
        y = model(x)
        assert y.shape == (6,)

    def test_flat_weights_round_trip(self):
        model = LukResidualNet(n_inputs=3, hidden_width=3, n_blocks=1)
        w = model.flat_weights().clone()
        model.load_flat_weights(w * 0.5)   # perturb
        model.load_flat_weights(w)          # restore
        assert torch.allclose(model.flat_weights(), w)

    def test_crystallize_makes_is_crystallized_true(self):
        model = LukResidualNet(n_inputs=3, hidden_width=3, n_blocks=1)
        assert not model.is_crystallized()
        model.crystallize()
        assert model.is_crystallized()


# ── Tests: formula extraction ──────────────────────────────────────────────────

class TestResidualExtraction:
    def _make_crystallized(self, merge_b: float) -> LukResidualNet:
        """2-input network, width=2, 1 block, no projection."""
        model = LukResidualNet(n_inputs=2, hidden_width=2, n_blocks=1, n_inner=1)
        # Configure inner layer as identity (diagonal weights 1, bias 0 → disjunction x_j)
        layer = model.blocks[0].inner_layers[0]
        with torch.no_grad():
            layer.weight.fill_(0.0)
            layer.weight[0, 0] = 1.0
            layer.weight[1, 1] = 1.0
            layer.bias.fill_(0.0)
        # Merge bias
        _force_merge_bias(model.blocks[0], merge_b)
        # Output layer: sum of both (disjunction), bias 0
        with torch.no_grad():
            model.output_layer.weight.fill_(1.0)
            model.output_layer.bias.fill_(0.0)
        model.crystallize()
        return model

    def test_disjunction_merge(self):
        model = self._make_crystallized(merge_b=0.0)
        result = extract_formula_residual(model, input_names=["a", "b"])
        assert result.representable
        assert "⊕" in result.formula

    def test_conjunction_merge(self):
        model = self._make_crystallized(merge_b=-1.0)
        result = extract_formula_residual(model, input_names=["a", "b"])
        assert result.representable
        assert "⊗" in result.formula

    def test_unrepresentable_merge_flagged(self):
        model = self._make_crystallized(merge_b=2.0)
        result = extract_formula_residual(model, input_names=["a", "b"])
        assert not result.representable

    def test_formula_contains_input_names(self):
        model = self._make_crystallized(merge_b=0.0)
        result = extract_formula_residual(model, input_names=["x1", "x2"])
        assert "x1" in result.formula or "x2" in result.formula

    def test_layer_formulas_populated(self):
        model = self._make_crystallized(merge_b=0.0)
        result = extract_formula_residual(model)
        assert len(result.layer_formulas) >= 2   # block + output
