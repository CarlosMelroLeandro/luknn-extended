import torch
import pytest
from luknn.network.luknn import LukNN, make_network
from luknn.extraction.classifier import classify_neuron, NeuronKind


class TestLukNN:
    def test_output_in_unit_interval(self):
        model = make_network(3, n_hidden_layers=2, hidden_width=4)
        x = torch.rand(20, 3)
        y = model(x)
        assert y.min().item() >= 0.0 - 1e-6
        assert y.max().item() <= 1.0 + 1e-6

    def test_flat_weights_roundtrip(self):
        model = make_network(2, n_hidden_layers=1, hidden_width=2)
        w = model.flat_weights().clone()
        w_noisy = w + 0.1
        model.load_flat_weights(w_noisy)
        w_back = model.flat_weights()
        assert torch.allclose(w_back, w_noisy)


class TestClassifier:
    def test_conjunction_2_inputs(self):
        # ψ_{-1}(x, y) = x ⊗ y  →  w=[1,1], b=-1
        cfg = classify_neuron(torch.tensor([1.0, 1.0]), torch.tensor(-1.0))
        assert cfg.kind == NeuronKind.CONJUNCTION

    def test_disjunction_2_inputs(self):
        # ψ_0(x, y) = x ⊕ y  →  w=[1,1], b=0
        cfg = classify_neuron(torch.tensor([1.0, 1.0]), torch.tensor(0.0))
        assert cfg.kind == NeuronKind.DISJUNCTION

    def test_negated_conjunction(self):
        # ψ_0(-x, y) = ¬x ⊗ y  →  n=1, p=1, b=-p+1=0  → conjunction
        cfg = classify_neuron(torch.tensor([-1.0, 1.0]), torch.tensor(0.0))
        assert cfg.kind == NeuronKind.CONJUNCTION

    def test_unrepresentable(self):
        # ψ_0(-x, y, z)  n=1, p=2, b_conj=-1, b_disj=1; b=0 → unrepresentable
        cfg = classify_neuron(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor(0.0))
        assert cfg.kind == NeuronKind.UNREPRESENTABLE
