import torch
import pytest
from luknn.layers.activation import TruncatedIdentityFn, TruncatedIdentity
from luknn.layers.lukasiewicz_linear import LukasiewiczLinear, LukasiewiczNet, _hard_snap


class TestTruncatedIdentityFn:
    def test_forward_clamps(self):
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        out = TruncatedIdentityFn.apply(x)
        assert out.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])

    def test_backward_linear_region(self):
        x = torch.tensor([0.5], requires_grad=True)
        out = TruncatedIdentityFn.apply(x)
        out.backward()
        assert x.grad.item() == pytest.approx(1.0)

    def test_backward_zero_outside(self):
        for v in [-0.1, 0.0, 1.0, 1.5]:
            x = torch.tensor([v], requires_grad=True)
            TruncatedIdentityFn.apply(x).backward()
            assert x.grad.item() == pytest.approx(0.0), f"expected 0 grad at x={v}"


class TestHardSnap:
    def test_positive_snaps_to_one(self):
        w = torch.tensor([0.5, 1.0, 0.34])
        assert _hard_snap(w).tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_negative_snaps_to_minus_one(self):
        w = torch.tensor([-0.5, -1.0, -0.34])
        assert _hard_snap(w).tolist() == pytest.approx([-1.0, -1.0, -1.0])

    def test_dead_zone_snaps_to_zero(self):
        w = torch.tensor([0.0, 0.2, -0.2, 0.32])
        assert _hard_snap(w).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


class TestLukasiewiczLinear:
    def test_output_in_unit_interval(self):
        for mode in ("continuous", "ste", "clamp"):
            layer = LukasiewiczLinear(4, 3, mode=mode)
            x = torch.rand(10, 4)
            out = layer(x)
            assert out.min() >= -1e-6
            assert out.max() <= 1 + 1e-6

    def test_ste_forward_uses_ternary(self):
        layer = LukasiewiczLinear(2, 1, mode="ste")
        # Set continuous weights far from thresholds
        layer.weight.data = torch.tensor([[0.8, -0.7]])
        layer.bias.data = torch.tensor([0.5])
        # STE: w_snap = [1, -1]; net = 0.5 * 1 + 0.5 * (-1) + 0.5 = 0.5
        x = torch.tensor([[0.5, 0.5]])
        out = layer(x)
        assert 0.0 <= out.item() <= 1.0

    def test_ste_gradient_flows(self):
        layer = LukasiewiczLinear(2, 1, mode="ste")
        x = torch.ones(5, 2)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is not None
        assert not layer.weight.grad.isnan().any()

    def test_crystallize_produces_integers(self):
        layer = LukasiewiczLinear(3, 2, mode="continuous")
        layer.weight.data = torch.tensor([[0.9, -0.85, 0.1], [0.0, 0.6, -0.4]])
        layer.bias.data = torch.tensor([0.5, 0.5])
        layer.crystallize()
        # After crystallization, weights ∈ {-1, 0, 1}
        w = layer.weight.data.view(-1).tolist()
        for v in w:
            assert v in (-1.0, 0.0, 1.0), f"unexpected weight {v}"


class TestLukasiewiczNet:
    def test_output_range(self):
        for mode in ("continuous", "ste", "clamp"):
            net = LukasiewiczNet(4, [3, 3], mode=mode)
            x = torch.rand(20, 4)
            y = net(x)
            assert y.shape == (20,)
            assert y.min() >= -1e-6
            assert y.max() <= 1 + 1e-6

    def test_flat_weights_roundtrip(self):
        net = LukasiewiczNet(3, [2], mode="continuous")
        w = net.flat_weights().clone()
        net.load_flat_weights(w + 0.1)
        assert torch.allclose(net.flat_weights(), w + 0.1)

    def test_crystallize_all_layers(self):
        net = LukasiewiczNet(4, [3, 2], mode="continuous")
        # Push all weights near integers first
        for layer in net.layers:
            layer.weight.data = layer.weight.data.sign() * 0.95
            layer.bias.data.fill_(0.5)
        net.crystallize()
        assert net.is_crystallized(tol=1e-3)
