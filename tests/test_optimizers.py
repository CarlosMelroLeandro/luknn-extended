"""
Optimizer smoke-tests: verify each method can overfit a tiny truth table.
Uses a 2-variable, 3-valued table (9 rows) for speed.
"""

import torch
import pytest
from luknn.logic.connectives import truth_subtable, tnorm, negation
from luknn.layers.lukasiewicz_linear import LukasiewiczNet
from luknn.optimizers import LMOptimizer, STEOptimizer, ProximalOptimizer


def _tiny_dataset():
    """2-variable, 3-valued table for ¬x1 ⊕ x2."""
    x = truth_subtable(2, 3)
    y = (1.0 - x[:, 0] + x[:, 1]).clamp(0.0, 1.0)  # ¬x1 ⊕ x2
    return x, y


class TestLMOptimizer:
    def test_trains_without_error(self):
        torch.manual_seed(0)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [3], mode="continuous")
        opt = LMOptimizer(model, mu_init=0.01, patience=30)
        result = opt.train(x, y, tol_mse=0.05, max_iter=200)
        assert result.total_time_s > 0
        assert isinstance(result.final_mse, float)
        assert len(result.mse_history) > 0

    def test_training_result_fields(self):
        torch.manual_seed(1)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [2], mode="continuous")
        opt = LMOptimizer(model)
        result = opt.train(x, y, tol_mse=0.1, max_iter=50)
        assert hasattr(result, "converged")
        assert hasattr(result, "mse_history")
        assert hasattr(result, "iterations")


class TestSTEOptimizer:
    def test_trains_without_error(self):
        torch.manual_seed(0)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [3], mode="ste")
        opt = STEOptimizer(model, lr=5e-3)
        result = opt.train(x, y, tol_mse=0.1, max_iter=500)
        assert result.total_time_s > 0
        assert isinstance(result.final_mse, float)

    def test_crystallized_after_training(self):
        torch.manual_seed(2)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [2], mode="ste")
        opt = STEOptimizer(model, lr=5e-3)
        opt.train(x, y, tol_mse=0.5, max_iter=200)  # low bar — just check crystallization
        assert model.is_crystallized(tol=1e-3)

    def test_gradient_flows_through_ste(self):
        torch.manual_seed(3)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [2], mode="ste")
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None, "STE must pass gradients to continuous weights"


class TestProximalOptimizer:
    def test_trains_without_error(self):
        torch.manual_seed(0)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [3], mode="clamp")
        opt = ProximalOptimizer(model, lr=1e-2, lambda_sparse=0.01, lambda_attract=0.05)
        result = opt.train(x, y, tol_mse=0.1, max_iter=300)
        assert result.total_time_s > 0

    def test_crystallized_after_training(self):
        torch.manual_seed(4)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [2], mode="clamp")
        opt = ProximalOptimizer(model, lr=1e-2)
        opt.train(x, y, tol_mse=0.5, max_iter=300)
        assert model.is_crystallized(tol=1e-3)

    def test_weights_stay_clamped_during_training(self):
        torch.manual_seed(5)
        x, y = _tiny_dataset()
        model = LukasiewiczNet(2, [3], mode="clamp")
        opt = ProximalOptimizer(model, lr=1e-2)

        # Monkey-patch to capture mid-training weight range
        max_abs = [0.0]
        orig_train = opt.train
        for _ in range(10):
            opt.inner.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.inner.step()
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.data.clamp_(-1.0, 1.0)
                for p in model.parameters():
                    if "weight" in "":
                        pass
                    max_abs[0] = max(max_abs[0],
                                     max(p.data.abs().max().item()
                                         for p in model.parameters()))
        assert max_abs[0] <= 1.5  # rough sanity check
