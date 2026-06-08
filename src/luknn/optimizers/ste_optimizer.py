"""
STEOptimizer — Straight-Through Estimator with ternary weight quantization.

Strategy
--------
Forward : weight tensor uses hard_snap({-1,0,1}) via STE (in LukasiewiczLinear).
Backward: gradient flows through as if weights were continuous (STE).
Adam/SGD updates the continuous weight store w_cont.
Post-training: hard crystallization snaps w_cont → {-1,0,1}.

Why STE works here
------------------
LukasiewiczLinear(mode='ste') computes:
    w_ternary = hard_snap(weight)
    w_ste     = (w_ternary - weight).detach() + weight
In forward: w_ste ≡ w_ternary (ternary values used in linear computation).
In backward: ∂loss/∂weight = ∂loss/∂w_ste (identity — STE assumption).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseOptimizer, TrainingResult
from ..layers.lukasiewicz_linear import LukasiewiczNet


class STEOptimizer(BaseOptimizer):
    """
    Adam + STE ternary quantization for ŁNNs.

    Parameters
    ----------
    model : LukasiewiczNet  (mode='ste')
    lr : float              Adam learning rate.
    weight_lr : float       Separate LR for biases (None → same as lr).
    clip_grad : float       Gradient norm clipping (0 = disabled).
    """

    def __init__(
        self,
        model: LukasiewiczNet,
        lr: float = 5e-3,
        weight_lr: float | None = None,
        clip_grad: float = 1.0,
    ):
        assert all(
            getattr(layer, "mode", None) == "ste"
            for layer in model.layers
        ), "STEOptimizer requires LukasiewiczNet with mode='ste'"

        self.model = model
        self.clip_grad = clip_grad

        # Optionally use different LR for biases
        if weight_lr is None:
            self.inner = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            weight_params = [p for n, p in model.named_parameters() if "weight" in n]
            bias_params = [p for n, p in model.named_parameters() if "bias" in n]
            self.inner = torch.optim.Adam([
                {"params": weight_params, "lr": lr},
                {"params": bias_params, "lr": weight_lr},
            ])

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 2000,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()
        mse_history: list[float] = []
        best_mse = float("inf")
        stagnation = 0
        patience = max(50, max_iter // 20)

        # Keep a checkpoint of the best model state (continuous weights)
        best_state: dict | None = None

        # Cosine-annealed LR: prevents the oscillation typical of STE with
        # fixed LR by reducing step size as the ternary network improves.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.inner, T_max=max_iter, eta_min=1e-5
        )

        for it in range(max_iter):
            self.inner.zero_grad()
            pred = self.model(x)
            if sample_weight is not None:
                loss = ((pred - y) ** 2 * sample_weight).mean()
            else:
                loss = F.mse_loss(pred, y)
            loss.backward()

            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.inner.step()
            scheduler.step()

            with torch.no_grad():
                for layer in self.model.layers:
                    layer.weight.data.clamp_(-1.5, 1.5)

            mse = loss.item()
            mse_history.append(mse)

            if mse < best_mse - 1e-6:
                best_mse = mse
                # Save continuous weight state at best ternary MSE
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break

            if verbose and it % 200 == 0:
                print(f"  STE iter {it:4d}  mse={mse:.6f}  best={best_mse:.6f}")

            if best_mse < tol_mse:
                break

        # Restore best weights before crystallizing
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Crystallize using hard_snap (matches forward pass convention)
        self.model.crystallize()

        final_mse = self._mse(self.model(x), y)
        converged = final_mse < tol_mse
        elapsed = time.perf_counter() - t0

        return TrainingResult(
            converged=converged,
            final_mse=final_mse,
            mse_history=mse_history,
            iterations=len(mse_history),
            total_time_s=elapsed,
            reason="converged" if converged else "stagnation",
        )
