"""
ProximalOptimizer — two-phase: MSE convergence → ternary regularization.

Phase 1 (2/3 of budget): pure MSE minimization.
  Learns the right function without regularization killing the weights.

Phase 2 (1/3 of budget): add ternary regularization with linear warm-up.
  λ grows from 0 → λ_target so weights are gently attracted to {-1,0,1}.

Phase 3 (hardening): short burst with 10× regularization to push weights
  very close to integers before crisp crystallization.

Why this works better than always-on regularization
----------------------------------------------------
With L1 active from the start, all weights collapse to 0 immediately
(constant-output network, MSE ≈ 0.16) because the regularization gradient
dominates the MSE gradient when weights are small. Separating the phases
lets Phase 1 find a genuine low-MSE solution first.
"""

import time
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseOptimizer, TrainingResult
from ..layers.lukasiewicz_linear import LukasiewiczNet


def _ternary_penalty(w: Tensor) -> Tensor:
    return (w.pow(2) * (1.0 - w.pow(2)).clamp(min=0.0)).sum()


def _ternary_regularization(model, lambda_sparse: float, lambda_attract: float) -> Tensor:
    reg = torch.tensor(0.0)
    for name, p in model.named_parameters():
        if "weight" in name:
            reg = reg + lambda_sparse * p.abs().sum()
            reg = reg + lambda_attract * _ternary_penalty(p)
    return reg


def _soft_threshold(w: Tensor, threshold: float) -> Tensor:
    return w.sign() * (w.abs() - threshold).clamp(min=0.0)


class ProximalOptimizer(BaseOptimizer):
    """
    Two-phase optimizer: MSE-only → ternary regularization.

    Parameters
    ----------
    model : LukasiewiczNet  (mode='clamp')
    lr : float
    lambda_sparse : float   L1 coeff (active only in Phase 2).
    lambda_attract : float  Ternary-attraction coeff (active in Phase 2).
    prox_threshold : float  Soft-threshold applied after each Phase-2 step.
    phase1_fraction : float Fraction of max_iter used for Phase 1 (default 0.6).
    """

    def __init__(
        self,
        model: LukasiewiczNet,
        lr: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        phase1_fraction: float = 0.6,
    ):
        assert all(getattr(layer, "mode", None) == "clamp" for layer in model.layers), \
            "ProximalOptimizer requires LukasiewiczNet with mode='clamp'"

        self.model = model
        self.lambda_sparse = lambda_sparse
        self.lambda_attract = lambda_attract
        self.prox_threshold = prox_threshold
        self.phase1_fraction = phase1_fraction
        self.inner = torch.optim.Adam(model.parameters(), lr=lr)

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 3000,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()
        mse_history: list[float] = []
        best_mse = float("inf")
        stagnation = 0
        patience = max(100, max_iter // 10)

        def _mse_loss(pred: Tensor) -> Tensor:
            if sample_weight is not None:
                return ((pred - y) ** 2 * sample_weight).mean()
            return F.mse_loss(pred, y)

        phase1_end = int(max_iter * self.phase1_fraction)

        # ── Phase 1: MSE only ─────────────────────────────────────────────
        for it in range(phase1_end):
            self.inner.zero_grad()
            pred = self.model(x)
            loss = _mse_loss(pred)
            loss.backward()
            self.inner.step()
            self._project()

            mse = loss.item()
            mse_history.append(mse)

            if verbose and it % 300 == 0:
                print(f"  Prox P1 iter {it:4d}  mse={mse:.6f}")

            if mse < tol_mse:
                break
            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break

        if verbose:
            print(f"  Phase 1 done: mse={best_mse:.5f} after {len(mse_history)} iters")

        # ── Phase 2: MSE + ternary regularization (linear warm-up) ───────
        phase2_iters = max_iter - phase1_end
        stagnation = 0
        patience2 = max(50, phase2_iters // 5)

        for step in range(phase2_iters):
            scale = (step + 1) / phase2_iters        # 0 → 1
            ls = self.lambda_sparse * scale
            la = self.lambda_attract * scale

            self.inner.zero_grad()
            pred = self.model(x)
            mse_loss = _mse_loss(pred)
            reg = _ternary_regularization(self.model, ls, la)
            (mse_loss + reg).backward()
            self.inner.step()

            if self.prox_threshold > 0:
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if "weight" in name:
                            p.data = _soft_threshold(p.data, self.prox_threshold * scale)

            self._project()

            mse = mse_loss.item()
            mse_history.append(mse)

            if verbose and step % 300 == 0:
                print(f"  Prox P2 step {step:4d}  mse={mse:.6f}  reg={reg.item():.5f}")

            if mse < tol_mse:
                break
            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience2:
                    break

        # ── Phase 3: hardening (10× reg, no MSE grad clipping) ───────────
        if best_mse < 0.15:
            for step in range(200):
                self.inner.zero_grad()
                pred = self.model(x)
                mse_loss = _mse_loss(pred)
                reg = _ternary_regularization(
                    self.model, self.lambda_sparse * 10, self.lambda_attract * 10
                )
                (mse_loss + reg).backward()
                self.inner.step()
                self._project()

        # Progressive crystallization
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

    def _project(self) -> None:
        with torch.no_grad():
            for layer in self.model.layers:
                layer.weight.data.clamp_(-1.0, 1.0)
