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

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

from .base import BaseOptimizer, TrainingResult
from ..layers.lukasiewicz_linear import LukasiewiczLinear


def _collect_luk_layers(model: nn.Module) -> list[LukasiewiczLinear]:
    """Return all LukasiewiczLinear sub-modules in any ŁNN architecture."""
    return [m for m in model.modules() if isinstance(m, LukasiewiczLinear)]


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


def _delta_n_norm(model: nn.Module) -> float:
    """
    Fraction of weights stuck in the middle: 0.1 < |w| < 0.9.

    A weight has 'decided' when it is either near zero (pruned, |w| <= 0.1)
    or near ±1 (active, |w| >= 0.9).  Weights in between are stuck and will
    not crystallize cleanly.  Phase 2 should run until this fraction is ~0.

    Note: the previous implementation computed mean(abs(w)), which equals
    floor-distance only because weights are clamped to [-1,1].  That metric
    decreases whenever L1 pushes weights toward 0, so tol_dn was satisfied
    by zeroing everything rather than by reaching ternary proximity.
    """
    parts = []
    for name, p in model.named_parameters():
        if "weight" in name:
            w_abs = p.data.abs()
            stuck = ((w_abs > 0.1) & (w_abs < 0.9)).float()
            parts.append(stuck)
    if not parts:
        return 0.0
    return torch.cat([s.flatten() for s in parts]).mean().item()


class ProximalOptimizerOLD(BaseOptimizer):
    """
    Original two-phase optimizer (preserved for comparison).
    Phase 2 stops on MSE only — see ProximalOptimizer for the corrected version.

    Parameters
    ----------
    model : nn.Module       Any ŁNN (LukasiewiczNet or LukResidualNet) with mode='clamp'.
    lr : float
    lambda_sparse : float   L1 coeff (active only in Phase 2).
    lambda_attract : float  Ternary-attraction coeff (active in Phase 2).
    prox_threshold : float  Soft-threshold applied after each Phase-2 step.
    phase1_fraction : float Fraction of max_iter used for Phase 1 (default 0.6).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        phase1_fraction: float = 0.6,
    ):
        luk_layers = _collect_luk_layers(model)
        assert luk_layers and all(
            layer.mode == "clamp" for layer in luk_layers
        ), "ProximalOptimizer requires all LukasiewiczLinear layers with mode='clamp'"

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
            for layer in _collect_luk_layers(self.model):
                layer.weight.data.clamp_(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ProximalTopK
# ─────────────────────────────────────────────────────────────────────────────

def _apply_topk_mask(model: nn.Module, k: int) -> dict:
    """
    For each output neuron (row of weight matrix), zero all but the top-k
    weights by magnitude.  Returns the binary mask so Phase 2 can enforce it
    every step (preventing gradient from reactivating pruned connections).
    """
    masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name and p.dim() == 2 and p.shape[1] > k:
                threshold = p.data.abs().topk(k, dim=1).values[:, -1:]
                mask = (p.data.abs() >= threshold).float()
                p.data *= mask
                masks[name] = mask
            else:
                masks[name] = torch.ones_like(p.data)
    return masks


class ProximalTopK(BaseOptimizer):
    """
    ProximalOptimizer + top-k fan-in pruning between Phase 1 and Phase 2.

    After Phase 1 converges, each output neuron retains only its k largest-
    magnitude weights; the rest are zeroed and masked for the remainder of
    training.  This forces the surviving weights to be large enough to survive
    crystallization to ±1, breaking the distributed-small-weight trap that
    collapses standard Proximal on high-fan-in inputs (e.g. Mushroom one-hot).

    Parameters
    ----------
    k_per_neuron : int   Max active inputs per output neuron after pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        phase1_fraction: float = 0.6,
        k_per_neuron: int = 10,
    ):
        luk_layers = _collect_luk_layers(model)
        assert luk_layers and all(
            layer.mode == "clamp" for layer in luk_layers
        ), "ProximalTopK requires all LukasiewiczLinear layers with mode='clamp'"

        self.model = model
        self.lambda_sparse = lambda_sparse
        self.lambda_attract = lambda_attract
        self.prox_threshold = prox_threshold
        self.phase1_fraction = phase1_fraction
        self.k_per_neuron = k_per_neuron
        self.inner = torch.optim.Adam(model.parameters(), lr=lr)

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        tol_dn: float = 0.05,
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

        # ── Phase 1: MSE only, runs until stagnation ──────────────────────
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
                print(f"  TopK P1 iter {it:4d}  mse={mse:.6f}")

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break

        # ── Top-k pruning: keep only k active connections per neuron ──────
        masks = _apply_topk_mask(self.model, self.k_per_neuron)

        if verbose:
            n_total = sum(m.numel() for m in masks.values())
            n_active = sum(m.sum().item() for m in masks.values())
            print(
                f"  Phase 1 done: mse={best_mse:.5f}  stuck={_delta_n_norm(self.model):.4f}"
                f"  after {len(mse_history)} iters"
            )
            print(
                f"  Top-k mask applied (k={self.k_per_neuron}): "
                f"{int(n_active)}/{n_total} weights active "
                f"({100*n_active/n_total:.1f}%)"
            )

        # ── Phase 2: MSE + ternary regularization, dual stopping ──────────
        # Reset tracker: post-mask MSE is higher than Phase 1 MSE, so Phase 2
        # must build its own stagnation baseline from scratch.
        phase2_iters = max_iter - phase1_end
        best_mse = float("inf")
        stagnation = 0
        patience2 = max(50, phase2_iters // 5)

        for step in range(phase2_iters):
            scale = (step + 1) / phase2_iters
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

            # Re-enforce the top-k mask so pruned connections stay at zero
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in masks:
                        p.data *= masks[name]

            mse = mse_loss.item()
            mse_history.append(mse)

            if verbose and step % 300 == 0:
                dn = _delta_n_norm(self.model)
                print(f"  TopK P2 step {step:4d}  mse={mse:.6f}  stuck={dn:.4f}  reg={reg.item():.5f}")

            dn = _delta_n_norm(self.model)
            if mse < tol_mse and dn < tol_dn:
                break

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience2:
                    break

        # ── Phase 3: hardening ────────────────────────────────────────────
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
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if name in masks:
                            p.data *= masks[name]

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
            for layer in _collect_luk_layers(self.model):
                layer.weight.data.clamp_(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Corrected ProximalOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class ProximalOptimizer(BaseOptimizer):
    """
    Two-phase optimizer with dual stopping condition.

    Fix over ProximalOptimizerOLD
    --------------------------------
    Phase 1 runs until stagnation (no premature tol_mse break), ensuring
    weights reach their continuous optimum before regularization starts.

    Phase 2 stops when BOTH conditions hold:
        mse  < tol_mse   (accuracy goal)
        dn   < tol_dn    (ternary proximity goal — normalised Δ(N)/n_weights)

    This prevents Phase 2 from exiting while weights are still far from
    {-1, 0, 1}, which would leave crystallization to make a destructive leap.

    Parameters
    ----------
    model           : Any ŁNN with mode='clamp'.
    lr              : Adam learning rate.
    lambda_sparse   : L1 coefficient (Phase 2 only).
    lambda_attract  : Ternary-attraction coefficient (Phase 2 only).
    prox_threshold  : Soft-threshold magnitude applied after each Phase-2 step.
    phase1_fraction : Fraction of max_iter budget allocated to Phase 1.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        phase1_fraction: float = 0.6,
    ):
        luk_layers = _collect_luk_layers(model)
        assert luk_layers and all(
            layer.mode == "clamp" for layer in luk_layers
        ), "ProximalOptimizer requires all LukasiewiczLinear layers with mode='clamp'"

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
        tol_dn: float = 0.05,
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

        # ── Phase 1: MSE only, runs until stagnation ──────────────────────
        # No tol_mse early exit: let the continuous solution fully converge
        # so weights are at their optimum before Phase 2 regularization starts.
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

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break

        if verbose:
            print(f"  Phase 1 done: mse={best_mse:.5f}  stuck={_delta_n_norm(self.model):.4f}"
                  f"  after {len(mse_history)} iters")

        # ── Phase 2: MSE + ternary regularization, dual stopping ──────────
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
                dn = _delta_n_norm(self.model)
                print(f"  Prox P2 step {step:4d}  mse={mse:.6f}  stuck={dn:.4f}  reg={reg.item():.5f}")

            # Dual stopping: accuracy AND ternary proximity
            dn = _delta_n_norm(self.model)
            if mse < tol_mse and dn < tol_dn:
                break

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience2:
                    break

        # ── Phase 3: hardening (10× reg) ──────────────────────────────────
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
            for layer in _collect_luk_layers(self.model):
                layer.weight.data.clamp_(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ProximalGroupLasso
# ─────────────────────────────────────────────────────────────────────────────

def _group_lasso_penalty(model: nn.Module, lambda_group: float) -> Tensor:
    """
    Group Lasso over input connections per output neuron.

    For each row w_i of each weight matrix (all inputs to neuron i), adds
    lambda_group * ||w_i||_2.  Unlike L1 (which shrinks each weight
    independently), this creates all-or-nothing pressure: once a row is
    small, the L2 gradient pulls the whole row toward zero together.
    """
    reg = torch.tensor(0.0)
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() == 2:
            reg = reg + lambda_group * p.pow(2).sum(dim=1).clamp(min=1e-12).sqrt().sum()
    return reg


class ProximalGroupLasso(BaseOptimizer):
    """
    ProximalOptimizer + Group Lasso (L2,1) during Phase 1.

    Adds lambda_group * sum_i ||w_i||_2 to the Phase 1 loss, where w_i is
    the vector of all input weights to output neuron i.  This creates
    structured sparsity: whole input connections are dropped rather than all
    weights shrinking uniformly (the L1 trap that causes Proximal to collapse
    on high fan-in inputs like Mushroom's 111 one-hot features).

    Parameters
    ----------
    lambda_group : float   Group Lasso coefficient for Phase 1 (default 0.01).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        phase1_fraction: float = 0.6,
        lambda_group: float = 0.01,
    ):
        luk_layers = _collect_luk_layers(model)
        assert luk_layers and all(
            layer.mode == "clamp" for layer in luk_layers
        ), "ProximalGroupLasso requires all LukasiewiczLinear layers with mode='clamp'"

        self.model = model
        self.lambda_sparse = lambda_sparse
        self.lambda_attract = lambda_attract
        self.prox_threshold = prox_threshold
        self.phase1_fraction = phase1_fraction
        self.lambda_group = lambda_group
        self.inner = torch.optim.Adam(model.parameters(), lr=lr)

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        tol_dn: float = 0.05,
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

        # ── Phase 1: MSE + Group Lasso ─────────────────────────────────────
        for it in range(phase1_end):
            self.inner.zero_grad()
            pred = self.model(x)
            mse_loss = _mse_loss(pred)
            gl = _group_lasso_penalty(self.model, self.lambda_group)
            (mse_loss + gl).backward()
            self.inner.step()
            self._project_gl()

            mse = mse_loss.item()
            mse_history.append(mse)

            if verbose and it % 300 == 0:
                print(f"  GL P1 iter {it:4d}  mse={mse:.6f}  gl={gl.item():.5f}")

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break

        if verbose:
            print(
                f"  Phase 1 done: mse={best_mse:.5f}  stuck={_delta_n_norm(self.model):.4f}"
                f"  after {len(mse_history)} iters"
            )

        # ── Phase 2: MSE + ternary regularization, dual stopping ──────────
        phase2_iters = max_iter - phase1_end
        stagnation = 0
        patience2 = max(50, phase2_iters // 5)

        for step in range(phase2_iters):
            scale = (step + 1) / phase2_iters
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

            self._project_gl()

            mse = mse_loss.item()
            mse_history.append(mse)

            if verbose and step % 300 == 0:
                dn = _delta_n_norm(self.model)
                print(f"  GL P2 step {step:4d}  mse={mse:.6f}  stuck={dn:.4f}  reg={reg.item():.5f}")

            dn = _delta_n_norm(self.model)
            if mse < tol_mse and dn < tol_dn:
                break

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience2:
                    break

        # ── Phase 3: hardening ────────────────────────────────────────────
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
                self._project_gl()

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

    def _project_gl(self) -> None:
        with torch.no_grad():
            for layer in _collect_luk_layers(self.model):
                layer.weight.data.clamp_(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ProximalL0
# ─────────────────────────────────────────────────────────────────────────────

class _HardConcreteGate(nn.Module):
    """
    Hard Concrete gate parametrization (Louizos et al., ICLR 2018).

    Wraps a weight tensor w → w * z, where z is sampled from a stretched
    sigmoid at train time and becomes a hard 0/1 gate at eval time.

    beta  = temperature (2/3 from paper)
    gamma = left stretch (-0.1), zeta = right stretch (1.1)
    These ensure z can exactly reach 0 and 1, unlike plain sigmoid.
    """
    BETA  =  2 / 3
    GAMMA = -0.1
    ZETA  =  1.1
    # Correction term for the L0 penalty: -beta * log(-gamma/zeta)
    _C = -BETA * math.log(-GAMMA / ZETA)

    def __init__(self, shape: tuple):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(shape, 0.5))

    def forward(self, w: Tensor) -> Tensor:
        if self.training:
            u = w.new_empty(w.shape).uniform_().clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid(
                (u.log() - (1 - u).log() + self.log_alpha) / self.BETA
            )
            z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        else:
            z = (self.log_alpha > 0).float()
        return w * z

    def right_inverse(self, w: Tensor) -> Tensor:
        return w

    def l0_penalty(self) -> Tensor:
        """Expected number of non-zero gates — differentiable w.r.t. log_alpha."""
        return torch.sigmoid(self.log_alpha + self._C).sum()


def _get_raw_weight(layer: LukasiewiczLinear) -> Tensor:
    """Return the pre-gate (original) weight, handling both normal and L0 layers."""
    if (
        hasattr(layer, "parametrizations")
        and "weight" in layer.parametrizations
    ):
        return layer.parametrizations.weight.original
    return layer.weight


def _l0_ternary_regularization(
    luk_layers: list, lambda_sparse: float, lambda_attract: float
) -> Tensor:
    """Ternary regularization on raw weights — ignores log_alpha tensors."""
    reg = torch.tensor(0.0)
    for layer in luk_layers:
        w = _get_raw_weight(layer)
        reg = reg + lambda_sparse * w.abs().sum()
        reg = reg + lambda_attract * _ternary_penalty(w)
    return reg


def _delta_n_norm_raw(luk_layers: list) -> float:
    """dn_stuck on raw weights, works with and without L0 gates."""
    parts = []
    for layer in luk_layers:
        w_abs = _get_raw_weight(layer).data.abs()
        stuck = ((w_abs > 0.1) & (w_abs < 0.9)).float()
        parts.append(stuck)
    if not parts:
        return 0.0
    return torch.cat([s.flatten() for s in parts]).mean().item()


class ProximalL0(BaseOptimizer):
    """
    ProximalOptimizer + Hard Concrete L0 gates (Louizos et al., ICLR 2018).

    Each weight w_ij has a learnable gate parameter log_alpha_ij.  During
    Phase 1 the gate is sampled from the Hard Concrete distribution (a
    differentiable approximation of Bernoulli).  The L0 penalty penalises the
    expected number of open gates, driving the network toward a sparse
    connectivity pattern that is determined by gradient information rather
    than mere magnitude (unlike TopK).

    After Phase 1: gates are binarised (log_alpha > 0 → z=1, else z=0).
    Phase 2 runs standard ternary regularisation on the surviving weights.
    Before crystallize(), the parametrisation is removed so the downstream
    crystallize() call operates on a normal weight tensor.

    Parameters
    ----------
    lambda_l0 : float   L0 penalty coefficient (default 1e-4 per gate).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        phase1_fraction: float = 0.6,
        lambda_l0: float = 1e-4,
    ):
        luk_layers = _collect_luk_layers(model)
        assert luk_layers and all(
            layer.mode == "clamp" for layer in luk_layers
        ), "ProximalL0 requires all LukasiewiczLinear layers with mode='clamp'"

        self.model = model
        self.luk_layers = luk_layers
        self.lambda_sparse = lambda_sparse
        self.lambda_attract = lambda_attract
        self.prox_threshold = prox_threshold
        self.phase1_fraction = phase1_fraction
        self.lambda_l0 = lambda_l0

        # Register Hard Concrete gates on every LukasiewiczLinear weight
        for layer in luk_layers:
            parametrize.register_parametrization(
                layer, "weight", _HardConcreteGate(layer.weight.shape)
            )

        # Optimizer covers model weights + all log_alpha tensors
        self.inner = torch.optim.Adam(model.parameters(), lr=lr)

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        tol_dn: float = 0.05,
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

        # ── Phase 1: MSE + L0 penalty, stochastic gates ───────────────────
        self.model.train()
        for it in range(phase1_end):
            self.inner.zero_grad()
            pred = self.model(x)          # forward uses sampled Hard Concrete z
            mse_loss = _mse_loss(pred)
            l0 = sum(
                layer.parametrizations.weight[0].l0_penalty()
                for layer in self.luk_layers
            )
            (mse_loss + self.lambda_l0 * l0).backward()
            self.inner.step()
            self._project_raw()

            mse = mse_loss.item()
            mse_history.append(mse)

            if verbose and it % 300 == 0:
                print(f"  L0 P1 iter {it:4d}  mse={mse:.6f}  l0={l0.item():.1f}")

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break

        # Switch to hard gates and prune gated-out raw weights
        # Zeroing gated-out weights before Phase 2 ensures:
        #   (a) _delta_n_norm_raw only measures active weights
        #   (b) ternary regularization is not wasted on pruned connections
        self.model.eval()
        with torch.no_grad():
            for layer in self.luk_layers:
                hard_gate = (
                    layer.parametrizations.weight[0].log_alpha.data > 0
                ).float()
                _get_raw_weight(layer).data *= hard_gate

        if verbose:
            n_open = sum(
                (layer.parametrizations.weight[0].log_alpha.data > 0).sum().item()
                for layer in self.luk_layers
            )
            n_total = sum(
                layer.parametrizations.weight.original.numel()
                for layer in self.luk_layers
            )
            print(
                f"  Phase 1 done: mse={best_mse:.5f}  stuck={_delta_n_norm_raw(self.luk_layers):.4f}"
                f"  after {len(mse_history)} iters"
            )
            print(
                f"  L0 gates: {int(n_open)}/{n_total} open "
                f"({100*n_open/n_total:.1f}%)"
            )

        # ── Phase 2: hard gates fixed, ternary reg on surviving weights ────
        phase2_iters = max_iter - phase1_end
        best_mse = float("inf")
        stagnation = 0
        patience2 = max(50, phase2_iters // 5)

        for step in range(phase2_iters):
            scale = (step + 1) / phase2_iters
            ls = self.lambda_sparse * scale
            la = self.lambda_attract * scale

            self.inner.zero_grad()
            pred = self.model(x)          # eval mode → hard gates
            mse_loss = _mse_loss(pred)
            reg = _l0_ternary_regularization(self.luk_layers, ls, la)
            (mse_loss + reg).backward()
            self.inner.step()

            if self.prox_threshold > 0:
                with torch.no_grad():
                    for layer in self.luk_layers:
                        w = _get_raw_weight(layer)
                        w.data = _soft_threshold(w.data, self.prox_threshold * scale)

            self._project_raw()

            mse = mse_loss.item()
            mse_history.append(mse)

            if verbose and step % 300 == 0:
                dn = _delta_n_norm_raw(self.luk_layers)
                print(f"  L0 P2 step {step:4d}  mse={mse:.6f}  stuck={dn:.4f}  reg={reg.item():.5f}")

            dn = _delta_n_norm_raw(self.luk_layers)
            if mse < tol_mse and dn < tol_dn:
                break

            if mse < best_mse - 1e-6:
                best_mse, stagnation = mse, 0
            else:
                stagnation += 1
                if stagnation >= patience2:
                    break

        # ── Phase 3: hardening ────────────────────────────────────────────
        if best_mse < 0.15:
            for step in range(200):
                self.inner.zero_grad()
                pred = self.model(x)
                mse_loss = _mse_loss(pred)
                reg = _l0_ternary_regularization(
                    self.luk_layers,
                    self.lambda_sparse * 10,
                    self.lambda_attract * 10,
                )
                (mse_loss + reg).backward()
                self.inner.step()
                self._project_raw()

        # ── Remove parametrisations, crystallize ──────────────────────────
        # leave_parametrized=True: layer.weight is set to current gated value
        for layer in self.luk_layers:
            parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)

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

    def _project_raw(self) -> None:
        """Clamp the raw (pre-gate) weights to [-1, 1]."""
        with torch.no_grad():
            for layer in self.luk_layers:
                _get_raw_weight(layer).data.clamp_(-1.0, 1.0)
