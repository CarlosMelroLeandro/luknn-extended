"""
LMOptimizer — paper baseline wrapped as a class.

Thin wrapper around `lm_train` (which uses `jacfwd` + Levenberg damping +
smooth crystallization after each accepted step).

Exposes the same interface as STEOptimizer and ProximalOptimizer so
BenchmarkRunner can call all three identically.
"""

import time
import torch
from torch import Tensor

from .base import BaseOptimizer, TrainingResult
from ..training.lm import (
    lm_train,
    lm_train_delayed,
    lm_train_progressive,
    lm_train_dual,
    lm_train_hybrid,
)
from ..training.obs_pruning import obs_prune
from ..network.crystallization import (
    progressive_crystallize,
    crisp_crystallize_weights,
    crisp_crystallize_bias,
    representation_error,
)
from ..layers.lukasiewicz_linear import LukasiewiczNet
import torch.nn as nn


class LMOptimizer(BaseOptimizer):
    """
    Modified Levenberg-Marquardt with smooth crystallization (§3.1 of paper).

    Parameters
    ----------
    model : LukasiewiczNet  (mode='continuous')
    mu_init : float         Initial LM damping factor.
    factor_up : float       μ multiplier on rejected step (default 10).
    factor_down : float     μ divisor on accepted step (default 10).
    crystallize_n : int     n for Υ_n after each accepted step (default 2).
    patience : int          Stagnation patience before early abort.
    prune : bool            Apply OBS pruning after crystallization.
    """

    def __init__(
        self,
        model: LukasiewiczNet,
        mu_init: float = 1e-2,
        factor_up: float = 10.0,
        factor_down: float = 10.0,
        crystallize_n: int = 2,
        patience: int = 50,
        prune: bool = True,
        batch_size: int = 0,
    ):
        self.model = model
        self.mu_init = mu_init
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.crystallize_n = crystallize_n
        self.patience = patience
        self.prune = prune
        self.batch_size = batch_size

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 400,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()

        raw = lm_train(
            self.model,
            x,
            y,
            max_iter=max_iter,
            mu_init=self.mu_init,
            tol_mse=tol_mse,
            crystallize_n=self.crystallize_n,
            patience=self.patience,
            batch_size=self.batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        if raw["converged"]:
            self._crystallize_model()
            mse_crisp = self._mse(self.model(x), y, sample_weight)
            delta_n = representation_error(self.model.flat_weights()).item()

            if mse_crisp <= tol_mse and delta_n < 0.01 and self.prune:
                obs_prune(self.model, x, y, mse_budget=tol_mse)

        elapsed = time.perf_counter() - t0
        final_mse = self._mse(self.model(x), y, sample_weight)
        reason = "converged" if raw["converged"] else raw.get("reason", "max_iter")

        return TrainingResult(
            converged=raw["converged"],
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=elapsed,
            reason=reason,
        )

    def _crystallize_model(self) -> None:
        # Prefer top-level crystallize() (LukasiewiczNet, LukResidualNet).
        if hasattr(self.model, "crystallize"):
            self.model.crystallize()
            return
        for m in self.model.modules():
            if hasattr(m, "crystallize"):
                m.crystallize()
            elif isinstance(m, nn.Linear):
                m.weight.data = crisp_crystallize_weights(
                    progressive_crystallize(m.weight.data)
                )
                m.bias.data = crisp_crystallize_bias(
                    progressive_crystallize(m.bias.data)
                )


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1 — Delayed crystallization
# ─────────────────────────────────────────────────────────────────────────────

class LMDelayedOptimizer(BaseOptimizer):
    """
    LM with delayed crystallization.

    Υ_n is not applied until (crystallize_start_fraction * max_iter) iterations
    have passed, letting weights find a continuous solution first.

    Parameters
    ----------
    crystallize_start_fraction : float
        Fraction of max_iter after which Υ_n is enabled (default 0.3).
    """

    def __init__(
        self,
        model,
        mu_init: float = 1e-2,
        crystallize_n: int = 2,
        crystallize_start_fraction: float = 0.3,
        patience: int = 50,
        prune: bool = True,
        batch_size: int = 0,
    ):
        self.model = model
        self.mu_init = mu_init
        self.crystallize_n = crystallize_n
        self.crystallize_start_fraction = crystallize_start_fraction
        self.patience = patience
        self.prune = prune
        self.batch_size = batch_size

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 400,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()

        raw = lm_train_delayed(
            self.model, x, y,
            max_iter=max_iter,
            mu_init=self.mu_init,
            tol_mse=tol_mse,
            crystallize_n=self.crystallize_n,
            crystallize_start_fraction=self.crystallize_start_fraction,
            patience=self.patience,
            batch_size=self.batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        if raw["converged"]:
            self._crystallize_model()
            mse_crisp = self._mse(self.model(x), y, sample_weight)
            delta_n = representation_error(self.model.flat_weights()).item()
            if mse_crisp <= tol_mse and delta_n < 0.01 and self.prune:
                from ..training.obs_pruning import obs_prune
                obs_prune(self.model, x, y, mse_budget=tol_mse)

        elapsed = time.perf_counter() - t0
        final_mse = self._mse(self.model(x), y, sample_weight)
        reason = "converged" if raw["converged"] else raw.get("reason", "max_iter")

        return TrainingResult(
            converged=raw["converged"],
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=elapsed,
            reason=reason,
        )

    def _crystallize_model(self) -> None:
        if hasattr(self.model, "crystallize"):
            self.model.crystallize()
            return
        for m in self.model.modules():
            if hasattr(m, "crystallize"):
                m.crystallize()
            elif isinstance(m, nn.Linear):
                m.weight.data = crisp_crystallize_weights(progressive_crystallize(m.weight.data))
                m.bias.data = crisp_crystallize_bias(progressive_crystallize(m.bias.data))


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — Progressive crystallization schedule
# ─────────────────────────────────────────────────────────────────────────────

class LMProgressiveOptimizer(BaseOptimizer):
    """
    LM with a progressive n schedule for Υ_n.

    n starts at n_schedule[0] and steps up at each schedule_fractions[i]
    threshold, finishing at n_schedule[-1].  Provides a warm-up analogous
    to Proximal's linear λ ramp.
    """

    def __init__(
        self,
        model,
        mu_init: float = 1e-2,
        n_schedule: "tuple[int, ...]" = (2, 4, 8, 16),
        schedule_fractions: "tuple[float, ...]" = (0.0, 0.5, 0.75, 0.9),
        patience: int = 50,
        prune: bool = True,
        batch_size: int = 0,
    ):
        self.model = model
        self.mu_init = mu_init
        self.n_schedule = n_schedule
        self.schedule_fractions = schedule_fractions
        self.patience = patience
        self.prune = prune
        self.batch_size = batch_size

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 400,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()

        raw = lm_train_progressive(
            self.model, x, y,
            max_iter=max_iter,
            mu_init=self.mu_init,
            tol_mse=tol_mse,
            n_schedule=self.n_schedule,
            schedule_fractions=self.schedule_fractions,
            patience=self.patience,
            batch_size=self.batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        if raw["converged"]:
            self._crystallize_model()
            mse_crisp = self._mse(self.model(x), y, sample_weight)
            delta_n = representation_error(self.model.flat_weights()).item()
            if mse_crisp <= tol_mse and delta_n < 0.01 and self.prune:
                from ..training.obs_pruning import obs_prune
                obs_prune(self.model, x, y, mse_budget=tol_mse)

        elapsed = time.perf_counter() - t0
        final_mse = self._mse(self.model(x), y, sample_weight)
        reason = "converged" if raw["converged"] else raw.get("reason", "max_iter")

        return TrainingResult(
            converged=raw["converged"],
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=elapsed,
            reason=reason,
        )

    def _crystallize_model(self) -> None:
        if hasattr(self.model, "crystallize"):
            self.model.crystallize()
            return
        for m in self.model.modules():
            if hasattr(m, "crystallize"):
                m.crystallize()
            elif isinstance(m, nn.Linear):
                m.weight.data = crisp_crystallize_weights(progressive_crystallize(m.weight.data))
                m.bias.data = crisp_crystallize_bias(progressive_crystallize(m.bias.data))


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3 — Dual stopping
# ─────────────────────────────────────────────────────────────────────────────

class LMDualOptimizer(BaseOptimizer):
    """
    LM with dual stopping: mse < tol_mse AND Δ(N)/P < tol_dn.

    Prevents early exit when weights satisfy the MSE goal but are still far
    from {-1, 0, 1}, which would leave crisp crystallization to make a
    destructive rounding step.

    Parameters
    ----------
    tol_dn : float   Normalised Δ(N) threshold (default 0.05).
    dn_patience : int   Max iters to wait for Δ(N) improvement after MSE is met.
    """

    def __init__(
        self,
        model,
        mu_init: float = 1e-2,
        crystallize_n: int = 2,
        tol_dn: float = 0.05,
        dn_patience: int = 50,
        patience: int = 50,
        prune: bool = True,
        batch_size: int = 0,
    ):
        self.model = model
        self.mu_init = mu_init
        self.crystallize_n = crystallize_n
        self.tol_dn = tol_dn
        self.dn_patience = dn_patience
        self.patience = patience
        self.prune = prune
        self.batch_size = batch_size

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 400,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()

        raw = lm_train_dual(
            self.model, x, y,
            max_iter=max_iter,
            mu_init=self.mu_init,
            tol_mse=tol_mse,
            tol_dn=self.tol_dn,
            crystallize_n=self.crystallize_n,
            patience=self.patience,
            dn_patience=self.dn_patience,
            batch_size=self.batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        if raw["converged"]:
            self._crystallize_model()
            mse_crisp = self._mse(self.model(x), y, sample_weight)
            delta_n = representation_error(self.model.flat_weights()).item()
            if mse_crisp <= tol_mse and delta_n < 0.01 and self.prune:
                from ..training.obs_pruning import obs_prune
                obs_prune(self.model, x, y, mse_budget=tol_mse)

        elapsed = time.perf_counter() - t0
        final_mse = self._mse(self.model(x), y, sample_weight)
        reason = "converged" if raw["converged"] else raw.get("reason", "max_iter")

        return TrainingResult(
            converged=raw["converged"],
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=elapsed,
            reason=reason,
            extra={"final_dn": raw.get("final_dn", None)},
        )

    def _crystallize_model(self) -> None:
        if hasattr(self.model, "crystallize"):
            self.model.crystallize()
            return
        for m in self.model.modules():
            if hasattr(m, "crystallize"):
                m.crystallize()
            elif isinstance(m, nn.Linear):
                m.weight.data = crisp_crystallize_weights(progressive_crystallize(m.weight.data))
                m.bias.data = crisp_crystallize_bias(progressive_crystallize(m.bias.data))


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid optimizer — Phase 1: LM  ·  Phase 2: Adam + ternary regularization
# ─────────────────────────────────────────────────────────────────────────────

class LMHybridOptimizer(BaseOptimizer):
    """
    Three-phase hybrid optimizer: LM → ternary reg → hardening.

    Phase 1 — LM with smooth crystallization (Υ_n).
        Fast second-order convergence; stops when mse < tol_mse, stagnation,
        or p1_fraction of the total budget is exhausted.

    Phase 2 — Adam with ternary regularization (Proximal-style, λ warm-up).
        Applies  λ_s·||w||₁ + λ_a·w²(1-w²)  with linear λ warm-up.
        Projects weights to [-1, 1] after every step.
        Stops when BOTH  mse < tol_mse  AND  stuck-fraction < tol_dn  (dual
        stopping), preventing a destructive crisp-crystallization leap.

    Phase 3 — hardening (10× reg, fixed p3_steps budget).
        Short burst with lambda_sparse×10 and lambda_attract×10.  No warm-up
        or early stopping.  Pushes near-integer weights over the threshold so
        crisp crystallization becomes non-destructive.
        Skipped if Phase 2 already dual-stopped or MSE ≥ 0.15 (hopeless).

    Motivation:
        LM alone fails to crystallize because Υ₂ is inert near 0.5.
        Proximal's ternary penalty has maximum gradient exactly at 0.5 and
        pulls weights toward {-1, 0, 1} without L1-collapse.  Starting
        Phase 2 from LM's continuous solution avoids the cold-start collapse
        that kills always-on regularization.  Phase 3 closes the residual gap
        where Phase 2 stagnates before all weights are fully integer.

    Parameters
    ----------
    p1_fraction     : float   Budget fraction for Phase 1 (default 0.4).
    p1_patience     : int     LM stagnation patience.
    lr_p2           : float   Adam learning rate for Phases 2 and 3.
    lambda_sparse   : float   L1 coefficient (×1 in Phase 2, ×10 in Phase 3).
    lambda_attract  : float   Ternary-attraction coefficient (×1/×10).
    prox_threshold  : float   Soft-threshold magnitude after each Phase-2 step.
    tol_dn          : float   Stuck-fraction threshold for dual stopping.
    p2_patience     : int     Phase-2 stagnation patience.
    p3_steps        : int     Fixed hardening steps in Phase 3 (default 200).
    """

    def __init__(
        self,
        model,
        mu_init: float = 1e-2,
        crystallize_n: int = 2,
        p1_fraction: float = 0.4,
        p1_patience: int = 30,
        lr_p2: float = 1e-2,
        lambda_sparse: float = 1e-3,
        lambda_attract: float = 0.1,
        prox_threshold: float = 5e-4,
        tol_dn: float = 0.05,
        p2_patience: int = 50,
        p3_steps: int = 200,
        prune: bool = True,
        batch_size: int = 0,
    ):
        self.model = model
        self.mu_init = mu_init
        self.crystallize_n = crystallize_n
        self.p1_fraction = p1_fraction
        self.p1_patience = p1_patience
        self.lr_p2 = lr_p2
        self.lambda_sparse = lambda_sparse
        self.lambda_attract = lambda_attract
        self.prox_threshold = prox_threshold
        self.tol_dn = tol_dn
        self.p2_patience = p2_patience
        self.p3_steps = p3_steps
        self.prune = prune
        self.batch_size = batch_size

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 1000,
        verbose: bool = False,
        sample_weight: "Tensor | None" = None,
    ) -> TrainingResult:
        t0 = time.perf_counter()

        raw = lm_train_hybrid(
            self.model, x, y,
            max_iter=max_iter,
            mu_init=self.mu_init,
            tol_mse=tol_mse,
            crystallize_n=self.crystallize_n,
            p1_patience=self.p1_patience,
            p1_fraction=self.p1_fraction,
            batch_size=self.batch_size,
            lr_p2=self.lr_p2,
            lambda_sparse=self.lambda_sparse,
            lambda_attract=self.lambda_attract,
            prox_threshold=self.prox_threshold,
            tol_dn=self.tol_dn,
            p2_patience=self.p2_patience,
            p3_steps=self.p3_steps,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        if raw["converged"]:
            self._crystallize_model()
            mse_crisp = self._mse(self.model(x), y, sample_weight)
            delta_n = representation_error(self.model.flat_weights()).item()
            if mse_crisp <= tol_mse and delta_n < 0.01 and self.prune:
                from ..training.obs_pruning import obs_prune
                obs_prune(self.model, x, y, mse_budget=tol_mse)

        elapsed = time.perf_counter() - t0
        final_mse = self._mse(self.model(x), y, sample_weight)
        reason = raw.get("reason", "max_iter")

        return TrainingResult(
            converged=raw["converged"],
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=elapsed,
            reason=reason,
            extra={
                "p1_iters":     raw.get("p1_iters"),
                "p1_converged": raw.get("p1_converged"),
            },
        )

    def _crystallize_model(self) -> None:
        if hasattr(self.model, "crystallize"):
            self.model.crystallize()
            return
        for m in self.model.modules():
            if hasattr(m, "crystallize"):
                m.crystallize()
            elif isinstance(m, nn.Linear):
                m.weight.data = crisp_crystallize_weights(progressive_crystallize(m.weight.data))
                m.bias.data = crisp_crystallize_bias(progressive_crystallize(m.bias.data))
