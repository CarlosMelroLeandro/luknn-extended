"""
STE optimizers — Straight-Through Estimator with ternary weight quantization.

Four variants of increasing sophistication:

  STEOptimizer          — original: Adam + cosine LR, MSE-only, [-1.5, 1.5] clamp
  STERegOptimizer       — + ternary reg w²(1-w²) with λ warm-up, [-1, 1] clamp
  STEDualOptimizer      — STEReg + dual stopping (mse AND boundary_frac)
  STEHybridOptimizer    — Phase 1 (pure MSE) → Phase 2 (reg warm-up + dual stop)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseOptimizer, TrainingResult
from ..layers.lukasiewicz_linear import LukasiewiczLinear
from ..training.ste import (
    ste_train_base,
    ste_train_reg,
    ste_train_dual,
    ste_train_hybrid,
)


def _collect_luk_layers(model: nn.Module) -> list[LukasiewiczLinear]:
    return [m for m in model.modules() if isinstance(m, LukasiewiczLinear)]


def _assert_ste_mode(model: nn.Module) -> None:
    layers = _collect_luk_layers(model)
    assert layers and all(
        l.mode == "ste" for l in layers
    ), "All LukasiewiczLinear layers must have mode='ste'"


# ── Base ──────────────────────────────────────────────────────────────────────

class STEOptimizer(BaseOptimizer):
    """
    Original STE optimizer: Adam + cosine LR annealing, MSE-only loss.

    Latent weights clamped to [-1.5, 1.5].  Best-state checkpoint restored
    before crystallization.

    Parameters
    ----------
    lr         : float  Adam learning rate (default 5e-3).
    clip_grad  : float  Gradient norm clip (0 = disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        clip_grad: float = 1.0,
    ):
        _assert_ste_mode(model)
        self.model = model
        self.lr = lr
        self.clip_grad = clip_grad

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

        raw = ste_train_base(
            self.model, x, y,
            max_iter=max_iter,
            tol_mse=tol_mse,
            lr=self.lr,
            clip_grad=self.clip_grad,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        self.model.crystallize()
        final_mse = self._mse(self.model(x), y, sample_weight)

        return TrainingResult(
            converged=final_mse < tol_mse,
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=time.perf_counter() - t0,
            reason=raw["reason"],
            extra={"boundary_frac_pre": raw.get("boundary_frac_pre", 0.0)},
        )


# ── Reg ───────────────────────────────────────────────────────────────────────

class STERegOptimizer(BaseOptimizer):
    """
    STE + ternary regularization w²(1-w²) with linear λ warm-up.

    The ternary penalty is zero at {-1, 0, 1} and maximal at ±0.5, pushing
    latent weights away from snap-threshold ambiguity without changing forward
    MSE.  Warm-up (0 → λ_attract) prevents cold-start collapse to 0.
    Latent weights clamped to [-1, 1].

    Parameters
    ----------
    lr             : float  Adam learning rate.
    lambda_attract : float  Peak ternary reg coefficient (reached at last iter).
    clip_grad      : float  Gradient norm clip.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        lambda_attract: float = 0.05,
        clip_grad: float = 1.0,
    ):
        _assert_ste_mode(model)
        self.model = model
        self.lr = lr
        self.lambda_attract = lambda_attract
        self.clip_grad = clip_grad

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

        raw = ste_train_reg(
            self.model, x, y,
            max_iter=max_iter,
            tol_mse=tol_mse,
            lr=self.lr,
            lambda_attract=self.lambda_attract,
            clip_grad=self.clip_grad,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        self.model.crystallize()
        final_mse = self._mse(self.model(x), y, sample_weight)

        return TrainingResult(
            converged=final_mse < tol_mse,
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=time.perf_counter() - t0,
            reason=raw["reason"],
            extra={"boundary_frac_pre": raw.get("boundary_frac_pre", 0.0)},
        )


# ── Dual ──────────────────────────────────────────────────────────────────────

class STEDualOptimizer(BaseOptimizer):
    """
    STEReg + dual stopping criterion.

    Stops when BOTH mse < mse_gate AND boundary_frac < tol_boundary.
    mse_gate uses a realistic threshold for STE (default 0.05 — much looser
    than tol_mse=2e-3 which is calibrated for continuous weights).
    boundary_frac is the fraction of latent weights within ±0.15 of the snap
    threshold (±0.33), indicating instability under crystallization.

    Parameters
    ----------
    lr             : float  Adam learning rate.
    lambda_attract : float  Peak ternary reg coefficient.
    mse_gate       : float  MSE threshold for the dual criterion (default 0.05).
    tol_boundary   : float  Max allowed boundary fraction at stopping (default 0.35).
    clip_grad      : float  Gradient norm clip.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        lambda_attract: float = 0.05,
        mse_gate: float = 0.05,
        tol_boundary: float = 0.35,
        clip_grad: float = 1.0,
    ):
        _assert_ste_mode(model)
        self.model = model
        self.lr = lr
        self.lambda_attract = lambda_attract
        self.mse_gate = mse_gate
        self.tol_boundary = tol_boundary
        self.clip_grad = clip_grad

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

        raw = ste_train_dual(
            self.model, x, y,
            max_iter=max_iter,
            tol_mse=tol_mse,
            mse_gate=self.mse_gate,
            lr=self.lr,
            lambda_attract=self.lambda_attract,
            tol_boundary=self.tol_boundary,
            clip_grad=self.clip_grad,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        self.model.crystallize()
        final_mse = self._mse(self.model(x), y, sample_weight)

        return TrainingResult(
            converged=final_mse < tol_mse,
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=time.perf_counter() - t0,
            reason=raw["reason"],
            extra={
                "boundary_frac_pre": raw.get("boundary_frac_pre", 0.0),
                "p1_iters": raw.get("p1_iters"),
            },
        )


# ── Hybrid ────────────────────────────────────────────────────────────────────

class STEHybridOptimizer(BaseOptimizer):
    """
    Two-phase STE hybrid.

    Phase 1 (~p1_fraction of budget): pure MSE + cosine LR.
        Finds a good ternary solution before regularization is applied,
        avoiding cold-start collapse where reg drives all latent weights to 0.

    Phase 2 (remaining budget): ternary reg warm-up (0 → λ_attract) + dual
        stopping (mse < tol AND boundary_frac < tol_boundary).  Uses half
        the Phase 1 LR to avoid undoing the Phase 1 solution.

    Motivation mirrors LM_hybrid: start from a good continuous (here ternary)
    solution, then apply pressure to stabilize the latent weights near
    {-1, 0, 1}.  The key difference from always-on regularization is that
    Phase 1 is free to explore the loss landscape without being pushed toward
    a sub-optimal ternary attractor.

    Parameters
    ----------
    lr             : float  Phase 1 Adam LR (Phase 2 uses lr/2).
    lambda_attract : float  Peak ternary reg coefficient in Phase 2.
    mse_gate       : float  MSE threshold for dual criterion (default 0.05).
    tol_boundary   : float  boundary_frac threshold for dual stopping (default 0.35).
    p1_fraction    : float  Fraction of max_iter allocated to Phase 1.
    clip_grad      : float  Gradient norm clip.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        lambda_attract: float = 0.05,
        mse_gate: float = 0.05,
        tol_boundary: float = 0.35,
        p1_fraction: float = 0.4,
        clip_grad: float = 1.0,
    ):
        _assert_ste_mode(model)
        self.model = model
        self.lr = lr
        self.lambda_attract = lambda_attract
        self.mse_gate = mse_gate
        self.tol_boundary = tol_boundary
        self.p1_fraction = p1_fraction
        self.clip_grad = clip_grad

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

        raw = ste_train_hybrid(
            self.model, x, y,
            max_iter=max_iter,
            tol_mse=tol_mse,
            mse_gate=self.mse_gate,
            lr=self.lr,
            lambda_attract=self.lambda_attract,
            tol_boundary=self.tol_boundary,
            p1_fraction=self.p1_fraction,
            clip_grad=self.clip_grad,
            verbose=verbose,
            sample_weight=sample_weight,
        )

        self.model.crystallize()
        final_mse = self._mse(self.model(x), y, sample_weight)

        return TrainingResult(
            converged=final_mse < tol_mse,
            final_mse=final_mse,
            mse_history=raw["mse_history"],
            iterations=raw["iterations"],
            total_time_s=time.perf_counter() - t0,
            reason=raw["reason"],
            extra={
                "boundary_frac_pre": raw.get("boundary_frac_pre", 0.0),
                "p1_iters": raw.get("p1_iters"),
            },
        )
