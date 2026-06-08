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
from ..training.lm import lm_train
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
