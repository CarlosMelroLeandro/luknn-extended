"""
DLMOptimizer — two-phase Adam with temperature annealing and entropy regularisation.

Phase 1 (p1_fraction of budget):
    Adam on BCE only.  Temperature = T_init (broad gate exploration).
    Best logit state checkpointed at minimum BCE.

Phase 2 (remaining budget):
    Adam with entropy-regularisation warm-up (0 → lambda_entropy).
    Temperature anneals T_init → T_final (sharpening toward argmax).
    Best logit state checkpointed at minimum BCE.
    Early stop: mse < tol_mse AND gate_confidence >= conf_threshold.

Post-training:
    Restore best-checkpoint logits, then crystallize().
    Representability is verified (100% for gate_set='rep').

Key fix: STE (Straight-Through Estimator) through gate clamps enables gradient
flow in hidden layers when inputs are binary {0,1}.  GateLayer activates STE
automatically in training mode.
"""

from __future__ import annotations

import copy
import time
import torch
import torch.nn.functional as F
from torch import Tensor

from ..optimizers.base import BaseOptimizer, TrainingResult
from .network import DLMNetwork, CrystallizedDLM


class DLMOptimizer(BaseOptimizer):
    """
    Two-phase optimizer for DLMNetwork.

    Parameters
    ----------
    model            : DLMNetwork
    lr               : Adam learning rate (default 5e-3)
    T_init           : initial softmax temperature (default 2.0 — broad)
    T_final          : final softmax temperature (default 0.1 — near-argmax)
    lambda_entropy   : peak entropy-regularisation coefficient (default 0.05)
    p1_fraction      : fraction of budget for Phase 1 (default 0.5)
    conf_threshold   : gate_confidence threshold for early exit (default 0.90)
    clip_grad        : gradient norm clip (default 1.0)
    loss             : 'bce' (default) or 'mse'
    """

    def __init__(
        self,
        model: DLMNetwork,
        lr: float = 5e-3,
        T_init: float = 2.0,
        T_final: float = 0.1,
        lambda_entropy: float = 0.05,
        p1_fraction: float = 0.5,
        conf_threshold: float = 0.90,
        clip_grad: float = 1.0,
        loss: str = "bce",
    ):
        assert isinstance(model, DLMNetwork), "DLMOptimizer requires a DLMNetwork"
        assert loss in ("mse", "bce"), f"loss must be 'mse' or 'bce', got {loss!r}"

        self.model = model
        self.lr = lr
        self.T_init = T_init
        self.T_final = T_final
        self.lambda_entropy = lambda_entropy
        self.p1_fraction = p1_fraction
        self.conf_threshold = conf_threshold
        self.clip_grad = clip_grad
        self.loss_type = loss

    def _compute_loss(self, pred: Tensor, y: Tensor, sample_weight: Tensor | None) -> Tensor:
        if self.loss_type == "bce":
            p = pred.clamp(1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(p, y, reduction="none")
        else:
            loss = (pred - y) ** 2

        if sample_weight is not None:
            return (loss * sample_weight).mean()
        return loss.mean()

    def _save_logits(self) -> list[Tensor]:
        """Snapshot all layer logits (detached copy)."""
        return [layer.logits.data.clone() for layer in self.model.all_layers]

    def _restore_logits(self, snapshot: list[Tensor]) -> None:
        """Restore logits from snapshot in-place."""
        for layer, saved in zip(self.model.all_layers, snapshot):
            layer.logits.data.copy_(saved)

    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 2000,
        verbose: bool = False,
        sample_weight: Tensor | None = None,
        batch_size: int | None = None,
    ) -> TrainingResult:
        """
        Parameters
        ----------
        x, y         : full training data
        tol_mse      : convergence threshold (on full-set MSE)
        max_iter     : total gradient steps
        verbose      : print progress every 200 steps
        sample_weight: per-sample weights
        batch_size   : if set, use mini-batch SGD (default: full dataset)
        """
        t0 = time.perf_counter()
        mse_history: list[float] = []
        best_bce = float("inf")
        best_checkpoint: list[Tensor] | None = None
        stagnation = 0
        patience = max(100, max_iter // 10)
        n_samples = x.shape[0]
        use_minibatch = batch_size is not None and batch_size < n_samples

        self.model.train()
        self.model.temperature = self.T_init
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        p1_end = int(max_iter * self.p1_fraction)
        reason = "max_iter"

        def _get_batch(x, y, sw, bs):
            idx = torch.randperm(x.shape[0])[:bs]
            sw_b = sw[idx] if sw is not None else None
            return x[idx], y[idx], sw_b

        def _full_mse(model, x, y, sw):
            with torch.no_grad():
                p = model(x)
            return float(self._mse(p, y, sw))

        # ── Phase 1: BCE only, high temperature, best-state checkpointing ─────
        for it in range(p1_end):
            if use_minibatch:
                xb, yb, swb = _get_batch(x, y, sample_weight, batch_size)
            else:
                xb, yb, swb = x, y, sample_weight

            optimiser.zero_grad()
            pred = self.model(xb)
            loss = self._compute_loss(pred, yb, swb)
            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            optimiser.step()

            # Full-dataset MSE every 50 steps when using mini-batches
            if use_minibatch and it % 50 == 0:
                mse = _full_mse(self.model, x, y, sample_weight)
            elif not use_minibatch:
                mse = float(self._mse(pred.detach(), y, sample_weight))
            else:
                mse = mse_history[-1] if mse_history else 0.25
            bce = float(loss.item())
            mse_history.append(mse)

            if bce < best_bce - 1e-6:
                best_bce = bce
                best_checkpoint = self._save_logits()
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    reason = "stagnation_p1"
                    break

            if verbose and it % 200 == 0:
                conf = self.model.gate_confidence()
                print(f"  DLM P1 {it:5d}  mse={mse:.5f}  bce={bce:.5f}  conf={conf:.3f}")

        if verbose:
            print(f"  Phase 1 done: bce={best_bce:.5f}  conf={self.model.gate_confidence():.3f}")

        # ── Phase 2: entropy reg + temperature annealing ──────────────────────
        if best_checkpoint is not None:
            self._restore_logits(best_checkpoint)

        p2_iters = max_iter - p1_end
        best_bce_p2 = float("inf")
        stagnation = 0
        patience2 = max(50, p2_iters // 5)

        for step in range(p2_iters):
            frac = (step + 1) / p2_iters

            T = self.T_init + (self.T_final - self.T_init) * frac
            self.model.temperature = T
            lam_H = self.lambda_entropy * frac

            if use_minibatch:
                xb, yb, swb = _get_batch(x, y, sample_weight, batch_size)
            else:
                xb, yb, swb = x, y, sample_weight

            optimiser.zero_grad()
            pred = self.model(xb)
            mse_loss = self._compute_loss(pred, yb, swb)
            h_loss = self.model.entropy_loss()
            total_loss = mse_loss + lam_H * h_loss
            total_loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            optimiser.step()

            if use_minibatch and step % 50 == 0:
                mse = _full_mse(self.model, x, y, sample_weight)
            elif not use_minibatch:
                mse = float(self._mse(pred.detach(), y, sample_weight))
            else:
                mse = mse_history[-1] if mse_history else 0.25
            bce = float(mse_loss.item())
            mse_history.append(mse)

            if bce < best_bce_p2 - 1e-6:
                best_bce_p2 = bce
                best_checkpoint = self._save_logits()
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= patience2:
                    reason = "stagnation_p2"
                    break

            if verbose and step % 200 == 0:
                conf = self.model.gate_confidence()
                h = float(h_loss.item())
                print(f"  DLM P2 {step:5d}  mse={mse:.5f}  bce={bce:.5f}  "
                      f"conf={conf:.3f}  H={h:.4f}  T={T:.3f}")

            conf = self.model.gate_confidence()
            if mse < tol_mse and conf >= self.conf_threshold:
                reason = "converged"
                break

        # ── Restore best checkpoint before crystallisation ────────────────────
        if best_checkpoint is not None:
            self._restore_logits(best_checkpoint)

        self.model.eval()

        # ── Crystallise to integer-weight CrystallizedDLM ────────────────────
        crys = self.model.crystallize()

        with torch.no_grad():
            pred_crys = crys(x)
        final_mse = float(self._mse(pred_crys, y, sample_weight))
        converged = final_mse < tol_mse and reason == "converged"

        elapsed = time.perf_counter() - t0

        if verbose:
            rep = crys.representability_fraction()
            conf = self.model.gate_confidence()
            print(f"  Crystallised: mse={final_mse:.5f}  "
                  f"representability={rep:.1%}  conf={conf:.3f}")

        return TrainingResult(
            converged=converged,
            final_mse=final_mse,
            mse_history=mse_history,
            iterations=len(mse_history),
            total_time_s=elapsed,
            reason=reason,
            extra={
                "gate_confidence": self.model.gate_confidence(),
                "representability": crys.representability_fraction(),
                "crystallized_model": crys,
                "gate_counts": self.model.representability_report()["gate_counts"],
            },
        )
