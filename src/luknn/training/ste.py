"""
STE variant training functions for Łukasiewicz NNs.

Four variants of increasing sophistication:

  ste_train_base   — original STE: Adam + cosine LR, MSE-only, stagnation stop
  ste_train_reg    — + ternary reg w²(1-w²) with λ warm-up, clamp [-1, 1]
  ste_train_dual   — ste_train_reg + dual stopping (mse AND boundary_frac)
  ste_train_hybrid — Phase 1 (pure MSE) → Phase 2 (reg warm-up + dual stop)

Key insight: in STE the forward pass already uses {-1,0,1}, so MSE already
measures ternary performance.  The latent weights can still sit near the snap
thresholds (±0.33), making them unstable — a small perturbation flips the bin.
The ternary penalty w²(1-w²) has gradient maximum at ±0.5 and zero at {-1,0,1},
pushing latent weights away from thresholds without changing forward-pass MSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.lukasiewicz_linear import LukasiewiczLinear


# ── Helpers ──────────────────────────────────────────────────────────────────

def _collect_luk_layers(model: nn.Module) -> list[LukasiewiczLinear]:
    return [m for m in model.modules() if isinstance(m, LukasiewiczLinear)]


def boundary_frac(model: nn.Module, thresh: float = 0.33, margin: float = 0.15) -> float:
    """Fraction of latent weights within `margin` of a snap threshold (±thresh).

    These are the weights most likely to flip bins when crystallized, since they
    sit near the decision boundary of hard_snap.  Lower is better.
    """
    parts = [layer.weight.data.view(-1) for layer in _collect_luk_layers(model)]
    if not parts:
        return 0.0
    w = torch.cat(parts).abs()
    near = (w > thresh - margin) & (w < thresh + margin)
    return near.float().mean().item()


def _ternary_reg(model: nn.Module, lam: float) -> Tensor:
    """λ · Σ w²(1-w²)  — zero at {-1,0,1}, maximum at ±0.5."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in _collect_luk_layers(model):
        w = layer.weight
        reg = reg + (w ** 2 * (1.0 - w ** 2)).sum()
    return lam * reg


def _clamp_latent(model: nn.Module, lo: float = -1.0, hi: float = 1.0) -> None:
    with torch.no_grad():
        for layer in _collect_luk_layers(model):
            layer.weight.data.clamp_(lo, hi)


# ── Training functions ────────────────────────────────────────────────────────

def ste_train_base(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 2000,
    tol_mse: float = 2e-3,
    lr: float = 5e-3,
    clip_grad: float = 1.0,
    patience_frac: float = 0.05,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """Original STE: Adam + cosine LR, MSE-only loss, stagnation stop.

    Latent weights clamped to [-1.5, 1.5] (original convention).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=1e-5
    )
    patience = max(50, int(max_iter * patience_frac))

    mse_history: list[float] = []
    best_mse = float("inf")
    best_state: dict | None = None
    stagnation = 0

    def _loss(pred: Tensor) -> Tensor:
        if sample_weight is not None:
            return ((pred - y) ** 2 * sample_weight).mean()
        return F.mse_loss(pred, y)

    for it in range(max_iter):
        optimizer.zero_grad()
        pred = model(x)
        loss = _loss(pred)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        _clamp_latent(model, -1.5, 1.5)

        mse = loss.item()
        mse_history.append(mse)

        if mse < best_mse - 1e-6:
            best_mse = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                break

        if verbose and it % 200 == 0:
            print(f"  base iter {it:4d}  mse={mse:.6f}")

        if best_mse < tol_mse:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    reason = "converged" if best_mse < tol_mse else "stagnation"
    return {
        "mse_history": mse_history,
        "converged": best_mse < tol_mse,
        "iterations": len(mse_history),
        "reason": reason,
        "boundary_frac_pre": boundary_frac(model),
    }


def ste_train_reg(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 2000,
    tol_mse: float = 2e-3,
    lr: float = 5e-3,
    clip_grad: float = 1.0,
    patience_frac: float = 0.05,
    lambda_attract: float = 0.05,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """STE + ternary reg w²(1-w²) with linear λ warm-up. Clamp to [-1, 1].

    The warm-up prevents the reg from collapsing weights to 0 before the
    network has found a good ternary solution.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=1e-5
    )
    patience = max(50, int(max_iter * patience_frac))

    mse_history: list[float] = []
    best_mse = float("inf")
    best_state: dict | None = None
    stagnation = 0

    def _mse_loss(pred: Tensor) -> Tensor:
        if sample_weight is not None:
            return ((pred - y) ** 2 * sample_weight).mean()
        return F.mse_loss(pred, y)

    for it in range(max_iter):
        scale = (it + 1) / max_iter      # 0 → 1
        optimizer.zero_grad()
        pred = model(x)
        mse_loss = _mse_loss(pred)
        reg = _ternary_reg(model, lambda_attract * scale)
        (mse_loss + reg).backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        _clamp_latent(model)

        mse = mse_loss.item()
        mse_history.append(mse)

        if mse < best_mse - 1e-6:
            best_mse = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                break

        if verbose and it % 200 == 0:
            bf = boundary_frac(model)
            print(f"  reg  iter {it:4d}  mse={mse:.6f}  bf={bf:.3f}  λ={lambda_attract*scale:.4f}")

        if best_mse < tol_mse:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    reason = "converged" if best_mse < tol_mse else "stagnation"
    return {
        "mse_history": mse_history,
        "converged": best_mse < tol_mse,
        "iterations": len(mse_history),
        "reason": reason,
        "boundary_frac_pre": boundary_frac(model),
    }


def ste_train_dual(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 2000,
    tol_mse: float = 2e-3,
    mse_gate: float = 0.05,
    lr: float = 5e-3,
    clip_grad: float = 1.0,
    patience_frac: float = 0.05,
    lambda_attract: float = 0.05,
    tol_boundary: float = 0.35,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """STE_reg + dual stopping.

    Stops when BOTH mse < tol_mse AND boundary_frac < tol_boundary.
    boundary_frac measures the fraction of latent weights near the snap
    threshold (±0.33 ± 0.15), which predicts instability under crystallization.
    The dual criterion ensures we only stop when the ternary solution is both
    accurate AND stable.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=1e-5
    )
    patience = max(50, int(max_iter * patience_frac))

    mse_history: list[float] = []
    best_mse = float("inf")
    best_state: dict | None = None
    dual_state: dict | None = None
    stagnation = 0
    reason = "stagnation"

    def _mse_loss(pred: Tensor) -> Tensor:
        if sample_weight is not None:
            return ((pred - y) ** 2 * sample_weight).mean()
        return F.mse_loss(pred, y)

    for it in range(max_iter):
        scale = (it + 1) / max_iter
        optimizer.zero_grad()
        pred = model(x)
        mse_loss = _mse_loss(pred)
        reg = _ternary_reg(model, lambda_attract * scale)
        (mse_loss + reg).backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        _clamp_latent(model)

        mse = mse_loss.item()
        mse_history.append(mse)
        bf = boundary_frac(model)

        if mse < best_mse - 1e-6:
            best_mse = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                break

        if verbose and it % 200 == 0:
            print(f"  dual iter {it:4d}  mse={mse:.6f}  bf={bf:.3f}")

        if mse < mse_gate and bf < tol_boundary:
            reason = "dual_stop"
            dual_state = {k: v.clone() for k, v in model.state_dict().items()}
            break

    if dual_state is not None:
        model.load_state_dict(dual_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    converged = reason == "dual_stop" or best_mse < tol_mse
    return {
        "mse_history": mse_history,
        "converged": converged,
        "iterations": len(mse_history),
        "reason": reason,
        "boundary_frac_pre": boundary_frac(model),
    }


def ste_train_hybrid(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 2000,
    tol_mse: float = 2e-3,
    mse_gate: float = 0.05,
    lr: float = 5e-3,
    clip_grad: float = 1.0,
    patience_frac: float = 0.05,
    lambda_attract: float = 0.05,
    tol_boundary: float = 0.35,
    p1_fraction: float = 0.4,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """Two-phase STE hybrid.

    Phase 1 (~40% of budget): pure MSE + cosine LR, no regularization.
        Lets the network find a good ternary solution before reg pressure is
        applied.  Avoids cold-start collapse where reg drives all weights to 0.

    Phase 2 (remaining 60%): ternary reg warm-up (0 → λ_attract) + dual
        stopping.  Half the Phase 1 LR to avoid undoing P1's solution.
    """
    mse_history: list[float] = []
    p1_end = int(max_iter * p1_fraction)
    p2_iters = max_iter - p1_end
    patience = max(50, int(max_iter * patience_frac))
    p1_converged = False
    reason = "max_iter"

    def _mse_loss(pred: Tensor) -> Tensor:
        if sample_weight is not None:
            return ((pred - y) ** 2 * sample_weight).mean()
        return F.mse_loss(pred, y)

    # ── Phase 1: pure MSE ────────────────────────────────────────────────────
    opt1 = torch.optim.Adam(model.parameters(), lr=lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, p1_end), eta_min=1e-5)
    best_mse = float("inf")
    best_state: dict | None = None
    stagnation = 0

    for it in range(p1_end):
        opt1.zero_grad()
        pred = model(x)
        loss = _mse_loss(pred)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt1.step()
        sch1.step()
        _clamp_latent(model)

        mse = loss.item()
        mse_history.append(mse)

        if mse < best_mse - 1e-6:
            best_mse = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                break

        if best_mse < tol_mse:
            p1_converged = True
            break

    p1_iters = len(mse_history)
    if verbose:
        print(f"  P1 done: mse={mse_history[-1]:.5f}  iters={p1_iters}  conv={p1_converged}")

    if p1_converged:
        if best_state is not None:
            model.load_state_dict(best_state)
        return {
            "mse_history": mse_history,
            "converged": True,
            "iterations": len(mse_history),
            "reason": "p1_converged",
            "p1_iters": p1_iters,
        }

    # Restore best Phase 1 state before Phase 2
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Phase 2: reg warm-up + dual stopping ─────────────────────────────────
    opt2 = torch.optim.Adam(model.parameters(), lr=lr * 0.5)
    best_mse2 = float("inf")
    stagnation2 = 0
    p2_patience = max(50, p2_iters // 5)
    dual_state: dict | None = None

    for step in range(p2_iters):
        scale = (step + 1) / p2_iters
        opt2.zero_grad()
        pred = model(x)
        mse_loss = _mse_loss(pred)
        reg = _ternary_reg(model, lambda_attract * scale)
        (mse_loss + reg).backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt2.step()
        _clamp_latent(model)

        mse = mse_loss.item()
        mse_history.append(mse)
        bf = boundary_frac(model)

        if verbose and step % 200 == 0:
            print(f"  P2 step {step:4d}  mse={mse:.6f}  bf={bf:.3f}")

        if mse < mse_gate and bf < tol_boundary:
            reason = "dual_stop"
            dual_state = {k: v.clone() for k, v in model.state_dict().items()}
            break

        if mse < best_mse2 - 1e-6:
            best_mse2 = mse
            stagnation2 = 0
        else:
            stagnation2 += 1
            if stagnation2 >= p2_patience:
                reason = "p2_stagnation"
                break

    if dual_state is not None:
        model.load_state_dict(dual_state)

    converged = reason == "dual_stop" or (mse_history[-1] if mse_history else float("inf")) < tol_mse
    return {
        "mse_history": mse_history,
        "converged": converged,
        "iterations": len(mse_history),
        "reason": reason,
        "p1_iters": p1_iters,
        "boundary_frac_pre": boundary_frac(model),
    }
