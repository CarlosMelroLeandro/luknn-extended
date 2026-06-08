"""
Optimal Brain Surgeon (OBS) pruning.

Hassibi, Stork & Wolf (ICNN 1993) — used as the final pruning step in
Leandro (ALT 2009) §3.

Algorithm:
  1.  Train to convergence.
  2.  Compute H ≈ J^T J  (Gauss–Newton approximation).
  3.  For each weight q: saliency_q = w_q² / (2 · [H⁻¹]_qq).
  4.  Remove weight q* = argmin saliency_q.
  5.  Compensate remaining weights: Δw = −(w_q* / [H⁻¹]_q*q*) · H⁻¹[:, q*].
  6.  Repeat until MSE budget is exceeded.
"""

import torch
from torch import Tensor
from ..network.luknn import LukNN
from .lm import _compute_jacobian


def obs_prune(
    model: LukNN,
    x: Tensor,
    y: Tensor,
    mse_budget: float = 2e-3,
    min_weights: int = 1,
    verbose: bool = False,
) -> list[int]:
    """
    Prune model in-place via OBS.  Returns list of pruned weight indices.

    Pruning stops when removing the next weight would push MSE above
    mse_budget, or when min_weights is reached.
    """
    pruned: list[int] = []
    mask = torch.ones(model.flat_weights().numel(), dtype=torch.bool)

    while mask.sum().item() > min_weights:
        e, J = _compute_jacobian(model, x, y)
        mse = (e ** 2).mean().item()

        H = J.T @ J                              # (P, P)
        try:
            H_inv = torch.linalg.inv(H + 1e-8 * torch.eye(H.shape[0]))
        except torch.linalg.LinAlgError:
            break

        w = model.flat_weights()
        diag_H_inv = torch.diag(H_inv)
        saliencies = (w ** 2) / (2.0 * diag_H_inv.clamp(min=1e-12))
        saliencies[~mask] = float("inf")         # already-pruned weights

        q = saliencies.argmin().item()
        # Estimate MSE increase before committing
        delta_mse = saliencies[q].item() / x.shape[0]
        if mse + delta_mse > mse_budget:
            break

        # Weight compensation
        w_q = w[q].item()
        h_inv_qq = H_inv[q, q].item()
        delta_w = -(w_q / h_inv_qq) * H_inv[:, q]

        w_new = w + delta_w
        w_new[q] = 0.0
        mask[q] = False
        model.load_flat_weights(w_new)
        pruned.append(q)

        if verbose:
            print(f"  Pruned weight {q}  saliency={saliencies[q]:.4e}  "
                  f"mse≈{mse + delta_mse:.6f}")

    return pruned
