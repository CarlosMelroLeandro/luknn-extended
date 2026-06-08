"""
Modified Levenberg-Marquardt optimizer with smooth crystallization.

Update rule (eq. from §3.1 of Leandro ALT 2009):

    w_{k+1} = Υ_2( w_k − [J^T J + μ·I]^{-1} J^T e )

Jacobian computed via forward-mode AD (jacfwd): P passes instead of N,
which is efficient when P ≪ N (always true for ŁNNs on truth tables).

For large datasets (mushroom, bAbI) use batch_size > 0 to sub-sample rows
for the Jacobian estimate while still checking convergence on the full set.
"""

import torch
from torch import Tensor
from torch.func import jacfwd, functional_call

from ..network.crystallization import smooth_crystallize


def _compute_jacobian(model, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return (e, J) where:
      e : (N,)   residuals  = model(x) − y
      J : (N, P) Jacobian   ∂output_i / ∂w_j  (forward-mode, P passes)
    """
    params = dict(model.named_parameters())

    def forward_fn(p):
        return functional_call(model, p, x)

    J_dict = jacfwd(forward_fn)(params)
    J = torch.cat([J_dict[k].reshape(len(x), -1) for k in params], dim=1)

    with torch.no_grad():
        e = functional_call(model, params, x) - y

    return e.detach(), J.detach()


def lm_train(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 1000,
    mu_init: float = 1e-2,
    mu_min: float = 1e-10,
    mu_max: float = 1e10,
    tol_mse: float = 2e-3,
    crystallize_n: int = 2,
    patience: int = 30,
    batch_size: int = 0,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """
    Train model in-place using the modified LM rule + smooth crystallization.

    batch_size > 0  enables mini-batch LM for large datasets: Jacobian is
    estimated from a random sub-batch each iteration; MSE check / acceptance
    test always uses the full dataset.

    Returns dict with keys: 'mse_history', 'converged', 'iterations'.
    """
    mu = mu_init
    mse_history: list[float] = []
    best_mse = float("inf")
    stagnation = 0
    N = len(x)
    use_minibatch = batch_size > 0 and batch_size < N

    params = dict(model.named_parameters())

    def _full_mse() -> float:
        with torch.no_grad():
            pred = functional_call(model, dict(model.named_parameters()), x)
            sq = (pred - y) ** 2
            if sample_weight is not None:
                return (sq * sample_weight).mean().item()
            return sq.mean().item()

    for it in range(max_iter):
        # ── Jacobian on batch (or full data) ──────────────────────────────
        if use_minibatch:
            idx = torch.randperm(N)[:batch_size]
            xb, yb = x[idx], y[idx]
            wb = sample_weight[idx] if sample_weight is not None else None
        else:
            xb, yb = x, y
            wb = sample_weight

        e, J = _compute_jacobian(model, xb, yb)

        # Apply weighted least squares: scale rows by sqrt(w)
        if wb is not None:
            sw = wb.sqrt()
            e = e * sw
            J = J * sw.unsqueeze(1)

        # ── Full-data MSE for history / convergence ───────────────────────
        # When sample_weight is set, e has been scaled by sqrt(w), so
        # e**2 == w*(pred-y)**2 and mean(e**2) == weighted MSE — correct.
        mse_full = _full_mse() if use_minibatch else (e ** 2).mean().item()
        mse_history.append(mse_full)

        if verbose and it % 50 == 0:
            print(f"  iter {it:4d}  mse={mse_full:.6f}  mu={mu:.2e}")

        if mse_full < tol_mse:
            if verbose:
                print(f"  Converged at iter {it}  mse={mse_full:.6f}")
            return {"mse_history": mse_history, "converged": True, "iterations": it}

        # ── Stagnation check on full-data MSE ────────────────────────────
        if mse_full < best_mse - 1e-6:
            best_mse = mse_full
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                return {"mse_history": mse_history, "converged": False,
                        "iterations": it, "reason": "stagnation"}

        # ── LM update ─────────────────────────────────────────────────────
        JtJ = J.T @ J
        Jte = J.T @ e
        P = JtJ.shape[0]
        A = JtJ + mu * torch.eye(P)

        try:
            delta = torch.linalg.solve(A, Jte)
        except torch.linalg.LinAlgError:
            mu = min(mu * 10.0, mu_max)
            continue

        w_old = model.flat_weights().clone()
        w_step = w_old - delta
        model.load_flat_weights(w_step)

        mse_new = _full_mse()

        if mse_new < mse_full:
            w_cryst = smooth_crystallize(w_step, n=crystallize_n)
            model.load_flat_weights(w_cryst)
            mu = max(mu / 10.0, mu_min)
        else:
            model.load_flat_weights(w_old)
            mu = min(mu * 10.0, mu_max)
            if mu >= mu_max:
                break

    return {"mse_history": mse_history, "converged": False, "iterations": max_iter}
