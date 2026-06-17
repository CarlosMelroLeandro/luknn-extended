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
import torch.nn.functional as F
from torch import Tensor
from torch.func import jacfwd, functional_call

from ..network.crystallization import smooth_crystallize, representation_error


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


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1 — Delayed crystallization
# ─────────────────────────────────────────────────────────────────────────────

def lm_train_delayed(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 1000,
    mu_init: float = 1e-2,
    mu_min: float = 1e-10,
    mu_max: float = 1e10,
    tol_mse: float = 2e-3,
    crystallize_n: int = 2,
    crystallize_start_fraction: float = 0.3,
    patience: int = 30,
    batch_size: int = 0,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """
    LM with delayed crystallization.

    Υ_n is suppressed for the first (crystallize_start_fraction * max_iter)
    iterations, letting weights reach a continuous optimum before integer
    attraction begins — analogous to Proximal's Phase 1.
    """
    mu = mu_init
    mse_history: list[float] = []
    best_mse = float("inf")
    stagnation = 0
    N = len(x)
    use_minibatch = batch_size > 0 and batch_size < N
    cryst_start = int(max_iter * crystallize_start_fraction)

    def _full_mse() -> float:
        with torch.no_grad():
            pred = functional_call(model, dict(model.named_parameters()), x)
            sq = (pred - y) ** 2
            if sample_weight is not None:
                return (sq * sample_weight).mean().item()
            return sq.mean().item()

    for it in range(max_iter):
        if use_minibatch:
            idx = torch.randperm(N)[:batch_size]
            xb, yb = x[idx], y[idx]
            wb = sample_weight[idx] if sample_weight is not None else None
        else:
            xb, yb = x, y
            wb = sample_weight

        e, J = _compute_jacobian(model, xb, yb)

        if wb is not None:
            sw = wb.sqrt()
            e = e * sw
            J = J * sw.unsqueeze(1)

        mse_full = _full_mse() if use_minibatch else (e ** 2).mean().item()
        mse_history.append(mse_full)

        if verbose and it % 50 == 0:
            phase = "warm" if it < cryst_start else "cryst"
            print(f"  iter {it:4d}  mse={mse_full:.6f}  mu={mu:.2e}  [{phase}]")

        if mse_full < tol_mse:
            if verbose:
                print(f"  Converged at iter {it}  mse={mse_full:.6f}")
            return {"mse_history": mse_history, "converged": True, "iterations": it}

        if mse_full < best_mse - 1e-6:
            best_mse = mse_full
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                return {"mse_history": mse_history, "converged": False,
                        "iterations": it, "reason": "stagnation"}

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
            if it >= cryst_start:
                w_cryst = smooth_crystallize(w_step, n=crystallize_n)
                model.load_flat_weights(w_cryst)
            # else: keep w_step unchanged (no crystallization yet)
            mu = max(mu / 10.0, mu_min)
        else:
            model.load_flat_weights(w_old)
            mu = min(mu * 10.0, mu_max)
            if mu >= mu_max:
                break

    return {"mse_history": mse_history, "converged": False, "iterations": max_iter}


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — Progressive crystallization schedule
# ─────────────────────────────────────────────────────────────────────────────

def lm_train_progressive(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 1000,
    mu_init: float = 1e-2,
    mu_min: float = 1e-10,
    mu_max: float = 1e10,
    tol_mse: float = 2e-3,
    n_schedule: "tuple[int, ...]" = (2, 4, 8, 16),
    schedule_fractions: "tuple[float, ...]" = (0.0, 0.5, 0.75, 0.9),
    patience: int = 30,
    batch_size: int = 0,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """
    LM with a progressive crystallization schedule.

    n in Υ_n increases over training — weak attraction early (n=2, almost
    linear), strong attraction late (n=16, near-binary).  Analogous to
    Proximal's λ warm-up.

    schedule_fractions[i] is the iteration fraction at which n_schedule[i]
    becomes active.  Both tuples must have the same length.
    """
    assert len(n_schedule) == len(schedule_fractions)
    mu = mu_init
    mse_history: list[float] = []
    best_mse = float("inf")
    stagnation = 0
    N = len(x)
    use_minibatch = batch_size > 0 and batch_size < N

    def _full_mse() -> float:
        with torch.no_grad():
            pred = functional_call(model, dict(model.named_parameters()), x)
            sq = (pred - y) ** 2
            if sample_weight is not None:
                return (sq * sample_weight).mean().item()
            return sq.mean().item()

    def _current_n(it: int) -> int:
        frac = it / max(max_iter, 1)
        n = n_schedule[0]
        for f, ni in zip(schedule_fractions, n_schedule):
            if frac >= f:
                n = ni
        return n

    for it in range(max_iter):
        if use_minibatch:
            idx = torch.randperm(N)[:batch_size]
            xb, yb = x[idx], y[idx]
            wb = sample_weight[idx] if sample_weight is not None else None
        else:
            xb, yb = x, y
            wb = sample_weight

        e, J = _compute_jacobian(model, xb, yb)

        if wb is not None:
            sw = wb.sqrt()
            e = e * sw
            J = J * sw.unsqueeze(1)

        mse_full = _full_mse() if use_minibatch else (e ** 2).mean().item()
        mse_history.append(mse_full)

        n = _current_n(it)
        if verbose and it % 50 == 0:
            print(f"  iter {it:4d}  mse={mse_full:.6f}  mu={mu:.2e}  n={n}")

        if mse_full < tol_mse:
            if verbose:
                print(f"  Converged at iter {it}  mse={mse_full:.6f}")
            return {"mse_history": mse_history, "converged": True, "iterations": it}

        if mse_full < best_mse - 1e-6:
            best_mse = mse_full
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                return {"mse_history": mse_history, "converged": False,
                        "iterations": it, "reason": "stagnation"}

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
            w_cryst = smooth_crystallize(w_step, n=n)
            model.load_flat_weights(w_cryst)
            mu = max(mu / 10.0, mu_min)
        else:
            model.load_flat_weights(w_old)
            mu = min(mu * 10.0, mu_max)
            if mu >= mu_max:
                break

    return {"mse_history": mse_history, "converged": False, "iterations": max_iter}


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3 — Dual stopping: mse + Δ(N)
# ─────────────────────────────────────────────────────────────────────────────

def lm_train_dual(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 1000,
    mu_init: float = 1e-2,
    mu_min: float = 1e-10,
    mu_max: float = 1e10,
    tol_mse: float = 2e-3,
    tol_dn: float = 0.05,
    crystallize_n: int = 2,
    patience: int = 30,
    dn_patience: int = 50,
    batch_size: int = 0,
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """
    LM with dual stopping condition: mse < tol_mse AND Δ(N)/P < tol_dn.

    When MSE is satisfied but weights are still far from integers, training
    continues.  A separate dn_patience counter aborts if Δ(N) stagnates
    after the MSE goal is reached.
    """
    mu = mu_init
    mse_history: list[float] = []
    best_mse = float("inf")
    stagnation = 0
    N = len(x)
    use_minibatch = batch_size > 0 and batch_size < N

    mse_satisfied = False
    best_dn = float("inf")
    dn_stagnation = 0

    def _full_mse() -> float:
        with torch.no_grad():
            pred = functional_call(model, dict(model.named_parameters()), x)
            sq = (pred - y) ** 2
            if sample_weight is not None:
                return (sq * sample_weight).mean().item()
            return sq.mean().item()

    def _dn() -> float:
        w = model.flat_weights()
        return representation_error(w).item() / max(w.numel(), 1)

    for it in range(max_iter):
        if use_minibatch:
            idx = torch.randperm(N)[:batch_size]
            xb, yb = x[idx], y[idx]
            wb = sample_weight[idx] if sample_weight is not None else None
        else:
            xb, yb = x, y
            wb = sample_weight

        e, J = _compute_jacobian(model, xb, yb)

        if wb is not None:
            sw = wb.sqrt()
            e = e * sw
            J = J * sw.unsqueeze(1)

        mse_full = _full_mse() if use_minibatch else (e ** 2).mean().item()
        mse_history.append(mse_full)

        if mse_full < tol_mse:
            mse_satisfied = True

        dn = _dn()
        if verbose and it % 50 == 0:
            print(f"  iter {it:4d}  mse={mse_full:.6f}  dn={dn:.4f}  mu={mu:.2e}")

        if mse_satisfied and dn < tol_dn:
            if verbose:
                print(f"  Converged at iter {it}  mse={mse_full:.6f}  dn={dn:.4f}")
            return {"mse_history": mse_history, "converged": True, "iterations": it,
                    "final_dn": dn}

        # Regular stagnation (on MSE)
        if mse_full < best_mse - 1e-6:
            best_mse = mse_full
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience and not mse_satisfied:
                return {"mse_history": mse_history, "converged": False,
                        "iterations": it, "reason": "stagnation"}

        # Δ(N) stagnation — only active once MSE is satisfied
        if mse_satisfied:
            if dn < best_dn - 1e-4:
                best_dn = dn
                dn_stagnation = 0
            else:
                dn_stagnation += 1
                if dn_stagnation >= dn_patience:
                    return {"mse_history": mse_history, "converged": False,
                            "iterations": it, "reason": "dn_stagnation",
                            "final_dn": dn}

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

    return {"mse_history": mse_history, "converged": False, "iterations": max_iter,
            "final_dn": _dn()}


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid — Phase 1: LM  ·  Phase 2: Adam + ternary regularization
# ─────────────────────────────────────────────────────────────────────────────

def _ternary_penalty(w: Tensor) -> Tensor:
    """w²(1-w²): zero at {-1,0,1}, positive in between."""
    return (w.pow(2) * (1.0 - w.pow(2)).clamp(min=0.0)).sum()


def _ternary_reg(model, lambda_sparse: float, lambda_attract: float) -> Tensor:
    reg = torch.tensor(0.0)
    for name, p in model.named_parameters():
        if "weight" in name:
            reg = reg + lambda_sparse * p.abs().sum()
            reg = reg + lambda_attract * _ternary_penalty(p)
    return reg


def _soft_threshold(w: Tensor, threshold: float) -> Tensor:
    return w.sign() * (w.abs() - threshold).clamp(min=0.0)


def _dn_stuck(model) -> float:
    """Fraction of weights stuck in (0.1, 0.9) — haven't committed to 0 or ±1."""
    parts = []
    for name, p in model.named_parameters():
        if "weight" in name:
            a = p.data.abs()
            parts.append(((a > 0.1) & (a < 0.9)).float())
    if not parts:
        return 0.0
    return torch.cat([s.flatten() for s in parts]).mean().item()


def lm_train_hybrid(
    model,
    x: Tensor,
    y: Tensor,
    max_iter: int = 1000,
    # ── Phase 1 (LM) ─────────────────────────────────────────────────────────
    mu_init: float = 1e-2,
    mu_min: float = 1e-10,
    mu_max: float = 1e10,
    tol_mse: float = 2e-3,
    crystallize_n: int = 2,
    p1_patience: int = 30,
    p1_fraction: float = 0.4,
    batch_size: int = 0,
    # ── Phase 2 (Adam + ternary reg, λ warm-up) ──────────────────────────────
    lr_p2: float = 1e-2,
    lambda_sparse: float = 1e-3,
    lambda_attract: float = 0.1,
    prox_threshold: float = 5e-4,
    tol_dn: float = 0.05,
    p2_patience: int = 50,
    # ── Phase 3 (hardening: 10× reg, fixed steps) ────────────────────────────
    p3_steps: int = 200,
    p3_mse_gate: float = 0.15,   # skip Phase 3 if best_mse ≥ this
    # ─────────────────────────────────────────────────────────────────────────
    verbose: bool = False,
    sample_weight: "Tensor | None" = None,
) -> dict:
    """
    Three-phase hybrid optimizer: LM → ternary reg → hardening.

    Phase 1 — LM with smooth crystallization.
        Fast second-order convergence toward a low-MSE continuous solution.
        Stops when mse < tol_mse, stagnation, or p1_fraction * max_iter exhausted.

    Phase 2 — Adam with ternary regularization (Proximal-style, λ warm-up).
        Applies  λ_s·||w||₁ + λ_a·w²(1-w²)  with linear warm-up from 0→λ.
        Dual stopping: exits when BOTH mse < tol_mse AND stuck < tol_dn.
        Projects weights to [-1, 1] after every step.

    Phase 3 — hardening (10× regularization, fixed budget).
        Short burst with lambda_sparse*10 and lambda_attract*10.  No warm-up,
        no stagnation check.  Pushes near-integer weights past the threshold
        so crisp crystallization becomes non-destructive.
        Skipped if best MSE seen so far ≥ p3_mse_gate (solution not useful).

    Rationale for Phase 3:
        After Phase 2, weights often reach mse < tol_mse but some remain in
        (0.1, 0.9) — too far from integers for crisp crystallization to be
        lossless.  A hardening burst with 10× reg pushes these stragglers
        over the edge at the cost of a small MSE increase that crisp
        rounding then absorbs.
    """
    mse_history: list[float] = []
    N = len(x)
    use_minibatch = batch_size > 0 and batch_size < N
    p1_end = int(max_iter * p1_fraction)

    def _full_mse() -> float:
        with torch.no_grad():
            pred = functional_call(model, dict(model.named_parameters()), x)
            sq = (pred - y) ** 2
            if sample_weight is not None:
                return (sq * sample_weight).mean().item()
            return sq.mean().item()

    def _project_weights() -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "weight" in name:
                    p.data.clamp_(-1.0, 1.0)

    def _mse_loss(pred: Tensor) -> Tensor:
        if sample_weight is not None:
            return ((pred - y) ** 2 * sample_weight).mean()
        return F.mse_loss(pred, y)

    # ── Phase 1: LM ──────────────────────────────────────────────────────────
    mu = mu_init
    best_mse = float("inf")
    stagnation = 0
    p1_converged = False

    for it in range(p1_end):
        if use_minibatch:
            idx = torch.randperm(N)[:batch_size]
            xb, yb = x[idx], y[idx]
            wb = sample_weight[idx] if sample_weight is not None else None
        else:
            xb, yb = x, y
            wb = sample_weight

        e, J = _compute_jacobian(model, xb, yb)
        if wb is not None:
            sw = wb.sqrt()
            e = e * sw
            J = J * sw.unsqueeze(1)

        mse_full = _full_mse() if use_minibatch else (e ** 2).mean().item()
        mse_history.append(mse_full)
        best_mse = min(best_mse, mse_full)

        if verbose and it % 50 == 0:
            print(f"  P1 iter {it:4d}  mse={mse_full:.6f}  mu={mu:.2e}")

        if mse_full < tol_mse:
            p1_converged = True
            if verbose:
                print(f"  P1 converged at iter {it}  mse={mse_full:.6f}")
            break

        if mse_full < best_mse - 1e-6:
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= p1_patience:
                if verbose:
                    print(f"  P1 stagnated at iter {it}  mse={mse_full:.6f}")
                break

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

    p1_iters = len(mse_history)
    p1_mse = mse_history[-1] if mse_history else float("inf")
    if verbose:
        print(f"  Phase 1 done: mse={p1_mse:.5f}  stuck={_dn_stuck(model):.3f}"
              f"  iters={p1_iters}  converged={p1_converged}")

    # ── Phase 2: Adam + ternary regularization ────────────────────────────────
    p2_iters = max_iter - p1_end
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_p2)

    best_mse2 = float("inf")
    stagnation2 = 0
    p2_patience_eff = max(p2_patience, p2_iters // 5)
    p2_reason = "max_iter"
    p2_dual_stop = False

    for step in range(p2_iters):
        scale = (step + 1) / p2_iters    # 0 → 1  (λ warm-up)
        ls = lambda_sparse * scale
        la = lambda_attract * scale

        optimizer.zero_grad()
        pred = model(x)
        mse_loss = _mse_loss(pred)
        reg = _ternary_reg(model, ls, la)
        (mse_loss + reg).backward()
        optimizer.step()

        if prox_threshold > 0:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "weight" in name:
                        p.data = _soft_threshold(p.data, prox_threshold * scale)

        _project_weights()

        mse = mse_loss.item()
        mse_history.append(mse)
        best_mse = min(best_mse, mse)

        if verbose and step % 100 == 0:
            dn = _dn_stuck(model)
            print(f"  P2 step {step:4d}  mse={mse:.6f}  stuck={dn:.3f}  "
                  f"reg={reg.item():.5f}")

        dn = _dn_stuck(model)
        if mse < tol_mse and dn < tol_dn:
            if verbose:
                print(f"  P2 dual-stop at step {step}  mse={mse:.6f}  stuck={dn:.4f}")
            p2_reason = "dual_stop"
            p2_dual_stop = True
            break

        if mse < best_mse2 - 1e-6:
            best_mse2 = mse
            stagnation2 = 0
        else:
            stagnation2 += 1
            if stagnation2 >= p2_patience_eff:
                p2_reason = "p2_stagnation"
                break

    if verbose:
        print(f"  Phase 2 done: mse={mse_history[-1]:.5f}  stuck={_dn_stuck(model):.3f}"
              f"  reason={p2_reason}")

    # ── Phase 3: hardening ────────────────────────────────────────────────────
    # Skip if already dual-stopped (fully crystallized) or MSE is hopeless.
    if not p2_dual_stop and best_mse < p3_mse_gate and p3_steps > 0:
        if verbose:
            print(f"  Phase 3: hardening ({p3_steps} steps, 10× reg)")
        for step in range(p3_steps):
            optimizer.zero_grad()
            pred = model(x)
            mse_loss = _mse_loss(pred)
            reg = _ternary_reg(model,
                               lambda_sparse * 3.0,
                               lambda_attract * 3.0)
            (mse_loss + reg).backward()
            optimizer.step()
            _project_weights()
            mse_history.append(mse_loss.item())

        final_dn = _dn_stuck(model)
        final_mse = _full_mse()
        converged = final_mse < tol_mse and final_dn < tol_dn
        reason = "hardening_converged" if converged else "hardening_done"
        if verbose:
            print(f"  Phase 3 done: mse={final_mse:.5f}  stuck={final_dn:.3f}"
                  f"  converged={converged}")
    else:
        converged = p2_dual_stop
        reason = p2_reason

    return {
        "mse_history": mse_history,
        "converged": converged,
        "iterations": len(mse_history),
        "reason": reason,
        "p1_iters": p1_iters,
        "p1_converged": p1_converged,
    }
