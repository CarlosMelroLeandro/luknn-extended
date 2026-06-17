"""
Statistical utilities for benchmark evaluation.

Provides:
  ci95(data)                          — mean + 95% CI via t-distribution
  format_ci(mean, lo, hi)             — compact string "mean [lo, hi]"
  wilcoxon_pairwise_holm(df, metric)  — pairwise Wilcoxon + Holm-Bonferroni table
  print_pairwise(table, metric)       — pretty-print the test table
  run_5x2cv(...)                      — 5×2 stratified CV for slow variants
"""

from __future__ import annotations
import itertools
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


# ── Confidence interval ───────────────────────────────────────────────────────

def ci95(data: np.ndarray | list[float]) -> tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) using the t-distribution (ddof=1)."""
    a = np.asarray(data, dtype=float)
    n = len(a)
    mean = float(np.mean(a))
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(np.std(a, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - t_crit * se, mean + t_crit * se


def format_ci(mean: float, lo: float, hi: float, d: int = 3) -> str:
    """Format as 'mean [lo, hi]' with d decimal places."""
    fmt = f".{d}f"
    return f"{mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


# ── Pairwise Wilcoxon + Holm-Bonferroni ──────────────────────────────────────

def _holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction (FWER control)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[order]
    adjusted = np.minimum(1.0, np.maximum.accumulate(sorted_p * (m - np.arange(m))))
    result = np.empty(m)
    result[order] = adjusted
    return result.tolist()


def wilcoxon_pairwise_holm(
    df: pd.DataFrame,
    metric: str,
    variants: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute pairwise two-sided Wilcoxon signed-rank tests between variants,
    then apply Holm-Bonferroni correction.

    Assumes df has columns 'variant' and ``metric``, with rows paired by
    insertion order within each variant (same seed / same fold).

    Returns a DataFrame with columns:
      v1, v2, p_raw, p_holm, significant (α=0.05)
    """
    if variants is None:
        variants = list(df["variant"].unique())

    pairs = list(itertools.combinations(variants, 2))
    raw_pvals: list[float] = []
    records: list[dict] = []

    for v1, v2 in pairs:
        s1 = df[df["variant"] == v1][metric].values
        s2 = df[df["variant"] == v2][metric].values
        n = min(len(s1), len(s2))
        d = s1[:n] - s2[:n]

        if np.all(d == 0):
            # identical distributions — no test possible
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(s1[:n], s2[:n],
                                      zero_method="wilcox",
                                      alternative="two-sided")
            except ValueError:
                p = 1.0

        raw_pvals.append(p)
        records.append({"v1": v1, "v2": v2, "p_raw": round(p, 5)})

    corrected = _holm_bonferroni(raw_pvals)
    for rec, p_holm in zip(records, corrected):
        rec["p_holm"] = round(float(p_holm), 5)
        rec["significant"] = float(p_holm) < 0.05

    return pd.DataFrame(records)


def print_pairwise(table: pd.DataFrame, metric: str, dataset: str = "") -> None:
    """Pretty-print the pairwise test table."""
    tag = f" — {dataset.upper()}" if dataset else ""
    print(f"\n  Pairwise Wilcoxon (metric={metric}, Holm-Bonferroni){tag}")
    print(f"  {'Pair':<40} {'p_raw':>8} {'p_holm':>8} {'sig':>5}")
    print("  " + "-" * 65)
    for _, row in table.iterrows():
        sig = "  *" if row["significant"] else "   "
        print(f"  {row['v1']:>20} vs {row['v2']:<20} "
              f"{row['p_raw']:8.4f} {row['p_holm']:8.4f}{sig}")


# ── 5×2 stratified cross-validation ──────────────────────────────────────────

def run_5x2cv(
    variant_name: str,
    make_model_fn: Callable,
    make_opt_fn: Callable,
    run_fold_fn: Callable,
    X: "torch.Tensor",
    y: "torch.Tensor",
    n_reps: int = 5,
    seed_base: int = 42,
) -> list[dict]:
    """
    5×2 stratified cross-validation runner.

    Args:
        variant_name:  name of the variant being tested
        make_model_fn: callable(n_features) → model
        make_opt_fn:   callable(model) → optimizer
        run_fold_fn:   callable(model, opt, X_tr, y_tr, X_te, y_te) → dict of metrics
        X, y:          full dataset (torch tensors)
        n_reps:        number of 2-fold repetitions (default 5 → 10 measurements)
        seed_base:     base seed for reproducibility

    Returns:
        List of dicts with 'rep', 'fold', 'variant', plus whatever run_fold_fn returns.
    """
    import torch
    from sklearn.model_selection import StratifiedKFold

    y_np = y.cpu().numpy().round().astype(int).ravel()
    records: list[dict] = []

    for rep in range(n_reps):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed_base + rep * 7)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X.cpu().numpy(), y_np)):
            seed = seed_base + rep * 100 + fold * 13
            torch.manual_seed(seed)

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            model = make_model_fn(X.shape[1])
            opt   = make_opt_fn(model)
            metrics = run_fold_fn(model, opt, X_tr, y_tr, X_te, y_te)

            rec = {"variant": variant_name, "rep": rep, "fold": fold}
            rec.update(metrics)
            records.append(rec)

    return records
