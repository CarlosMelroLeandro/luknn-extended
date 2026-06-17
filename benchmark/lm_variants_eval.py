"""
Evaluation of 4 LM variants on MONK-1, Mushroom, Spambase, and Musk v2.

Variants:
  LM_base        — original (n=2, crystallize from iter 0, mse-only stop)
  LM_delayed     — delayed crystallization (start_fraction=0.3)
  LM_progressive — progressive n schedule (2→4→8→16)
  LM_dual        — dual stopping: mse + Δ(N)/P < tol_dn
  LM_hybrid      — Phase 1 (LM) → Phase 2 (Adam proximal)

Statistical robustness strategy:
  • monk / mushroom: 30 independent trials (fast enough; ~3–8 s/trial)
  • spambase / musk: 5×2 stratified cross-validation (10 paired measurements
    per variant) — LM Jacobian cost makes 30 full-dataset trials infeasible.

95% CI via t-distribution; pairwise Wilcoxon signed-rank with
Holm-Bonferroni correction for multiple comparisons.

Usage:
  python benchmark/lm_variants_eval.py                          # all datasets
  python benchmark/lm_variants_eval.py --dataset monk
  python benchmark/lm_variants_eval.py --dataset spambase       # uses 5×2 CV
  python benchmark/lm_variants_eval.py --dataset monk --trials 30
  python benchmark/lm_variants_eval.py --dataset spambase --cv_reps 5
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from luknn.network.luknn import make_network
from luknn.benchmark.datasets import load_monk, load_mushroom, load_spambase, load_musk
from luknn.benchmark.metrics import compute_f1, compute_accuracy, compute_delta_n
from luknn.benchmark.stats import ci95, format_ci, wilcoxon_pairwise_holm, print_pairwise
from luknn.network.crystallization import (
    representation_error,
    progressive_crystallize,
    crisp_crystallize_weights,
    crisp_crystallize_bias,
)
from luknn.optimizers import (
    LMOptimizer,
    LMDelayedOptimizer,
    LMProgressiveOptimizer,
    LMDualOptimizer,
    LMHybridOptimizer,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def force_crystallize(model) -> None:
    for m in model.modules():
        if hasattr(m, "crystallize"):
            m.crystallize()
        elif isinstance(m, nn.Linear):
            m.weight.data = crisp_crystallize_weights(
                progressive_crystallize(m.weight.data)
            )
            m.bias.data = crisp_crystallize_bias(
                progressive_crystallize(m.bias.data)
            )


def is_crystallized(model) -> bool:
    w = model.flat_weights()
    return representation_error(w).item() < 1e-2


VARIANTS = {
    "LM_base": lambda model: LMOptimizer(
        model, mu_init=1e-2, crystallize_n=2, patience=50, prune=False
    ),
    "LM_delayed": lambda model: LMDelayedOptimizer(
        model, mu_init=1e-2, crystallize_n=2,
        crystallize_start_fraction=0.3, patience=50, prune=False
    ),
    "LM_progressive": lambda model: LMProgressiveOptimizer(
        model, mu_init=1e-2,
        n_schedule=(2, 4, 8, 16),
        schedule_fractions=(0.0, 0.5, 0.75, 0.9),
        patience=50, prune=False
    ),
    "LM_dual": lambda model: LMDualOptimizer(
        model, mu_init=1e-2, crystallize_n=2,
        tol_dn=0.05, dn_patience=50, patience=50, prune=False
    ),
    "LM_hybrid": lambda model: LMHybridOptimizer(
        model, mu_init=1e-2, crystallize_n=2,
        p1_fraction=0.4, p1_patience=30,
        lr_p2=1e-2, lambda_sparse=1e-3, lambda_attract=0.1,
        prox_threshold=5e-4, tol_dn=0.05, p2_patience=50,
        prune=False
    ),
}


def _lm_hidden_width(n_features: int) -> int:
    # LM Jacobian cost ∝ n_params (forward-mode AD, P passes per iter).
    # Aggressive caps to keep each step tractable when running sequentially.
    if n_features > 80:   # mushroom (111), musk (166)
        return 8
    if n_features > 40:   # spambase (57)
        return 12
    return n_features     # monk (17) — no cap needed


# ── Single-trial runner ───────────────────────────────────────────────────────

def run_trial(
    variant_name: str,
    make_opt,
    ds,
    seed: int,
    max_iter: int,
    tol_mse: float,
    batch_size: int,
) -> dict:
    torch.manual_seed(seed)
    hw = _lm_hidden_width(ds.n_features)
    model = make_network(ds.n_features, n_hidden_layers=2, hidden_width=hw)

    opt = make_opt(model)
    if hasattr(opt, "batch_size"):
        opt.batch_size = batch_size

    t0 = time.perf_counter()
    res = opt.train(ds.X_train, ds.y_train, tol_mse=tol_mse, max_iter=max_iter)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        pred_cont = model(ds.X_test)
    f1_cont = compute_f1(pred_cont, ds.y_test)
    dn_pre = compute_delta_n(model)

    force_crystallize(model)
    with torch.no_grad():
        pred_test = model(ds.X_test)

    f1_crisp = compute_f1(pred_test, ds.y_test)
    acc      = compute_accuracy(pred_test, ds.y_test)
    dn_post  = compute_delta_n(model)
    cryst    = is_crystallized(model)

    return {
        "variant":      variant_name,
        "seed":         seed,
        "f1_cont":      round(f1_cont, 4),
        "f1_crisp":     round(f1_crisp, 4),
        "acc":          round(acc, 4),
        "dn_pre":       round(dn_pre, 4),
        "dn_post":      round(dn_post, 4),
        "crystallized": cryst,
        "converged":    res.converged,
        "iterations":   res.iterations,
        "final_mse":    round(res.final_mse, 6),
        "time_s":       round(elapsed, 2),
        "reason":       res.reason,
    }


# ── 30-trial dataset runner ──────────────────────────────────────────────────

def run_dataset_trials(
    dataset: str,
    n_trials: int,
    max_iter: int,
    tol_mse: float,
    batch_size: int,
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}  mode=trials  n={n_trials}  max_iter={max_iter}")
    print(f"{'='*60}")

    if dataset == "monk":
        ds = load_monk(problem=1, seed=42)
    elif dataset == "mushroom":
        ds = load_mushroom(seed=42)
    elif dataset == "spambase":
        ds = load_spambase(seed=42)
    elif dataset == "musk":
        ds = load_musk(seed=42)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"  features={ds.n_features}  train={len(ds.X_train)}  test={len(ds.X_test)}")

    rows = []
    for vname, make_opt in VARIANTS.items():
        print(f"\n  [{vname}]")
        for trial in range(n_trials):
            seed = 42 + trial * 17
            r = run_trial(vname, make_opt, ds, seed, max_iter, tol_mse, batch_size)
            print(f"    trial {trial:2d}: F1_cont={r['f1_cont']:.3f}  "
                  f"F1_crisp={r['f1_crisp']:.3f}  cryst={r['crystallized']}  "
                  f"ΔN_pre={r['dn_pre']:.1f}  iters={r['iterations']:4d}  "
                  f"t={r['time_s']:.1f}s  {r['reason']}")
            rows.append(r)

    return pd.DataFrame(rows)


# ── 5×2 CV dataset runner ────────────────────────────────────────────────────

def run_dataset_5x2cv(
    dataset: str,
    cv_reps: int,
    max_iter: int,
    tol_mse: float,
    batch_size: int,
) -> pd.DataFrame:
    """
    5×2 stratified cross-validation for slow datasets (spambase, musk).
    Each repetition produces 2 paired measurements (one per fold) per variant.
    Total: cv_reps × 2 = 10 observations per variant.
    """
    from sklearn.model_selection import StratifiedKFold

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}  mode=5x2cv  reps={cv_reps}  max_iter={max_iter}")
    print(f"{'='*60}")

    if dataset == "spambase":
        ds = load_spambase(seed=42)
    elif dataset == "musk":
        ds = load_musk(seed=42)
    else:
        raise ValueError(f"5×2 CV only for spambase/musk; got: {dataset}")

    X_all = torch.cat([ds.X_train, ds.X_test], dim=0)
    y_all = torch.cat([ds.y_train, ds.y_test], dim=0)
    y_np = y_all.cpu().numpy().round().astype(int).ravel()

    print(f"  features={ds.n_features}  total_samples={len(X_all)}")

    hw = _lm_hidden_width(ds.n_features)

    rows = []
    for vname, make_opt_fn in VARIANTS.items():
        print(f"\n  [{vname}]")
        for rep in range(cv_reps):
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 + rep * 7)
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all.cpu().numpy(), y_np)):
                seed = 42 + rep * 100 + fold * 13
                torch.manual_seed(seed)

                X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
                X_te, y_te = X_all[te_idx], y_all[te_idx]

                model = make_network(ds.n_features, n_hidden_layers=2, hidden_width=hw)
                opt = make_opt_fn(model)
                if hasattr(opt, "batch_size"):
                    opt.batch_size = batch_size

                t0 = time.perf_counter()
                res = opt.train(X_tr, y_tr, tol_mse=tol_mse, max_iter=max_iter)
                elapsed = time.perf_counter() - t0

                with torch.no_grad():
                    pred_cont = model(X_te)
                f1_cont = compute_f1(pred_cont, y_te)
                dn_pre  = compute_delta_n(model)

                force_crystallize(model)
                with torch.no_grad():
                    pred_test = model(X_te)

                f1_crisp = compute_f1(pred_test, y_te)
                acc      = compute_accuracy(pred_test, y_te)
                dn_post  = compute_delta_n(model)
                cryst    = is_crystallized(model)

                r = {
                    "variant":      vname,
                    "rep":          rep,
                    "fold":         fold,
                    "f1_cont":      round(f1_cont, 4),
                    "f1_crisp":     round(f1_crisp, 4),
                    "acc":          round(acc, 4),
                    "dn_pre":       round(dn_pre, 4),
                    "dn_post":      round(dn_post, 4),
                    "crystallized": cryst,
                    "converged":    res.converged,
                    "iterations":   res.iterations,
                    "final_mse":    round(res.final_mse, 6),
                    "time_s":       round(elapsed, 2),
                    "reason":       res.reason,
                }
                print(f"    rep {rep} fold {fold}: F1_cont={f1_cont:.3f}  "
                      f"F1_crisp={f1_crisp:.3f}  cryst={cryst}  "
                      f"ΔN_pre={dn_pre:.1f}  iters={res.iterations:4d}  "
                      f"t={elapsed:.1f}s")
                rows.append(r)

    df = pd.DataFrame(rows)
    # add 'seed' column for compatibility with summary/test functions
    df["seed"] = df["rep"] * 100 + df["fold"]
    return df


# ── Summary with 95% CI ───────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, dataset: str, mode: str) -> None:
    n_obs = len(df[df["variant"] == list(VARIANTS.keys())[0]])
    print(f"\n{'='*60}")
    print(f"SUMMARY — {dataset.upper()}  mode={mode}  (n={n_obs} obs per variant)")
    print(f"{'='*60}")
    print(f"  {'Variant':<16}  {'F1_crisp [95% CI]':>30}  {'Cryst%':>7}  {'Conv%':>6}  {'Iters':>6}")
    print("  " + "-" * 75)
    for vname in VARIANTS:
        g = df[df["variant"] == vname]
        f1_mean, f1_lo, f1_hi = ci95(g["f1_crisp"].values)
        cryst_pct = 100 * g["crystallized"].mean()
        conv_pct  = 100 * g["converged"].mean()
        iters_mean = g["iterations"].mean()
        ci_str = format_ci(f1_mean, f1_lo, f1_hi)
        print(f"  {vname:<16}  {ci_str:>30}  {cryst_pct:6.0f}%  {conv_pct:5.0f}%  {iters_mean:6.0f}")


def print_tests(df: pd.DataFrame, dataset: str) -> None:
    table = wilcoxon_pairwise_holm(df, metric="f1_crisp", variants=list(VARIANTS.keys()))
    print_pairwise(table, metric="f1_crisp", dataset=dataset)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    ALL_DATASETS = ["monk", "mushroom", "spambase", "musk"]
    p.add_argument("--dataset", choices=ALL_DATASETS + ["all"], default="all")
    p.add_argument("--trials",   type=int,   default=None,
                   help="Trials (monk/mushroom only; default 30)")
    p.add_argument("--cv_reps",  type=int,   default=None,
                   help="CV repetitions (spambase/musk only; default 5 → 10 obs)")
    p.add_argument("--max_iter", type=int,   default=None,
                   help="Max LM iterations (per-dataset default applies)")
    p.add_argument("--tol_mse",  type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=0,
                   help="Jacobian mini-batch (0=auto)")
    p.add_argument("--out",      type=str,   default=None)
    p.add_argument("--no_tests", action="store_true",
                   help="Skip pairwise statistical tests")
    args = p.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    # monk/mushroom: 30 independent trials
    _trials   = {"monk": 30, "mushroom": 30}
    # Reduced max_iter: hw caps mean fewer params → converges or stagnates faster.
    # spambase/musk use 5×2 CV; keep iters low to stay tractable sequentially.
    _max_iter = {"monk": 300, "mushroom": 150, "spambase": 80, "musk": 50}
    _batch    = {"monk": 0, "mushroom": 128, "spambase": 64, "musk": 32}
    # spambase/musk: 5×2 CV (10 paired observations)
    _cv_reps  = {"spambase": 5, "musk": 5}

    # Datasets needing 5×2 CV due to LM cost
    _large_datasets = {"spambase", "musk"}

    results_dir = Path(__file__).parent.parent / "results" / "lm_variants"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for ds_name in datasets:
        max_iter = args.max_iter or _max_iter[ds_name]
        batch_sz = args.batch_size if args.batch_size > 0 else _batch[ds_name]

        if ds_name in _large_datasets:
            cv_reps = args.cv_reps or _cv_reps[ds_name]
            df = run_dataset_5x2cv(ds_name, cv_reps, max_iter, args.tol_mse, batch_sz)
            mode = f"5x2cv"
        else:
            n_trials = args.trials or _trials[ds_name]
            df = run_dataset_trials(ds_name, n_trials, max_iter, args.tol_mse, batch_sz)
            mode = "trials"

        df.insert(0, "dataset", ds_name)
        print_summary(df, ds_name, mode)
        if not args.no_tests:
            print_tests(df, ds_name)

        per_ds_path = results_dir / f"{ds_name}_lm_variants.csv"
        df.to_csv(per_ds_path, index=False)
        print(f"\n  → {per_ds_path}")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = args.out or str(results_dir / "results.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nResultados guardados em {out_path}")


if __name__ == "__main__":
    main()
