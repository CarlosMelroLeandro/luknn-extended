"""
Evaluation of Proximal variants on MONK-1, Mushroom, Spambase, and Musk v2.

Variants:
  Proximal        — standard two-phase: MSE → ternary reg warm-up + Phase 3 hardening
  ProximalTopK    — same phases but restricts each neuron to top-K active weights
  ProximalGroupLasso — group-lasso penalty per input neuron (fan-in restriction)
  ProximalL0      — L0 pseudo-norm via continuous relaxation (straight-through)

Statistical robustness: 30 independent trials per variant/dataset.
95% CI via t-distribution; pairwise Wilcoxon signed-rank with Holm-Bonferroni
correction for multiple comparisons.

Usage:
  python benchmark/proximal_variants_eval.py
  python benchmark/proximal_variants_eval.py --dataset monk --trials 30
  python benchmark/proximal_variants_eval.py --dataset spambase --trials 30
  python benchmark/proximal_variants_eval.py --dataset all
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

from luknn.layers.lukasiewicz_linear import make_lukasiewicz_net
from luknn.benchmark.datasets import load_monk, load_mushroom, load_spambase, load_musk
from luknn.benchmark.metrics import compute_f1, compute_accuracy, compute_delta_n
from luknn.benchmark.stats import ci95, format_ci, wilcoxon_pairwise_holm, print_pairwise
from luknn.optimizers import (
    ProximalOptimizer,
    ProximalTopK,
    ProximalGroupLasso,
    ProximalL0,
)


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(n_features: int):
    return make_lukasiewicz_net(
        n_features, n_hidden_layers=2, hidden_width=n_features, mode="clamp"
    )


# ── Variant registry ──────────────────────────────────────────────────────────

VARIANTS = {
    "Proximal": lambda model: ProximalOptimizer(
        model, lr=5e-3, lambda_sparse=1e-3, lambda_attract=0.1,
        phase1_fraction=0.6,
    ),
    "ProximalTopK": lambda model: ProximalTopK(
        model, lr=5e-3, lambda_sparse=1e-3, lambda_attract=0.1,
        phase1_fraction=0.6,
    ),
    "ProximalGroupLasso": lambda model: ProximalGroupLasso(
        model, lr=5e-3, lambda_sparse=1e-3, lambda_attract=0.1,
        phase1_fraction=0.6,
    ),
    "ProximalL0": lambda model: ProximalL0(
        model, lr=5e-3, lambda_sparse=1e-3, lambda_attract=0.1,
        phase1_fraction=0.6,
    ),
}


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    variant_name: str,
    make_opt,
    ds,
    seed: int,
    max_iter: int,
    tol_mse: float,
) -> dict:
    torch.manual_seed(seed)
    model = _make_model(ds.n_features)
    opt = make_opt(model)

    t0 = time.perf_counter()
    res = opt.train(ds.X_train, ds.y_train, tol_mse=tol_mse, max_iter=max_iter)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        pred = model(ds.X_test)

    f1  = compute_f1(pred, ds.y_test)
    acc = compute_accuracy(pred, ds.y_test)
    dn  = compute_delta_n(model)

    return {
        "variant":    variant_name,
        "seed":       seed,
        "f1":         round(f1, 4),
        "acc":        round(acc, 4),
        "dn_post":    round(dn, 4),
        "converged":  res.converged,
        "iterations": res.iterations,
        "final_mse":  round(res.final_mse, 6),
        "time_s":     round(elapsed, 2),
        "reason":     res.reason,
    }


# ── Dataset runner ────────────────────────────────────────────────────────────

def run_dataset(
    dataset: str,
    n_trials: int,
    max_iter: int,
    tol_mse: float,
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}  trials={n_trials}  max_iter={max_iter}")
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
            r = run_trial(vname, make_opt, ds, seed, max_iter, tol_mse)
            print(f"    trial {trial:2d}: F1={r['f1']:.3f}  dn={r['dn_post']:.3f}  "
                  f"iters={r['iterations']:4d}  t={r['time_s']:.1f}s  {r['reason']}")
            rows.append(r)

    return pd.DataFrame(rows)


# ── Summary with 95% CI ───────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, dataset: str) -> None:
    print(f"\n{'='*60}")
    print(f"SUMMARY — {dataset.upper()}  (n={len(df)//len(VARIANTS)} trials per variant)")
    print(f"{'='*60}")
    print(f"  {'Variant':<22}  {'F1 [95% CI]':>28}  {'dn_post':>8}  {'Conv%':>6}  {'Iters':>6}")
    print("  " + "-" * 78)
    for vname in VARIANTS:
        g = df[df["variant"] == vname]
        f1_mean, f1_lo, f1_hi = ci95(g["f1"].values)
        dn_mean = g["dn_post"].mean()
        conv_pct = 100 * g["converged"].mean()
        iters_mean = g["iterations"].mean()
        ci_str = format_ci(f1_mean, f1_lo, f1_hi)
        print(f"  {vname:<22}  {ci_str:>28}  {dn_mean:8.3f}  {conv_pct:5.0f}%  {iters_mean:6.0f}")


def print_tests(df: pd.DataFrame, dataset: str) -> None:
    table = wilcoxon_pairwise_holm(df, metric="f1", variants=list(VARIANTS.keys()))
    print_pairwise(table, metric="f1", dataset=dataset)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ALL_DATASETS = ["monk", "mushroom", "spambase", "musk"]
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ALL_DATASETS + ["all"], default="all")
    p.add_argument("--trials",   type=int,   default=None,
                   help="Trials per variant (default: 30 for all datasets)")
    p.add_argument("--max_iter", type=int,   default=None,
                   help="Max iterations (default: 500 monk, 300 others)")
    p.add_argument("--tol_mse",  type=float, default=2e-3)
    p.add_argument("--out",      type=str,   default=None)
    p.add_argument("--no_tests", action="store_true",
                   help="Skip pairwise statistical tests")
    args = p.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    _trials   = {"monk": 30, "mushroom": 30, "spambase": 30, "musk": 30}
    _max_iter = {"monk": 500, "mushroom": 300, "spambase": 300, "musk": 300}

    results_dir = Path(__file__).parent.parent / "results" / "proximal_variants"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for ds_name in datasets:
        n_trials = args.trials   or _trials[ds_name]
        max_iter = args.max_iter or _max_iter[ds_name]
        df = run_dataset(ds_name, n_trials, max_iter, args.tol_mse)
        df.insert(0, "dataset", ds_name)
        print_summary(df, ds_name)
        if not args.no_tests:
            print_tests(df, ds_name)
        per_ds_path = results_dir / f"{ds_name}_proximal_variants.csv"
        df.to_csv(per_ds_path, index=False)
        print(f"\n  → {per_ds_path}")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = args.out or str(results_dir / "results.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nResultados guardados em {out_path}")


if __name__ == "__main__":
    main()
