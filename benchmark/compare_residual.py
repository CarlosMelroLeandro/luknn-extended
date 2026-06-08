"""
Baseline vs residual comparison on classification datasets.

Usage:
  # Mushroom
  python benchmark/compare_residual.py --dataset mushroom

  # Heart Disease
  python benchmark/compare_residual.py --dataset heart

  # Both
  python benchmark/compare_residual.py --dataset all

  # Verbose with more trials
  python benchmark/compare_residual.py --dataset mushroom --n_trials 10 --verbose

Produces:
  results/residual/comparison_<dataset>_<timestamp>.csv
  results/residual/comparison_<dataset>_<timestamp>.json
  (and prints a comparison table to stdout)
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
import numpy as np
from scipy import stats

from luknn.benchmark.config import load_config, ExperimentConfig
from luknn.benchmark.runner import BenchmarkRunner
from luknn.benchmark.metrics import save_results, BenchmarkResult


DATASET_PAIRS = {
    "mushroom": (
        "configs/lm_mushroom.yaml",
        "configs/lm_residual_mushroom.yaml",
    ),
    "heart": (
        "configs/lm_heart.yaml",
        "configs/lm_residual_heart.yaml",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Baseline vs Residual comparison")
    p.add_argument("--dataset", default="all",
                   choices=["mushroom", "heart", "all"])
    p.add_argument("--n_trials", type=int, default=None,
                   help="Override n_trials in config (default: use config value)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--results_dir", default="results/residual")
    return p.parse_args()


def _apply_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    from dataclasses import replace
    kw: dict = {}
    if args.n_trials:   kw["n_trials"]    = args.n_trials
    if args.seed:       kw["seed"]        = args.seed
    if args.verbose:    kw["verbose"]     = True
    kw["results_dir"] = args.results_dir
    return replace(cfg, **kw) if kw else cfg


def _stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std)."""
    a = np.array(values)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def _wilcoxon(a: list[float], b: list[float]) -> float | None:
    """Wilcoxon signed-rank test; returns p-value or None if insufficient data."""
    if len(a) < 2 or len(a) != len(b):
        return None
    try:
        _, p = stats.wilcoxon(a, b)
        return float(p)
    except ValueError:
        return None


def print_comparison(
    baseline: list[BenchmarkResult],
    residual: list[BenchmarkResult],
    dataset: str,
) -> None:
    metrics = [
        ("MSE final",    "final_mse",          True),   # True = lower is better
        ("Accuracy",     "accuracy",           False),
        ("F1",           "f1",                 False),
        ("Iters",        "iterations",         True),
        ("Time (s)",     "total_time_s",       True),
        ("Crystallized", "is_crystallized",    False),
        ("λ-similar",    "lambda_similarity",  False),
    ]

    print(f"\n{'='*72}")
    print(f"  Dataset: {dataset.upper()}   "
          f"  Baseline n={len(baseline)}   Residual n={len(residual)}")
    print(f"{'='*72}")
    print(f"{'Metric':<20} {'Baseline':>14} {'Residual':>14} {'p-value':>10} {'Best':>8}")
    print(f"{'-'*72}")

    for label, attr, lower_is_better in metrics:
        b_vals = [getattr(r, attr) for r in baseline]
        r_vals = [getattr(r, attr) for r in residual]

        if isinstance(b_vals[0], bool):
            b_str = f"{sum(b_vals)}/{len(b_vals)} ✓"
            r_str = f"{sum(r_vals)}/{len(r_vals)} ✓"
            p_str = "—"
            winner = "—"
        else:
            bm, bs = _stats(b_vals)
            rm, rs = _stats(r_vals)
            b_str = f"{bm:.4f} ± {bs:.4f}"
            r_str = f"{rm:.4f} ± {rs:.4f}"
            p = _wilcoxon(b_vals, r_vals)
            p_str = f"{p:.3f}" if p is not None else "—"
            sig = "*" if (p is not None and p < 0.05) else ""
            if lower_is_better:
                winner = ("Residual" if rm < bm else "Baseline") + sig
            else:
                winner = ("Residual" if rm > bm else "Baseline") + sig

        print(f"{label:<20} {b_str:>14} {r_str:>14} {p_str:>10} {winner:>8}")

    print(f"{'='*72}\n")


def run_comparison(dataset: str, args) -> list[BenchmarkResult]:
    base_cfg_path, res_cfg_path = DATASET_PAIRS[dataset]
    root = Path(__file__).parents[1]

    base_cfg = _apply_overrides(load_config(root / base_cfg_path), args)
    res_cfg  = _apply_overrides(load_config(root / res_cfg_path),  args)

    print(f"\n[Baseline]  {base_cfg.name}")
    baseline = BenchmarkRunner(base_cfg).run()

    print(f"\n[Residual]  {res_cfg.name}")
    residual = BenchmarkRunner(res_cfg).run()

    print_comparison(baseline, residual, dataset)

    all_results = baseline + residual
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    saved = save_results(all_results, args.results_dir)
    print(f"  Results → {saved}")

    return all_results


def main():
    args = parse_args()
    datasets = list(DATASET_PAIRS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_comparison(ds, args)


if __name__ == "__main__":
    main()
