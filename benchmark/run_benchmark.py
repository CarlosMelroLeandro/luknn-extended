"""
Benchmark entry-point.

Usage
-----
# Single config
python benchmark/run_benchmark.py configs/lm_baseline.yaml

# Compare all three optimizers on same dataset
python benchmark/run_benchmark.py --compare configs/lm_baseline.yaml configs/ste.yaml configs/proximal.yaml

# Custom formula / n_values overrides
python benchmark/run_benchmark.py configs/lm_baseline.yaml --formula f2 --n_values 4 --n_trials 5
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
from luknn.benchmark.config import load_config, ExperimentConfig
from luknn.benchmark.runner import BenchmarkRunner
from luknn.benchmark.metrics import save_results, BenchmarkResult


def parse_args():
    p = argparse.ArgumentParser(description="LNN Optimizer Benchmark")
    p.add_argument("configs", nargs="+", help="YAML config files")
    p.add_argument("--compare", action="store_true",
                   help="Compare multiple methods and print summary table")
    p.add_argument("--formula", default=None)
    p.add_argument("--n_values", type=int, default=None)
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--results_dir", default=None)
    return p.parse_args()


def apply_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    """Apply CLI overrides on top of YAML config."""
    from dataclasses import replace
    kwargs = {}
    if args.formula:    kwargs["formula"]     = args.formula
    if args.n_values:   kwargs["n_values"]    = args.n_values
    if args.n_trials:   kwargs["n_trials"]    = args.n_trials
    if args.seed:       kwargs["seed"]        = args.seed
    if args.verbose:    kwargs["verbose"]     = True
    if args.results_dir: kwargs["results_dir"] = args.results_dir
    return replace(cfg, **kwargs) if kwargs else cfg


def print_table(all_results: list[BenchmarkResult]) -> None:
    """Print a compact comparison table."""
    col_w = 12
    headers = ["Method", "Dataset", "MSE", "Acc", "F1", "Cryst", "λ", "Time(s)", "Iters"]
    print("\n" + "─" * (col_w * len(headers)))
    print("  ".join(f"{h:<{col_w}}" for h in headers))
    print("─" * (col_w * len(headers)))

    for r in all_results:
        row = [
            r.method,
            r.dataset[:col_w],
            f"{r.final_mse:.5f}",
            f"{r.accuracy:.3f}",
            f"{r.f1:.3f}" if r.f1 == r.f1 else "nan",
            "✓" if r.is_crystallized else "✗",
            f"{r.lambda_similarity:.3f}",
            f"{r.total_time_s:.1f}",
            str(r.iterations),
        ]
        print("  ".join(f"{v:<{col_w}}" for v in row))
    print("─" * (col_w * len(headers)))


def main():
    args = parse_args()
    all_results: list[BenchmarkResult] = []

    for cfg_path in args.configs:
        cfg = apply_overrides(load_config(cfg_path), args)
        print(f"\n{'='*60}")
        print(f"  {cfg.name}  [{cfg.optimizer_method}]")
        print(f"{'='*60}")

        runner = BenchmarkRunner(cfg)
        results = runner.run()
        all_results.extend(results)

        for r in results:
            print(r.summary())

    # Save all results
    if all_results:
        out_dir = all_results[0].config.get("results_dir", "results")
        saved = save_results(all_results, out_dir)
        print(f"\nResults saved → {saved}")

    if args.compare and len(all_results) > 1:
        print_table(all_results)


if __name__ == "__main__":
    main()
