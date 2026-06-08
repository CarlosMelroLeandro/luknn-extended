"""
Re-train with the best configs from the grid search and compare against the baseline.

For each dataset:
  1. Load the best hyperparameters from the tuning JSON.
  2. Build the Residual config (LM_Residual) with those params.
  3. Build the Baseline config (LM) with the same training budget.
  4. Run BenchmarkRunner × 10 trials for each.
  5. Print a comparison table + Wilcoxon p-values.
  6. Save to results/final/<dataset>_<timestamp>.json

Usage:
  python benchmark/retrain_best.py
  python benchmark/retrain_best.py --datasets mushroom heart
  python benchmark/retrain_best.py --n_trials 10 --results_dir results/final
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
from scipy import stats
import torch

from luknn.benchmark.config import ExperimentConfig
from luknn.benchmark.runner import BenchmarkRunner
from luknn.benchmark.metrics import BenchmarkResult, save_results


# ── Tuning files (grid-search results) ───────────────────────────────────────

ROOT = Path(__file__).parents[1]

TUNING_FILES = {
    "mushroom":      ROOT / "results/tuning/mushroom_20260604_091426.json",
    "heart":         ROOT / "results/tuning/heart_20260604_091426.json",
    "monk_1":        ROOT / "results/tuning/monk_1_20260604_091426.json",
    "monk_2":        ROOT / "results/tuning/monk_2_20260604_103838.json",
    "monk_3":        ROOT / "results/tuning/monk_3_20260604_114059.json",
    "breast_cancer": ROOT / "results/tuning/breast_cancer_20260604_091426.json",
}


def _load_best_params(dataset_key: str) -> dict:
    path = TUNING_FILES[dataset_key]
    return json.loads(path.read_text())["best_params"]


# ── Base configs per dataset ──────────────────────────────────────────────────
# Base configs define everything except params that vary with tuning.
# Baseline uses LM (standard network); residual uses LM_Residual with the
# best params found.  Training budget is equal for both.

def _mushroom_configs(best: dict, n_trials: int) -> tuple[ExperimentConfig, ExperimentConfig]:
    common = dict(
        seed=42, n_inputs=111,
        dataset_type="mushroom",
        tol_mse=0.15, max_iter=600, n_trials=n_trials, verbose=False,
    )
    opt_common = dict(
        patience=80, crystallize_n=2, prune=False, batch_size=512,
        mu_init=best["mu_init"],
    )
    baseline = ExperimentConfig(
        name="LM — Mushroom [final]",
        hidden_layers=[6, 4],
        optimizer_method="LM",
        optimizer_params=opt_common,
        **common,
    )
    residual = ExperimentConfig(
        name="LM_Residual — Mushroom [final]",
        hidden_layers=[6, 4],
        optimizer_method="LM_Residual",
        optimizer_params=opt_common,
        hidden_width=best["hidden_width"],
        n_blocks=best["n_blocks"],
        n_inner=1,
        **common,
    )
    return baseline, residual


def _heart_configs(best: dict, n_trials: int) -> tuple[ExperimentConfig, ExperimentConfig]:
    common = dict(
        seed=42, n_inputs=13,
        dataset_type="heart_disease", heart_subset="cleveland",
        tol_mse=0.15, max_iter=800, n_trials=n_trials, verbose=False,
    )
    opt_common = dict(
        patience=100, crystallize_n=2, batch_size=0,
        mu_init=best["mu_init"],
        prune=best.get("prune", True),
    )
    baseline = ExperimentConfig(
        name="LM — Heart [final]",
        hidden_layers=[6, 4],
        optimizer_method="LM",
        optimizer_params=opt_common,
        **common,
    )
    residual = ExperimentConfig(
        name="LM_Residual — Heart [final]",
        hidden_layers=[6, 4],
        optimizer_method="LM_Residual",
        optimizer_params=opt_common,
        hidden_width=best["hidden_width"],
        n_blocks=best["n_blocks"],
        n_inner=1,
        **common,
    )
    return baseline, residual


def _monk_configs(problem: int, best: dict, n_trials: int) -> tuple[ExperimentConfig, ExperimentConfig]:
    common = dict(
        seed=42, n_inputs=17,
        dataset_type="monk", monk_problem=problem,
        tol_mse=0.10, max_iter=500, n_trials=n_trials, verbose=False,
    )
    opt_common = dict(
        patience=80, crystallize_n=2, prune=False, batch_size=0,
        mu_init=best["mu_init"],
    )
    baseline = ExperimentConfig(
        name=f"LM — MONK-{problem} [final]",
        hidden_layers=[8, 4],       # architecture used in experiments/monk/run.py
        optimizer_method="LM",
        optimizer_params=opt_common,
        **common,
    )
    residual = ExperimentConfig(
        name=f"LM_Residual — MONK-{problem} [final]",
        hidden_layers=[8, 4],
        optimizer_method="LM_Residual",
        optimizer_params=opt_common,
        hidden_width=best["hidden_width"],
        n_blocks=best["n_blocks"],
        n_inner=1,
        **common,
    )
    return baseline, residual


def _breast_cancer_configs(best: dict, n_trials: int) -> tuple[ExperimentConfig, ExperimentConfig]:
    common = dict(
        seed=42, n_inputs=20,
        dataset_type="breast_cancer",
        tol_mse=0.15, max_iter=800, n_trials=n_trials, verbose=False,
    )
    opt_common = dict(
        patience=100, crystallize_n=2, batch_size=0,
        mu_init=best["mu_init"],
        prune=best.get("prune", True),
    )
    baseline = ExperimentConfig(
        name="LM — Breast Cancer [final]",
        hidden_layers=[8, 8],       # architecture used in experiments/breast_cancer/run.py
        optimizer_method="LM",
        optimizer_params=opt_common,
        **common,
    )
    residual = ExperimentConfig(
        name="LM_Residual — Breast Cancer [final]",
        hidden_layers=[8, 8],
        optimizer_method="LM_Residual",
        optimizer_params=opt_common,
        hidden_width=best["hidden_width"],
        n_blocks=best["n_blocks"],
        n_inner=1,
        **common,
    )
    return baseline, residual


# ── Statistics ────────────────────────────────────────────────────────────────

def _stats(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def _wilcoxon(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    try:
        _, p = stats.wilcoxon(a, b)
        return float(p)
    except ValueError:
        return None


# ── Comparison table ──────────────────────────────────────────────────────────

METRICS = [
    ("MSE final",    "final_mse",          True),
    ("Accuracy",     "accuracy",           False),
    ("F1",           "f1",                 False),
    ("Iters",        "iterations",         True),
    ("Time (s)",     "total_time_s",       True),
    ("Crystallized", "is_crystallized",    False),
    ("λ-similar",    "lambda_similarity",  False),
]


def print_comparison(
    label: str,
    baseline: list[BenchmarkResult],
    residual: list[BenchmarkResult],
    best_params: dict,
) -> dict:
    print(f"\n{'='*74}")
    print(f"  {label}")
    print(f"  Residual params: {best_params}")
    print(f"  Baseline n={len(baseline)}   Residual n={len(residual)}")
    print(f"{'='*74}")
    print(f"{'Metric':<22} {'Baseline':>14} {'Residual':>14} {'p-value':>10} {'Best':>10}")
    print(f"{'-'*74}")

    row_data = {}
    for label_m, attr, lower_better in METRICS:
        b_vals = [getattr(r, attr) for r in baseline]
        r_vals = [getattr(r, attr) for r in residual]

        if isinstance(b_vals[0], bool):
            b_str  = f"{sum(b_vals)}/{len(b_vals)} ✓"
            r_str  = f"{sum(r_vals)}/{len(r_vals)} ✓"
            p_str  = "—"
            winner = "—"
            row_data[attr] = {"baseline": sum(b_vals)/len(b_vals), "residual": sum(r_vals)/len(r_vals)}
        else:
            bm, bs = _stats(b_vals)
            rm, rs = _stats(r_vals)
            b_str = f"{bm:.4f} ± {bs:.4f}"
            r_str = f"{rm:.4f} ± {rs:.4f}"
            p = _wilcoxon(b_vals, r_vals)
            p_str = f"{p:.3f}" if p is not None else "—"
            sig = "*" if (p is not None and p < 0.05) else ""
            if lower_better:
                winner = ("Residual" if rm < bm else "Baseline") + sig
            else:
                winner = ("Residual" if rm > bm else "Baseline") + sig
            row_data[attr] = {"baseline_mean": bm, "baseline_std": bs,
                               "residual_mean": rm, "residual_std": rs, "p_value": p}

        print(f"{label_m:<22} {b_str:>14} {r_str:>14} {p_str:>10} {winner:>10}")

    print(f"{'='*74}")
    return row_data


# ── Main runner ───────────────────────────────────────────────────────────────

def run_dataset(
    label: str,
    base_cfg: ExperimentConfig,
    res_cfg: ExperimentConfig,
    best_params: dict,
    results_dir: Path,
) -> dict:
    t0 = time.perf_counter()
    print(f"\n>>> [Baseline]  {base_cfg.name}")
    baseline = BenchmarkRunner(base_cfg).run()

    print(f">>> [Residual]  {res_cfg.name}")
    residual = BenchmarkRunner(res_cfg).run()

    stats_table = print_comparison(label, baseline, residual, best_params)
    elapsed = time.perf_counter() - t0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"{label.replace(' ', '_').replace('-','_')}_{timestamp}.json"
    payload = {
        "label":       label,
        "best_params": best_params,
        "elapsed_s":   elapsed,
        "stats":       stats_table,
        "baseline": [
            {k: getattr(r, k) for k in
             ["method","dataset","trial","final_mse","accuracy","f1",
              "is_crystallized","delta_n","lambda_similarity",
              "total_time_s","iterations","converged"]}
            for r in baseline
        ],
        "residual": [
            {k: getattr(r, k) for k in
             ["method","dataset","trial","final_mse","accuracy","f1",
              "is_crystallized","delta_n","lambda_similarity",
              "total_time_s","iterations","converged"]}
            for r in residual
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"  Results → {out}")
    return payload


# ── Final summary ─────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    print(f"\n\n{'#'*74}")
    print("  FINAL SUMMARY — all datasets")
    print(f"{'#'*74}")
    print(f"{'Dataset':<22} {'Acc Base':>10} {'Acc Res':>10} {'p-value':>10} {'Cryst Base':>12} {'Cryst Res':>10}")
    print(f"{'-'*74}")
    for r in all_results:
        acc  = r["stats"].get("accuracy", {})
        cryst = r["stats"].get("is_crystallized", {})
        bm   = acc.get("baseline_mean", float("nan"))
        rm   = acc.get("residual_mean", float("nan"))
        p    = acc.get("p_value")
        p_s  = f"{p:.3f}" if p is not None else "—"
        bc   = cryst.get("baseline", float("nan"))
        rc   = cryst.get("residual", float("nan"))
        win  = "Res" if rm > bm else ("Base" if bm > rm else "=")
        sig  = "*" if (p is not None and p < 0.05) else ""
        print(f"{r['label']:<22} {bm:>10.4f} {rm:>10.4f} {p_s:>10} {bc:>12.2f} {rc:>10.2f}  {win}{sig}")
    print(f"{'#'*74}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+",
        default=["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"],
        choices=["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer", "all"],
    )
    p.add_argument("--n_trials",    type=int, default=10)
    p.add_argument("--results_dir", default="results/final")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = (
        ["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"]
        if "all" in args.datasets else args.datasets
    )
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds in datasets:
        best = _load_best_params(ds)
        print(f"\n{'*'*74}")
        print(f"  Dataset: {ds.upper()}   best_params: {best}")
        print(f"{'*'*74}")

        if ds == "mushroom":
            base_cfg, res_cfg = _mushroom_configs(best, args.n_trials)
        elif ds == "heart":
            base_cfg, res_cfg = _heart_configs(best, args.n_trials)
        elif ds.startswith("monk_"):
            problem = int(ds[-1])
            base_cfg, res_cfg = _monk_configs(problem, best, args.n_trials)
        elif ds == "breast_cancer":
            base_cfg, res_cfg = _breast_cancer_configs(best, args.n_trials)
        else:
            raise ValueError(f"Unknown dataset: {ds!r}")

        result = run_dataset(ds, base_cfg, res_cfg, best, results_dir)
        all_results.append({"label": ds, **result})

    if len(all_results) > 1:
        print_summary(all_results)

    # Save consolidated summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"summary_{ts}.json"
    summary_path.write_text(json.dumps(
        [{"label": r["label"], "best_params": r["best_params"], "stats": r["stats"]}
         for r in all_results],
        indent=2,
    ))
    print(f"Consolidated summary → {summary_path}\n")


if __name__ == "__main__":
    main()
