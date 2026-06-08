"""
Final retrain and comparison across 3 optimizers: LM_Residual, STE, Proximal.

For each dataset:
  1. Load best tuning params for each optimizer.
  2. Run 10 independent trials.
  3. Print a comparison table (accuracy, MSE, crystallization, time).
  4. Save to results/final3/<dataset>_<timestamp>.json

Usage:
  python benchmark/retrain_all_optimizers.py
  python benchmark/retrain_all_optimizers.py --datasets mushroom heart
  python benchmark/retrain_all_optimizers.py --n_trials 10
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
sys.path.insert(0, str(Path(__file__).parents[1] / "tuning"))

import numpy as np
from scipy import stats

from luknn.benchmark.config import ExperimentConfig
from luknn.benchmark.runner import BenchmarkRunner
from luknn.benchmark.metrics import BenchmarkResult
from tune import _apply_params                    # reuses the tuning engine
from tune_ste import _BASES as _STE_BASES         # STE base configs
from tune_proximal import _BASES as _PRX_BASES    # Proximal base configs

ROOT = Path(__file__).parents[1]

# ── Tuning files ──────────────────────────────────────────────────────────────

_TUNING = {
    "lm_residual": {
        "mushroom":      ROOT / "results/tuning/mushroom_20260604_091426.json",
        "heart":         ROOT / "results/tuning/heart_20260604_091426.json",
        "monk_1":        ROOT / "results/tuning/monk_1_20260604_091426.json",
        "monk_2":        ROOT / "results/tuning/monk_2_20260604_103838.json",
        "monk_3":        ROOT / "results/tuning/monk_3_20260604_114059.json",
        "breast_cancer": ROOT / "results/tuning/breast_cancer_20260604_091426.json",
    },
    "ste": {
        "mushroom":      ROOT / "results/tuning/ste_mushroom_20260604_204141.json",
        "heart":         ROOT / "results/tuning/ste_heart_20260604_204140.json",
        "monk_1":        ROOT / "results/tuning/ste_monk_1_20260604_204140.json",
        "monk_2":        ROOT / "results/tuning/ste_monk_2_20260604_204139.json",
        "monk_3":        ROOT / "results/tuning/ste_monk_3_20260604_204141.json",
        "breast_cancer": ROOT / "results/tuning/ste_breast_cancer_20260604_204140.json",
    },
    "proximal": {
        "mushroom":      ROOT / "results/tuning/proximal_mushroom_20260604_213900.json",
        "heart":         ROOT / "results/tuning/proximal_heart_20260604_213905.json",
        "monk_1":        ROOT / "results/tuning/proximal_monk_1_20260604_213904.json",
        "monk_2":        ROOT / "results/tuning/proximal_monk_2_20260604_213901.json",
        "monk_3":        ROOT / "results/tuning/proximal_monk_3_20260604_213902.json",
        "breast_cancer": ROOT / "results/tuning/proximal_breast_cancer_20260604_213903.json",
    },
}

# ── LM_Residual base configs (reused from retrain_best.py) ───────────────────

def _lm_residual_base(ds: str) -> ExperimentConfig:
    """LM_Residual base config per dataset (identical budget to tuning)."""
    if ds == "mushroom":
        return ExperimentConfig(
            name=f"LM_Residual — Mushroom [final]", seed=42, n_inputs=111,
            hidden_layers=[6,4], optimizer_method="LM_Residual",
            optimizer_params={"patience":80,"crystallize_n":2,"prune":False,"batch_size":512,"mu_init":0.01},
            dataset_type="mushroom", tol_mse=0.15, max_iter=600, verbose=False,
        )
    if ds == "heart":
        return ExperimentConfig(
            name=f"LM_Residual — Heart [final]", seed=42, n_inputs=13,
            hidden_layers=[6,4], optimizer_method="LM_Residual",
            optimizer_params={"patience":100,"crystallize_n":2,"prune":True,"batch_size":0,"mu_init":0.01},
            dataset_type="heart_disease", heart_subset="cleveland",
            tol_mse=0.15, max_iter=800, verbose=False,
        )
    if ds.startswith("monk_"):
        prob = int(ds[-1])
        return ExperimentConfig(
            name=f"LM_Residual — MONK-{prob} [final]", seed=42, n_inputs=17,
            hidden_layers=[8,4], optimizer_method="LM_Residual",
            optimizer_params={"patience":80,"crystallize_n":2,"prune":False,"batch_size":0,"mu_init":0.01},
            dataset_type="monk", monk_problem=prob, tol_mse=0.10, max_iter=500, verbose=False,
        )
    # breast_cancer
    return ExperimentConfig(
        name=f"LM_Residual — Breast Cancer [final]", seed=42, n_inputs=20,
        hidden_layers=[8,8], optimizer_method="LM_Residual",
        optimizer_params={"patience":100,"crystallize_n":2,"prune":True,"batch_size":0,"mu_init":0.01},
        dataset_type="breast_cancer", tol_mse=0.15, max_iter=800, verbose=False,
    )


def _build_cfg(optimizer: str, ds: str, n_trials: int) -> ExperimentConfig:
    """Build the final ExperimentConfig applying the best tuning params."""
    best = json.loads(_TUNING[optimizer][ds].read_text())["best_params"]

    if optimizer == "lm_residual":
        base = _lm_residual_base(ds)
    elif optimizer == "ste":
        base = _STE_BASES[ds]
    else:
        base = _PRX_BASES[ds]

    cfg = _apply_params(replace(base, n_trials=n_trials), best)
    return replace(cfg, name=f"{optimizer.upper()} — {ds} [final]")


# ── Statistics ────────────────────────────────────────────────────────────────

def _stats(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals)
    return float(a.mean()), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def _wilcoxon(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    try:
        _, p = stats.wilcoxon(a, b)
        return float(p)
    except ValueError:
        return None


# ── Comparison table ──────────────────────────────────────────────────────────

METHODS = ["lm_residual", "ste", "proximal"]
LABELS  = {"lm_residual": "LM_Res", "ste": "STE", "proximal": "Proximal"}

METRICS = [
    ("Accuracy",     "accuracy",          False),
    ("F1",           "f1",                False),
    ("MSE final",    "final_mse",         True),
    ("Crystallized", "is_crystallized",   False),
    ("λ-similar",    "lambda_similarity", False),
    ("Time (s)",     "total_time_s",      True),
    ("Iterations",   "iterations",        True),
]


def print_comparison(ds: str, results: dict[str, list[BenchmarkResult]],
                     best_params: dict[str, dict]) -> dict:
    w = 14
    hdr = f"{'Metric':<22}" + "".join(f"{LABELS[m]:>{w}}" for m in METHODS)
    p_pairs = [("LM/STE", "lm_residual", "ste"),
               ("LM/PRX", "lm_residual", "proximal"),
               ("STE/PRX","ste",         "proximal")]
    hdr += "".join(f"  p({lbl})".rjust(11) for lbl, *_ in p_pairs) + "  Best"

    print(f"\n{'='*len(hdr)}")
    print(f"  {ds.upper()}")
    for m, bp in best_params.items():
        print(f"  {LABELS[m]}: {bp}")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    row_data: dict = {}
    for label, attr, lower_better in METRICS:
        vals = {m: [getattr(r, attr) for r in results[m]] for m in METHODS}

        if isinstance(vals[METHODS[0]][0], bool):
            cells = {m: f"{sum(vals[m])}/{len(vals[m])} ✓" for m in METHODS}
            p_vals = {lbl: None for lbl, *_ in p_pairs}
            best_lbl = "—"
        else:
            means = {m: _stats(vals[m])[0] for m in METHODS}
            stds  = {m: _stats(vals[m])[1] for m in METHODS}
            cells = {m: f"{means[m]:.4f}±{stds[m]:.4f}" for m in METHODS}
            p_vals = {lbl: _wilcoxon(vals[a], vals[b]) for lbl, a, b in p_pairs}
            best_m = min(means, key=means.__getitem__) if lower_better \
                     else max(means, key=means.__getitem__)
            sig = "*" if any(p is not None and p < 0.05 for p in p_vals.values()) else ""
            best_lbl = LABELS[best_m] + sig
            row_data[attr] = {m: {"mean": means[m], "std": stds[m]} for m in METHODS}
            row_data[attr]["p_values"] = p_vals

        line = f"{label:<22}" + "".join(f"{cells[m]:>{w}}" for m in METHODS)
        for lbl, *_ in p_pairs:
            p = p_vals[lbl]
            line += f"  {'—' if p is None else f'{p:.3f}':>9}"
        line += f"  {best_lbl}"
        print(line)

    print("=" * len(hdr))
    return row_data


# ── Runner ────────────────────────────────────────────────────────────────────

def run_dataset(ds: str, n_trials: int, results_dir: Path) -> dict:
    all_results: dict[str, list[BenchmarkResult]] = {}
    best_params: dict[str, dict] = {}

    for opt in METHODS:
        cfg = _build_cfg(opt, ds, n_trials)
        best_params[opt] = json.loads(_TUNING[opt][ds].read_text())["best_params"]
        print(f"\n  [{LABELS[opt]}] {cfg.name}")
        all_results[opt] = BenchmarkRunner(cfg).run()

    row_data = print_comparison(ds, all_results, best_params)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"{ds}_{ts}.json"
    out.write_text(json.dumps({
        "dataset":     ds,
        "best_params": best_params,
        "stats":       row_data,
        **{opt: [
                {k: getattr(r, k) for k in
                 ["method","dataset","trial","final_mse","accuracy","f1",
                  "is_crystallized","delta_n","lambda_similarity",
                  "total_time_s","iterations","converged"]}
                for r in all_results[opt]
           ] for opt in METHODS
        },
    }, indent=2))
    print(f"  Saved → {out}")
    return {"dataset": ds, "stats": row_data, "best_params": best_params}


# ── Final summary ─────────────────────────────────────────────────────────────

def print_summary(all_ds: list[dict]) -> None:
    print(f"\n\n{'#'*80}")
    print("  FINAL SUMMARY — Accuracy per dataset and optimizer")
    print(f"{'#'*80}")
    hdr = f"{'Dataset':<22}" + "".join(f"{LABELS[m]:>12}" for m in METHODS) \
          + f"  {'Best':>10}"
    print(hdr)
    print("-" * len(hdr))
    for d in all_ds:
        acc = d["stats"].get("accuracy", {})
        means = {m: acc.get(m, {}).get("mean", float("nan")) for m in METHODS}
        best  = max(means, key=means.__getitem__)
        row   = f"{d['dataset']:<22}" \
              + "".join(f"{means[m]:>12.4f}" for m in METHODS) \
              + f"  {LABELS[best]:>10}"
        print(row)
    print("#" * 80)

    # Crystallization
    print(f"\n{'#'*80}")
    print("  FINAL SUMMARY — Crystallization rate")
    print(f"{'#'*80}")
    print(hdr.replace("Accuracy","Cryst"))
    print("-" * len(hdr))
    for d in all_ds:
        cryst = d["stats"].get("is_crystallized", {})
        vals  = {m: cryst.get(m, float("nan"))
                 if not isinstance(cryst.get(m), dict)
                 else cryst[m].get("mean", float("nan")) for m in METHODS}
        best  = max(vals, key=vals.__getitem__)
        row   = f"{d['dataset']:<22}" \
              + "".join(f"{vals[m]:>12.2f}" for m in METHODS) \
              + f"  {LABELS[best]:>10}"
        print(row)
    print("#" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_DATASETS = ["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=ALL_DATASETS + ["all"])
    p.add_argument("--n_trials",    type=int, default=10)
    p.add_argument("--results_dir", default="results/final3")
    return p.parse_args()


def main():
    args   = parse_args()
    ds_list = ALL_DATASETS if "all" in args.datasets else args.datasets
    rd      = ROOT / args.results_dir
    rd.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"  Final retrain — LM_Residual vs STE vs Proximal")
    print(f"  Datasets : {ds_list}")
    print(f"  Trials   : {args.n_trials}")
    print(f"{'#'*80}")

    summary = []
    for ds in ds_list:
        t0 = time.perf_counter()
        entry = run_dataset(ds, args.n_trials, rd)
        entry["elapsed_s"] = time.perf_counter() - t0
        summary.append(entry)

    if len(summary) > 1:
        print_summary(summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (rd / f"summary_{ts}.json").write_text(json.dumps(
        [{"dataset": e["dataset"], "best_params": e["best_params"],
          "stats": e["stats"]} for e in summary],
        indent=2))
    print(f"Consolidated summary → {rd}/summary_{ts}.json\n")


if __name__ == "__main__":
    main()
