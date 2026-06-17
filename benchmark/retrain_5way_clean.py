"""
Tier C final evaluation — 5-way comparison with proper train/val/test splits.

For mushroom, heart, breast_cancer: HP selection was done on a separate
validation set (results/tuning_val/).  Final evaluation scores on the held-out
test set only (use_val_split=False).

For MONK-1/2/3: no contamination issue (fixed UCI train/test split with no
tuning done on the test set); uses original results/tuning/ as before.

Output: results/final5_clean/<dataset>_<timestamp>.json

Usage:
  python benchmark/retrain_5way_clean.py
  python benchmark/retrain_5way_clean.py --datasets mushroom heart
  python benchmark/retrain_5way_clean.py --n_trials 10
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
from tune import _apply_params
from tune_ste             import _BASES as _STE_BASES
from tune_proximal        import _BASES as _PRX_BASES
from tune_ste_residual    import _BASES as _STER_BASES
from tune_proximal_residual import _BASES as _PRXR_BASES

ROOT         = Path(__file__).parents[1]
TUNING_DIR   = ROOT / "results/tuning"      # MONK
TUNING_VAL   = ROOT / "results/tuning_val"  # mushroom / heart / breast_cancer

# Datasets where HP selection used a val split (not the test set)
_VAL_TUNED = {"mushroom", "heart", "breast_cancer"}

# ── Tuning file discovery ─────────────────────────────────────────────────────

_LABEL_PREFIX = {
    "lm_residual":        "",
    "ste":                "ste_",
    "ste_residual":       "ste_residual_",
    "proximal":           "proximal_",
    "proximal_residual":  "proximal_residual_",
}


def _latest_tuning(optimizer: str, ds: str) -> Path:
    tdir = TUNING_VAL if ds in _VAL_TUNED else TUNING_DIR
    prefix = _LABEL_PREFIX[optimizer]
    pattern = f"{prefix}{ds}_*.json"
    candidates = sorted(tdir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No tuning file for {optimizer}/{ds} in {tdir}/{pattern}"
        )
    return candidates[-1]


# ── LM_Residual base configs ──────────────────────────────────────────────────

def _lm_residual_base(ds: str) -> ExperimentConfig:
    if ds == "mushroom":
        return ExperimentConfig(
            name="LM_Residual — Mushroom [clean-final]", seed=42, n_inputs=111,
            hidden_layers=[6, 4], optimizer_method="LM_Residual",
            optimizer_params={"patience": 80, "crystallize_n": 2, "prune": False,
                              "batch_size": 512, "mu_init": 0.01},
            dataset_type="mushroom", tol_mse=0.15, max_iter=600, verbose=False,
        )
    if ds == "heart":
        return ExperimentConfig(
            name="LM_Residual — Heart [clean-final]", seed=42, n_inputs=13,
            hidden_layers=[6, 4], optimizer_method="LM_Residual",
            optimizer_params={"patience": 100, "crystallize_n": 2, "prune": True,
                              "batch_size": 0, "mu_init": 0.01},
            dataset_type="heart_disease", heart_subset="cleveland",
            tol_mse=0.15, max_iter=800, verbose=False,
        )
    if ds.startswith("monk_"):
        prob = int(ds[-1])
        return ExperimentConfig(
            name=f"LM_Residual — MONK-{prob} [clean-final]", seed=42, n_inputs=17,
            hidden_layers=[8, 4], optimizer_method="LM_Residual",
            optimizer_params={"patience": 80, "crystallize_n": 2, "prune": False,
                              "batch_size": 0, "mu_init": 0.01},
            dataset_type="monk", monk_problem=prob, tol_mse=0.10, max_iter=500, verbose=False,
        )
    # breast_cancer
    return ExperimentConfig(
        name="LM_Residual — Breast Cancer [clean-final]", seed=42, n_inputs=20,
        hidden_layers=[8, 8], optimizer_method="LM_Residual",
        optimizer_params={"patience": 100, "crystallize_n": 2, "prune": True,
                          "batch_size": 0, "mu_init": 0.01},
        dataset_type="breast_cancer", tol_mse=0.15, max_iter=800, verbose=False,
    )


_BASE_FN = {
    "lm_residual":       _lm_residual_base,
    "ste":               lambda ds: _STE_BASES[ds],
    "ste_residual":      lambda ds: _STER_BASES[ds],
    "proximal":          lambda ds: _PRX_BASES[ds],
    "proximal_residual": lambda ds: _PRXR_BASES[ds],
}


def _build_cfg(optimizer: str, ds: str, n_trials: int) -> ExperimentConfig:
    tuning_path = _latest_tuning(optimizer, ds)
    best = json.loads(tuning_path.read_text())["best_params"]
    base = _BASE_FN[optimizer](ds)
    cfg  = _apply_params(replace(base, n_trials=n_trials), best)
    # Final eval always uses test set (never val)
    cfg  = replace(cfg, use_val_split=False, val_fraction=0.0)
    return replace(cfg, name=f"{optimizer.upper()} — {ds} [clean-final]")


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

METHODS = ["lm_residual", "ste", "ste_residual", "proximal", "proximal_residual"]
LABELS  = {
    "lm_residual":       "LM_Res",
    "ste":               "STE",
    "ste_residual":      "STE_Res",
    "proximal":          "Proximal",
    "proximal_residual": "Prx_Res",
}

METRICS = [
    ("Accuracy",     "accuracy",          False),
    ("F1",           "f1",                False),
    ("MSE final",    "final_mse",         True),
    ("Crystallized", "is_crystallized",   False),
    ("λ-similar",    "lambda_similarity", False),
    ("Time (s)",     "total_time_s",      True),
    ("Iterations",   "iterations",        True),
]

_PAIRS = [
    ("LM/STE",    "lm_residual",  "ste"),
    ("STE/STEr",  "ste",          "ste_residual"),
    ("PRX/PRXr",  "proximal",     "proximal_residual"),
    ("STEr/PRXr", "ste_residual", "proximal_residual"),
]


def print_comparison(ds: str, results: dict, best_params: dict) -> dict:
    w = 12
    hdr = f"{'Metric':<18}" + "".join(f"{LABELS[m]:>{w}}" for m in METHODS)
    p_part = "".join(f"  {lbl}".rjust(10) for lbl, *_ in _PAIRS) + "  Best"
    hdr += p_part
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(f"  {ds.upper()}  [test-set evaluation — val-selected HPs]")
    for m, bp in best_params.items():
        print(f"  {LABELS[m]}: {bp}")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    row_data: dict = {}
    for label, attr, lower_better in METRICS:
        vals = {m: [getattr(r, attr) for r in results[m]] for m in METHODS}
        if isinstance(vals[METHODS[0]][0], bool):
            cells  = {m: f"{sum(vals[m])}/{len(vals[m])} ✓" for m in METHODS}
            p_vals = {lbl: None for lbl, *_ in _PAIRS}
            best_lbl = "—"
        else:
            means    = {m: _stats(vals[m])[0] for m in METHODS}
            stds     = {m: _stats(vals[m])[1] for m in METHODS}
            cells    = {m: f"{means[m]:.4f}±{stds[m]:.4f}" for m in METHODS}
            p_vals   = {lbl: _wilcoxon(vals[a], vals[b]) for lbl, a, b in _PAIRS}
            best_m   = min(means, key=means.__getitem__) if lower_better \
                       else max(means, key=means.__getitem__)
            sig      = "*" if any(p is not None and p < 0.05 for p in p_vals.values()) else ""
            best_lbl = LABELS[best_m] + sig
            row_data[attr] = {m: {"mean": means[m], "std": stds[m]} for m in METHODS}
            row_data[attr]["p_values"] = p_vals

        line = f"{label:<18}" + "".join(f"{cells[m]:>{w}}" for m in METHODS)
        for lbl, *_ in _PAIRS:
            p = p_vals[lbl]
            line += f"  {'—' if p is None else f'{p:.3f}':>8}"
        line += f"  {best_lbl}"
        print(line)

    print(sep)
    return row_data


def run_dataset(ds: str, n_trials: int, results_dir: Path) -> dict:
    all_results: dict[str, list[BenchmarkResult]] = {}
    best_params: dict[str, dict] = {}

    for opt in METHODS:
        cfg = _build_cfg(opt, ds, n_trials)
        best_params[opt] = json.loads(_latest_tuning(opt, ds).read_text())["best_params"]
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
                 ["method", "dataset", "trial", "final_mse", "accuracy", "f1",
                  "is_crystallized", "delta_n", "lambda_similarity",
                  "total_time_s", "iterations", "converged"]}
                for r in all_results[opt]
           ] for opt in METHODS
        },
    }, indent=2))
    print(f"  Saved → {out}")
    return {"dataset": ds, "stats": row_data, "best_params": best_params}


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_ds: list[dict]) -> None:
    print(f"\n\n{'#'*90}")
    print("  CLEAN FINAL SUMMARY — Accuracy (HP selected on val, scored on test)")
    print(f"{'#'*90}")
    hdr = f"{'Dataset':<18}" + "".join(f"{LABELS[m]:>12}" for m in METHODS) + "  Best"
    print(hdr)
    print("-" * len(hdr))
    for d in all_ds:
        acc   = d["stats"].get("accuracy", {})
        means = {m: acc.get(m, {}).get("mean", float("nan")) for m in METHODS}
        best  = max(means, key=means.__getitem__)
        print(f"{d['dataset']:<18}" + "".join(f"{means[m]:>12.4f}" for m in METHODS)
              + f"  {LABELS[best]}")
    print("#" * 90 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_DATASETS = ["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",    nargs="+", default=["all"],
                   choices=ALL_DATASETS + ["all"])
    p.add_argument("--n_trials",    type=int, default=10)
    p.add_argument("--results_dir", default="results/final5_clean")
    return p.parse_args()


def main():
    args    = parse_args()
    ds_list = ALL_DATASETS if "all" in args.datasets else args.datasets
    rd      = ROOT / args.results_dir
    rd.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*90}")
    print(f"  Tier C clean final retrain — test-set evaluation")
    print(f"  HP source: tuning_val (mushroom/heart/breast_cancer), tuning (MONK)")
    print(f"  Datasets : {ds_list}")
    print(f"  Trials   : {args.n_trials}")
    print(f"{'#'*90}")

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
