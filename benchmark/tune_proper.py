"""
Tier C HP tuning — mushroom, heart, breast_cancer with proper train/val/test splits.

Scores HP combinations on a held-out VALIDATION set (20% of train), never on the
test set.  Tuning results are saved to results/tuning_val/ so that the original
results/tuning/ files (used for MONK) remain untouched.

Usage:
  python benchmark/tune_proper.py
  python benchmark/tune_proper.py --datasets mushroom heart
  python benchmark/tune_proper.py --n_trials 5 --results_dir results/tuning_val
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "tuning"))

from luknn.benchmark.config import ExperimentConfig
from tune import run_grid

# ── Method-specific imports ───────────────────────────────────────────────────

from tune_ste             import _BASES as _STE_BASES,       GRID as _STE_GRID
from tune_ste_residual    import _BASES as _STER_BASES,      GRID as _STER_GRID
from tune_proximal        import _BASES as _PRX_BASES,       GRID as _PRX_GRID
from tune_proximal_residual import _BASES as _PRXR_BASES,   GRID as _PRXR_GRID

# ── LM_Residual base configs ──────────────────────────────────────────────────

_LM_BASES: dict[str, ExperimentConfig] = {
    "mushroom": ExperimentConfig(
        name="LM_Residual — Mushroom [val-tuning]",
        seed=42, n_inputs=111, hidden_layers=[6, 4],
        optimizer_method="LM_Residual",
        optimizer_params={"mu_init": 0.01, "patience": 80,
                          "crystallize_n": 2, "prune": False, "batch_size": 512},
        dataset_type="mushroom", hidden_width=6, n_blocks=1, n_inner=1,
        tol_mse=0.15, max_iter=600, verbose=False,
    ),
    "heart": ExperimentConfig(
        name="LM_Residual — Heart Disease [val-tuning]",
        seed=42, n_inputs=13, hidden_layers=[6, 4],
        optimizer_method="LM_Residual",
        optimizer_params={"mu_init": 0.01, "patience": 100,
                          "crystallize_n": 2, "prune": True, "batch_size": 0},
        dataset_type="heart_disease", heart_subset="cleveland",
        hidden_width=6, n_blocks=1, n_inner=1,
        tol_mse=0.15, max_iter=800, verbose=False,
    ),
    "breast_cancer": ExperimentConfig(
        name="LM_Residual — Breast Cancer [val-tuning]",
        seed=42, n_inputs=20, hidden_layers=[4, 4],
        optimizer_method="LM_Residual",
        optimizer_params={"mu_init": 0.01, "patience": 100,
                          "crystallize_n": 2, "prune": True, "batch_size": 0},
        dataset_type="breast_cancer",
        hidden_width=6, n_blocks=1, n_inner=1,
        tol_mse=0.15, max_iter=800, verbose=False,
    ),
}

_LM_GRIDS: dict[str, dict] = {
    "mushroom":     {"hidden_width": [4, 6, 8], "n_blocks": [1, 2], "mu_init": [0.001, 0.01]},
    "heart":        {"hidden_width": [4, 6, 8], "n_blocks": [1, 2], "mu_init": [0.001, 0.01, 0.1], "prune": [True, False]},
    "breast_cancer": {"hidden_width": [4, 6, 8], "n_blocks": [1, 2], "mu_init": [0.001, 0.01, 0.1], "prune": [True, False]},
}

# ── Per-method config + grid tables ──────────────────────────────────────────

_METHOD_TABLE = {
    "lm_residual":       (_LM_BASES,   _LM_GRIDS,   lambda ds: f"mushroom"   if ds == "mushroom" else ds),
    "ste":               (_STE_BASES,  {"mushroom": _STE_GRID, "heart": _STE_GRID, "breast_cancer": _STE_GRID},   None),
    "ste_residual":      (_STER_BASES, {"mushroom": _STER_GRID, "heart": _STER_GRID, "breast_cancer": _STER_GRID}, None),
    "proximal":          (_PRX_BASES,  {"mushroom": _PRX_GRID, "heart": _PRX_GRID, "breast_cancer": _PRX_GRID},   None),
    "proximal_residual": (_PRXR_BASES, {"mushroom": _PRXR_GRID, "heart": _PRXR_GRID, "breast_cancer": _PRXR_GRID}, None),
}

_LABEL_MAP = {
    "lm_residual":       "",
    "ste":               "ste_",
    "ste_residual":      "ste_residual_",
    "proximal":          "proximal_",
    "proximal_residual": "proximal_residual_",
}

ALL_DATASETS = ["mushroom", "heart", "breast_cancer"]
ALL_METHODS  = ["lm_residual", "ste", "ste_residual", "proximal", "proximal_residual"]

VAL_FRACTION = 0.2


def tune_one(method: str, ds: str, n_trials: int, results_dir: str) -> None:
    bases, grids, _ = _METHOD_TABLE[method]
    base  = bases[ds]
    grid  = grids[ds] if isinstance(grids, dict) else grids
    label = f"{_LABEL_MAP[method]}{ds}"

    cfg = replace(base, val_fraction=VAL_FRACTION, use_val_split=True)
    print(f"\n  Tuning {method} / {ds}  (val={VAL_FRACTION:.0%} of train, "
          f"{len(list(__import__('itertools').product(*grid.values())))} combos × {n_trials} trials)")

    run_grid(cfg, grid, n_trials=n_trials, results_dir=results_dir, label=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",    nargs="+", default=["all"],
                   choices=ALL_DATASETS + ["all"])
    p.add_argument("--methods",     nargs="+", default=["all"],
                   choices=ALL_METHODS + ["all"])
    p.add_argument("--n_trials",    type=int, default=5)
    p.add_argument("--results_dir", default="results/tuning_val")
    args = p.parse_args()

    ds_list = ALL_DATASETS if "all" in args.datasets else args.datasets
    mt_list = ALL_METHODS  if "all" in args.methods  else args.methods

    ROOT = Path(__file__).parents[1]
    (ROOT / args.results_dir).mkdir(parents=True, exist_ok=True)
    rd = str(ROOT / args.results_dir)

    print(f"\n{'#'*80}")
    print(f"  Tier C HP tuning — val-split  (val_fraction={VAL_FRACTION:.0%})")
    print(f"  Datasets : {ds_list}")
    print(f"  Methods  : {mt_list}")
    print(f"  Output   : {rd}")
    print(f"{'#'*80}")

    for ds in ds_list:
        for mt in mt_list:
            tune_one(mt, ds, args.n_trials, rd)

    print(f"\n  Done. Tuning files written to {rd}/\n")


if __name__ == "__main__":
    main()
