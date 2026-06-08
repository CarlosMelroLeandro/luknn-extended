"""
Shared grid-search engine for ŁNN residual hyperparameter tuning.

Score metric (maximised):
    score = 0.6 * mean_accuracy + 0.4 * crystallization_rate

    Balances predictive quality with interpretability: a crystallized
    network maps directly to a Łukasiewicz formula (Prop. 3 of the paper).

Usage (from dataset-specific scripts):
    from tune import run_grid
"""

from __future__ import annotations
import itertools
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
from luknn.benchmark.config import ExperimentConfig
from luknn.benchmark.runner import BenchmarkRunner
from luknn.benchmark.metrics import BenchmarkResult

# Keys that live inside ExperimentConfig.optimizer_params (not top-level fields).
# Covers LM, STE and Proximal optimizers.
_OPT_KEYS = {
    # LM
    "mu_init", "patience", "crystallize_n", "prune", "batch_size",
    # STE
    "lr", "weight_lr", "clip_grad",
    # Proximal
    "lambda_sparse", "lambda_attract", "prox_threshold", "phase1_fraction",
}


def _apply_params(cfg: ExperimentConfig, params: dict) -> ExperimentConfig:
    """Split params into optimizer-level vs config-level and apply both."""
    opt = {k: v for k, v in params.items() if k in _OPT_KEYS}
    top = {k: v for k, v in params.items() if k not in _OPT_KEYS}
    if opt:
        cfg = replace(cfg, optimizer_params={**cfg.optimizer_params, **opt})
    if top:
        cfg = replace(cfg, **top)
    return cfg


def _score(results: list[BenchmarkResult]) -> float:
    acc   = float(np.mean([r.accuracy for r in results]))
    cryst = float(np.mean([r.is_crystallized for r in results]))
    return 0.6 * acc + 0.4 * cryst


def _fmt_eta(seconds: float) -> str:
    return f"{seconds/60:.1f}min" if seconds >= 60 else f"{seconds:.0f}s"


def _save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def run_grid(
    base_config: ExperimentConfig,
    grid: dict[str, list],
    n_trials: int = 5,
    results_dir: str = "results/tuning",
    label: str = "tuning",
) -> dict[str, Any]:
    """
    Exhaustive grid search over all combinations in `grid`.

    Parameters
    ----------
    base_config  : ExperimentConfig base (fixed params not in grid)
    grid         : mapping param_name → list of candidate values
    n_trials     : independent restarts per combination
    results_dir  : where to write the JSON result file
    label        : prefix for the output filename (e.g. "mushroom")

    Returns
    -------
    dict with keys: label, best_score, best_params, top10, all
    """
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = Path(results_dir) / f"{label}_{timestamp}.json"

    all_entries: list[dict] = []
    best_score  = -1.0
    best_params: dict = {}
    best_entry: dict  = {}

    print(f"\n{'='*64}")
    print(f"  Grid search  : {label}")
    print(f"  Combinations : {total}   Trials/combo : {n_trials}")
    print(f"  Output       : {out_path}")
    print(f"{'='*64}\n", flush=True)

    t_start = time.perf_counter()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg    = _apply_params(replace(base_config, n_trials=n_trials), params)

        elapsed = time.perf_counter() - t_start
        eta     = (elapsed / i) * (total - i) if i > 1 else 0.0
        print(f"[{i:3d}/{total}]  {params}  ETA {_fmt_eta(eta)}", flush=True)

        try:
            results = BenchmarkRunner(cfg).run()
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            continue

        score      = _score(results)
        acc_vals   = [r.accuracy       for r in results]
        cryst_vals = [r.is_crystallized for r in results]
        mse_vals   = [r.final_mse      for r in results]

        acc_mean  = float(np.mean(acc_vals))
        acc_std   = float(np.std(acc_vals,  ddof=1) if len(acc_vals)  > 1 else 0.0)
        mse_mean  = float(np.mean(mse_vals))
        cryst_rt  = float(np.mean(cryst_vals))

        marker = "  ** NEW BEST **" if score > best_score else ""
        print(
            f"  score={score:.4f}  acc={acc_mean:.4f}±{acc_std:.4f}"
            f"  cryst={cryst_rt:.1f}  mse={mse_mean:.4f}{marker}",
            flush=True,
        )

        entry = {
            "rank":               0,
            "params":             params,
            "score":              score,
            "accuracy_mean":      acc_mean,
            "accuracy_std":       acc_std,
            "crystallization_rate": cryst_rt,
            "mse_mean":           mse_mean,
            "n_trials":           n_trials,
        }
        all_entries.append(entry)

        if score > best_score:
            best_score  = score
            best_params = params
            best_entry  = entry

        # Intermediate save — survives interruption
        _save(out_path, {
            "label":        label,
            "status":       "running",
            "n_completed":  i,
            "n_total":      total,
            "best_score":   best_score,
            "best_params":  best_params,
            "completed":    all_entries,
        })

    # Final: sort and annotate ranks
    all_entries.sort(key=lambda e: -e["score"])
    for rank, entry in enumerate(all_entries, 1):
        entry["rank"] = rank

    final = {
        "label":              label,
        "timestamp":          timestamp,
        "status":             "done",
        "n_combos":           total,
        "n_trials_per_combo": n_trials,
        "best_score":         best_score,
        "best_params":        best_params,
        "top10":              all_entries[:10],
        "all":                all_entries,
    }
    _save(out_path, final)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*64}")
    print(f"  Completed in {_fmt_eta(elapsed_total)}")
    print(f"  Best score   : {best_score:.4f}")
    print(f"  Best params  : {best_params}")
    print(f"  Results      : {out_path}")
    print(f"{'='*64}\n", flush=True)

    return final
