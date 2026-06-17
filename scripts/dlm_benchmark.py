"""
DLM Benchmark — 30-trial evaluation of the Differentiable Łukasiewicz Machine
on MONK-1, Mushroom, Spambase, and Musk.

Results are saved to results/dlm_variants/{dataset}_dlm.csv

Usage:
    python scripts/dlm_benchmark.py [--dataset monk|mushroom|spambase|musk] [--n_trials 30]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.luknn.dlm.network import make_dlm_net
from src.luknn.dlm.optimizer import DLMOptimizer
from src.luknn.benchmark.datasets import (
    load_monk, load_mushroom, load_spambase, load_musk,
)

# ── per-dataset hyperparameters ───────────────────────────────────────────────
# hidden_width = 4 × n_features (capped at 512 for memory)
# n_output_heads = n_features (= hidden_width // 4), caps at 128

_DATASET_CONFIG = {
    "monk": dict(
        loader=lambda: load_monk(problem=1, seed=0),
        hidden_width=68,       # 4 × 17
        n_output_heads=17,
        max_iter=3000,
        batch_size=None,       # full batch (124 samples)
    ),
    "mushroom": dict(
        loader=load_mushroom,
        hidden_width=256,      # cap (4×111=444 too slow)
        n_output_heads=64,
        max_iter=2000,
        batch_size=512,
    ),
    "spambase": dict(
        loader=load_spambase,
        hidden_width=228,      # 4 × 57
        n_output_heads=57,
        max_iter=2000,
        batch_size=512,
    ),
    "musk": dict(
        loader=load_musk,
        hidden_width=256,      # cap
        n_output_heads=64,
        max_iter=1500,
        batch_size=256,
    ),
}

_OPTIMIZER_DEFAULTS = dict(
    lr=5e-3,
    T_init=2.0,
    T_final=0.05,
    lambda_entropy=0.15,
    p1_fraction=0.5,
    conf_threshold=0.90,
    loss="bce",
)


def run_trial(dataset_name: str, cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    ds = cfg["loader"]()
    n_f = ds.n_features

    x_tr, y_tr = ds.X_train, ds.y_train.unsqueeze(-1)
    x_te, y_te = ds.X_test,  ds.y_test

    model = make_dlm_net(
        n_features=n_f,
        n_hidden_layers=2,
        hidden_width=cfg["hidden_width"],
        temperature=_OPTIMIZER_DEFAULTS["T_init"],
        gate_set="rep",
        seed=seed,
        n_output_heads=cfg["n_output_heads"],
    )

    opt = DLMOptimizer(model, **_OPTIMIZER_DEFAULTS)
    t0 = time.perf_counter()
    result = opt.train(
        x_tr, y_tr,
        max_iter=cfg["max_iter"],
        tol_mse=5e-3,
        batch_size=cfg.get("batch_size"),
    )
    elapsed = time.perf_counter() - t0

    crys = result.extra["crystallized_model"]
    with torch.no_grad():
        pred_te = crys(x_te).squeeze()
    pred_bin = (pred_te > 0.5).float()

    acc = (pred_bin == y_te).float().mean().item()
    try:
        f1 = float(f1_score(y_te.numpy(), pred_bin.numpy(), zero_division=0))
    except Exception:
        f1 = float("nan")

    return {
        "dataset": dataset_name,
        "method": "DLM",
        "trial": seed,
        "seed": seed,
        "n_features": n_f,
        "n_neurons": model.n_neurons(),
        "n_output_heads": cfg["n_output_heads"],
        "accuracy": acc,
        "f1": f1,
        "final_mse": result.final_mse,
        "gate_confidence": result.extra["gate_confidence"],
        "representability": result.extra["representability"],
        "converged": result.converged,
        "iterations": result.iterations,
        "reason": result.reason,
        "total_time_s": elapsed,
        "gate_counts": str(result.extra["gate_counts"]),
    }


def run_benchmark(dataset_name: str, n_trials: int = 30, verbose: bool = True) -> None:
    if dataset_name not in _DATASET_CONFIG:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Choose from {list(_DATASET_CONFIG)}")

    cfg = _DATASET_CONFIG[dataset_name]
    out_dir = ROOT / "results" / "dlm_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_dlm.csv"

    # Resume: detect already-completed trials by reading existing CSV.
    done_seeds: set[int] = set()
    fieldnames: list[str] | None = None
    if out_file.exists():
        existing = list(csv.DictReader(open(out_file)))
        done_seeds = {int(r["seed"]) for r in existing}
        if existing:
            fieldnames = list(existing[0].keys())

    print(f"\n{'='*60}")
    print(f"DLM Benchmark: {dataset_name}  ({n_trials} trials)")
    print(f"  width={cfg['hidden_width']}  heads={cfg['n_output_heads']}"
          f"  max_iter={cfg['max_iter']}  batch={cfg.get('batch_size', 'full')}")
    print(f"  output → {out_file.name}")
    if done_seeds:
        print(f"  Resuming: {len(done_seeds)} trials already done, {n_trials - len(done_seeds)} remaining")
    print(f"{'='*60}", flush=True)

    rows: list[dict] = []
    completed = len(done_seeds)
    for trial in range(n_trials):
        seed = trial * 1000 + 42
        if seed in done_seeds:
            continue

        t_trial = time.perf_counter()
        row = run_trial(dataset_name, cfg, seed)
        dt = time.perf_counter() - t_trial
        rows.append(row)
        completed += 1

        # Write incrementally — append a single row immediately after each trial.
        if fieldnames is None:
            fieldnames = list(row.keys())
        write_header = not out_file.exists() or out_file.stat().st_size == 0
        with open(out_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        if verbose:
            print(
                f"  [{completed:2d}/{n_trials}] seed={seed}  "
                f"acc={row['accuracy']:.3f}  F1={row['f1']:.3f}  "
                f"mse={row['final_mse']:.4f}  conf={row['gate_confidence']:.3f}  "
                f"{row['reason']}  ({dt:.1f}s)",
                flush=True,
            )

    # Summary statistics over all rows (including previously completed ones).
    all_rows = list(csv.DictReader(open(out_file)))
    f1s  = [float(r["f1"]) for r in all_rows if r["f1"] != "nan"]
    accs = [float(r["accuracy"]) for r in all_rows]
    confs = [float(r["gate_confidence"]) for r in all_rows]
    print(f"\nSummary ({dataset_name}, {len(all_rows)} trials):")
    print(f"  F1  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
          f"  [min={np.min(f1s):.3f}  max={np.max(f1s):.3f}]")
    print(f"  Acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Conf: {np.mean(confs):.3f} ± {np.std(confs):.3f}")
    print(f"  Saved: {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLM benchmark runner")
    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_CONFIG) + ["all"],
        default="monk",
        help="Dataset to evaluate (default: monk)",
    )
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials (default: 30)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trial output")
    args = parser.parse_args()

    datasets = list(_DATASET_CONFIG) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_benchmark(ds, n_trials=args.n_trials, verbose=not args.quiet)
