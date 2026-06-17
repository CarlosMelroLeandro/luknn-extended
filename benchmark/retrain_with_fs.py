"""
Benchmark with XGBoost feature selection pre-processing.

Evaluates all ŁNN methods + DLM on all datasets after reducing features to the
minimum set that covers 90% of XGBoost gain importance.

Results saved to results/fs_variants/{dataset}_{method}.csv (incremental).

Usage
-----
    python benchmark/retrain_with_fs.py [--dataset all|mushroom|spambase|musk|monk]
                                        [--method all|LM_Residual|STE|...]
                                        [--n_trials 30]
                                        [--threshold 0.90]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from luknn.benchmark.datasets import load_monk, load_mushroom, load_spambase, load_musk
from luknn.benchmark.config import ExperimentConfig
from luknn.benchmark.runner import BenchmarkRunner
from luknn.preprocessing import XGBFeatureSelector
from luknn.dlm.network import make_dlm_net
from luknn.dlm.optimizer import DLMOptimizer

# ── dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "monk":     {"loader": lambda: load_monk(problem=1, seed=0), "name": "monk_1"},
    "mushroom": {"loader": load_mushroom, "name": "mushroom"},
    "spambase": {"loader": load_spambase, "name": "spambase"},
    "musk":     {"loader": load_musk,     "name": "musk"},
}

# ── ŁNN method configs (best params from final5_clean) ───────────────────────
LNN_METHODS = {
    "LM_Residual": {
        "optimizer_method": "LM_Residual",
        "optimizer_params": {"mu_init": 0.01},
        "hidden_width": 8, "n_blocks": 1, "n_inner": 1,
        "hidden_layers": [],
        "max_iter": 400,
    },
    "STE": {
        "optimizer_method": "STE",
        "optimizer_params": {"lr": 0.01},
        "hidden_layers": [8, 4],
        "hidden_width": 8, "n_blocks": 1,
        "max_iter": 2000,
    },
    "STE_Residual": {
        "optimizer_method": "STE_Residual",
        "optimizer_params": {"lr": 0.01},
        "hidden_width": 8, "n_blocks": 1, "n_inner": 1,
        "hidden_layers": [],
        "max_iter": 2000,
    },
    "Proximal": {
        "optimizer_method": "Proximal",
        "optimizer_params": {"lr": 0.005, "lambda_sparse": 0.0001},
        "hidden_layers": [4, 4],
        "hidden_width": 4, "n_blocks": 1,
        "max_iter": 2000,
    },
    "Proximal_Residual": {
        "optimizer_method": "Proximal_Residual",
        "optimizer_params": {"lr": 0.005, "lambda_sparse": 0.0001},
        "hidden_width": 6, "n_blocks": 1, "n_inner": 1,
        "hidden_layers": [],
        "max_iter": 2000,
    },
}

# ── DLM config per dataset (adapted for reduced k) ───────────────────────────
DLM_BASE = dict(
    n_hidden_layers=2,
    temperature=2.0,
    gate_set="rep",
    max_iter=2000,
    tol_mse=5e-3,
    batch_size=512,
    lr=5e-3, T_init=2.0, T_final=0.05,
    lambda_entropy=0.15, p1_fraction=0.5, conf_threshold=0.90,
)


def _dlm_heads(k: int) -> int:
    """n_output_heads = k (same as input, capped at 64)."""
    return min(k, 64)


def _dlm_width(k: int) -> int:
    """hidden_width = 4×k capped at 256."""
    return min(4 * k, 256)


# ── incremental CSV writer ────────────────────────────────────────────────────
def _append_row(out_file: Path, row: dict, write_header: bool) -> None:
    with open(out_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _done_seeds(out_file: Path) -> set[int]:
    if not out_file.exists():
        return set()
    rows = list(csv.DictReader(open(out_file)))
    return {int(r["seed"]) for r in rows}


# ── ŁNN trial ─────────────────────────────────────────────────────────────────
def run_lnn_trial(ds, method_name: str, mcfg: dict, seed: int,
                  fs: XGBFeatureSelector, threshold: float) -> dict:
    torch.manual_seed(seed)

    x_tr = fs.transform(ds.X_train)
    x_te = fs.transform(ds.X_test)
    k = fs.k_

    cfg = ExperimentConfig(
        name=f"{method_name}_fs",
        seed=seed,
        n_inputs=k,
        hidden_layers=mcfg.get("hidden_layers", [k, k]),
        optimizer_method=mcfg["optimizer_method"],
        optimizer_params=mcfg["optimizer_params"],
        hidden_width=min(mcfg.get("hidden_width", k), k),
        n_blocks=mcfg.get("n_blocks", 1),
        n_inner=mcfg.get("n_inner", 1),
        tol_mse=2e-3,
        max_iter=mcfg.get("max_iter", 400),
        n_trials=1,
        use_feature_selection=False,  # FS already applied manually
        verbose=False,
    )

    # Build model and run through BenchmarkRunner internals manually
    from luknn.layers.lukasiewicz_linear import LukasiewiczNet
    from luknn.network.residual_luknn import LukResidualNet
    from luknn.optimizers import LMOptimizer, STEOptimizer, ProximalOptimizer
    from luknn.benchmark.metrics import compute_accuracy, compute_f1

    _MODE = {"LM": "continuous", "LM_Residual": "continuous",
              "STE": "ste", "STE_Residual": "ste",
              "Proximal": "clamp", "Proximal_Residual": "clamp"}
    _RESIDUAL = {"LM_Residual", "STE_Residual", "Proximal_Residual"}

    mode = _MODE[method_name]
    if method_name in _RESIDUAL:
        model = LukResidualNet(k, cfg.hidden_width, cfg.n_blocks, cfg.n_inner, mode)
    else:
        model = LukasiewiczNet(k, cfg.hidden_layers, mode)

    if method_name in ("LM", "LM_Residual"):
        opt = LMOptimizer(model, **mcfg["optimizer_params"])
    elif method_name in ("STE", "STE_Residual"):
        opt = STEOptimizer(model, **mcfg["optimizer_params"])
    else:
        opt = ProximalOptimizer(model, **mcfg["optimizer_params"])

    t0 = time.perf_counter()
    res = opt.train(x_tr, ds.y_train, tol_mse=cfg.tol_mse, max_iter=cfg.max_iter)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        pred = model(x_te)

    return {
        "dataset": ds.name,
        "method": method_name,
        "seed": seed,
        "fs_k": k,
        "fs_threshold": threshold,
        "n_features_orig": ds.n_features,
        "accuracy": compute_accuracy(pred, ds.y_test),
        "f1": compute_f1(pred, ds.y_test),
        "final_mse": res.final_mse,
        "converged": res.converged,
        "iterations": res.iterations,
        "total_time_s": elapsed,
    }


# ── DLM trial ─────────────────────────────────────────────────────────────────
def run_dlm_trial(ds, seed: int, fs: XGBFeatureSelector, threshold: float) -> dict:
    torch.manual_seed(seed)
    k = fs.k_

    x_tr = fs.transform(ds.X_train)
    x_te = fs.transform(ds.X_test)
    y_tr = ds.y_train.unsqueeze(-1)

    model = make_dlm_net(
        n_features=k,
        n_hidden_layers=DLM_BASE["n_hidden_layers"],
        hidden_width=_dlm_width(k),
        temperature=DLM_BASE["temperature"],
        gate_set=DLM_BASE["gate_set"],
        seed=seed,
        n_output_heads=_dlm_heads(k),
    )

    opt = DLMOptimizer(
        model,
        lr=DLM_BASE["lr"], T_init=DLM_BASE["T_init"], T_final=DLM_BASE["T_final"],
        lambda_entropy=DLM_BASE["lambda_entropy"],
        p1_fraction=DLM_BASE["p1_fraction"],
        conf_threshold=DLM_BASE["conf_threshold"],
    )

    t0 = time.perf_counter()
    res = opt.train(x_tr, y_tr, max_iter=DLM_BASE["max_iter"],
                    tol_mse=DLM_BASE["tol_mse"], batch_size=DLM_BASE["batch_size"])
    elapsed = time.perf_counter() - t0

    crys = res.extra["crystallized_model"]
    with torch.no_grad():
        pred = (crys(x_te).squeeze() > 0.5).float()

    acc = (pred == ds.y_test).float().mean().item()
    try:
        f1 = float(f1_score(ds.y_test.numpy(), pred.numpy(), zero_division=0))
    except Exception:
        f1 = float("nan")

    return {
        "dataset": ds.name,
        "method": "DLM",
        "seed": seed,
        "fs_k": k,
        "fs_threshold": threshold,
        "n_features_orig": ds.n_features,
        "accuracy": acc,
        "f1": f1,
        "final_mse": res.final_mse,
        "gate_confidence": res.extra["gate_confidence"],
        "representability": res.extra["representability"],
        "converged": res.converged,
        "iterations": res.iterations,
        "reason": res.reason,
        "total_time_s": elapsed,
    }


# ── benchmark runner ──────────────────────────────────────────────────────────
def run_benchmark(ds_key: str, method_name: str, n_trials: int,
                  threshold: float, out_dir: Path) -> None:
    ds_cfg = DATASETS[ds_key]
    ds = ds_cfg["loader"]()

    out_file = out_dir / f"{ds_cfg['name']}_{method_name}_fs.csv"
    done = _done_seeds(out_file)
    remaining = [i for i in range(n_trials) if (i * 1000 + 42) not in done]

    if not remaining:
        print(f"  [{ds_cfg['name']} / {method_name}] already complete ({n_trials} trials)")
        return

    print(f"\n{'='*62}")
    print(f"  {ds_cfg['name'].upper()} / {method_name}  "
          f"({len(remaining)}/{n_trials} trials remaining)")

    # Fit FS once on the full training set (seed-independent)
    fs = XGBFeatureSelector(threshold=threshold, importance_type="gain")
    fs.fit(ds.X_train, ds.y_train)
    print(f"  {fs.summary()}")
    print(f"{'='*62}", flush=True)

    for trial_idx in remaining:
        seed = trial_idx * 1000 + 42
        t0 = time.perf_counter()

        try:
            if method_name == "DLM":
                row = run_dlm_trial(ds, seed, fs, threshold)
            else:
                row = run_lnn_trial(ds, method_name, LNN_METHODS[method_name],
                                    seed, fs, threshold)
        except Exception as e:
            print(f"  [!] seed={seed} failed: {e}", flush=True)
            continue

        dt = time.perf_counter() - t0
        write_header = not out_file.exists() or out_file.stat().st_size == 0
        _append_row(out_file, row, write_header)

        f1_str = f"{row['f1']:.3f}" if row["f1"] == row["f1"] else "nan"
        print(f"  [{trial_idx+1:2d}/{n_trials}] seed={seed}  "
              f"k={row['fs_k']}  acc={row['accuracy']:.3f}  F1={f1_str}  "
              f"mse={row['final_mse']:.4f}  ({dt:.1f}s)", flush=True)

    # Summary
    if out_file.exists():
        rows = list(csv.DictReader(open(out_file)))
        f1s  = [float(r["f1"]) for r in rows if r["f1"] not in ("nan", "")]
        accs = [float(r["accuracy"]) for r in rows]
        k    = int(rows[0]["fs_k"])
        n_orig = int(rows[0]["n_features_orig"])
        print(f"\n  Summary ({ds_cfg['name']} / {method_name}, {len(rows)} trials, "
              f"k={k}/{n_orig}):")
        if f1s:
            print(f"    F1 : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}  "
                  f"[{np.min(f1s):.3f}, {np.max(f1s):.3f}]")
        print(f"    Acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS) + ["all"])
    parser.add_argument("--method",  default="all",
                        choices=list(LNN_METHODS) + ["DLM", "all"])
    parser.add_argument("--n_trials",  type=int,   default=30)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "fs_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_keys  = list(DATASETS) if args.dataset == "all" else [args.dataset]
    methods  = list(LNN_METHODS) + ["DLM"] if args.method == "all" else [args.method]

    for ds_key in ds_keys:
        for method_name in methods:
            run_benchmark(ds_key, method_name, args.n_trials, args.threshold, out_dir)
