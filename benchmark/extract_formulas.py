"""
Łukasiewicz formula extraction from crystallized models.

For each dataset:
  • Trains up to n_trials (baseline LM + residual LM_Residual).
  • Collects all crystallized models.
  • Calls extract_formula / extract_formula_residual.
  • Prints the formula layer by layer.
  • Saves to results/formulas/<dataset>_<timestamp>.json

Usage:
  python benchmark/extract_formulas.py
  python benchmark/extract_formulas.py --datasets heart monk_1 monk_2 monk_3
  python benchmark/extract_formulas.py --n_trials 10
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

import torch
import numpy as np

from luknn.benchmark.config import ExperimentConfig
from luknn.benchmark.datasets import load_dataset
from luknn.benchmark.metrics import compute_delta_n
from luknn.layers.lukasiewicz_linear import LukasiewiczNet
from luknn.network.residual_luknn import LukResidualNet
from luknn.optimizers import LMOptimizer
from luknn.extraction.extractor import extract_formula
from luknn.extraction.residual_extractor import extract_formula_residual


ROOT = Path(__file__).parents[1]

TUNING_FILES = {
    "mushroom":      ROOT / "results/tuning/mushroom_20260604_091426.json",
    "heart":         ROOT / "results/tuning/heart_20260604_091426.json",
    "monk_1":        ROOT / "results/tuning/monk_1_20260604_091426.json",
    "monk_2":        ROOT / "results/tuning/monk_2_20260604_103838.json",
    "monk_3":        ROOT / "results/tuning/monk_3_20260604_114059.json",
    "breast_cancer": ROOT / "results/tuning/breast_cancer_20260604_091426.json",
}

MONK_RULES = {
    1: "(a1==a2) OR (a5==1)",
    2: "exactly 2 of {a1..a6} are 1",
    3: "(a5==3 AND a4==1) OR (a5!=4 AND a2!=3)  [5% noise]",
}


# ── Configs (reuses logic from retrain_best) ──────────────────────────────────

def _load_best(key: str) -> dict:
    return json.loads(TUNING_FILES[key].read_text())["best_params"]


def _make_configs(dataset_key: str, n_trials: int) -> tuple[ExperimentConfig, ExperimentConfig]:
    """Return (baseline_cfg, residual_cfg) for the given dataset."""
    best = _load_best(dataset_key)

    if dataset_key == "mushroom":
        common = dict(seed=42, n_inputs=111, dataset_type="mushroom",
                      tol_mse=0.15, max_iter=600, n_trials=n_trials, verbose=False)
        opt = dict(patience=80, crystallize_n=2, prune=False, batch_size=512,
                   mu_init=best["mu_init"])
        b = ExperimentConfig(name="LM — Mushroom", hidden_layers=[6,4],
                             optimizer_method="LM", optimizer_params=opt, **common)
        r = ExperimentConfig(name="LM_Residual — Mushroom", hidden_layers=[6,4],
                             optimizer_method="LM_Residual", optimizer_params=opt,
                             hidden_width=best["hidden_width"], n_blocks=best["n_blocks"],
                             n_inner=1, **common)

    elif dataset_key == "heart":
        common = dict(seed=42, n_inputs=13, dataset_type="heart_disease",
                      heart_subset="cleveland", tol_mse=0.15, max_iter=800,
                      n_trials=n_trials, verbose=False)
        opt = dict(patience=100, crystallize_n=2, batch_size=0,
                   mu_init=best["mu_init"], prune=best.get("prune", True))
        b = ExperimentConfig(name="LM — Heart", hidden_layers=[6,4],
                             optimizer_method="LM", optimizer_params=opt, **common)
        r = ExperimentConfig(name="LM_Residual — Heart", hidden_layers=[6,4],
                             optimizer_method="LM_Residual", optimizer_params=opt,
                             hidden_width=best["hidden_width"], n_blocks=best["n_blocks"],
                             n_inner=1, **common)

    elif dataset_key.startswith("monk_"):
        prob = int(dataset_key[-1])
        common = dict(seed=42, n_inputs=17, dataset_type="monk", monk_problem=prob,
                      tol_mse=0.10, max_iter=500, n_trials=n_trials, verbose=False)
        opt = dict(patience=80, crystallize_n=2, prune=False, batch_size=0,
                   mu_init=best["mu_init"])
        b = ExperimentConfig(name=f"LM — MONK-{prob}", hidden_layers=[8,4],
                             optimizer_method="LM", optimizer_params=opt, **common)
        r = ExperimentConfig(name=f"LM_Residual — MONK-{prob}", hidden_layers=[8,4],
                             optimizer_method="LM_Residual", optimizer_params=opt,
                             hidden_width=best["hidden_width"], n_blocks=best["n_blocks"],
                             n_inner=1, **common)

    elif dataset_key == "breast_cancer":
        common = dict(seed=42, n_inputs=20, dataset_type="breast_cancer",
                      tol_mse=0.15, max_iter=800, n_trials=n_trials, verbose=False)
        opt = dict(patience=100, crystallize_n=2, batch_size=0,
                   mu_init=best["mu_init"], prune=best.get("prune", True))
        b = ExperimentConfig(name="LM — Breast Cancer", hidden_layers=[8,8],
                             optimizer_method="LM", optimizer_params=opt, **common)
        r = ExperimentConfig(name="LM_Residual — Breast Cancer", hidden_layers=[8,8],
                             optimizer_method="LM_Residual", optimizer_params=opt,
                             hidden_width=best["hidden_width"], n_blocks=best["n_blocks"],
                             n_inner=1, **common)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key!r}")

    return b, r


# ── Training + extraction engine ──────────────────────────────────────────────

def _train_one(cfg: ExperimentConfig, dataset, trial: int):
    """
    Train one model and return (model, final_mse, crystallized, accuracy).
    The model is crystallized if delta_n < 1e-3.
    """
    seed = cfg.seed + trial * 1000
    torch.manual_seed(seed)
    n_inputs = dataset.n_features

    if cfg.optimizer_method == "LM_Residual":
        model = LukResidualNet(n_inputs=n_inputs, hidden_width=cfg.hidden_width,
                               n_blocks=cfg.n_blocks, n_inner=cfg.n_inner,
                               mode="continuous")
    else:
        model = LukasiewiczNet(n_inputs, cfg.hidden_layers, mode="continuous")

    opt = LMOptimizer(model, **cfg.optimizer_params)
    result = opt.train(dataset.X_train, dataset.y_train,
                       tol_mse=cfg.tol_mse, max_iter=cfg.max_iter)

    with torch.no_grad():
        pred = model(dataset.X_test)
        acc = float(((pred >= 0.5).float() == dataset.y_test).float().mean())

    delta = compute_delta_n(model)
    return model, result.final_mse, delta < 1e-3, acc


def _do_extract(model, feature_names: list[str], is_residual: bool) -> dict:
    """Extract formula and return a serializable dict.

    n_values=2: uses binary truth table (2^n rows).
    Datasets with many features (MONK=17, Mushroom=111) make n_values>=3
    computationally infeasible (3^17 = 129M, 4^17 = 17B rows).
    n_values=2 is correct for binary classification {0,1}.
    """
    if is_residual:
        res = extract_formula_residual(model, input_names=feature_names, n_values=2)
    else:
        res = extract_formula(model, input_names=feature_names, n_values=2)

    return {
        "formula":         res.formula,
        "representable":   res.representable,
        "layer_formulas":  res.layer_formulas,
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def _print_formula_block(title: str, entries: list[dict], ref_rule: str | None = None) -> None:
    print(f"\n  ┌─ {title}")
    if not entries:
        print("  │   (no model crystallized in this run)")
        print("  └─")
        return

    if ref_rule:
        print(f"  │   Reference rule: {ref_rule}")

    for i, e in enumerate(entries, 1):
        rep_mark = "✓ representable" if e["representable"] else "~ λ-approximation"
        print(f"  │")
        print(f"  │   Trial {e['trial']}  acc={e['acc']:.4f}  mse={e['mse']:.4f}  [{rep_mark}]")

        # Layer by layer
        for layer_i, layer_syms in enumerate(e["layer_formulas"]):
            tag = f"layer {layer_i+1}"
            if layer_i == len(e["layer_formulas"]) - 1:
                tag = "output"
            for j, sym in enumerate(layer_syms):
                label = f"h{layer_i+1}_{j+1}" if tag != "output" else "F"
                print(f"  │     {label} = {sym}")

        print(f"  │   → F = {e['formula']}")

        if i < len(entries):
            # Check if formula is identical to the previous one (stability)
            if entries[i-1]["formula"] == entries[i]["formula"] if i > 0 else False:
                print("  │   (formula identical to previous ✓)")

    # Show whether all formulas are equal
    formulas = [e["formula"] for e in entries]
    if len(set(formulas)) == 1 and len(formulas) > 1:
        print(f"  │")
        print(f"  │   *** All {len(entries)} formulas are identical — stable convergence ***")
    elif len(set(formulas)) > 1:
        print(f"  │")
        print(f"  │   ({len(set(formulas))} distinct formulas across {len(entries)} trials)")
    print("  └─")


def _print_dataset_header(ds_key: str, n_base: int, n_res: int) -> None:
    print(f"\n{'='*70}")
    print(f"  {ds_key.upper()}")
    if ds_key.startswith("monk_"):
        p = int(ds_key[-1])
        print(f"  True rule: {MONK_RULES[p]}")
    print(f"  Baseline crystallized: {n_base} trials  |  Residual crystallized: {n_res} trials")
    print(f"{'='*70}")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(
    ds_key: str,
    n_trials: int,
    results_dir: Path,
) -> dict:
    base_cfg, res_cfg = _make_configs(ds_key, n_trials)
    dataset = load_dataset(base_cfg)
    feat_names = dataset.feature_names or [f"x{i+1}" for i in range(dataset.n_features)]

    # Truncate names to 10 chars to avoid cluttering the formula
    short_names = [n[:10] for n in feat_names]

    base_entries: list[dict] = []
    res_entries:  list[dict] = []

    print(f"\n  Training baseline ({n_trials} trials)…", flush=True)
    for t in range(n_trials):
        model, mse, cryst, acc = _train_one(base_cfg, dataset, t)
        status = "✓ cryst" if cryst else "  ----"
        print(f"    trial {t+1:2d}  mse={mse:.4f}  acc={acc:.4f}  {status}", flush=True)
        if cryst:
            extr = _do_extract(model, short_names, is_residual=False)
            base_entries.append({"trial": t+1, "mse": mse, "acc": acc, **extr})

    print(f"\n  Training residual ({n_trials} trials)…", flush=True)
    for t in range(n_trials):
        model, mse, cryst, acc = _train_one(res_cfg, dataset, t)
        status = "✓ cryst" if cryst else "  ----"
        print(f"    trial {t+1:2d}  mse={mse:.4f}  acc={acc:.4f}  {status}", flush=True)
        if cryst:
            extr = _do_extract(model, short_names, is_residual=True)
            res_entries.append({"trial": t+1, "mse": mse, "acc": acc, **extr})

    ref = MONK_RULES.get(int(ds_key[-1])) if ds_key.startswith("monk_") else None
    _print_dataset_header(ds_key, len(base_entries), len(res_entries))
    _print_formula_block("BASELINE (LM)", base_entries, ref_rule=ref)
    _print_formula_block("RESIDUAL (LM_Residual)", res_entries, ref_rule=ref)

    payload = {
        "dataset":  ds_key,
        "baseline": base_entries,
        "residual": res_entries,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"{ds_key}_{ts}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n  Saved to {out}")
    return payload


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+",
        default=["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"],
        choices=["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer", "all"],
    )
    p.add_argument("--n_trials",    type=int, default=10)
    p.add_argument("--results_dir", default="results/formulas")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = (
        ["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"]
        if "all" in args.datasets else args.datasets
    )
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print("  Łukasiewicz Formula Extraction")
    print(f"  Datasets : {datasets}")
    print(f"  Trials   : {args.n_trials}")
    print(f"{'#'*70}")

    for ds in datasets:
        process_dataset(ds, args.n_trials, results_dir)

    print(f"\n{'#'*70}")
    print("  Done.")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
