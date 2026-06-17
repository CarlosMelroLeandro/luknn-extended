"""
BenchmarkRunner — orchestrates a full experiment from config to results.

Flow per trial
--------------
1.  Set random seed.
2.  Load dataset.
3.  Build LukasiewiczNet (mode depends on optimizer).
4.  Instantiate optimizer.
5.  Train (with memory tracking).
6.  Compute all metrics.
7.  Accumulate BenchmarkResult.
"""

from __future__ import annotations
import time
import torch
from torch import Tensor

from .config import ExperimentConfig
from .datasets import load_dataset, Dataset
from ..preprocessing import XGBFeatureSelector
from .metrics import (
    BenchmarkResult,
    MemoryTracker,
    compute_accuracy,
    compute_f1,
    compute_lambda_similarity,
    compute_delta_n,
    iter_to_threshold,
    save_results,
)
from ..layers.lukasiewicz_linear import LukasiewiczNet
from ..network.residual_luknn import LukResidualNet
from ..optimizers import LMOptimizer, STEOptimizer, ProximalOptimizer

_OPTIMIZER_MODE = {
    "LM": "continuous",
    "LM_Residual": "continuous",
    "STE": "ste",
    "STE_Residual": "ste",
    "Proximal": "clamp",
    "Proximal_Residual": "clamp",
}

_RESIDUAL_METHODS = {"LM_Residual", "STE_Residual", "Proximal_Residual"}


class BenchmarkRunner:
    """
    Run N trials of a single (method, dataset) combination.

    Parameters
    ----------
    config : ExperimentConfig
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> list[BenchmarkResult]:
        dataset = load_dataset(self.config)
        results = []

        for trial in range(self.config.n_trials):
            seed = self.config.seed + trial * 1000
            torch.manual_seed(seed)

            result = self._run_trial(dataset, trial, seed)
            results.append(result)
            if self.config.verbose:
                print(result.summary())

        return results

    def _run_trial(
        self, dataset: Dataset, trial: int, seed: int
    ) -> BenchmarkResult:
        cfg = self.config
        method = cfg.optimizer_method
        mode = _OPTIMIZER_MODE.get(method, "continuous")

        # Feature selection (fitted on train split only — no leakage)
        x_train = dataset.X_train
        x_test  = dataset.X_test
        x_val   = dataset.X_val
        fs_k: int | None = None
        fs_indices: list[int] | None = None

        if cfg.use_feature_selection:
            fs = XGBFeatureSelector(
                threshold=cfg.fs_threshold,
                importance_type=cfg.fs_importance_type,
                max_features=cfg.fs_max_features,
            )
            x_train = fs.fit_transform(dataset.X_train, dataset.y_train)
            x_test  = fs.transform(dataset.X_test)
            if x_val is not None:
                x_val = fs.transform(x_val)
            fs_k = fs.k_
            fs_indices = fs.selected_indices_.tolist()
            if cfg.verbose:
                print(f"  [FS] {fs.summary()}")

        # Use selected feature count as n_inputs
        n_inputs = x_train.shape[1]

        if method in _RESIDUAL_METHODS:
            model = LukResidualNet(
                n_inputs=n_inputs,
                hidden_width=cfg.hidden_width,
                n_blocks=cfg.n_blocks,
                n_inner=cfg.n_inner,
                mode=mode,
            )
        else:
            model = LukasiewiczNet(n_inputs, cfg.hidden_layers, mode=mode)

        optimizer = self._build_optimizer(model, method, cfg.optimizer_params)

        y_train = dataset.y_train

        with MemoryTracker() as mem:
            t0 = time.perf_counter()
            train_result = optimizer.train(
                x_train,
                y_train,
                tol_mse=cfg.tol_mse,
                max_iter=cfg.max_iter,
                verbose=cfg.verbose,
            )
            mem.update()
            elapsed = time.perf_counter() - t0

        # Evaluate on val set during HP tuning; on test set for final evaluation
        if cfg.use_val_split and x_val is not None:
            eval_X, eval_y = x_val, dataset.y_val
        else:
            eval_X, eval_y = x_test, dataset.y_test

        with torch.no_grad():
            pred_test = model(eval_X)

        accuracy = compute_accuracy(pred_test, eval_y)
        f1 = compute_f1(pred_test, eval_y)
        lam = compute_lambda_similarity(model, eval_X, eval_y)
        delta_n = compute_delta_n(model)
        crystallized = delta_n < 1e-3
        n_iter = train_result.iterations
        time_per_iter = elapsed / max(n_iter, 1)
        i_thresh = iter_to_threshold(train_result.mse_history, cfg.tol_mse)

        return BenchmarkResult(
            method=method,
            dataset=dataset.name,
            trial=trial,
            final_mse=train_result.final_mse,
            accuracy=accuracy,
            f1=f1,
            is_crystallized=crystallized,
            delta_n=delta_n,
            lambda_similarity=lam,
            total_time_s=elapsed,
            time_per_iter_s=time_per_iter,
            peak_memory_mb=mem.peak_delta_mb,
            converged=train_result.converged,
            iterations=n_iter,
            iter_to_threshold=i_thresh,
            mse_history=train_result.mse_history,
            config={**cfg.to_dict(), "fs_k": fs_k, "fs_indices": fs_indices},
        )

    @staticmethod
    def _build_optimizer(model, method: str, params: dict):
        if method in ("LM", "LM_Residual"):
            return LMOptimizer(model, **params)
        elif method in ("STE", "STE_Residual"):
            return STEOptimizer(model, **params)
        elif method in ("Proximal", "Proximal_Residual"):
            return ProximalOptimizer(model, **params)
        else:
            raise ValueError(f"Unknown optimizer method: {method!r}")
