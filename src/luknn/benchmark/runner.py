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
    "Proximal": "clamp",
}


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

        # Use actual dataset feature count (overrides config n_inputs for real datasets)
        n_inputs = dataset.n_features if dataset.n_features != cfg.n_inputs else cfg.n_inputs

        if method == "LM_Residual":
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

        x_train, y_train = dataset.X_train, dataset.y_train

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

        # --- Metrics on test set ---
        with torch.no_grad():
            pred_test = model(dataset.X_test)

        accuracy = compute_accuracy(pred_test, dataset.y_test)
        f1 = compute_f1(pred_test, dataset.y_test)
        lam = compute_lambda_similarity(model, dataset.X_test, dataset.y_test)
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
            config=cfg.to_dict(),
        )

    @staticmethod
    def _build_optimizer(model, method: str, params: dict):
        if method in ("LM", "LM_Residual"):
            return LMOptimizer(model, **params)
        elif method == "STE":
            return STEOptimizer(model, **params)
        elif method == "Proximal":
            return ProximalOptimizer(model, **params)
        else:
            raise ValueError(f"Unknown optimizer method: {method!r}")
