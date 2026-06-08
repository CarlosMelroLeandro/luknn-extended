"""Base class and shared data structures for all three optimizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class TrainingResult:
    converged: bool
    final_mse: float
    mse_history: list[float]
    iterations: int
    total_time_s: float
    reason: str = ""          # "converged" | "stagnation" | "mu_max" | "max_iter"
    extra: dict = field(default_factory=dict)


class BaseOptimizer(ABC):
    """Common interface for LM, STE, and Proximal optimizers."""

    @abstractmethod
    def train(
        self,
        x: Tensor,
        y: Tensor,
        tol_mse: float = 2e-3,
        max_iter: int = 400,
        verbose: bool = False,
        sample_weight: Tensor | None = None,
    ) -> TrainingResult: ...

    @staticmethod
    def _mse(pred: Tensor, y: Tensor, w: Tensor | None = None) -> float:
        sq = (pred - y) ** 2
        if w is not None:
            return (sq * w).mean().item()
        return sq.mean().item()
