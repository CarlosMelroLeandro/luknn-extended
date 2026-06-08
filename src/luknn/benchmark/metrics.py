"""
Benchmark metrics: performance, representability, efficiency, convergence.
"""

from __future__ import annotations
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from torch import Tensor

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from sklearn.metrics import f1_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    trial: int
    # --- Performance ---
    final_mse: float
    accuracy: float
    f1: float
    # --- Representability ---
    is_crystallized: bool
    delta_n: float               # Representation error Δ(N) — 0 = pure CNN
    lambda_similarity: float     # exp(-MAE) to nearest representable formula
    # --- Efficiency ---
    total_time_s: float
    time_per_iter_s: float
    peak_memory_mb: float
    # --- Convergence ---
    converged: bool
    iterations: int
    iter_to_threshold: int | None   # first iter where mse < tol_mse
    mse_history: list[float] = field(default_factory=list)
    # --- Config snapshot ---
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mse_history", None)   # too long for summary tables
        return d

    def summary(self) -> str:
        lines = [
            f"  method={self.method}  dataset={self.dataset}  trial={self.trial}",
            f"  mse={self.final_mse:.5f}  acc={self.accuracy:.3f}  f1={self.f1:.3f}",
            f"  crystallized={self.is_crystallized}  Δ(N)={self.delta_n:.4f}  λ={self.lambda_similarity:.4f}",
            f"  time={self.total_time_s:.1f}s  iters={self.iterations}  "
            f"iter_to_thr={self.iter_to_threshold}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_accuracy(pred: Tensor, y: Tensor, tol: float = 0.1) -> float:
    """
    Tolerance-based accuracy: fraction of samples where |pred − y| < tol.

    For binary targets (y ∈ {0,1}) this is equivalent to correct classification.
    For continuous truth-table targets it measures closeness to ground truth.
    """
    return ((pred.detach() - y).abs() < tol).float().mean().item()


def compute_f1(pred: Tensor, y: Tensor, threshold: float = 0.5) -> float:
    """
    F1 score. Only meaningful for binary targets; returns nan otherwise.
    """
    if not _HAS_SKLEARN:
        return float("nan")
    t = y.cpu().numpy()
    # Only compute F1 if y is binary
    unique = set(t.round().tolist())
    if not unique.issubset({0.0, 1.0}):
        return float("nan")
    p = (pred.detach().cpu().numpy() >= threshold).astype(float)
    try:
        return float(f1_score(t.round(), p, zero_division=0))
    except Exception:
        return float("nan")


def compute_lambda_similarity(
    model,
    x: Tensor,
    y: Tensor,
) -> float:
    """
    λ-similarity = exp(−MAE) between model output and target.
    λ = 1.0 means perfect match; λ = 0 means maximum error.
    """
    with torch.no_grad():
        pred = model(x)
    mae = (pred - y).abs().mean().item()
    return math.exp(-mae)


def compute_delta_n(model) -> float:
    """Representation error Δ(N) = Σ(w − floor(w)) over all parameters."""
    from ..network.crystallization import representation_error
    all_w = torch.cat([p.data.view(-1) for p in model.parameters()])
    return representation_error(all_w).item()


def iter_to_threshold(mse_history: list[float], tol: float) -> int | None:
    for i, mse in enumerate(mse_history):
        if mse < tol:
            return i
    return None


class MemoryTracker:
    """Tracks peak RSS memory on CPU (GPU tracking requires CUDA)."""

    def __init__(self):
        self._start_mb: float = 0.0
        self._peak_mb: float = 0.0

    def __enter__(self):
        self._start_mb = self._rss()
        self._peak_mb = self._start_mb
        return self

    def __exit__(self, *_):
        self._peak_mb = max(self._peak_mb, self._rss())

    def update(self):
        self._peak_mb = max(self._peak_mb, self._rss())

    @property
    def peak_delta_mb(self) -> float:
        return max(0.0, self._peak_mb - self._start_mb)

    @staticmethod
    def _rss() -> float:
        if _HAS_PSUTIL:
            return psutil.Process(os.getpid()).memory_info().rss / 1024**2
        return 0.0


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def save_results(results: list[BenchmarkResult], out_dir: str | Path) -> Path:
    """Save results to JSON (full) and CSV (summary)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"results_{timestamp}.json"
    csv_path = out_dir / f"results_{timestamp}.csv"

    # JSON
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # CSV summary
    import csv
    if results:
        keys = list(results[0].to_dict().keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r.to_dict())

    return json_path
