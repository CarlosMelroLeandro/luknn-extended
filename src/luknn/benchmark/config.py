"""
Experiment configuration: dataclass + YAML loader.

YAML schema
-----------
experiment:
  name: str
  seed: int

architecture:
  n_inputs: int
  hidden_layers: list[int]   # e.g. [4, 4]

optimizer:
  method: "LM" | "STE" | "Proximal"
  params: dict               # method-specific kwargs

dataset:
  type: "truth_table" | "mushroom" | "babi"
  # truth_table only:
  formula: str               # "f1" | "f2" | "f6" | "random"
  n_values: int              # 2 | 3 | 4 | 5
  n_vars: int                # number of propositional variables
  # mushroom / babi: no extra fields needed

training:
  tol_mse: float
  max_iter: int
  n_trials: int              # independent random restarts

logging:
  results_dir: str
  verbose: bool
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    # Architecture
    n_inputs: int
    hidden_layers: list[int]
    # Optimizer
    optimizer_method: str
    optimizer_params: dict = field(default_factory=dict)
    # Dataset
    dataset_type: str = "truth_table"
    formula: str | None = "f1"
    n_values: int = 3
    n_vars: int = 6
    heart_subset: str = "cleveland"   # "cleveland" | "all"
    monk_problem: int = 1             # 1 | 2 | 3
    # Residual architecture (used when optimizer_method == "LM_Residual")
    hidden_width: int = 8
    n_blocks: int = 1
    n_inner: int = 1
    # Training
    tol_mse: float = 2e-3
    max_iter: int = 400
    n_trials: int = 3
    # Logging
    results_dir: str = "results"
    verbose: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        exp = raw.get("experiment", {})
        arch = raw.get("architecture", {})
        opt = raw.get("optimizer", {})
        ds = raw.get("dataset", {})
        tr = raw.get("training", {})
        lg = raw.get("logging", {})

        return cls(
            name=exp.get("name", "experiment"),
            seed=exp.get("seed", 42),
            n_inputs=arch.get("n_inputs", 6),
            hidden_layers=arch.get("hidden_layers", [4, 4]),
            optimizer_method=opt.get("method", "LM"),
            optimizer_params=opt.get("params", {}),
            dataset_type=ds.get("type", "truth_table"),
            formula=ds.get("formula", "f1"),
            n_values=ds.get("n_values", 3),
            n_vars=ds.get("n_vars", 6),
            heart_subset=ds.get("heart_subset", "cleveland"),
            monk_problem=ds.get("monk_problem", 1),
            hidden_width=arch.get("hidden_width", 8),
            n_blocks=arch.get("n_blocks", 1),
            n_inner=arch.get("n_inner", 1),
            tol_mse=tr.get("tol_mse", 2e-3),
            max_iter=tr.get("max_iter", 400),
            n_trials=tr.get("n_trials", 3),
            results_dir=lg.get("results_dir", "results"),
            verbose=lg.get("verbose", False),
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)
