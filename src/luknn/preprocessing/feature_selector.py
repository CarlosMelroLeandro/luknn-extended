"""
XGBFeatureSelector — XGBoost-based feature selection for ŁNN pipelines.

Strategy: fit XGBoost on training data, rank features by cumulative 'gain'
importance, retain the minimum set that covers >= threshold (default 90%) of
total gain.

Invariant: fit() uses only X_train/y_train — no data leakage from test set.

Usage
-----
    fs = XGBFeatureSelector(threshold=0.90, importance_type="gain")
    X_tr_sel = fs.fit_transform(X_train, y_train)   # → Tensor[n_train, k]
    X_te_sel = fs.transform(X_test)                  # → Tensor[n_test, k]
    print(fs.k_, fs.selected_indices_)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class XGBFeatureSelector:
    """
    Parameters
    ----------
    threshold      : float   Minimum cumulative gain fraction to cover (default 0.90).
    importance_type: str     XGBoost importance type: 'gain' | 'weight' | 'cover'.
    xgb_params     : dict    Extra kwargs forwarded to xgboost.XGBClassifier.
    min_features   : int     Lower bound on k (default 2).
    max_features   : int | None  Upper bound on k (None = no cap).
    """

    def __init__(
        self,
        threshold: float = 0.90,
        importance_type: str = "gain",
        xgb_params: dict | None = None,
        min_features: int = 2,
        max_features: int | None = None,
    ):
        self.threshold = threshold
        self.importance_type = importance_type
        self.xgb_params = xgb_params or {}
        self.min_features = min_features
        self.max_features = max_features

        # Set after fit()
        self.selected_indices_: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None  # all features, sorted by index
        self.k_: int | None = None
        self._fitted = False

    def fit(self, X_train: Tensor, y_train: Tensor) -> "XGBFeatureSelector":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        X_np = X_train.detach().cpu().numpy().astype(np.float32)
        y_np = y_train.detach().cpu().numpy().astype(np.int32)

        n_features = X_np.shape[1]

        params = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            importance_type=self.importance_type,
            verbosity=0,
        )
        params.update(self.xgb_params)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_np, y_np)

        # Raw importances (one per feature, indexed 0..n_features-1)
        raw = clf.feature_importances_  # shape (n_features,)
        self.feature_importances_ = raw.copy()

        total = raw.sum()
        if total == 0:
            # Degenerate: all importances zero — keep all features
            self.selected_indices_ = np.arange(n_features)
            self.k_ = n_features
            self._fitted = True
            return self

        # Sort descending by importance, compute cumulative fraction
        order = np.argsort(raw)[::-1]
        cumulative = np.cumsum(raw[order]) / total

        # Minimum k such that cumulative >= threshold
        k = int(np.searchsorted(cumulative, self.threshold)) + 1
        k = max(k, self.min_features)
        if self.max_features is not None:
            k = min(k, self.max_features)
        k = min(k, n_features)

        self.selected_indices_ = np.sort(order[:k])  # keep original order
        self.k_ = k
        self._fitted = True
        return self

    def transform(self, X: Tensor) -> Tensor:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        idx = torch.tensor(self.selected_indices_, dtype=torch.long)
        return X[:, idx]

    def fit_transform(self, X_train: Tensor, y_train: Tensor) -> Tensor:
        return self.fit(X_train, y_train).transform(X_train)

    def summary(self) -> str:
        if not self._fitted:
            return "XGBFeatureSelector (not fitted)"
        imp = self.feature_importances_[self.selected_indices_]
        cov = imp.sum() / self.feature_importances_.sum()
        return (
            f"XGBFeatureSelector: k={self.k_}  "
            f"threshold={self.threshold:.0%}  "
            f"actual_coverage={cov:.1%}  "
            f"importance_type={self.importance_type!r}"
        )
