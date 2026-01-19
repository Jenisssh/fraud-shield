"""LightGBM gradient-boosted classifier for fraud detection.

This is the headline model — every other piece of work (calibration,
threshold tuning, SHAP, drift monitoring) wraps around it. The wrapper
exposes a small sklearn-shaped surface (``fit``, ``predict_proba``,
``predict``) but uses native ``lgb.train`` under the hood so we can:

- run **early stopping** against a held-out val set on PR-AUC
- auto-compute ``scale_pos_weight`` from class counts when not supplied
- pin ``deterministic=True`` and ``force_col_wise=True`` for
  bit-reproducible runs across reruns and across machines

Tree models don't need scaling, so unlike the logistic baseline we feed
the engineered DataFrame straight into LightGBM after Time/Amount have
been replaced with their engineered counterparts.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraud_shield.config import settings
from fraud_shield.features.transformers import TimeAmountFeatures

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


class LightGBMFraudClassifier:
    """LightGBM wrapper with a sklearn-like surface.

    Parameters
    ----------
    params:
        Overrides merged on top of :data:`DEFAULT_PARAMS`. Anything LightGBM
        understands works (``num_leaves``, ``learning_rate``, etc.).
    num_boost_round:
        Maximum number of boosting iterations. Early stopping usually
        ends training well before this.
    early_stopping_rounds:
        Stop if val PR-AUC hasn't improved for this many rounds.
        Ignored when ``fit`` is called without a validation set.
    random_state:
        Seed for LightGBM's RNG. Defaults to ``settings.random_seed``.

    Attributes
    ----------
    booster:
        The trained :class:`lightgbm.Booster`. ``None`` until ``fit`` is called.
    feature_names_:
        Engineered column order seen at fit time. Used to align test
        frames at prediction time.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        num_boost_round: int = 2_000,
        early_stopping_rounds: int = 50,
        random_state: int | None = None,
    ) -> None:
        seed = settings.random_seed if random_state is None else random_state
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.transformer = TimeAmountFeatures(drop_original=True)
        self.booster: lgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | NDArray[Any],
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | NDArray[Any] | None = None,
    ) -> LightGBMFraudClassifier:
        """Fit on ``(X_train, y_train)``, early-stop against ``(X_val, y_val)``."""
        params = dict(self.params)
        if "scale_pos_weight" not in params:
            pos = int(np.sum(np.asarray(y_train) == 1))
            neg = int(np.sum(np.asarray(y_train) == 0))
            params["scale_pos_weight"] = neg / max(pos, 1)

        X_train_t = self.transformer.fit_transform(X_train)
        self.feature_names_ = list(X_train_t.columns)
        train_set = lgb.Dataset(X_train_t, label=np.asarray(y_train))

        valid_sets: list[lgb.Dataset] = [train_set]
        valid_names: list[str] = ["train"]
        callbacks: list[Any] = [lgb.log_evaluation(0)]

        if X_val is not None and y_val is not None:
            X_val_t = self.transformer.transform(X_val)
            val_set = lgb.Dataset(X_val_t, label=np.asarray(y_val), reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster.best_iteration or None
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        """Return shape (n, 2) — column 0 is P(normal), column 1 is P(fraud)."""
        if self.booster is None:
            raise RuntimeError("LightGBMFraudClassifier is not fitted yet")
        X_t = self.transformer.transform(X)
        p1 = np.asarray(self.booster.predict(X_t, num_iteration=self.best_iteration_)).astype(
            "float64"
        )
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> NDArray[Any]:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
