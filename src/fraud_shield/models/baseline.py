"""Logistic regression baseline.

The role of this model is to *fail well*. Any more sophisticated approach
(LightGBM, calibrated boosting, stacking) has to beat this on the same
evaluation harness. If the LightGBM PR-AUC is only marginally better, the
extra complexity isn't earning its keep.

Pipeline steps:

1. :class:`fraud_shield.features.transformers.TimeAmountFeatures` with
   ``drop_original=True`` — replaces ``Time`` and ``Amount`` with four
   engineered columns.
2. :class:`sklearn.preprocessing.StandardScaler` — zero-mean / unit-variance
   per column. Linear models need it; the V features are roughly
   pre-scaled by ULB's PCA but ``log_amount`` is not.
3. :class:`sklearn.linear_model.LogisticRegression` with
   ``class_weight='balanced'`` — the simplest valid imbalance handler.
   Notebook 03 (Day 8) compares this to SMOTE and undersampling.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_shield.config import settings
from fraud_shield.features.transformers import TimeAmountFeatures

ClassWeight = Literal["balanced"] | dict[int, float] | None


def build_baseline_pipeline(
    *,
    class_weight: ClassWeight = "balanced",
    max_iter: int = 1_000,
    C: float = 1.0,
    random_state: int | None = None,
) -> Pipeline:
    """Construct the unfit baseline pipeline.

    Parameters
    ----------
    class_weight:
        Passed straight through to ``LogisticRegression``. Default
        ``'balanced'`` multiplies the per-class loss by
        ``n_samples / (n_classes * class_count)`` — critical at 0.17% imbalance.
        Pass ``None`` to disable.
    max_iter:
        L-BFGS iteration cap. 1000 should be enough for ULB; bump if
        sklearn raises ``ConvergenceWarning``.
    C:
        Inverse L2 regularization strength. Default 1.0.
    random_state:
        Defaults to ``settings.random_seed``.
    """
    seed = settings.random_seed if random_state is None else random_state
    return Pipeline(
        [
            ("features", TimeAmountFeatures(drop_original=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=max_iter,
                    C=C,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **pipeline_kwargs: Any,
) -> Pipeline:
    """Fit the baseline pipeline on ``(X_train, y_train)`` and return it."""
    pipe = build_baseline_pipeline(**pipeline_kwargs)
    pipe.fit(X_train, y_train)
    return pipe
