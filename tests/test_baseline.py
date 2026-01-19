"""Tests for fraud_shield.models.baseline.

For end-to-end behaviour we use the ``predictable_fraud_df`` fixture
(defined in conftest) where V14 sign drives the label — that way the
baseline should score PR-AUC well above chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_shield.evaluation.metrics import pr_auc
from fraud_shield.features.transformers import TimeAmountFeatures
from fraud_shield.models.baseline import build_baseline_pipeline, train_baseline


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["Class"]), df["Class"]


class TestBuildBaselinePipeline:
    def test_returns_sklearn_pipeline(self) -> None:
        pipe = build_baseline_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_expected_steps_in_order(self) -> None:
        pipe = build_baseline_pipeline()
        names = [name for name, _ in pipe.steps]
        assert names == ["features", "scaler", "clf"]

    def test_features_step_drops_originals(self) -> None:
        pipe = build_baseline_pipeline()
        features_step = pipe.named_steps["features"]
        assert isinstance(features_step, TimeAmountFeatures)
        assert features_step.drop_original is True

    def test_clf_step_is_logistic_regression(self) -> None:
        pipe = build_baseline_pipeline()
        assert isinstance(pipe.named_steps["clf"], LogisticRegression)

    def test_default_class_weight_is_balanced(self) -> None:
        pipe = build_baseline_pipeline()
        assert pipe.named_steps["clf"].class_weight == "balanced"

    def test_class_weight_can_be_overridden(self) -> None:
        pipe = build_baseline_pipeline(class_weight=None)
        assert pipe.named_steps["clf"].class_weight is None

    def test_C_can_be_overridden(self) -> None:
        pipe = build_baseline_pipeline(C=0.01)
        assert pytest.approx(0.01) == pipe.named_steps["clf"].C


class TestTrainBaseline:
    def test_returns_fitted_pipeline(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        pipe = train_baseline(X, y)
        # check_is_fitted style: predict shouldn't raise NotFittedError
        proba = pipe.predict_proba(X.head(5))
        assert proba.shape == (5, 2)

    def test_predict_proba_in_unit_interval(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        pipe = train_baseline(X, y)
        proba = pipe.predict_proba(X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_beats_random_on_predictable_signal(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        pipe = train_baseline(X, y, max_iter=2_000)
        scores = pipe.predict_proba(X)[:, 1]
        # V14 sign drives Y, baseline should ace this
        assert pr_auc(y, scores) > 0.95

    def test_class_weight_balanced_lifts_recall_for_minority(
        self,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        # Skew the data more so the difference between weighted/unweighted is visible
        X, y = _split_xy(predictable_fraud_df)
        pipe_weighted = train_baseline(X, y, class_weight="balanced", max_iter=2_000)
        pipe_unweighted = train_baseline(X, y, class_weight=None, max_iter=2_000)

        pred_w = pipe_weighted.predict(X)
        pred_u = pipe_unweighted.predict(X)
        recall_w = ((pred_w == 1) & (y == 1)).sum() / (y == 1).sum()
        recall_u = ((pred_u == 1) & (y == 1)).sum() / (y == 1).sum()
        # Balanced should not be worse at catching positives
        assert recall_w >= recall_u

    def test_seed_makes_run_reproducible(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        a = train_baseline(X, y, random_state=7).predict_proba(X.head(20))[:, 1]
        b = train_baseline(X, y, random_state=7).predict_proba(X.head(20))[:, 1]
        np.testing.assert_array_equal(a, b)
