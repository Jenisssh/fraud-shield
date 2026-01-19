"""Tests for fraud_shield.models.lightgbm_model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_shield.evaluation.metrics import pr_auc
from fraud_shield.models.lightgbm_model import DEFAULT_PARAMS, LightGBMFraudClassifier


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["Class"]), df["Class"]


class TestLightGBMFraudClassifier:
    def test_default_params_present(self) -> None:
        model = LightGBMFraudClassifier()
        for key in ("objective", "learning_rate", "num_leaves", "deterministic"):
            assert key in model.params
        assert model.params["deterministic"] is True

    def test_user_params_override_defaults(self) -> None:
        model = LightGBMFraudClassifier(params={"num_leaves": 7, "learning_rate": 0.1})
        assert model.params["num_leaves"] == 7
        assert pytest.approx(0.1) == model.params["learning_rate"]
        # untouched defaults still present
        assert model.params["objective"] == DEFAULT_PARAMS["objective"]

    def test_booster_is_none_before_fit(self) -> None:
        model = LightGBMFraudClassifier()
        assert model.booster is None

    def test_predict_proba_before_fit_raises(self, tiny_fraud_df: pd.DataFrame) -> None:
        model = LightGBMFraudClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(tiny_fraud_df)

    def test_fit_sets_booster_and_feature_names(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        model = LightGBMFraudClassifier(num_boost_round=20).fit(X, y)
        assert model.booster is not None
        assert model.feature_names_ is not None
        # engineered columns should be present, originals dropped
        assert "hour" in model.feature_names_
        assert "log_amount" in model.feature_names_
        assert "Time" not in model.feature_names_
        assert "Amount" not in model.feature_names_

    def test_predict_proba_shape_and_range(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        model = LightGBMFraudClassifier(num_boost_round=20).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_thresholds_at_half_by_default(
        self, predictable_fraud_df: pd.DataFrame
    ) -> None:
        X, y = _split_xy(predictable_fraud_df)
        model = LightGBMFraudClassifier(num_boost_round=20).fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, (scores >= 0.5).astype(int))

    def test_beats_random_on_predictable_signal(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        model = LightGBMFraudClassifier(num_boost_round=50).fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        assert pr_auc(y, scores) > 0.95

    def test_early_stopping_with_validation_set(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        # First 80% train, last 20% val
        cut = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:cut], X.iloc[cut:]
        y_train, y_val = y.iloc[:cut], y.iloc[cut:]

        model = LightGBMFraudClassifier(
            num_boost_round=1_000,
            early_stopping_rounds=10,
        ).fit(X_train, y_train, X_val, y_val)
        # Should have stopped well before the cap
        assert model.best_iteration_ is not None
        assert model.best_iteration_ < 1_000

    def test_auto_scale_pos_weight_lifts_recall(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        # auto: scale_pos_weight not given → wrapper computes neg/pos
        auto = LightGBMFraudClassifier(num_boost_round=50)
        assert "scale_pos_weight" not in auto.params
        auto.fit(X, y)
        # explicit no-upweight baseline (1.0 = treat classes equally)
        flat = LightGBMFraudClassifier(params={"scale_pos_weight": 1.0}, num_boost_round=50).fit(
            X, y
        )

        recall_auto = ((auto.predict(X) == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        recall_flat = ((flat.predict(X) == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        # auto-weighting should catch at least as many positives at threshold 0.5
        assert recall_auto >= recall_flat

    def test_seed_reproducibility(self, predictable_fraud_df: pd.DataFrame) -> None:
        X, y = _split_xy(predictable_fraud_df)
        a = LightGBMFraudClassifier(num_boost_round=30, random_state=7).fit(X, y)
        b = LightGBMFraudClassifier(num_boost_round=30, random_state=7).fit(X, y)
        np.testing.assert_array_equal(
            a.predict_proba(X.head(50))[:, 1],
            b.predict_proba(X.head(50))[:, 1],
        )

    def test_different_seeds_produce_different_models(
        self, predictable_fraud_df: pd.DataFrame
    ) -> None:
        X, y = _split_xy(predictable_fraud_df)
        # Use feature_fraction < 1 and bagging so seeds actually matter
        params = {"feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1}
        a = LightGBMFraudClassifier(params=params, num_boost_round=50, random_state=1).fit(X, y)
        b = LightGBMFraudClassifier(params=params, num_boost_round=50, random_state=2).fit(X, y)
        # Probabilities should differ on most rows
        diffs = np.abs(a.predict_proba(X)[:, 1] - b.predict_proba(X)[:, 1])
        assert (diffs > 1e-6).sum() > len(X) // 10
