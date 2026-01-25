"""Tests for fraud_shield.evaluation.interpretability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_shield.evaluation.interpretability import (
    ExplanationResult,
    FeatureContribution,
    FraudExplainer,
)
from fraud_shield.models.lightgbm_model import LightGBMFraudClassifier


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["Class"]), df["Class"]


@pytest.fixture
def fitted_model(predictable_fraud_df: pd.DataFrame) -> LightGBMFraudClassifier:
    X, y = _split_xy(predictable_fraud_df)
    return LightGBMFraudClassifier(num_boost_round=40).fit(X, y)


class TestConstruction:
    def test_rejects_unfitted_model(self) -> None:
        model = LightGBMFraudClassifier()
        with pytest.raises(RuntimeError, match="must be fitted"):
            FraudExplainer(model)

    def test_construction_succeeds_for_fitted_model(
        self, fitted_model: LightGBMFraudClassifier
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        assert explainer.feature_names == fitted_model.feature_names_

    def test_explainer_has_expected_value(self, fitted_model: LightGBMFraudClassifier) -> None:
        explainer = FraudExplainer(fitted_model)
        # expected_value should be a finite scalar
        assert np.isfinite(np.asarray(explainer.explainer.expected_value).flatten()[0])


class TestExplain:
    def test_shape_matches_input_and_features(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        sv = explainer.explain(X.head(20))
        assert sv.shape == (20, len(explainer.feature_names))

    def test_single_row_returns_expected_shape(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        sv = explainer.explain(X.head(1))
        assert sv.shape == (1, len(explainer.feature_names))


class TestExplainOne:
    def test_returns_explanation_result(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.head(1))
        assert isinstance(result, ExplanationResult)

    def test_rejects_multi_row_input(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        with pytest.raises(ValueError, match="single row"):
            explainer.explain_one(X.head(3))

    def test_accepts_series(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.iloc[0])
        assert isinstance(result, ExplanationResult)

    def test_additivity_margin_equals_expected_plus_shap_sum(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        # The defining property of SHAP: margin = expected_value + sum(shap)
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.head(1))
        sum_shap = sum(c.shap_value for c in result.contributions)
        assert pytest.approx(result.raw_score) == result.expected_value + sum_shap

    def test_contributions_cover_all_features(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.head(1))
        features = {c.feature for c in result.contributions}
        assert features == set(explainer.feature_names)


class TestTopK:
    def test_returns_at_most_k(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.head(1))
        top = result.top_k(5)
        assert len(top) <= 5

    def test_returns_sorted_by_abs_shap(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.head(1))
        top = result.top_k(8)
        abs_vals = [abs(c.shap_value) for c in top]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_top_k_larger_than_available_returns_all(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        result = explainer.explain_one(X.head(1))
        # request more than features count
        top = result.top_k(10_000)
        assert len(top) == len(explainer.feature_names)


class TestGlobalImportance:
    def test_returns_dataframe_with_one_row_per_feature(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        df = explainer.global_importance(X.head(50))
        assert set(df.columns) == {"feature", "mean_abs_shap"}
        assert len(df) == len(explainer.feature_names)

    def test_sorted_descending_by_importance(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        df = explainer.global_importance(X.head(50))
        assert df["mean_abs_shap"].is_monotonic_decreasing

    def test_all_importance_values_non_negative(
        self,
        fitted_model: LightGBMFraudClassifier,
        predictable_fraud_df: pd.DataFrame,
    ) -> None:
        explainer = FraudExplainer(fitted_model)
        X, _ = _split_xy(predictable_fraud_df)
        df = explainer.global_importance(X.head(50))
        assert (df["mean_abs_shap"] >= 0).all()


class TestImmutability:
    def test_feature_contribution_is_frozen(self) -> None:
        c = FeatureContribution(feature="V1", value=0.5, shap_value=0.1)
        with pytest.raises((AttributeError, Exception)):
            c.shap_value = 0.9  # type: ignore[misc]

    def test_explanation_result_is_frozen(self) -> None:
        r = ExplanationResult(expected_value=0.0, raw_score=0.0, contributions=[])
        with pytest.raises((AttributeError, Exception)):
            r.raw_score = 1.0  # type: ignore[misc]
