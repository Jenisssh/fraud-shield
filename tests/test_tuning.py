"""Tests for fraud_shield.models.tuning."""

from __future__ import annotations

import optuna
import pandas as pd
import pytest

from fraud_shield.models.tuning import suggest_lightgbm_params, tune_lightgbm


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["Class"]), df["Class"]


def _train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    cut = int(len(df) * 0.8)
    X, y = _split_xy(df)
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


class TestSuggestLightGBMParams:
    def test_returns_expected_keys(self) -> None:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_lightgbm_params(trial)
        expected_keys = {
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
        }
        assert set(params.keys()) == expected_keys

    def test_params_within_documented_ranges(self) -> None:
        study = optuna.create_study(direction="maximize")
        # Run a few trials so the sampler explores
        for _ in range(5):
            trial = study.ask()
            p = suggest_lightgbm_params(trial)
            assert 0.01 <= p["learning_rate"] <= 0.2
            assert 15 <= p["num_leaves"] <= 255
            assert 5 <= p["min_child_samples"] <= 100
            assert 0.5 <= p["feature_fraction"] <= 1.0
            assert 0.5 <= p["bagging_fraction"] <= 1.0
            assert 1 <= p["bagging_freq"] <= 10
            assert 1e-8 <= p["lambda_l1"] <= 10.0
            assert 1e-8 <= p["lambda_l2"] <= 10.0
            study.tell(trial, 0.0)


class TestTuneLightGBM:
    def test_returns_optuna_study(self, predictable_fraud_df: pd.DataFrame) -> None:
        X_tr, y_tr, X_va, y_va = _train_val(predictable_fraud_df)
        study = tune_lightgbm(
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_trials=3,
            num_boost_round=30,
            use_pruner=False,
        )
        assert isinstance(study, optuna.Study)

    def test_runs_requested_trial_count(self, predictable_fraud_df: pd.DataFrame) -> None:
        X_tr, y_tr, X_va, y_va = _train_val(predictable_fraud_df)
        study = tune_lightgbm(
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_trials=4,
            num_boost_round=30,
            use_pruner=False,
        )
        assert len(study.trials) == 4

    def test_best_value_is_valid_pr_auc(self, predictable_fraud_df: pd.DataFrame) -> None:
        X_tr, y_tr, X_va, y_va = _train_val(predictable_fraud_df)
        study = tune_lightgbm(
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_trials=3,
            num_boost_round=30,
            use_pruner=False,
        )
        assert 0.0 <= study.best_value <= 1.0
        # On predictable V14 signal, PR-AUC should be high even with 3 trials
        assert study.best_value > 0.7

    def test_best_params_compatible_with_classifier(
        self, predictable_fraud_df: pd.DataFrame
    ) -> None:
        from fraud_shield.models.lightgbm_model import LightGBMFraudClassifier

        X_tr, y_tr, X_va, y_va = _train_val(predictable_fraud_df)
        study = tune_lightgbm(
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_trials=3,
            num_boost_round=30,
            use_pruner=False,
        )
        # The best params should plug straight into the classifier
        model = LightGBMFraudClassifier(params=study.best_params, num_boost_round=20).fit(
            X_tr, y_tr
        )
        assert model.booster is not None

    def test_seed_makes_search_reproducible(self, predictable_fraud_df: pd.DataFrame) -> None:
        X_tr, y_tr, X_va, y_va = _train_val(predictable_fraud_df)
        kwargs = {
            "n_trials": 3,
            "num_boost_round": 30,
            "use_pruner": False,
            "random_state": 17,
        }
        a = tune_lightgbm(X_tr, y_tr, X_va, y_va, **kwargs)
        b = tune_lightgbm(X_tr, y_tr, X_va, y_va, **kwargs)
        assert a.best_params == b.best_params
        assert pytest.approx(a.best_value) == b.best_value
