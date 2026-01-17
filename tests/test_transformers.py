"""Tests for fraud_shield.features.transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_shield.features.transformers import (
    ENGINEERED_COLUMNS,
    REQUIRED_COLUMNS,
    TimeAmountFeatures,
)


def _make_minimal_df(time: list[float], amount: list[float]) -> pd.DataFrame:
    """Build a frame with the right columns but only a few rows for hand-computed tests."""
    n = len(time)
    data: dict[str, list[float]] = {f"V{i}": [0.0] * n for i in range(1, 29)}
    data["Time"] = time
    data["Amount"] = amount
    data["Class"] = [0] * n
    return pd.DataFrame(data)


class TestTimeAmountFeatures:
    def test_required_columns_are_time_and_amount(self) -> None:
        assert frozenset({"Time", "Amount"}) == REQUIRED_COLUMNS

    def test_engineered_columns_constant_lists_four(self) -> None:
        assert ENGINEERED_COLUMNS == ("hour", "day", "log_amount", "amount_is_zero")

    def test_adds_four_engineered_columns(self, tiny_fraud_df: pd.DataFrame) -> None:
        out = TimeAmountFeatures().fit_transform(tiny_fraud_df)
        for col in ENGINEERED_COLUMNS:
            assert col in out.columns

    def test_keeps_originals_by_default(self, tiny_fraud_df: pd.DataFrame) -> None:
        out = TimeAmountFeatures().fit_transform(tiny_fraud_df)
        assert "Time" in out.columns
        assert "Amount" in out.columns

    def test_drops_originals_when_requested(self, tiny_fraud_df: pd.DataFrame) -> None:
        out = TimeAmountFeatures(drop_original=True).fit_transform(tiny_fraud_df)
        assert "Time" not in out.columns
        assert "Amount" not in out.columns
        assert "hour" in out.columns

    def test_hour_within_24h_range(self, tiny_fraud_df: pd.DataFrame) -> None:
        out = TimeAmountFeatures().fit_transform(tiny_fraud_df)
        assert out["hour"].min() >= 0
        assert out["hour"].max() <= 23

    def test_hour_hand_computed(self) -> None:
        # 0s -> hour 0; 3600s -> 1; 7199s -> 1 (rounding down); 86400s (1 day) -> 0;
        # 87000s -> 0 (24h wrap)
        df = _make_minimal_df(
            time=[0.0, 3600.0, 7199.0, 86400.0, 87000.0],
            amount=[10.0, 20.0, 30.0, 40.0, 50.0],
        )
        out = TimeAmountFeatures().fit_transform(df)
        assert list(out["hour"]) == [0, 1, 1, 0, 0]

    def test_day_hand_computed(self) -> None:
        df = _make_minimal_df(
            time=[0.0, 86399.0, 86400.0, 172800.0],
            amount=[1.0, 1.0, 1.0, 1.0],
        )
        out = TimeAmountFeatures().fit_transform(df)
        assert list(out["day"]) == [0, 0, 1, 2]

    def test_log_amount_matches_log1p(self, tiny_fraud_df: pd.DataFrame) -> None:
        out = TimeAmountFeatures().fit_transform(tiny_fraud_df)
        expected = np.log1p(tiny_fraud_df["Amount"]).astype("float64")
        np.testing.assert_array_almost_equal(out["log_amount"].to_numpy(), expected.to_numpy())

    def test_amount_is_zero_flag(self) -> None:
        df = _make_minimal_df(time=[0.0, 0.0, 0.0], amount=[0.0, 1.0, 1e-9])
        out = TimeAmountFeatures().fit_transform(df)
        # 1e-9 is non-zero — only exact zero gets the flag
        assert list(out["amount_is_zero"]) == [1, 0, 0]

    def test_fit_returns_self(self, tiny_fraud_df: pd.DataFrame) -> None:
        t = TimeAmountFeatures()
        assert t.fit(tiny_fraud_df) is t

    def test_does_not_mutate_input(self, tiny_fraud_df: pd.DataFrame) -> None:
        snapshot = tiny_fraud_df.copy()
        TimeAmountFeatures().fit_transform(tiny_fraud_df)
        pd.testing.assert_frame_equal(tiny_fraud_df, snapshot)

    def test_fit_rejects_missing_time(self) -> None:
        df = pd.DataFrame({"Amount": [1.0, 2.0]})
        with pytest.raises(KeyError, match="Time"):
            TimeAmountFeatures().fit(df)

    def test_transform_rejects_missing_amount(self) -> None:
        t = TimeAmountFeatures()
        good = _make_minimal_df(time=[0.0], amount=[1.0])
        t.fit(good)
        bad = good.drop(columns=["Amount"])
        with pytest.raises(KeyError, match="Amount"):
            t.transform(bad)

    def test_get_feature_names_out_includes_originals_by_default(
        self, tiny_fraud_df: pd.DataFrame
    ) -> None:
        t = TimeAmountFeatures()
        names = t.get_feature_names_out(input_features=list(tiny_fraud_df.columns))
        assert "Time" in names
        assert "Amount" in names
        for col in ENGINEERED_COLUMNS:
            assert col in names

    def test_get_feature_names_out_excludes_originals_when_dropped(
        self, tiny_fraud_df: pd.DataFrame
    ) -> None:
        t = TimeAmountFeatures(drop_original=True)
        names = t.get_feature_names_out(input_features=list(tiny_fraud_df.columns))
        assert "Time" not in names
        assert "Amount" not in names
        for col in ENGINEERED_COLUMNS:
            assert col in names

    def test_works_inside_sklearn_pipeline(self, tiny_fraud_df: pd.DataFrame) -> None:
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("features", TimeAmountFeatures())])
        out = pipe.fit_transform(tiny_fraud_df)
        assert isinstance(out, pd.DataFrame)
        assert "hour" in out.columns
