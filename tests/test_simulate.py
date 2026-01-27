"""Tests for fraud_shield.monitoring.simulate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_shield.monitoring.drift import detect_drift
from fraud_shield.monitoring.simulate import inject_drift


@pytest.fixture
def base_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1_000
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )


class TestInjectDrift:
    def test_returns_same_shape(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"])
        assert out.shape == base_frame.shape
        assert list(out.columns) == list(base_frame.columns)

    def test_does_not_mutate_input(self, base_frame: pd.DataFrame) -> None:
        snapshot = base_frame.copy()
        inject_drift(base_frame, ["a"], magnitude=2.0)
        pd.testing.assert_frame_equal(base_frame, snapshot)

    def test_unaffected_features_pass_through(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], magnitude=2.0)
        pd.testing.assert_series_equal(out["b"], base_frame["b"])
        pd.testing.assert_series_equal(out["c"], base_frame["c"])

    def test_rejects_unknown_kind(self, base_frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown drift kind"):
            inject_drift(base_frame, ["a"], kind="bogus")  # type: ignore[arg-type]

    def test_rejects_missing_feature(self, base_frame: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="not in dataframe"):
            inject_drift(base_frame, ["does_not_exist"])


class TestMeanShift:
    def test_adds_expected_amount(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="mean_shift", magnitude=1.0)
        expected_shift = base_frame["a"].std()
        actual_shift = out["a"].mean() - base_frame["a"].mean()
        assert pytest.approx(expected_shift, rel=1e-6) == actual_shift

    def test_is_deterministic(self, base_frame: pd.DataFrame) -> None:
        a = inject_drift(base_frame, ["a"], kind="mean_shift", magnitude=0.5)
        b = inject_drift(base_frame, ["a"], kind="mean_shift", magnitude=0.5)
        pd.testing.assert_frame_equal(a, b)

    def test_detector_flags_the_shift(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="mean_shift", magnitude=2.0)
        report = detect_drift(base_frame, out)
        row = report[report["feature"] == "a"].iloc[0]
        assert row["drifted"]


class TestVarianceScale:
    def test_doubling_variance_doubles_std(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="variance_scale", magnitude=2.0)
        assert pytest.approx(2.0, rel=1e-6) == out["a"].std() / base_frame["a"].std()

    def test_preserves_mean(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="variance_scale", magnitude=3.0)
        assert pytest.approx(base_frame["a"].mean(), abs=1e-10) == out["a"].mean()

    def test_detector_flags_the_change(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="variance_scale", magnitude=3.0)
        report = detect_drift(base_frame, out)
        row = report[report["feature"] == "a"].iloc[0]
        assert row["drifted"]


class TestMixture:
    def test_changes_some_but_not_all_rows(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="mixture", magnitude=2.0, random_state=42)
        diff = (out["a"] != base_frame["a"]).sum()
        # Roughly half the rows; allow a wide tolerance for random variation
        assert 350 < diff < 650

    def test_seed_makes_mixture_reproducible(self, base_frame: pd.DataFrame) -> None:
        a = inject_drift(base_frame, ["a"], kind="mixture", magnitude=1.0, random_state=7)
        b = inject_drift(base_frame, ["a"], kind="mixture", magnitude=1.0, random_state=7)
        pd.testing.assert_frame_equal(a, b)

    def test_detector_flags_the_mixture(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="mixture", magnitude=3.0, random_state=7)
        report = detect_drift(base_frame, out)
        row = report[report["feature"] == "a"].iloc[0]
        assert row["drifted"]


class TestEndToEndWithDriftDetector:
    def test_one_feature_drifts_others_dont(self, base_frame: pd.DataFrame) -> None:
        out = inject_drift(base_frame, ["a"], kind="mean_shift", magnitude=2.0)
        report = detect_drift(base_frame, out)
        drifted = set(report[report["drifted"]]["feature"])
        assert drifted == {"a"}
