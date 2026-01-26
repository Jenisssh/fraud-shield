"""Tests for fraud_shield.monitoring.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_shield.monitoring.drift import (
    DriftResult,
    detect_drift,
    ks_drift_test,
    population_stability_index,
)


@pytest.fixture
def stable_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=2_000)
    b = rng.normal(size=2_000)
    return a, b


@pytest.fixture
def shifted_pair() -> tuple[np.ndarray, np.ndarray]:
    """Same shape, mean shifted by 1.5 standard deviations."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=2_000)
    b = rng.normal(loc=1.5, size=2_000)
    return a, b


class TestPSI:
    def test_identical_distributions_give_small_psi(
        self, stable_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = stable_pair
        assert population_stability_index(a, b) < 0.05

    def test_shifted_distributions_give_large_psi(
        self, shifted_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = shifted_pair
        assert population_stability_index(a, b) > 0.2

    def test_psi_is_non_negative(self, shifted_pair: tuple[np.ndarray, np.ndarray]) -> None:
        a, b = shifted_pair
        assert population_stability_index(a, b) >= 0.0

    def test_psi_handles_zero_inflated_feature(self) -> None:
        # Most values are 0; rare positives — common pattern for "is_zero" flags
        a = np.concatenate([np.zeros(900), np.ones(100)])
        b = np.concatenate([np.zeros(900), np.ones(100)])
        # PSI must be finite (no log(0) blow-up) and small for the same distribution
        assert np.isfinite(population_stability_index(a, b))
        assert population_stability_index(a, b) < 0.05

    def test_psi_handles_empty_distributions(self) -> None:
        assert population_stability_index(np.array([]), np.array([])) == 0.0
        assert population_stability_index(np.array([1.0, 2.0]), np.array([])) == 0.0


class TestKSDriftTest:
    def test_identical_distributions_give_high_pvalue(
        self, stable_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = stable_pair
        _, p = ks_drift_test(a, b)
        assert p > 0.05

    def test_shifted_distributions_give_low_pvalue(
        self, shifted_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = shifted_pair
        _, p = ks_drift_test(a, b)
        assert p < 0.001

    def test_statistic_is_in_unit_interval(
        self, shifted_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = shifted_pair
        stat, _ = ks_drift_test(a, b)
        assert 0.0 <= stat <= 1.0


class TestDetectDrift:
    @pytest.fixture
    def two_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        n = 1_000
        ref = pd.DataFrame(
            {
                "stable_a": rng.normal(size=n),
                "stable_b": rng.normal(size=n),
                "drifted_c": rng.normal(size=n),
            }
        )
        cur = pd.DataFrame(
            {
                "stable_a": rng.normal(size=n),
                "stable_b": rng.normal(size=n),
                "drifted_c": rng.normal(loc=2.0, size=n),  # mean-shifted
            }
        )
        return ref, cur

    def test_returns_dataframe_with_expected_columns(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        out = detect_drift(ref, cur)
        assert set(out.columns) == {
            "feature",
            "psi",
            "ks_statistic",
            "ks_pvalue",
            "drifted",
        }

    def test_one_row_per_shared_feature(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        out = detect_drift(ref, cur)
        assert set(out["feature"]) == set(ref.columns)

    def test_sorted_by_psi_descending(self, two_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, cur = two_frames
        out = detect_drift(ref, cur)
        assert out["psi"].is_monotonic_decreasing

    def test_flags_drifted_feature(self, two_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, cur = two_frames
        out = detect_drift(ref, cur)
        row = out[out["feature"] == "drifted_c"].iloc[0]
        assert row["drifted"]
        assert row["psi"] > 0.2

    def test_does_not_flag_stable_features(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        out = detect_drift(ref, cur)
        stable_rows = out[out["feature"].str.startswith("stable_")]
        assert not stable_rows["drifted"].any()

    def test_custom_feature_subset_respected(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        out = detect_drift(ref, cur, features=["drifted_c"])
        assert list(out["feature"]) == ["drifted_c"]

    def test_threshold_parameter_lowers_alert_bar(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        strict = detect_drift(ref, cur, psi_threshold=0.05)
        lenient = detect_drift(ref, cur, psi_threshold=1.0)
        # strict threshold catches at least as many drifted features
        assert strict["drifted"].sum() >= lenient["drifted"].sum()

    def test_missing_feature_in_reference_raises(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        with pytest.raises(KeyError, match="not in reference"):
            detect_drift(ref, cur, features=["does_not_exist"])

    def test_feature_only_in_reference_raises(
        self, two_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_frames
        ref_extra = ref.assign(extra=1.0)
        with pytest.raises(KeyError, match="not in current"):
            detect_drift(ref_extra, cur, features=["extra"])


class TestDriftResultDataclass:
    def test_as_dict_returns_expected_keys(self) -> None:
        r = DriftResult(
            feature="V1",
            psi=0.05,
            ks_statistic=0.01,
            ks_pvalue=0.5,
            drifted=False,
        )
        assert set(r.as_dict().keys()) == {
            "feature",
            "psi",
            "ks_statistic",
            "ks_pvalue",
            "drifted",
        }

    def test_is_frozen(self) -> None:
        r = DriftResult(feature="V1", psi=0.0, ks_statistic=0.0, ks_pvalue=1.0, drifted=False)
        with pytest.raises((AttributeError, Exception)):
            r.psi = 99.0  # type: ignore[misc]
