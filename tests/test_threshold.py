"""Tests for fraud_shield.evaluation.threshold."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_shield.evaluation.threshold import (
    CostMatrix,
    ThresholdResult,
    expected_cost_at_threshold,
    optimize_threshold,
    sweep_thresholds,
)


class TestCostMatrix:
    def test_accepts_non_negative_costs(self) -> None:
        cm = CostMatrix(fn=100.0, fp=5.0)
        assert pytest.approx(100.0) == cm.fn
        assert pytest.approx(5.0) == cm.fp

    def test_rejects_negative_fn(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostMatrix(fn=-1.0, fp=5.0)

    def test_rejects_negative_fp(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostMatrix(fn=10.0, fp=-1.0)

    def test_is_frozen(self) -> None:
        cm = CostMatrix(fn=10.0, fp=1.0)
        with pytest.raises((AttributeError, Exception)):
            cm.fn = 20.0  # type: ignore[misc]


class TestExpectedCostAtThreshold:
    def test_threshold_at_zero_predicts_all_positive(self) -> None:
        y = np.array([1, 1, 0, 0])
        s = np.array([0.1, 0.4, 0.6, 0.9])
        c, tp, fp, fn, tn = expected_cost_at_threshold(y, s, 0.0, CostMatrix(fn=10, fp=1))
        assert tp == 2  # both positives predicted positive
        assert fp == 2  # both negatives predicted positive
        assert fn == 0
        assert tn == 0
        assert pytest.approx(2.0) == c  # 0 * 10 + 2 * 1

    def test_threshold_above_max_predicts_all_negative(self) -> None:
        y = np.array([1, 1, 0, 0])
        s = np.array([0.1, 0.4, 0.6, 0.9])
        c, tp, fp, fn, tn = expected_cost_at_threshold(y, s, 1.0, CostMatrix(fn=10, fp=1))
        assert tp == 0
        assert fp == 0
        assert fn == 2  # both positives missed
        assert tn == 2
        assert pytest.approx(20.0) == c  # 2 * 10 + 0 * 1

    def test_mid_threshold_hand_computed(self) -> None:
        # y      = [1,    1,    0,    0]
        # scores = [0.1,  0.4,  0.6,  0.9]   threshold 0.5
        # pred   = [0,    0,    1,    1]
        # tp=0, fp=2 (rows 2,3 wrong), fn=2 (rows 0,1 wrong), tn=0
        y = np.array([1, 1, 0, 0])
        s = np.array([0.1, 0.4, 0.6, 0.9])
        c, tp, fp, fn, tn = expected_cost_at_threshold(y, s, 0.5, CostMatrix(fn=10, fp=1))
        assert tp == 0
        assert fp == 2
        assert fn == 2
        assert tn == 0
        assert pytest.approx(22.0) == c  # 2 * 10 + 2 * 1


class TestSweepThresholds:
    def test_returns_dataframe_with_expected_columns(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.1, size=200)
        s = rng.uniform(size=200)
        df = sweep_thresholds(y, s, CostMatrix(fn=10, fp=1))
        for col in (
            "threshold",
            "expected_cost",
            "n_tp",
            "n_fp",
            "n_fn",
            "n_tn",
            "precision",
            "recall",
        ):
            assert col in df.columns
        assert len(df) > 1

    def test_accepts_custom_candidates(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.1, size=100)
        s = rng.uniform(size=100)
        df = sweep_thresholds(y, s, CostMatrix(fn=10, fp=1), candidates=np.linspace(0, 1, 11))
        assert len(df) == 11

    def test_tp_plus_fn_equals_total_positives(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.2, size=200)
        s = rng.uniform(size=200)
        df = sweep_thresholds(y, s, CostMatrix(fn=10, fp=1))
        total_pos = int(y.sum())
        assert (df["n_tp"] + df["n_fn"] == total_pos).all()

    def test_tn_plus_fp_equals_total_negatives(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.2, size=200)
        s = rng.uniform(size=200)
        df = sweep_thresholds(y, s, CostMatrix(fn=10, fp=1))
        total_neg = int((y == 0).sum())
        assert (df["n_tn"] + df["n_fp"] == total_neg).all()


class TestOptimizeThreshold:
    def test_returns_threshold_result(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.1, size=200)
        s = rng.uniform(size=200)
        result = optimize_threshold(y, s, CostMatrix(fn=10, fp=1))
        assert isinstance(result, ThresholdResult)
        assert 0.0 <= result.threshold <= 1.0 + 1e-6

    def test_expected_cost_no_greater_than_any_sweep_row(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.1, size=200)
        s = rng.uniform(size=200)
        cost = CostMatrix(fn=20, fp=1)
        result = optimize_threshold(y, s, cost)
        df = sweep_thresholds(y, s, cost)
        assert result.expected_cost == df["expected_cost"].min()

    def test_high_fn_cost_lowers_threshold(self) -> None:
        # When missing a fraud is very expensive, the optimizer should pick
        # a lower threshold (catch more, accept more FPs).
        rng = np.random.default_rng(2)
        y = rng.binomial(1, 0.1, size=500)
        s = np.clip(y + rng.normal(scale=0.5, size=500), 0, 1)  # noisy positive correlation

        cheap_fn = optimize_threshold(y, s, CostMatrix(fn=2, fp=1))
        expensive_fn = optimize_threshold(y, s, CostMatrix(fn=200, fp=1))
        assert expensive_fn.threshold <= cheap_fn.threshold

    def test_high_fp_cost_raises_threshold(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.1, size=500)
        s = np.clip(y + rng.normal(scale=0.5, size=500), 0, 1)

        cheap_fp = optimize_threshold(y, s, CostMatrix(fn=10, fp=1))
        expensive_fp = optimize_threshold(y, s, CostMatrix(fn=10, fp=100))
        assert expensive_fp.threshold >= cheap_fp.threshold

    def test_tie_breaks_toward_higher_threshold(self) -> None:
        # All wrong + all costs equal: cost is the same at every threshold
        # that produces the same FP count. With our tie-break, the optimizer
        # should prefer the highest threshold (fewest FPs).
        y = np.array([0, 0, 0, 0])
        s = np.array([0.1, 0.3, 0.6, 0.9])
        # With all-negatives there are no FN to penalize, so cost = FP * cost.fp
        # Threshold above max → 0 FP, cost 0. That's the optimum.
        result = optimize_threshold(y, s, CostMatrix(fn=10, fp=1))
        assert result.threshold >= 0.9  # high threshold preferred

    def test_as_dict_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(4)
        y = rng.binomial(1, 0.1, size=100)
        s = rng.uniform(size=100)
        result = optimize_threshold(y, s, CostMatrix(fn=10, fp=1))
        d = result.as_dict()
        assert set(d.keys()) == {
            "threshold",
            "expected_cost",
            "n_tp",
            "n_fp",
            "n_fn",
            "n_tn",
            "precision",
            "recall",
        }
