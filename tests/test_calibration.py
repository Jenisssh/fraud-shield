"""Tests for fraud_shield.models.calibration.

We don't fit a real classifier here — the wrapper just needs anything
with ``predict_proba``. A ``MockOverconfidentModel`` lets us write a
deterministic mis-calibration scenario where the score-to-probability
mapping is known wrong, so we can assert the calibrator actually fixes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fraud_shield.evaluation.metrics import brier
from fraud_shield.models.calibration import CalibratedFraudClassifier


class MockOverconfidentModel:
    """Returns pre-set scores from a lookup mirroring the row index.

    Lets a test construct a deterministic miscalibration: scores stored
    here vs labels supplied at calibrate() time.
    """

    def __init__(self, scores: NDArray[Any]) -> None:
        self.scores = np.asarray(scores).astype("float64")

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        # Use the row index of X as a positional lookup into self.scores
        idx = np.asarray(X.index).astype(int)
        p1 = self.scores[idx]
        return np.column_stack([1.0 - p1, p1])


@pytest.fixture
def miscalibrated_setup() -> tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray]:
    """Build a model whose scores systematically overshoot reality.

    Strategy:
      - raw_score = 0.95 → empirical positive rate 0.5
      - raw_score = 0.05 → empirical positive rate 0.05
    The base model is *informative* (ranks correctly) but *miscalibrated*
    (over-predicts probabilities in the high range).
    """
    rng = np.random.default_rng(0)
    n = 1_000
    # Half the rows are "high score" group, half "low score" group
    raw = np.where(rng.random(n) < 0.5, 0.95, 0.05)
    # True labels: high-score group has 0.5 positive rate, low-score 0.05
    labels = np.where(
        raw > 0.5,
        rng.binomial(1, 0.5, size=n),
        rng.binomial(1, 0.05, size=n),
    )
    model = MockOverconfidentModel(raw)
    X = pd.DataFrame({"dummy": np.zeros(n)})
    return model, X, labels


class TestConstruction:
    def test_default_method_is_isotonic(self) -> None:
        cal = CalibratedFraudClassifier(MockOverconfidentModel(np.zeros(5)))
        assert cal.method == "isotonic"

    def test_unknown_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            CalibratedFraudClassifier(
                MockOverconfidentModel(np.zeros(5)),
                method="bogus",  # type: ignore[arg-type]
            )

    def test_calibrator_is_none_before_calibrate(self) -> None:
        cal = CalibratedFraudClassifier(MockOverconfidentModel(np.zeros(5)))
        assert cal.calibrator is None

    def test_predict_proba_before_calibrate_raises(self) -> None:
        cal = CalibratedFraudClassifier(MockOverconfidentModel(np.zeros(5)))
        with pytest.raises(RuntimeError, match="not calibrated"):
            cal.predict_proba(pd.DataFrame({"x": [0.0]}))


class TestIsotonicCalibration:
    def test_calibrate_sets_isotonic_regressor(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model, method="isotonic").calibrate(X, y)
        assert isinstance(cal.calibrator, IsotonicRegression)

    def test_predict_proba_returns_valid_probabilities(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model, method="isotonic").calibrate(X, y)
        proba = cal.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_calibration_reduces_brier_on_miscalibrated_model(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        raw_scores = model.predict_proba(X)[:, 1]
        raw_brier = brier(y, raw_scores)

        cal = CalibratedFraudClassifier(model, method="isotonic").calibrate(X, y)
        calibrated_scores = cal.predict_proba(X)[:, 1]
        calibrated_brier = brier(y, calibrated_scores)

        # Isotonic minimizes Brier on its training set, so this must hold
        assert calibrated_brier < raw_brier

    def test_isotonic_is_monotone_in_raw_score(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model, method="isotonic").calibrate(X, y)
        # Probe: low raw → low calibrated; high raw → higher calibrated
        low = pd.DataFrame({"dummy": [0.0]}, index=[0])  # raw=0.05
        high_idx = int(np.argmax(model.scores > 0.5))
        high = pd.DataFrame({"dummy": [0.0]}, index=[high_idx])  # raw=0.95
        p_low = cal.predict_proba(low)[0, 1]
        p_high = cal.predict_proba(high)[0, 1]
        assert p_high > p_low


class TestSigmoidCalibration:
    def test_calibrate_sets_logistic_regression(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model, method="sigmoid").calibrate(X, y)
        assert isinstance(cal.calibrator, LogisticRegression)

    def test_predict_proba_returns_valid_probabilities(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model, method="sigmoid").calibrate(X, y)
        proba = cal.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_calibration_reduces_brier_on_miscalibrated_model(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        raw_brier = brier(y, model.predict_proba(X)[:, 1])
        cal = CalibratedFraudClassifier(model, method="sigmoid").calibrate(X, y)
        cal_brier = brier(y, cal.predict_proba(X)[:, 1])
        assert cal_brier < raw_brier


class TestPredict:
    def test_predict_thresholds_at_half_by_default(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model).calibrate(X, y)
        scores = cal.predict_proba(X)[:, 1]
        preds = cal.predict(X)
        np.testing.assert_array_equal(preds, (scores >= 0.5).astype(int))

    def test_predict_accepts_custom_threshold(
        self,
        miscalibrated_setup: tuple[MockOverconfidentModel, pd.DataFrame, np.ndarray],
    ) -> None:
        model, X, y = miscalibrated_setup
        cal = CalibratedFraudClassifier(model).calibrate(X, y)
        # Low threshold → predict positive for more rows
        preds_low = cal.predict(X, threshold=0.1)
        preds_high = cal.predict(X, threshold=0.9)
        assert preds_low.sum() >= preds_high.sum()
