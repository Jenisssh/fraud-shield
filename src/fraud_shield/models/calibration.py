"""Post-hoc probability calibration for any classifier with ``predict_proba``.

Why calibration matters for fraud:

A gradient-boosted model trained with ``scale_pos_weight`` produces
*scores* — values in [0, 1] that rank correctly but don't behave as
probabilities. If the deployment owner wants the rule "block when
``P(fraud) > 0.8``", that statement is only meaningful if 0.8 actually
means "80% likely fraud" in the empirical sense. Without calibration,
"0.8" usually means something closer to "rank in the top 5%."

Two methods:

- **Isotonic regression** — non-parametric monotone mapping from raw
  score to calibrated probability. More flexible; needs enough val data
  (rule of thumb: at least a few hundred positives) or it overfits.
- **Sigmoid (Platt scaling)** — a one-parameter logistic fit. Less data
  needed, but assumes the miscalibration looks sigmoidal. Wins on small
  validation sets.

We don't use ``sklearn.calibration.CalibratedClassifierCV`` directly
because ``LightGBMFraudClassifier`` isn't strictly sklearn-compatible
(no ``BaseEstimator`` subclass). Wrapping it manually with a Protocol
keeps the interface narrow and the calibration step explicit.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityModel(Protocol):
    """Anything that returns shape-(n, 2) probabilities for binary classification."""

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:  # pragma: no cover
        ...


CalibrationMethod = Literal["isotonic", "sigmoid"]


class CalibratedFraudClassifier:
    """Wrap a *fitted* probability model with a learned calibration map.

    Workflow:

    1. Train your base model on the training set.
    2. ``CalibratedFraudClassifier(base, method='isotonic')``
    3. ``.calibrate(X_val, y_val)`` — fits the calibration map on the
       validation set's predictions vs labels.
    4. ``.predict_proba(X_test)`` — base scores get mapped through the
       calibrator before being returned.

    The base model is *not* re-fit. The val set should be different from
    the train set the base model saw, or the calibration map will be
    over-optimistic.

    Attributes
    ----------
    calibrator:
        The fitted calibrator (``IsotonicRegression`` or
        ``LogisticRegression``). ``None`` until ``calibrate`` is called.
    """

    def __init__(
        self,
        base_model: ProbabilityModel,
        *,
        method: CalibrationMethod = "isotonic",
    ) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid'; got {method!r}")
        self.base_model = base_model
        self.method = method
        self.calibrator: IsotonicRegression | LogisticRegression | None = None

    def calibrate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series | NDArray[Any],
    ) -> CalibratedFraudClassifier:
        """Fit the calibration map on held-out predictions."""
        raw_scores = np.asarray(self.base_model.predict_proba(X_val)[:, 1]).astype("float64")
        y_val_arr = np.asarray(y_val).astype(int)

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_scores, y_val_arr)
            self.calibrator = iso
        else:
            sig = LogisticRegression(solver="lbfgs", max_iter=1_000)
            sig.fit(raw_scores.reshape(-1, 1), y_val_arr)
            self.calibrator = sig
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        """Return shape-(n, 2) calibrated probabilities."""
        if self.calibrator is None:
            raise RuntimeError("CalibratedFraudClassifier is not calibrated yet")
        raw = np.asarray(self.base_model.predict_proba(X)[:, 1]).astype("float64")

        if isinstance(self.calibrator, IsotonicRegression):
            calibrated = np.asarray(self.calibrator.transform(raw)).astype("float64")
        else:
            calibrated = np.asarray(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]).astype(
                "float64"
            )

        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> NDArray[Any]:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
