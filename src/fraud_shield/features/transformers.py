"""Feature engineering transformers for the ULB fraud dataset.

The V1..V28 columns are already centered around zero by the ULB team's PCA,
so the engineering effort goes into the two raw columns:

- ``Time`` (seconds since first transaction) → ``hour`` and ``day``
- ``Amount`` (USD-ish, heavy right-skew) → ``log_amount`` and a ``amount_is_zero`` flag

Transformers expose the standard scikit-learn ``fit`` / ``transform`` /
``fit_transform`` / ``get_feature_names_out`` surface so they slot directly
into ``Pipeline`` and ``ColumnTransformer``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

REQUIRED_COLUMNS: frozenset[str] = frozenset({"Time", "Amount"})
ENGINEERED_COLUMNS: tuple[str, ...] = ("hour", "day", "log_amount", "amount_is_zero")


class TimeAmountFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Append engineered features derived from ``Time`` and ``Amount``.

    Parameters
    ----------
    drop_original:
        When True, drop ``Time`` and ``Amount`` after engineering. Default
        ``False`` so downstream code can keep both the raw and the
        engineered representations.

    Notes
    -----
    The transformer is stateless — ``fit`` only runs column validation and
    is safe to call on the held-out test set. ``transform`` never mutates
    its input.
    """

    def __init__(self, *, drop_original: bool = False) -> None:
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Any = None) -> TimeAmountFeatures:
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        out = X.copy()
        out["hour"] = ((X["Time"] // 3600) % 24).astype("int64")
        out["day"] = (X["Time"] // 86400).astype("int64")
        out["log_amount"] = np.log1p(X["Amount"]).astype("float64")
        out["amount_is_zero"] = (X["Amount"] == 0).astype("int64")
        if self.drop_original:
            out = out.drop(columns=["Time", "Amount"])
        return out

    def get_feature_names_out(
        self, input_features: list[str] | NDArray[Any] | None = None
    ) -> NDArray[Any]:
        if input_features is None:
            base: list[str] = []
        else:
            base = list(input_features)
        if self.drop_original:
            base = [c for c in base if c not in {"Time", "Amount"}]
        return np.array([*base, *ENGINEERED_COLUMNS])

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(X.columns)
        if missing:
            raise KeyError(
                f"TimeAmountFeatures requires columns {sorted(REQUIRED_COLUMNS)}; "
                f"missing: {sorted(missing)}"
            )
