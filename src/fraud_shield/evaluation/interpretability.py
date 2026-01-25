"""SHAP-based explanations for the LightGBM fraud classifier.

Two interfaces:

- :meth:`FraudExplainer.explain_one` — per-transaction explanation
  suitable for the FastAPI ``/explain`` endpoint. Returns a sorted list
  of feature contributions so the API can return e.g. "top 10 reasons
  this transaction was flagged".

- :meth:`FraudExplainer.global_importance` — mean(|SHAP|) per feature
  across a population. Used in the README's "what does the model
  actually look at" section and in the Streamlit dashboard.

Important detail about scale: ``TreeExplainer`` returns SHAP values on
the **margin** (log-odds) scale, not the probability scale. The
additivity property holds:

    margin = expected_value + sum(shap_values)
    probability = sigmoid(margin)

We surface the margin (``raw_score`` on :class:`ExplanationResult`) so
downstream callers can choose whether to transform to probability
themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray

from fraud_shield.models.lightgbm_model import LightGBMFraudClassifier


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's contribution to a single prediction."""

    feature: str
    value: float
    shap_value: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    """Full explanation for one transaction."""

    expected_value: float
    raw_score: float
    contributions: list[FeatureContribution] = field(default_factory=list)

    def top_k(self, k: int = 10) -> list[FeatureContribution]:
        """Return the ``k`` contributions with the largest absolute SHAP value."""
        return sorted(self.contributions, key=lambda c: abs(c.shap_value), reverse=True)[:k]


def _coerce_binary_shap(raw: Any) -> NDArray[Any]:
    """Older SHAP returns a 2-element list for binary; newer returns one array."""
    if isinstance(raw, list):
        return np.asarray(raw[1])
    return np.asarray(raw)


def _coerce_expected_value(raw: Any) -> float:
    """Same shape juggling for expected_value."""
    if isinstance(raw, list | np.ndarray):
        arr = np.asarray(raw)
        if arr.ndim == 0:
            return float(arr)
        return float(arr[1] if arr.size > 1 else arr[0])
    return float(raw)


class FraudExplainer:
    """Wraps a fitted :class:`LightGBMFraudClassifier` with a SHAP TreeExplainer.

    Cheap to construct (TreeExplainer initialization is just a tree walk),
    but the explainer should be cached across requests in the API to avoid
    redoing it on every call.
    """

    def __init__(self, model: LightGBMFraudClassifier) -> None:
        if model.booster is None:
            raise RuntimeError("model must be fitted before building an explainer")
        if not model.feature_names_:
            raise RuntimeError("model has no feature_names_ — refit before explaining")
        self.model = model
        self.feature_names: list[str] = list(model.feature_names_)
        self.explainer = shap.TreeExplainer(
            model.booster,
            feature_perturbation="tree_path_dependent",
        )

    def explain(self, X: pd.DataFrame) -> NDArray[Any]:
        """Return SHAP values for each (row, feature) pair.

        Shape: ``(n_rows, n_features)`` in the order of ``self.feature_names``.
        Values are on the margin (log-odds) scale.
        """
        X_t = self.model.transformer.transform(X)[self.feature_names]
        return _coerce_binary_shap(self.explainer.shap_values(X_t))

    def explain_one(self, x: pd.DataFrame | pd.Series) -> ExplanationResult:
        """Explain a single transaction.

        Accepts a one-row DataFrame or a Series (which gets wrapped).
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T
        if len(x) != 1:
            raise ValueError(f"explain_one expects a single row, got {len(x)}")

        shap_values = self.explain(x)[0]
        expected = _coerce_expected_value(self.explainer.expected_value)
        raw_score = expected + float(shap_values.sum())

        x_t = self.model.transformer.transform(x)[self.feature_names]
        feature_values = x_t.iloc[0]

        contributions = [
            FeatureContribution(
                feature=name,
                value=float(feature_values[name]),
                shap_value=float(sv),
            )
            for name, sv in zip(self.feature_names, shap_values, strict=True)
        ]
        return ExplanationResult(
            expected_value=expected,
            raw_score=raw_score,
            contributions=contributions,
        )

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ``mean(|SHAP|)`` per feature across the rows of ``X``.

        Returns a 2-column DataFrame sorted descending, ready to plot.
        """
        shap_values = self.explain(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
