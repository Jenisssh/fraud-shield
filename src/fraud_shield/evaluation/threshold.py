"""Cost-aware threshold optimization for fraud classifiers.

Fraud teams don't pick a decision threshold by F1 or Youden's J — they
pick it by **expected cost**:

    E[cost] = C_FN * FN + C_FP * FP

where ``C_FN`` is the average loss from a missed fraud (chargeback +
the cost of goods + churn) and ``C_FP`` is the cost of investigating a
false alarm (analyst time, customer friction, declined-card support
calls). Typical fraud teams operate with ``C_FN / C_FP`` somewhere
between 10 and 100 — i.e. catching a fraud is worth a lot of false
alarms.

This module:

- represents the cost configuration as an immutable :class:`CostMatrix`
- exposes :func:`expected_cost_at_threshold` for one-off scoring
- exposes :func:`sweep_thresholds` for plotting (the full DataFrame
  drives the cost-vs-threshold chart in the demo Streamlit dashboard)
- exposes :func:`optimize_threshold` for the headline production
  threshold

Reproducibility note: tie-breaking on the optimum prefers the *higher*
threshold — fewer false positives at equal cost is the safer
production choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """Per-error costs in arbitrary units (typically currency).

    Costs must be non-negative. Setting either to zero is allowed (the
    optimizer will then drive the corresponding error count to zero
    regardless of threshold), though probably not what you want.
    """

    fn: float
    fp: float

    def __post_init__(self) -> None:
        if self.fn < 0 or self.fp < 0:
            raise ValueError(f"costs must be non-negative; got fn={self.fn}, fp={self.fp}")


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """One row of the optimization output — the chosen operating point."""

    threshold: float
    expected_cost: float
    n_tp: int
    n_fp: int
    n_fn: int
    n_tn: int
    precision: float
    recall: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_cost_at_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
    cost: CostMatrix,
) -> tuple[float, int, int, int, int]:
    """Compute expected cost and confusion-matrix counts at one threshold.

    Returns ``(expected_cost, n_tp, n_fp, n_fn, n_tn)``.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    pred = (y_score_arr >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true_arr == 1)).sum())
    fp = int(((pred == 1) & (y_true_arr == 0)).sum())
    fn = int(((pred == 0) & (y_true_arr == 1)).sum())
    tn = int(((pred == 0) & (y_true_arr == 0)).sum())
    total_cost = cost.fn * fn + cost.fp * fp
    return total_cost, tp, fp, fn, tn


def sweep_thresholds(
    y_true: ArrayLike,
    y_score: ArrayLike,
    cost: CostMatrix,
    *,
    candidates: ArrayLike | None = None,
) -> pd.DataFrame:
    """Compute the cost surface across candidate thresholds.

    Parameters
    ----------
    y_true, y_score:
        Ground truth labels and the model's positive-class probability
        (or score).
    cost:
        Cost configuration. The same matrix is applied to every row in
        the output.
    candidates:
        Thresholds to evaluate. When ``None``, uses the minimal set of
        thresholds where the predicted label actually changes (the
        sklearn PR-curve thresholds), which is faster than scanning a
        uniform grid for the same coverage.

    Returns
    -------
    pd.DataFrame
        Columns: ``threshold``, ``expected_cost``, ``n_tp``, ``n_fp``,
        ``n_fn``, ``n_tn``, ``precision``, ``recall``.
    """
    y_score_arr = np.asarray(y_score).astype(float)
    if candidates is None:
        _, _, pr_thresholds = precision_recall_curve(y_true, y_score_arr)
        # PR-curve omits the trivial "predict all positive" point; add it
        # by including 0.0, and also include 1.0+eps to capture the
        # "predict nothing" extreme.
        candidates = np.unique(np.concatenate([[0.0], pr_thresholds, [1.0 + 1e-9]]))
    candidates_arr = np.asarray(candidates).astype(float)

    rows: list[dict[str, float]] = []
    for t in candidates_arr:
        c, tp, fp, fn, tn = expected_cost_at_threshold(y_true, y_score_arr, float(t), cost)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                "threshold": float(t),
                "expected_cost": float(c),
                "n_tp": tp,
                "n_fp": fp,
                "n_fn": fn,
                "n_tn": tn,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows)


def optimize_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    cost: CostMatrix,
    *,
    candidates: ArrayLike | None = None,
) -> ThresholdResult:
    """Return the threshold that minimizes expected cost.

    Ties are broken by preferring the *higher* threshold — at equal
    expected cost, fewer false positives is the safer production choice.
    """
    df = sweep_thresholds(y_true, y_score, cost, candidates=candidates)
    df_sorted = df.sort_values(["expected_cost", "threshold"], ascending=[True, False])
    best = df_sorted.iloc[0]
    return ThresholdResult(
        threshold=float(best["threshold"]),
        expected_cost=float(best["expected_cost"]),
        n_tp=int(best["n_tp"]),
        n_fp=int(best["n_fp"]),
        n_fn=int(best["n_fn"]),
        n_tn=int(best["n_tn"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
    )
