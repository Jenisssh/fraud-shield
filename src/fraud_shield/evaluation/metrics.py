"""Imbalance-aware metrics for fraud detection.

Why not accuracy or ROC-AUC alone:

- 0.17% positive class → predicting all-zero gives 99.83% accuracy. Useless.
- ROC-AUC is inflated under heavy imbalance — the FPR denominator
  (≈ 99.83% of rows) absorbs huge numbers of false positives without the
  metric moving much. Two models with very different operational behavior
  can have indistinguishable ROC-AUCs.

What we report instead:

- **PR-AUC** (average precision) — area under the precision-recall curve.
  Focuses on the positive class; the right headline number under imbalance.
- **recall @ precision = p** — the operational metric a fraud team cares
  about. "Given we want to keep false-alarm rate manageable (precision
  ≥ p), what fraction of frauds do we catch?"
- **precision @ recall = r** — the inverse view, useful when leadership
  fixes a target catch-rate.
- **Brier score** — mean squared error between predicted probability and
  ground truth. Drops as calibration improves; the headline calibration
  number once we wrap the model with ``CalibratedClassifierCV``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Compact bundle of the metrics we report for every model."""

    pr_auc: float
    roc_auc: float
    recall_at_precision: float
    precision_at_recall: float
    brier: float
    positive_rate: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Average precision — equivalent to area under the precision-recall curve."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Reported for reference only — see module docstring for caveats."""
    return float(roc_auc_score(y_true, y_score))


def recall_at_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    target_precision: float = 0.9,
) -> float:
    """Maximum recall achievable while keeping precision ≥ ``target_precision``.

    Returns 0.0 if the model can never hit the target precision (the
    threshold-tuning surface doesn't contain a valid operating point).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def precision_at_recall(
    y_true: ArrayLike,
    y_score: ArrayLike,
    target_recall: float = 0.9,
) -> float:
    """Maximum precision achievable while keeping recall ≥ ``target_recall``.

    Returns 0.0 if the target recall is unachievable.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= target_recall
    if not mask.any():
        return 0.0
    return float(precision[mask].max())


def brier(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error between predicted probability and {0,1} label."""
    return float(brier_score_loss(y_true, y_prob))


def evaluate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    target_precision: float = 0.9,
    target_recall: float = 0.9,
) -> EvaluationReport:
    """One-shot computation of all metrics, returned as an EvaluationReport."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    return EvaluationReport(
        pr_auc=pr_auc(y_true_arr, y_score_arr),
        roc_auc=roc_auc(y_true_arr, y_score_arr),
        recall_at_precision=recall_at_precision(y_true_arr, y_score_arr, target_precision),
        precision_at_recall=precision_at_recall(y_true_arr, y_score_arr, target_recall),
        brier=brier(y_true_arr, y_score_arr),
        positive_rate=float(np.mean(y_true_arr)),
    )
