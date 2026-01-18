"""Tests for fraud_shield.evaluation.metrics.

For most assertions we construct deterministic score/label pairs that have
known-good metric values, so the tests don't depend on a particular sklearn
implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_shield.evaluation.metrics import (
    EvaluationReport,
    brier,
    evaluate,
    pr_auc,
    precision_at_recall,
    recall_at_precision,
    roc_auc,
)


@pytest.fixture
def perfect_scores() -> tuple[np.ndarray, np.ndarray]:
    """Scores that perfectly separate positives from negatives."""
    rng = np.random.default_rng(0)
    n_pos, n_neg = 50, 950
    y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    y_score = np.concatenate(
        [
            rng.uniform(0.7, 1.0, size=n_pos),
            rng.uniform(0.0, 0.3, size=n_neg),
        ]
    )
    return y_true, y_score


@pytest.fixture
def random_scores() -> tuple[np.ndarray, np.ndarray]:
    """Random scores — should give AUCs near 0.5."""
    rng = np.random.default_rng(1)
    n = 1000
    y_true = rng.binomial(1, 0.02, size=n)
    y_score = rng.uniform(size=n)
    return y_true, y_score


class TestPRAUC:
    def test_perfect_separation_gives_one(
        self, perfect_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = perfect_scores
        assert pr_auc(y_true, y_score) == pytest.approx(1.0)

    def test_random_scores_give_near_positive_rate(
        self, random_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        # PR-AUC of random scores ≈ positive rate
        y_true, y_score = random_scores
        positive_rate = y_true.mean()
        assert pr_auc(y_true, y_score) == pytest.approx(positive_rate, abs=0.05)

    def test_inverted_scores_give_low_score(
        self, perfect_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = perfect_scores
        assert pr_auc(y_true, 1.0 - y_score) < 0.1


class TestROCAUC:
    def test_perfect_separation_gives_one(
        self, perfect_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = perfect_scores
        assert roc_auc(y_true, y_score) == pytest.approx(1.0)

    def test_inverted_scores_give_zero(self, perfect_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_score = perfect_scores
        assert roc_auc(y_true, 1.0 - y_score) == pytest.approx(0.0)


class TestRecallAtPrecision:
    def test_perfect_separation_recalls_all_at_p9(
        self, perfect_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = perfect_scores
        assert recall_at_precision(y_true, y_score, 0.9) == pytest.approx(1.0)

    def test_returns_zero_when_target_unachievable(self) -> None:
        # Scores are anti-correlated with labels — precision = 0.9 unreachable
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert recall_at_precision(y_true, y_score, 0.9) == 0.0


class TestPrecisionAtRecall:
    def test_perfect_separation_keeps_precision_one_at_r9(
        self, perfect_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = perfect_scores
        assert precision_at_recall(y_true, y_score, 0.9) == pytest.approx(1.0)

    def test_returns_zero_when_target_recall_unachievable(self) -> None:
        # Only 1 positive that the model gets right; recall maxes at 1.0
        # but request impossible 1.1 to trigger unachievable
        y_true = np.array([1, 0, 0])
        y_score = np.array([0.9, 0.1, 0.2])
        assert precision_at_recall(y_true, y_score, 1.1) == 0.0


class TestBrier:
    def test_perfect_probabilities_give_zero(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier(y_true, y_prob) == pytest.approx(0.0)

    def test_all_half_probability_gives_quarter(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert brier(y_true, y_prob) == pytest.approx(0.25)


class TestEvaluate:
    def test_returns_evaluation_report(self, perfect_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_score = perfect_scores
        report = evaluate(y_true, y_score)
        assert isinstance(report, EvaluationReport)

    def test_report_fields_match_individual_computations(
        self, perfect_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = perfect_scores
        report = evaluate(y_true, y_score, target_precision=0.8, target_recall=0.8)
        assert report.pr_auc == pr_auc(y_true, y_score)
        assert report.roc_auc == roc_auc(y_true, y_score)
        assert report.recall_at_precision == recall_at_precision(y_true, y_score, 0.8)
        assert report.precision_at_recall == precision_at_recall(y_true, y_score, 0.8)
        assert report.brier == brier(y_true, y_score)

    def test_report_positive_rate_correct(
        self, random_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_score = random_scores
        report = evaluate(y_true, y_score)
        assert report.positive_rate == pytest.approx(y_true.mean())

    def test_as_dict_returns_flat_dict(self, perfect_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_score = perfect_scores
        d = evaluate(y_true, y_score).as_dict()
        assert set(d.keys()) == {
            "pr_auc",
            "roc_auc",
            "recall_at_precision",
            "precision_at_recall",
            "brier",
            "positive_rate",
        }
        for v in d.values():
            assert isinstance(v, float)

    def test_report_is_immutable(self, perfect_scores: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_score = perfect_scores
        report = evaluate(y_true, y_score)
        with pytest.raises((AttributeError, Exception)):  # frozen dataclass
            report.pr_auc = 0.5  # type: ignore[misc]
