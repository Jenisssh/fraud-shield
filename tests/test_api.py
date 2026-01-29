"""Integration tests for the FastAPI inference service.

Uses ``TestClient`` and ``app.dependency_overrides`` to swap in stub
artifacts so the tests don't need a trained model on disk. The stubs
implement just enough of the model / explainer protocol to drive the
endpoints — see :class:`StubModel` and :class:`StubExplainer`.

A separate ``test_503_when_artifacts_missing`` test deliberately *does
not* override the dependency, so the real ``get_artifacts`` path runs
and returns 503 because ``app.state.artifacts`` is unset.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from fraud_shield.api.dependencies import ModelArtifacts, get_artifacts
from fraud_shield.api.main import app
from fraud_shield.evaluation.interpretability import (
    ExplanationResult,
    FeatureContribution,
)

V_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]
ENGINEERED_NAMES = ["hour", "day", "log_amount", "amount_is_zero"]
ALL_FEATURE_NAMES = V_FEATURE_NAMES + ENGINEERED_NAMES


class StubModel:
    """Returns a fixed positive-class probability for every input row."""

    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.score), np.full(n, self.score)])


class StubExplainer:
    """Returns a canned :class:`ExplanationResult` regardless of input."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names

    def explain_one(self, x: pd.DataFrame | pd.Series) -> ExplanationResult:
        contributions = [
            FeatureContribution(feature=name, value=float(i), shap_value=0.1 * (i + 1))
            for i, name in enumerate(self.feature_names)
        ]
        return ExplanationResult(
            expected_value=-3.0,
            raw_score=-3.0 + sum(c.shap_value for c in contributions),
            contributions=contributions,
        )


def make_payload(**overrides: float) -> dict[str, float]:
    """Build a valid Transaction payload with neutral defaults."""
    payload: dict[str, float] = {name: 0.0 for name in V_FEATURE_NAMES}
    payload["Time"] = 0.0
    payload["Amount"] = 10.0
    payload.update(overrides)
    return payload


@pytest.fixture
def fraud_artifacts() -> ModelArtifacts:
    return ModelArtifacts(
        model=StubModel(score=0.92),
        explainer=StubExplainer(ALL_FEATURE_NAMES),
        threshold=0.5,
        version="v0.1.0-test",
    )


@pytest.fixture
def safe_artifacts() -> ModelArtifacts:
    """Score below threshold — the OK branch."""
    return ModelArtifacts(
        model=StubModel(score=0.1),
        explainer=StubExplainer(ALL_FEATURE_NAMES),
        threshold=0.5,
        version="v0.1.0-test",
    )


@pytest.fixture
def client_with(fraud_artifacts: ModelArtifacts) -> Iterator[TestClient]:
    app.dependency_overrides[get_artifacts] = lambda: fraud_artifacts
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def safe_client(safe_artifacts: ModelArtifacts) -> Iterator[TestClient]:
    app.dependency_overrides[get_artifacts] = lambda: safe_artifacts
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


class TestHealth:
    def test_returns_200_with_status_ok(self, client_with: TestClient) -> None:
        r = client_with.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_version"] == "v0.1.0-test"
        assert body["threshold"] == 0.5

    def test_response_keys_are_exactly_expected(self, client_with: TestClient) -> None:
        body = client_with.get("/health").json()
        assert set(body.keys()) == {"status", "model_version", "threshold"}


class TestPredict:
    def test_returns_200_with_calibrated_score(self, client_with: TestClient) -> None:
        r = client_with.post("/predict", json=make_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["score"] == pytest.approx(0.92)
        assert body["threshold"] == 0.5
        assert body["model_version"] == "v0.1.0-test"

    def test_decision_is_fraud_when_score_above_threshold(self, client_with: TestClient) -> None:
        body = client_with.post("/predict", json=make_payload()).json()
        assert body["decision"] == "FRAUD"

    def test_decision_is_ok_when_score_below_threshold(self, safe_client: TestClient) -> None:
        body = safe_client.post("/predict", json=make_payload()).json()
        assert body["decision"] == "OK"
        assert body["score"] == pytest.approx(0.1)

    def test_422_on_missing_required_field(self, client_with: TestClient) -> None:
        payload = make_payload()
        del payload["V1"]
        r = client_with.post("/predict", json=payload)
        assert r.status_code == 422

    def test_422_on_extra_field(self, client_with: TestClient) -> None:
        payload = make_payload()
        payload["unauthorized_feature"] = 99.0
        r = client_with.post("/predict", json=payload)
        assert r.status_code == 422

    def test_422_on_negative_amount(self, client_with: TestClient) -> None:
        r = client_with.post("/predict", json=make_payload(Amount=-1.0))
        assert r.status_code == 422

    def test_422_on_negative_time(self, client_with: TestClient) -> None:
        r = client_with.post("/predict", json=make_payload(Time=-1.0))
        assert r.status_code == 422

    def test_422_on_wrong_type(self, client_with: TestClient) -> None:
        payload: dict[str, Any] = dict(make_payload())
        payload["V1"] = "not-a-number"
        r = client_with.post("/predict", json=payload)
        assert r.status_code == 422

    def test_response_keys_are_exactly_expected(self, client_with: TestClient) -> None:
        body = client_with.post("/predict", json=make_payload()).json()
        assert set(body.keys()) == {"score", "decision", "threshold", "model_version"}


class TestExplain:
    def test_returns_top_contributions_sorted_by_abs_shap(self, client_with: TestClient) -> None:
        r = client_with.post("/explain", json=make_payload())
        assert r.status_code == 200
        body = r.json()
        assert len(body["top_contributions"]) == 10
        abs_vals = [abs(c["shap_value"]) for c in body["top_contributions"]]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_each_contribution_has_three_fields(self, client_with: TestClient) -> None:
        body = client_with.post("/explain", json=make_payload()).json()
        for c in body["top_contributions"]:
            assert set(c.keys()) == {"feature", "value", "shap_value"}

    def test_decision_matches_predict_endpoint(self, client_with: TestClient) -> None:
        predict_body = client_with.post("/predict", json=make_payload()).json()
        explain_body = client_with.post("/explain", json=make_payload()).json()
        assert predict_body["decision"] == explain_body["decision"]
        assert predict_body["score"] == pytest.approx(explain_body["score"])

    def test_includes_expected_value_and_raw_score(self, client_with: TestClient) -> None:
        body = client_with.post("/explain", json=make_payload()).json()
        assert "expected_value" in body
        assert "raw_score" in body
        # additivity: raw_score = expected_value + sum(shap)
        contributions_in_explainer = 32  # 28 V + 4 engineered
        full_sum = 0.1 * sum(range(1, contributions_in_explainer + 1))
        assert body["raw_score"] == pytest.approx(-3.0 + full_sum)

    def test_422_on_bad_payload(self, client_with: TestClient) -> None:
        r = client_with.post("/explain", json={"oops": True})
        assert r.status_code == 422


class TestDegradedMode:
    def test_503_when_artifacts_not_loaded(self) -> None:
        # No dependency override, no lifespan (TestClient without 'with')
        # → app.state.artifacts is unset → get_artifacts raises 503
        client = TestClient(app)
        for path, method in (
            ("/health", "get"),
            ("/predict", "post"),
            ("/explain", "post"),
        ):
            r = client.get(path) if method == "get" else client.post(path, json=make_payload())
            assert r.status_code == 503, f"{path} should return 503 when degraded"
            assert "not loaded" in r.json()["detail"].lower()


class TestOpenAPI:
    def test_openapi_schema_exposed(self, client_with: TestClient) -> None:
        r = client_with.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        # Three documented endpoints
        assert "/health" in schema["paths"]
        assert "/predict" in schema["paths"]
        assert "/explain" in schema["paths"]
