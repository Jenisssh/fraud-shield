"""Structural smoke tests for the FastAPI app.

Confirms the app imports, declares the routes it should, and that the
schemas can roundtrip an example transaction. Full integration tests
(actual HTTP calls with TestClient and a stub model) land on Day 14.
"""

from __future__ import annotations

from fraud_shield.api.main import app
from fraud_shield.api.schemas import (
    ExplainResponse,
    HealthResponse,
    PredictResponse,
    Transaction,
)


def test_app_declares_expected_routes() -> None:
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/health" in paths
    assert "/predict" in paths
    assert "/explain" in paths


def test_app_metadata() -> None:
    assert app.title == "FraudShield API"
    assert app.version  # populated from package __version__


def test_transaction_schema_accepts_valid_payload() -> None:
    payload: dict[str, float] = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Time"] = 0.0
    payload["Amount"] = 10.0
    t = Transaction.model_validate(payload)
    assert t.Time == 0.0
    assert t.Amount == 10.0


def test_transaction_schema_rejects_extra_field() -> None:
    payload: dict[str, float] = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Time"] = 0.0
    payload["Amount"] = 10.0
    payload["sneaky"] = 1.0  # extra='forbid' should reject
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Transaction.model_validate(payload)


def test_response_schemas_round_trip() -> None:
    # Confirm the response models accept the values the endpoints will produce
    h = HealthResponse(status="ok", model_version="v0.1.0", threshold=0.5)
    assert h.status == "ok"

    p = PredictResponse(score=0.8, decision="FRAUD", threshold=0.5, model_version="v0.1.0")
    assert p.decision == "FRAUD"

    e = ExplainResponse(
        score=0.8,
        decision="FRAUD",
        threshold=0.5,
        expected_value=-3.0,
        raw_score=1.4,
        top_contributions=[],
        model_version="v0.1.0",
    )
    assert e.expected_value == -3.0
