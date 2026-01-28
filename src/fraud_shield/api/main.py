"""FastAPI inference service for FraudShield.

Three endpoints:

- ``GET  /health`` — liveness + model version + active threshold
- ``POST /predict`` — calibrated P(fraud) and a binary decision
- ``POST /explain`` — same prediction plus the top-10 SHAP contributions

The service loads the model bundle once on startup via the lifespan
hook and keeps it on ``app.state.artifacts``. Each request is logged
as a structured JSON line with method, path, status, and duration_ms.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response

from fraud_shield import __version__
from fraud_shield.api.dependencies import ModelArtifacts, get_artifacts, load_artifacts
from fraud_shield.api.schemas import (
    Decision,
    ExplainResponse,
    FeatureContributionDTO,
    HealthResponse,
    PredictResponse,
    Transaction,
)
from fraud_shield.utils.logging import get_logger

log = get_logger("fraud_shield.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once at process start; clear them at shutdown."""
    try:
        artifacts = load_artifacts()
        app.state.artifacts = artifacts
        log.info(
            "model_loaded",
            version=artifacts.version,
            threshold=artifacts.threshold,
        )
    except FileNotFoundError as e:
        app.state.artifacts = None
        log.warning("model_artifacts_missing", error=str(e))
    yield
    app.state.artifacts = None


app = FastAPI(
    title="FraudShield API",
    description=(
        "Calibrated credit-card fraud scoring with SHAP explanations. "
        "Powered by LightGBM + isotonic calibration."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    """Structured request log: method, path, status, duration_ms."""
    started = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


def _decision(score: float, threshold: float) -> Decision:
    return "FRAUD" if score >= threshold else "OK"


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(artifacts: ModelArtifacts = Depends(get_artifacts)) -> HealthResponse:
    """Liveness check with the model version and operating threshold."""
    return HealthResponse(
        status="ok",
        model_version=artifacts.version,
        threshold=artifacts.threshold,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    transaction: Transaction,
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> PredictResponse:
    """Score a single transaction. Returns calibrated P(fraud) + binary decision."""
    df = pd.DataFrame([transaction.model_dump()])
    score = float(artifacts.model.predict_proba(df)[0, 1])
    return PredictResponse(
        score=score,
        decision=_decision(score, artifacts.threshold),
        threshold=artifacts.threshold,
        model_version=artifacts.version,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
def explain(
    transaction: Transaction,
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> ExplainResponse:
    """Score plus SHAP-based top-10 feature contributions."""
    df = pd.DataFrame([transaction.model_dump()])
    score = float(artifacts.model.predict_proba(df)[0, 1])
    explanation = artifacts.explainer.explain_one(df)
    top = explanation.top_k(10)

    return ExplainResponse(
        score=score,
        decision=_decision(score, artifacts.threshold),
        threshold=artifacts.threshold,
        expected_value=explanation.expected_value,
        raw_score=explanation.raw_score,
        top_contributions=[
            FeatureContributionDTO(
                feature=c.feature,
                value=c.value,
                shap_value=c.shap_value,
            )
            for c in top
        ],
        model_version=artifacts.version,
    )
