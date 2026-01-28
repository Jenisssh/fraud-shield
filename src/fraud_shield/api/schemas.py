"""Pydantic request/response models for the FastAPI service.

Defining each V1..V28 column explicitly is verbose, but it gives us:

- per-field FastAPI/OpenAPI docs out of the box
- automatic 422 responses on missing or malformed fields
- a single source of truth that the Streamlit form can introspect
- ``extra='forbid'`` rejects unknown keys, so callers can't sneak in
  features the model never saw

All response models exclude internal-only fields and freeze their
configuration so they're safe to ship as JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["FRAUD", "OK"]


class Transaction(BaseModel):
    """A single credit card transaction in the ULB schema."""

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0, description="Seconds since the first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="Transaction amount")


class PredictResponse(BaseModel):
    """The /predict response — score plus a binary decision at the configured threshold."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1, description="Calibrated P(fraud)")
    decision: Decision
    threshold: float = Field(ge=0, le=1, description="Threshold the decision was made against")
    model_version: str


class FeatureContributionDTO(BaseModel):
    """One feature's contribution to a single prediction, suitable for JSON."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float
    shap_value: float = Field(description="SHAP value on the margin (log-odds) scale")


class ExplainResponse(BaseModel):
    """The /explain response — prediction plus the top contributing features."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    decision: Decision
    threshold: float = Field(ge=0, le=1)
    expected_value: float = Field(description="SHAP base value on the margin scale")
    raw_score: float = Field(description="Margin score = expected_value + sum(shap_values)")
    top_contributions: list[FeatureContributionDTO]
    model_version: str


class HealthResponse(BaseModel):
    """The /health response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    model_version: str
    threshold: float = Field(ge=0, le=1)
