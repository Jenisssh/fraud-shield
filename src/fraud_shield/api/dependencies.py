"""FastAPI dependencies — artifact loading and per-request access.

The service holds a *single* set of model artifacts for the lifetime of
the process. They're loaded once in the ``lifespan`` hook (see
``main.py``) and stored on ``app.state.artifacts``. ``get_artifacts`` is
the dependency every endpoint declares, and integration tests override
it to inject a mock so they don't need a trained model on disk.

The expected on-disk layout:

    models/
    ├── calibrated.joblib     # dict with keys: model, explainer, version
    └── threshold.txt          # single float, the chosen decision threshold

Either file missing causes the API to come up in *degraded* mode and
return 503 from the endpoints — preferable to silently serving with
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from fastapi import HTTPException, Request, status

from fraud_shield.config import settings

DEFAULT_MODEL_VERSION = "v0.1.0"


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    """Bundle the inference service needs at request time.

    Both ``model`` and ``explainer`` are typed as ``Any`` because the
    concrete classes (``CalibratedFraudClassifier``,
    ``FraudExplainer``) come from different modules — keeping this DTO
    decoupled from them makes test stubbing trivial.
    """

    model: Any
    explainer: Any
    threshold: float
    version: str


def load_artifacts(models_dir: Path | None = None) -> ModelArtifacts:
    """Load the on-disk model bundle.

    Raises :class:`FileNotFoundError` if ``calibrated.joblib`` is missing.
    Falls back to a 0.5 threshold when ``threshold.txt`` is absent — the
    `/health` endpoint surfaces this with ``status='degraded'``.
    """
    models_dir = models_dir or settings.models_dir
    bundle_path = models_dir / "calibrated.joblib"
    threshold_path = models_dir / "threshold.txt"

    if not bundle_path.exists():
        raise FileNotFoundError(f"model bundle not found at {bundle_path}")

    bundle = joblib.load(bundle_path)
    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5

    return ModelArtifacts(
        model=bundle["model"],
        explainer=bundle["explainer"],
        threshold=threshold,
        version=str(bundle.get("version", DEFAULT_MODEL_VERSION)),
    )


def get_artifacts(request: Request) -> ModelArtifacts:
    """FastAPI dependency that returns the per-app artifacts.

    Returns 503 if the lifespan failed to load anything. Tests override
    this with ``app.dependency_overrides[get_artifacts] = stub``.
    """
    artifacts: ModelArtifacts | None = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded — train and persist before serving",
        )
    return artifacts
