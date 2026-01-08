"""Project-wide configuration loaded from environment variables.

All paths are absolute and resolved relative to the project root so that scripts
behave identically whether invoked from `D:\\MLE projects\\fraud-shield` or from
a notebook one level deeper.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    All fields can be overridden via environment variables with the
    ``FRAUDSHIELD_`` prefix, e.g. ``FRAUDSHIELD_RANDOM_SEED=7``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FRAUDSHIELD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    configs_dir: Path = PROJECT_ROOT / "configs"

    random_seed: int = 42
    test_size: float = Field(default=0.20, ge=0.05, le=0.40)
    val_size: float = Field(default=0.15, ge=0.05, le=0.40)

    target_column: str = "Class"


settings = Settings()
