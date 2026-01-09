"""Smoke tests — verify the package imports and the basic fixture works.

These keep CI green on Day 1, before any real modules exist. Real unit tests
land alongside the modules they cover (test_data.py, test_features.py, ...).
"""

from __future__ import annotations

import pandas as pd

import fraud_shield
from fraud_shield.config import settings


def test_version_is_set() -> None:
    assert fraud_shield.__version__ == "0.1.0"


def test_settings_has_project_root_with_pyproject() -> None:
    assert (settings.project_root / "pyproject.toml").is_file()


def test_settings_seed_is_42() -> None:
    assert settings.random_seed == 42


def test_tiny_fraud_df_shape(tiny_fraud_df: pd.DataFrame) -> None:
    assert tiny_fraud_df.shape == (1_000, 31)
    assert "Class" in tiny_fraud_df.columns
    assert tiny_fraud_df["Class"].isin([0, 1]).all()


def test_tiny_fraud_df_columns_match_ulb_schema(tiny_fraud_df: pd.DataFrame) -> None:
    expected = {f"V{i}" for i in range(1, 29)} | {"Time", "Amount", "Class"}
    assert set(tiny_fraud_df.columns) == expected
