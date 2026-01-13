"""Tests for the pandera transaction schema.

Covers the happy path (valid frame round-trips), structural rejections
(missing or extra columns under strict mode), and per-column constraints
(non-negative Amount, Class in {0, 1}).
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from fraud_shield.data.schema import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    V_COLUMNS,
    validate,
)


def test_v_columns_has_28_entries() -> None:
    assert len(V_COLUMNS) == 28
    assert V_COLUMNS[0] == "V1"
    assert V_COLUMNS[-1] == "V28"


def test_all_columns_is_31_columns() -> None:
    assert len(ALL_COLUMNS) == 31
    assert ALL_COLUMNS[0] == "Time"
    assert ALL_COLUMNS[-1] == TARGET_COLUMN
    assert "Amount" in ALL_COLUMNS


def test_feature_columns_excludes_target() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 30


def test_validate_passes_for_tiny_fraud_df(tiny_fraud_df: pd.DataFrame) -> None:
    result = validate(tiny_fraud_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == tiny_fraud_df.shape
    assert set(result.columns) == set(ALL_COLUMNS)


def test_validate_rejects_missing_v_column(tiny_fraud_df: pd.DataFrame) -> None:
    bad = tiny_fraud_df.drop(columns=["V14"])
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(bad)


def test_validate_rejects_extra_column(tiny_fraud_df: pd.DataFrame) -> None:
    bad = tiny_fraud_df.assign(merchant_id="abc")
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(bad)


def test_validate_rejects_negative_amount(tiny_fraud_df: pd.DataFrame) -> None:
    bad = tiny_fraud_df.copy()
    bad.loc[0, "Amount"] = -1.0
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(bad)


def test_validate_rejects_negative_time(tiny_fraud_df: pd.DataFrame) -> None:
    bad = tiny_fraud_df.copy()
    bad.loc[0, "Time"] = -100.0
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(bad)


def test_validate_rejects_invalid_class(tiny_fraud_df: pd.DataFrame) -> None:
    bad = tiny_fraud_df.copy()
    bad.loc[0, "Class"] = 2
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(bad)


def test_validate_coerces_integer_v_features_to_float(tiny_fraud_df: pd.DataFrame) -> None:
    bad = tiny_fraud_df.copy()
    bad["V1"] = bad["V1"].round().astype("int64")
    result = validate(bad)
    assert result["V1"].dtype.kind == "f"
