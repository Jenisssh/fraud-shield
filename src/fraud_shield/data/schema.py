"""Pandera schema for the ULB credit-card transaction dataframe.

The dataset has 31 columns: ``Time``, ``V1``..``V28`` (PCA-anonymized),
``Amount``, and ``Class``. The schema enforces dtypes, non-negativity for
``Time`` and ``Amount``, and ``Class`` ∈ {0, 1}. ``strict=True`` rejects
unexpected columns; ``coerce=True`` upcasts ints to floats where possible
so downstream code can assume float dtypes for the V/Amount features.

Use ``validate(df)`` for the common case — it returns the (coerced)
DataFrame, or raises ``SchemaErrors`` listing every violation at once.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pandera as pa

V_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))
FEATURE_COLUMNS: tuple[str, ...] = ("Time", *V_COLUMNS, "Amount")
TARGET_COLUMN: str = "Class"
ALL_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)

TRANSACTION_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        "Time": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        **{v: pa.Column(float, nullable=False) for v in V_COLUMNS},
        "Amount": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "Class": pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=True,
    ordered=False,
)


def validate(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Validate a transaction dataframe and return the coerced copy.

    Parameters
    ----------
    df:
        Candidate dataframe. Must have the 31 ULB columns and nothing else.
    lazy:
        When True (default) collect every violation and raise once at the end.
        When False, fail on the first violation.

    Returns
    -------
    pd.DataFrame
        The validated, dtype-coerced dataframe. Caller should treat the
        return value as authoritative; the input may have had different
        dtypes.
    """
    return cast(pd.DataFrame, TRANSACTION_SCHEMA.validate(df, lazy=lazy))
