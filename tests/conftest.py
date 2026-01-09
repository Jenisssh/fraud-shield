"""Shared pytest fixtures.

The ``tiny_fraud_df`` fixture mimics the schema of the ULB credit-card dataset
(V1..V28 + Time + Amount + Class) at ~1k rows so unit tests run in milliseconds
without needing the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_fraud_df() -> pd.DataFrame:
    """Synthetic dataset with the same columns as ULB creditcard.csv."""
    rng = np.random.default_rng(42)
    n = 1_000
    data: dict[str, np.ndarray] = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = np.arange(n, dtype=float) * 7.3
    data["Amount"] = rng.exponential(scale=50.0, size=n)
    data["Class"] = (rng.random(n) < 0.01).astype(int)
    return pd.DataFrame(data)
