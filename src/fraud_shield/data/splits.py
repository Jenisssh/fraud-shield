"""Train / val / test split strategies for the transaction dataset.

Two flavours live here:

- :func:`stratified_random_split` — IID random sampling with class
  stratification. The right choice for the *headline* model score, since
  it gives every transaction an equal chance of landing in any partition.

- :func:`time_aware_split` (Day 4b) — chronological split. Train on the
  earliest transactions, test on the latest. Reveals temporal generalization
  — the gap between this score and the stratified score is one of the
  numbers we'll report as a sanity check on whether the model would survive
  deployment.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_shield.config import settings


def stratified_random_split(
    df: pd.DataFrame,
    *,
    target: str = "Class",
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random 3-way split.

    Holds out ``test`` first, then carves ``val`` out of the remainder so
    the val fraction is computed against the full dataset, not against the
    train+val intermediate.

    Parameters
    ----------
    df:
        Source dataframe. Must contain ``target``.
    target:
        Column to stratify on. Default ``"Class"``.
    test_size, val_size:
        Fractions of the *original* dataframe. Defaults come from
        :class:`fraud_shield.config.Settings`.
    random_state:
        Seed; defaults to ``settings.random_seed``.

    Returns
    -------
    tuple of (train, val, test)
        Index-reset copies. The three are disjoint and union to ``df``.
    """
    test_size = settings.test_size if test_size is None else test_size
    val_size = settings.val_size if val_size is None else val_size
    random_state = settings.random_seed if random_state is None else random_state

    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size + val_size must be < 1; got {test_size} + {val_size}"
        )

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )
    val_fraction_of_remainder = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remainder,
        stratify=train_val[target],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
