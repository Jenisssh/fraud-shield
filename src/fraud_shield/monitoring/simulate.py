"""Synthetic drift injection for demos, tests, and the dashboard.

The Streamlit drift dashboard ships with a "simulate drift" slider so
the demo can show the detector lighting up on cue. This module backs
that slider. It also makes the drift module testable end-to-end —
inject known drift, confirm the detector flags it.

Three drift kinds:

- ``mean_shift`` — add ``magnitude * std`` to the feature. The
  textbook case: a sensor reads systematically high, or a vendor
  changed how amounts are reported.
- ``variance_scale`` — multiply the centered values by ``magnitude``.
  Models a distribution that has stretched (volatility increase) or
  compressed (regularization upstream) without shifting the mean.
- ``mixture`` — replace half the rows (random sample) with mean-shifted
  versions. Models a partial population change, e.g. a new merchant
  segment coming online.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

DriftKind = Literal["mean_shift", "variance_scale", "mixture"]


def inject_drift(
    df: pd.DataFrame,
    features: list[str],
    *,
    kind: DriftKind = "mean_shift",
    magnitude: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return a new DataFrame with synthetic drift applied to ``features``.

    Parameters
    ----------
    df:
        Source dataframe. Not mutated.
    features:
        Columns to perturb. Other columns pass through unchanged.
    kind:
        Which drift pattern to apply (see module docstring).
    magnitude:
        - ``mean_shift``: number of standard deviations to add.
        - ``variance_scale``: multiplier for the centered values
          (>1 spreads, <1 contracts).
        - ``mixture``: standard-deviation shift applied to the sampled
          half of rows.
    random_state:
        Seed for the row sampler in ``mixture`` mode. Ignored for the
        deterministic ``mean_shift`` / ``variance_scale`` modes.

    Returns
    -------
    pd.DataFrame
        A fresh dataframe with the same shape as ``df`` and the same
        column order; only the listed features have changed values.
    """
    if kind not in ("mean_shift", "variance_scale", "mixture"):
        raise ValueError(f"unknown drift kind: {kind!r}")

    rng = np.random.default_rng(random_state)
    out = df.copy()

    for f in features:
        if f not in df.columns:
            raise KeyError(f"feature {f!r} not in dataframe")

        col = df[f].astype(float)
        mean = float(col.mean())
        std = float(col.std() or 1.0)

        if kind == "mean_shift":
            out[f] = col + magnitude * std
        elif kind == "variance_scale":
            out[f] = (col - mean) * magnitude + mean
        else:  # mixture
            mask = rng.random(len(df)) < 0.5
            shifted = col + magnitude * std
            out[f] = np.where(mask, shifted, col)

    return out
