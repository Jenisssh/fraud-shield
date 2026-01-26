"""Covariate-drift detection: PSI and two-sample KS test per feature.

Drift monitoring is a critical part of any deployed fraud system. The
question we're answering is *"does the feature distribution in production
look different from the distribution the model trained on?"*. If yes,
the model's calibration assumptions break down and the operating
threshold may no longer minimize expected cost.

This module exposes two complementary tests:

- **PSI (Population Stability Index)** — bin-based summary. PSI < 0.1
  is no concern, 0.1-0.2 is a watch, ≥ 0.2 is a typical alert
  threshold. Industry-standard for tabular features.
- **Two-sample KS test** — non-parametric distribution comparison.
  Returns a statistic (max gap between CDFs) and a p-value.
  Complements PSI for smaller samples where binning is unstable.

For convenience, :func:`detect_drift` runs both per-feature and returns
a sorted DataFrame ready to drop into a dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ks_2samp


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Per-feature drift result."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quantile_edges(reference: NDArray[Any], n_bins: int) -> NDArray[Any]:
    """Build bin edges from the reference distribution's quantiles.

    Collapses ties (e.g. zero-inflated features) and extends the outer
    edges to ±inf so the bins cover any value the current distribution
    might present.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    return np.concatenate([[-np.inf], edges[1:-1], [np.inf]])


def population_stability_index(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute the Population Stability Index between two 1-D distributions.

    The reference distribution defines the bin edges (deciles by default).
    Empty bins are clipped to ``eps`` to avoid ``log(0)``.

    Returns
    -------
    float
        Non-negative PSI. 0 means identical distributions.
    """
    ref = np.asarray(reference).astype(float)
    cur = np.asarray(current).astype(float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = _quantile_edges(ref, n_bins)
    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_prop = ref_counts / max(len(ref), 1)
    cur_prop = cur_counts / max(len(cur), 1)

    ref_prop = np.clip(ref_prop, eps, None)
    cur_prop = np.clip(cur_prop, eps, None)

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_drift_test(reference: ArrayLike, current: ArrayLike) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov drift test.

    Returns
    -------
    (statistic, p_value)
        ``statistic`` is the max gap between empirical CDFs (in [0, 1]).
        ``p_value`` is the probability of observing a gap that large
        under the null hypothesis of equal distributions — low values
        indicate drift.
    """
    result = ks_2samp(np.asarray(reference), np.asarray(current))
    return float(result.statistic), float(result.pvalue)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: list[str] | None = None,
    psi_threshold: float = 0.2,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute drift metrics for every shared numeric feature.

    Parameters
    ----------
    reference:
        The 'known good' distribution — usually the training set.
    current:
        The distribution to check — usually a recent production window.
    features:
        Subset to evaluate. ``None`` uses the intersection of both frames.
    psi_threshold:
        PSI cutoff above which the feature is marked drifted.
        Industry default is 0.2.
    n_bins:
        Number of bins for the PSI computation. 10 (deciles) is standard.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``psi``, ``ks_statistic``, ``ks_pvalue``,
        ``drifted``. Sorted by PSI descending so dashboards lead with
        the worst offenders.
    """
    if features is None:
        features = [c for c in reference.columns if c in current.columns]

    rows: list[dict[str, Any]] = []
    for f in features:
        if f not in reference.columns:
            raise KeyError(f"feature {f!r} not in reference")
        if f not in current.columns:
            raise KeyError(f"feature {f!r} not in current")
        psi = population_stability_index(reference[f], current[f], n_bins=n_bins)
        ks_stat, ks_p = ks_drift_test(reference[f], current[f])
        rows.append(
            {
                "feature": f,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "drifted": psi >= psi_threshold,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
    )
