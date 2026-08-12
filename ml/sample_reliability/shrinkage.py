"""Bayesian shrinkage for per-90 / rate-based features (PR3).

A naive ``goals / minutes * 90`` on a 100-minute sample with 3 goals
yields an absurd ``2.7 goals/90``.  This module provides a deterministic
shrinkage estimator that pulls extreme small-sample rates toward a
population prior.

Formula (single-binomial, additive shrinkage):

    adjusted_rate = (observed_rate * minutes + prior_rate * prior_strength)
                    / (minutes + prior_strength)

Where:

* ``observed_rate`` is the empirical per-90 rate,
* ``minutes`` is the per-90 denominator (sum of minutes / 90),
* ``prior_rate`` is a population-level reference rate (typically the
  median of the standard cohort), and
* ``prior_strength`` is the equivalent "minutes of pseudo-counts" the
  prior contributes (default ``DEFAULT_PRIOR_STRENGTH = 300`).

The function is stateless, deterministic, and works on both scalars
and ``pd.Series`` inputs.  It is shared between training and inference
so the same shrinkage is applied on both sides (no train/serve skew).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_PRIOR_STRENGTH: Final[int] = 300


def apply_shrinkage(
    observed_rate: float | pd.Series,
    *,
    minutes: float | pd.Series,
    prior_rate: float,
    prior_strength: int = DEFAULT_PRIOR_STRENGTH,
) -> float | pd.Series:
    """Return the shrinkage-adjusted rate.

    Args:
        observed_rate: Empirical rate (per-90, per-appearance, …).
        minutes: Sample-size proxy (minutes, appearances …).  Must be
            non-negative; the function does not silently repair negatives.
        prior_rate: Population prior (e.g. median of the standard
            cohort).  Must be non-negative.
        prior_strength: Number of pseudo-observations the prior
            contributes.  Higher → stronger pull toward the prior.

    Returns:
        Adjusted rate, same type as ``observed_rate`` (scalar or
        ``pd.Series``).
    """
    if prior_rate < 0:
        raise ValueError("prior_rate must be non-negative")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")

    # Defensive normalisation: convert negative minutes to NaN rather than
    # silently clamping — a negative denominator is a data-quality bug.
    minutes_arr = _ensure_array(minutes)
    observed_arr = _ensure_array(observed_rate)

    safe_minutes = np.where(minutes_arr < 0, np.nan, minutes_arr)
    safe_minutes = np.where(np.isnan(safe_minutes), 0.0, safe_minutes)

    adjusted = (observed_arr * safe_minutes + prior_rate * prior_strength) / (
        safe_minutes + prior_strength
    )
    if isinstance(observed_rate, pd.Series):
        return pd.Series(adjusted, index=observed_rate.index)
    return float(adjusted)


def estimate_prior_rate(
    observed_rates: pd.Series,
    *,
    minutes: pd.Series,
    min_minutes: int = 800,
) -> float:
    """Estimate the population prior as the median of the standard cohort.

    Computed lazily from the observed rates restricted to rows with
    ``minutes >= min_minutes``.  Returns ``0.0`` if the cohort is empty
    (the caller should treat that as a degenerate-data signal, not a
    reason to fall back silently).
    """
    mask = (minutes >= min_minutes) & observed_rates.notna()
    if not mask.any():
        log.warning(
            "estimate_prior_rate: empty cohort (minutes>=%d); returning 0.0", min_minutes
        )
        return 0.0
    return float(observed_rates[mask].median())


def _ensure_array(value: float | pd.Series) -> np.ndarray:
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float, copy=False)
    return np.asarray(value, dtype=float)
