"""Cohort classification and ``SampleReliability`` DTO.

This module implements the foundational data-quality layer for the
low-sample player modelling feature.  The historical threshold
(``min_minutes = 800``) is preserved as the boundary for the
high-confidence cohort.  Players between ``100`` and ``799`` minutes are
classified as ``LIMITED`` and can be optionally included in the training
set with a reduced sample weight (see :mod:`ml.sample_reliability.weights`).

Classification rules (single source of truth — see ``plan.md`` §0.1 and §5):

* ``minutes < 100`` → ``INSUFFICIENT`` (excluded from main training by
  default; may still be used for identity / role signals).
* ``100 <= minutes < 800`` → ``LIMITED`` (training-eligible with
  reduced weight, opt-in via feature flag).
* ``minutes >= 800`` → ``STANDARD`` (weight = 1.0).

This module never mutates the input DataFrame; all functions are pure
and deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Public cohort constants ───────────────────────────────────────────────────

COHORT_INSUFFICIENT: Final[str] = "INSUFFICIENT"
COHORT_LIMITED: Final[str] = "LIMITED"
COHORT_STANDARD: Final[str] = "STANDARD"
SAMPLE_COHORTS: Final[tuple[str, ...]] = (
    COHORT_INSUFFICIENT,
    COHORT_LIMITED,
    COHORT_STANDARD,
)

# Decision-layer reliability weights applied to Optimizer (ILP objective) and
# Auction when a player belongs to a low-sample cohort.  STANDARD keeps full
# weight; LIMITED / INSUFFICIENT are penalised so inflated raw predictions do
# not compete on equal footing.  Values are conservative defaults — calibrate
# via backtest before treating them as final (see reliability rollout plan).
RELIABILITY_WEIGHT_BY_COHORT: Final[dict[str, float]] = {
    COHORT_STANDARD: 1.0,
    COHORT_LIMITED: 0.65,
    COHORT_INSUFFICIENT: 0.30,
}

# Type alias for static type checkers.
Cohort = str


# ── DTO ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SampleReliability:
    """Reliability metadata for a single (player, season) observation.

    Attributes:
        minutes: Minutes played in the season.  ``< 0`` is treated as NaN.
        appearances: Number of matches played.  ``< 0`` is treated as NaN.
        starts: Number of starts (``None`` when unknown, e.g. pre-2024 data).
        cohort: One of ``INSUFFICIENT`` / ``LIMITED`` / ``STANDARD``.
        weight: Sample weight in ``[0.0, 1.0]``.  ``0.0`` means
            "do not use for training", ``1.0`` is the high-confidence
            reference weight.
    """

    minutes: int
    appearances: int
    starts: int | None
    cohort: Cohort
    weight: float

    def __post_init__(self) -> None:  # pragma: no cover - invariant guard
        if self.cohort not in SAMPLE_COHORTS:
            raise ValueError(
                f"Invalid cohort '{self.cohort}'; expected one of {SAMPLE_COHORTS}"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"Sample weight must be in [0, 1], got {self.weight}"
            )


# ── Pure classification ─────────────────────────────────────────────────────

def classify_cohort(
    minutes: int | float | None,
    *,
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
) -> Cohort:
    """Classify a single observation into a sample cohort.

    Args:
        minutes: Minutes played.  ``None``/NaN/negative → ``INSUFFICIENT``.
        min_minutes_hard: Lower bound for the ``LIMITED`` cohort.  Must
            be strictly positive and strictly less than ``standard_minutes``.
        standard_minutes: Lower bound for the ``STANDARD`` cohort.

    Returns:
        The cohort label as a string constant (see ``SAMPLE_COHORTS``).

    Raises:
        ValueError: if ``min_minutes_hard >= standard_minutes``.
    """
    if min_minutes_hard < 0:
        raise ValueError("min_minutes_hard must be non-negative")
    if min_minutes_hard >= standard_minutes:
        raise ValueError(
            "min_minutes_hard must be strictly less than standard_minutes"
        )

    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return COHORT_INSUFFICIENT
    if minutes < 0:
        return COHORT_INSUFFICIENT
    if minutes < min_minutes_hard:
        return COHORT_INSUFFICIENT
    if minutes < standard_minutes:
        return COHORT_LIMITED
    return COHORT_STANDARD


def build_sample_reliability(
    row: pd.Series,
    *,
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
    weight: float | None = None,
) -> SampleReliability:
    """Build a :class:`SampleReliability` from a single row.

    Helper used by the trainer to attach reliability metadata to each
    (player, season) observation.  ``weight`` defaults to ``1.0`` for
    STANDARD, ``0.0`` for INSUFFICIENT; LIMITED rows are usually
    re-weighted by :func:`ml.sample_reliability.weights.compute_sample_weight`.
    """
    minutes = _coerce_int(row.get("mins_played"))
    appearances = _coerce_int(row.get("appearances"))
    starts_raw = row.get("starts")
    starts: int | None
    if starts_raw is None or (isinstance(starts_raw, float) and np.isnan(starts_raw)):
        starts = None
    else:
        try:
            starts = max(0, int(starts_raw))
        except (TypeError, ValueError):
            starts = None

    cohort = classify_cohort(
        minutes,
        min_minutes_hard=min_minutes_hard,
        standard_minutes=standard_minutes,
    )
    if weight is None:
        weight = 0.0 if cohort == COHORT_INSUFFICIENT else 1.0
    return SampleReliability(
        minutes=minutes,
        appearances=appearances,
        starts=starts,
        cohort=cohort,
        weight=float(weight),
    )


# ── Dataset profiling ───────────────────────────────────────────────────────

def profile_dataset(
    df: pd.DataFrame,
    *,
    minutes_col: str = "mins_played",
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
) -> dict[str, int | float]:
    """Return a summary of cohort counts for a training-ready DataFrame.

    Args:
        df: DataFrame containing at minimum a minutes column.
        minutes_col: Column to read minutes from.
        min_minutes_hard: Hard eligibility cutoff for limited samples.
        standard_minutes: Reference high-confidence threshold.

    Returns:
        Dictionary with cohort counts, total weight, and ratio fields.
        Safe to serialise as JSON.
    """
    if minutes_col not in df.columns:
        raise KeyError(f"Column '{minutes_col}' not found in DataFrame")

    minutes = pd.to_numeric(df[minutes_col], errors="coerce")
    cohorts = minutes.apply(
        lambda m: classify_cohort(
            m,
            min_minutes_hard=min_minutes_hard,
            standard_minutes=standard_minutes,
        )
    )
    counts = {c: int((cohorts == c).sum()) for c in SAMPLE_COHORTS}
    total = sum(counts.values())
    return {
        "n_total": total,
        "n_insufficient": counts[COHORT_INSUFFICIENT],
        "n_limited": counts[COHORT_LIMITED],
        "n_standard": counts[COHORT_STANDARD],
        "share_insufficient": counts[COHORT_INSUFFICIENT] / total if total else 0.0,
        "share_limited": counts[COHORT_LIMITED] / total if total else 0.0,
        "share_standard": counts[COHORT_STANDARD] / total if total else 0.0,
        "min_minutes_hard": int(min_minutes_hard),
        "standard_minutes": int(standard_minutes),
    }


# ── Internal helpers ────────────────────────────────────────────────────────

def _coerce_int(value: object) -> int:
    """Coerce a value to a non-negative int, returning ``0`` for null/NaN/garbage."""
    if value is None:
        return 0
    if isinstance(value, float) and np.isnan(value):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
