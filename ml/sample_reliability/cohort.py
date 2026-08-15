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
#
# Legacy step-function path.  Prefer :func:`continuous_reliability_weight` /
# :func:`get_reliability_weight` (mode="continuous") for new code; the dict
# remains as a documented fallback and for bit-identical behaviour when
# ``reliability_weight_mode="bucket"``.
RELIABILITY_WEIGHT_BY_COHORT: Final[dict[str, float]] = {
    COHORT_STANDARD: 1.0,
    COHORT_LIMITED: 0.65,
    COHORT_INSUFFICIENT: 0.30,
}

# Default floor for the continuous decision weight (never fully zero out an
# eligible LIMITED player).  Matches the historical INSUFFICIENT bucket.
DEFAULT_RELIABILITY_FLOOR: Final[float] = 0.30

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


# ── Continuous decision-layer reliability weight (WS2) ──────────────────────

def continuous_reliability_weight(
    minutes: int | float | None,
    *,
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
    floor: float = DEFAULT_RELIABILITY_FLOOR,
    strategy: str = "sqrt",
) -> float:
    """Continuous decision-layer reliability weight in ``[floor, 1.0]``.

    Replaces the three-bucket step function for Optimizer / Auction ranking
    so that a player at 105' is discounted more heavily than one at 795'.
    Shape mirrors :func:`ml.sample_reliability.weights.compute_sample_weight`
    (default ``sqrt``) but never drops below *floor* for eligible players
    (minutes >= min_minutes_hard).

    Invariants (enforced by tests):
    * result ∈ [floor, 1.0]
    * monotonically non-decreasing in minutes
    * weight(standard_minutes) == 1.0 (and above)
    * weight(minutes < min_minutes_hard) == floor  (INSUFFICIENT still
      receives the floor rather than 0 so the player remains visible)
    """
    import math

    if floor < 0.0 or floor > 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if min_minutes_hard < 0:
        raise ValueError("min_minutes_hard must be non-negative")
    if min_minutes_hard >= standard_minutes:
        raise ValueError(
            "min_minutes_hard must be strictly less than standard_minutes"
        )
    if strategy not in {"sqrt", "linear"}:
        raise ValueError(
            f"Unsupported strategy for continuous reliability weight: {strategy!r}. "
            "Supported: 'sqrt', 'linear'."
        )

    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return float(floor)
    try:
        m = float(minutes)
    except (TypeError, ValueError):
        return float(floor)
    if m < 0 or math.isnan(m):
        return float(floor)
    if m < min_minutes_hard:
        return float(floor)
    if m >= standard_minutes:
        return 1.0

    # Interpolate between floor (at min_minutes_hard) and 1.0 (at standard).
    # Using the same functional form as sample weighting, then rescaling
    # into [floor, 1] so the curve is continuous at the hard cutoff.
    ratio = m / float(standard_minutes)
    if strategy == "sqrt":
        raw = math.sqrt(ratio)
    else:
        raw = ratio
    # raw at min_minutes_hard is sqrt(min/std) or min/std; map [raw_at_hard, 1] → [floor, 1]
    raw_at_hard = (
        math.sqrt(min_minutes_hard / float(standard_minutes))
        if strategy == "sqrt"
        else (min_minutes_hard / float(standard_minutes))
    )
    if raw_at_hard >= 1.0:
        return 1.0
    # Linear map of raw from [raw_at_hard, 1] onto [floor, 1]
    t = (raw - raw_at_hard) / (1.0 - raw_at_hard)
    t = max(0.0, min(1.0, t))
    return float(floor + t * (1.0 - floor))


def get_reliability_weight(
    minutes: int | float | None = None,
    cohort: Cohort | None = None,
    *,
    mode: str = "bucket",
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
    floor: float = DEFAULT_RELIABILITY_FLOOR,
    strategy: str = "sqrt",
) -> float:
    """Dispatch to continuous or legacy bucket reliability weight.

    * mode="bucket" (default): uses :data:`RELIABILITY_WEIGHT_BY_COHORT`
      keyed by *cohort* (falls back to 1.0).  Bit-identical to pre-WS2.
    * mode="continuous": uses :func:`continuous_reliability_weight` on
      *minutes*.  If minutes is missing, falls back to the bucket value
      for the provided cohort (or STANDARD).
    """
    if mode == "continuous":
        if minutes is not None:
            return continuous_reliability_weight(
                minutes,
                min_minutes_hard=min_minutes_hard,
                standard_minutes=standard_minutes,
                floor=floor,
                strategy=strategy,
            )
        # minutes unavailable → degrade to bucket for the known cohort
        if cohort is not None:
            return float(RELIABILITY_WEIGHT_BY_COHORT.get(cohort, 1.0))
        return 1.0

    # Legacy bucket path
    if cohort is not None:
        return float(RELIABILITY_WEIGHT_BY_COHORT.get(cohort, 1.0))
    if minutes is not None:
        c = classify_cohort(
            minutes,
            min_minutes_hard=min_minutes_hard,
            standard_minutes=standard_minutes,
        )
        return float(RELIABILITY_WEIGHT_BY_COHORT.get(c, 1.0))
    return 1.0


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
