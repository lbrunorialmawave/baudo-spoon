"""Sample-weighting strategies for low-sample training observations.

Implements PR2 from ``plan.md`` (Low-Sample Player & Breakout Modeling).

The trainer must never contain hard-coded weight formulas scattered in
``if`` branches.  All weighting logic lives in
:func:`compute_sample_weight`, which is deterministic, fully typed, and
side-effect free.

Available strategies (selected via :class:`WeightingStrategy`):

* ``"constant"`` — weight = 1.0 for ``STANDARD``/``LIMITED``,
  weight = 0.0 for ``INSUFFICIENT`` (today's implicit behaviour).
* ``"linear"`` — ``min(1, minutes / standard_minutes)``.
* ``"sqrt"`` — ``min(1, sqrt(minutes / standard_minutes))`` (default
  candidate; ``plan.md`` §9.2).
* ``"bucketed"`` — discrete bucket weights (100–399 → w1, 400–799 → w2,
  800+ → 1.0).  Bucket boundaries are exposed as module-level constants
  so they can be referenced from configuration.

Important invariants (validated by tests):

* ``weight in [0, 1]`` for every supported ``minutes`` value.
* ``weight`` is monotonically non-decreasing in ``minutes``.
* ``weight(standard_minutes) == 1.0`` and
  ``weight(minutes > standard_minutes) == 1.0``.
* For ``minutes < min_minutes_hard`` the weight is exactly ``0.0`` (the
  row is excluded from training by design).
"""

from __future__ import annotations

import math
from typing import Final

from .cohort import (
    COHORT_INSUFFICIENT,
    COHORT_LIMITED,
    COHORT_STANDARD,
    Cohort,
)

# ── Public strategy constants ───────────────────────────────────────────────

STRATEGY_CONSTANT: Final[str] = "constant"
STRATEGY_LINEAR: Final[str] = "linear"
STRATEGY_SQRT: Final[str] = "sqrt"
STRATEGY_BUCKETED: Final[str] = "bucketed"

# Strategy names that keep LIMITED rows in the training set.
ENABLED_STRATEGIES: Final[frozenset[str]] = frozenset(
    {STRATEGY_LINEAR, STRATEGY_SQRT, STRATEGY_BUCKETED}
)

# Discrete bucket weights for the "bucketed" strategy.
_BUCKET_WEIGHTS: Final[dict[tuple[int, int], float]] = {
    (100, 399): 0.35,
    (400, 799): 0.65,
    (800, 10_000): 1.00,
}

WeightingStrategy = str


# ── Public API ──────────────────────────────────────────────────────────────

def compute_sample_weight(
    minutes: int | float | None,
    *,
    strategy: WeightingStrategy = STRATEGY_SQRT,
    standard_minutes: int = 800,
    min_minutes_hard: int = 100,
) -> float:
    """Return the sample weight for a single observation.

    Args:
        minutes: Minutes played by the player.  ``None``, NaN, or
            negative values are treated as ``INSUFFICIENT`` (weight 0.0).
        strategy: One of the supported strategy constants.
        standard_minutes: Reference threshold for ``STANDARD`` cohort.
        min_minutes_hard: Lower eligibility cutoff.  Below this value
            the weight is exactly ``0.0``.

    Returns:
        Sample weight in ``[0.0, 1.0]``.

    Raises:
        ValueError: if *strategy* is not a supported name.
    """
    _validate_params(strategy, standard_minutes, min_minutes_hard)

    if minutes is None or _is_non_positive(minutes):
        return 0.0
    if minutes < min_minutes_hard:
        return 0.0
    if minutes >= standard_minutes:
        return 1.0

    if strategy == STRATEGY_CONSTANT:
        return 1.0
    if strategy == STRATEGY_LINEAR:
        return min(1.0, float(minutes) / float(standard_minutes))
    if strategy == STRATEGY_SQRT:
        return min(1.0, math.sqrt(float(minutes) / float(standard_minutes)))
    if strategy == STRATEGY_BUCKETED:
        return _bucket_weight(int(minutes))

    raise ValueError(f"Unknown weighting strategy: {strategy!r}")  # pragma: no cover


def cohort_for_minutes(
    minutes: int | float | None,
    *,
    min_minutes_hard: int = 100,
    standard_minutes: int = 800,
) -> Cohort:
    """Convenience wrapper that maps minutes → cohort without re-importing
    :mod:`ml.sample_reliability.cohort`.

    Provided here to keep the weighting API self-contained for trainer
    integration.  Delegates to :func:`ml.sample_reliability.cohort.classify_cohort`.
    """
    # Local import avoids a circular dependency at module-load time.
    from .cohort import classify_cohort

    return classify_cohort(
        minutes,
        min_minutes_hard=min_minutes_hard,
        standard_minutes=standard_minutes,
    )


# ── Internal helpers ────────────────────────────────────────────────────────

def _bucket_weight(minutes: int) -> float:
    for (lo, hi), w in _BUCKET_WEIGHTS.items():
        if lo <= minutes <= hi:
            return w
    return 0.0


def _validate_params(
    strategy: str, standard_minutes: int, min_minutes_hard: int
) -> None:
    if strategy not in {STRATEGY_CONSTANT, STRATEGY_LINEAR, STRATEGY_SQRT, STRATEGY_BUCKETED}:
        raise ValueError(
            f"Unknown weighting strategy: {strategy!r}. "
            f"Supported: constant, linear, sqrt, bucketed."
        )
    if standard_minutes <= 0:
        raise ValueError("standard_minutes must be positive")
    if min_minutes_hard < 0:
        raise ValueError("min_minutes_hard must be non-negative")
    if min_minutes_hard >= standard_minutes:
        raise ValueError(
            "min_minutes_hard must be strictly less than standard_minutes"
        )


def _is_non_positive(value: int | float) -> bool:
    """Return True for None, NaN, or values that are <= 0.

    ``NaN`` comparisons always evaluate to False, so we must short-circuit
    on ``math.isnan`` explicitly before the ``<=`` test.
    """
    if value is None:
        return True
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True
    if math.isnan(f):
        return True
    return f <= 0.0