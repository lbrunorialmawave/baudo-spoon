"""Schedule difficulty coefficient computation.

.. deprecated::
    This module is not connected to the training pipeline or any feature set.
    ``opponent_elo`` / ``opponent_expected_points`` columns are not produced by
    any current data ingestion step, so ``compute_difficulty_coefficients`` is
    never called outside of tests.  Wire it into ``ml/features/`` or the
    trainer's feature engineering step before removing this notice.

``difficulty_coefficient`` is normalised to ``[coeff_min, coeff_max]`` where:

  - 1.0 = average-strength opponent
  - > 1.0 = stronger-than-average opponent (stats adjusted UPWARD)
  - < 1.0 = weaker-than-average opponent (stats adjusted DOWNWARD)

This direction is intentional: a player who performs well against strong
opponents deserves a higher adjusted score than one who performs the same
against weak opponents.

Components (all normalised within [0, 1] per season before weighting):
  - elo: opponent ELO rating
  - expected_points: opponent xPts from recent form
  - league_position: opponent league position (inverted: bottom = 0, top = 1)
  - goal_difference: opponent goal difference
  - squad_value: opponent transfer market value (not always available)

If a component column is absent its weight is redistributed proportionally
across the remaining present components rather than silently zeroed.  Zeroing
would bias the coefficient toward 1.0 incorrectly when components with high
weight are missing.
"""

from __future__ import annotations

import logging

import polars as pl

from ml.domain.config import DEFAULT_SCHEDULE_ADJUSTMENT, ScheduleAdjustmentConfig

__all__ = ["compute_difficulty_coefficients"]

log = logging.getLogger(__name__)

# Maps ScheduleAdjustmentConfig field name → expected DataFrame column name.
_COMPONENT_COLUMNS: dict[str, str] = {
    "elo_weight": "opponent_elo",
    "expected_points_weight": "opponent_expected_points",
    "league_position_weight": "opponent_league_position",
    "goal_difference_weight": "opponent_goal_difference",
    "squad_value_weight": "opponent_squad_value",
}

# Higher raw value = weaker opponent for these → invert before normalising.
_INVERT_COMPONENTS: frozenset[str] = frozenset(["opponent_league_position"])


def compute_difficulty_coefficients(
    data: pl.DataFrame,
    config: ScheduleAdjustmentConfig = DEFAULT_SCHEDULE_ADJUSTMENT,
    season_col: str = "season_start",
) -> pl.Series:
    """Compute per-row difficulty coefficients in ``[config.coeff_min, config.coeff_max]``.

    Args:
        data: DataFrame with opponent stat columns and (optionally) a season column.
        config: ScheduleAdjustmentConfig with component weights and range.
        season_col: Column name for within-season normalisation; falls back to
            global normalisation when absent.

    Returns:
        ``pl.Series`` named ``"difficulty_coefficient"`` with shape ``(len(data),)``
        and all values in ``[coeff_min, coeff_max]``.
        Returns all-1.0 (neutral) when no opponent columns are present.
    """
    # Identify which component columns are present in the data.
    available: dict[str, float] = {
        col: getattr(config, weight_field)
        for weight_field, col in _COMPONENT_COLUMNS.items()
        if col in data.columns
    }

    if not available:
        log.warning(
            "compute_difficulty_coefficients: no opponent columns found; "
            "returning neutral coefficients (1.0). Expected: %s",
            sorted(_COMPONENT_COLUMNS.values()),
        )
        return pl.Series("difficulty_coefficient", [1.0] * len(data), dtype=pl.Float64)

    # Redistribute weights across available components proportionally.
    total_weight = sum(available.values())
    rescaled: dict[str, float] = {col: w / total_weight for col, w in available.items()}

    missing_cols = sorted(set(_COMPONENT_COLUMNS.values()) - set(available))
    if missing_cols:
        log.info(
            "compute_difficulty_coefficients: missing columns %s; "
            "weights redistributed to present components.",
            missing_cols,
        )

    # Build composite score ∈ [0, 1] as weighted sum of per-season normalised components.
    raw_score = pl.Series("_score", [0.0] * len(data), dtype=pl.Float64)

    for col, weight in rescaled.items():
        series = data[col].cast(pl.Float64)
        if col in _INVERT_COMPONENTS:
            series = -series

        normalised = _normalise_per_season(series, data, season_col)
        raw_score = raw_score + normalised * weight

    # Scale from [0, 1] to [coeff_min, coeff_max].
    coeff_range = config.coeff_max - config.coeff_min
    coefficients = raw_score * coeff_range + config.coeff_min
    return coefficients.rename("difficulty_coefficient")


def _normalise_per_season(
    series: pl.Series,
    data: pl.DataFrame,
    season_col: str,
) -> pl.Series:
    """Min-max normalise *series* within each season.  Falls back to global."""
    if season_col not in data.columns:
        return _minmax(series)

    seasons = data[season_col]
    result_list: list[float] = [0.0] * len(data)

    # ponytail: O(n_seasons) Python loop; vectorise with group_by if n_seasons >> 10
    for season in seasons.unique().to_list():
        mask = seasons == season
        true_indices = mask.arg_true().to_list()
        vals = series.filter(mask)
        norm_vals = _minmax(vals).to_list()
        for i, idx in enumerate(true_indices):
            result_list[idx] = norm_vals[i]

    return pl.Series("_norm", result_list, dtype=pl.Float64)


def _minmax(s: pl.Series) -> pl.Series:
    """Min-max normalise a series to [0, 1]; returns 0.5 when range is zero."""
    s_min = s.min()
    s_max = s.max()
    if s_min is None or s_max is None or s_max == s_min:
        return pl.Series("_norm", [0.5] * len(s), dtype=pl.Float64)
    return (s - s_min) / (s_max - s_min)
