"""Rolling temporal trend features (rolling mean and YoY delta).

Uses ``shift(1)`` before ``rolling_mean`` to enforce ``closed='left'``
semantics: the window for season *t* includes only seasons *t-1*, *t-2*, …
Never the current season.  This eliminates look-ahead bias and makes the
features safe inside cross-validation pipelines.

Input DataFrame must contain per-90 columns (run ``ml/features/per90.py``
first) and identifier columns ``player_fotmob_id`` + ``season_start``.
"""
from __future__ import annotations

import polars as pl

from ml.domain.features import Feature, MissingDataPolicy

__all__ = [
    "RollingMeanFeature",
    "YoYDeltaFeature",
    "ALL_ROLLING_FEATURES",
    "ALL_DELTA_FEATURES",
    "ALL_TREND_FEATURES",
]

# Stats for which trend features are computed (must already be per-90 columns).
_TREND_CANDIDATES: list[str] = [
    "goals_per90",
    "goal_assist_per90",
    "total_scoring_att_per90",
    "ontarget_scoring_att_per90",
    "yellow_card_per90",
    "won_contest_per90",
    "total_att_assist_per90",
    "interception_per90",
    "saves_per90",
    "_goals_prevented_per90",
]


class RollingMeanFeature(Feature):
    """Rolling window mean of a per-90 stat, strictly historical (closed='left').

    ``shift(1)`` before ``rolling_mean`` ensures the window for row *i*
    contains only values from rows 0 … i-1 (sorted by season), never the
    current-season value.
    """

    stat_col: str
    window: int = 2
    missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.stat_col}_roll{self.window}"

    @property
    def required_columns(self) -> frozenset[str]:  # type: ignore[override]
        return frozenset([self.stat_col, "player_fotmob_id", "season_start"])

    def compute(self, data: pl.DataFrame) -> pl.Series:
        df = (
            data
            .with_row_index("__row_idx__")
            .sort(["player_fotmob_id", "season_start"])
            .with_columns(pl.col(self.stat_col).cast(pl.Float64).fill_null(0.0))
            .with_columns(
                pl.col(self.stat_col)
                .shift(1)
                .rolling_mean(window_size=self.window, min_samples=1)
                .over("player_fotmob_id")
                .alias(self.name)
            )
        )
        return df.sort("__row_idx__")[self.name]


class YoYDeltaFeature(Feature):
    """Year-over-year change in a per-90 stat (current season minus previous)."""

    stat_col: str
    missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.stat_col}_delta1"

    @property
    def required_columns(self) -> frozenset[str]:  # type: ignore[override]
        return frozenset([self.stat_col, "player_fotmob_id", "season_start"])

    def compute(self, data: pl.DataFrame) -> pl.Series:
        df = (
            data
            .with_row_index("__row_idx__")
            .sort(["player_fotmob_id", "season_start"])
            .with_columns(pl.col(self.stat_col).cast(pl.Float64).fill_null(0.0))
            .with_columns(
                pl.col(self.stat_col)
                .diff(1)
                .over("player_fotmob_id")
                .alias(self.name)
            )
        )
        return df.sort("__row_idx__")[self.name]


# ── Concrete instances ────────────────────────────────────────────────────────

ALL_ROLLING_FEATURES: list[Feature] = []
ALL_DELTA_FEATURES: list[Feature] = []

for _stat in _TREND_CANDIDATES:
    _roll = type(
        f"Roll_{_stat}",
        (RollingMeanFeature,),
        {"stat_col": _stat, "window": 2},
    )()
    _delta = type(
        f"Delta_{_stat}",
        (YoYDeltaFeature,),
        {"stat_col": _stat},
    )()
    ALL_ROLLING_FEATURES.append(_roll)
    ALL_DELTA_FEATURES.append(_delta)

ALL_TREND_FEATURES: list[Feature] = ALL_ROLLING_FEATURES + ALL_DELTA_FEATURES
